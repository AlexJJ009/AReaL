# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "examples/swe/qwen38_flash_next"


def load_entry():
    spec = importlib.util.spec_from_file_location(
        "qwen_swe_entry", RECIPE / "runtime/train256_claude_recover_isolated.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    return entry


def test_pinned_swe_subset_excludes_heldout_and_rejects_inventory_drift():
    entry = load_entry()
    split = json.loads((RECIPE / "fixtures/split.json").read_text())
    selected = json.loads((RECIPE / "fixtures/parallel-canary-split.json").read_text())[
        "selected"
    ]
    split["rl_acceptance_ids"] = selected
    rows = [{"data_id": key} for key in split["all_data_ids"]]
    actual = entry.select_training_rows(rows, split)
    assert [row["data_id"] for row in actual] == selected
    assert len(actual) == 16
    assert not set(selected).intersection(split["heldout"])
    with pytest.raises(ValueError, match="Live inventory"):
        entry.select_training_rows(rows[:-1], split)


@pytest.mark.parametrize("profile", ["swe", "rlvr"])
def test_submit_preserves_arguments_and_packages_runtime(tmp_path, profile):
    recorder = tmp_path / "sbatch"
    capture = tmp_path / "captured.json"
    recorder.write_text(
        '#!/usr/bin/env python3\nimport json, os, sys\nfrom pathlib import Path\nPath(os.environ["CAPTURE"]).write_text(json.dumps({"args":sys.argv[1:],"actor":os.environ["QWEN_ACTOR_PYTHONPATH"],"rollout":os.environ["QWEN_ROLLOUT_PYTHONPATH"]}))\n'
    )
    recorder.chmod(0o755)
    fixture = tmp_path / "acceptance.json"
    fixture.write_text("{}")
    env = dict(
        os.environ,
        PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
        CAPTURE=str(capture),
    )
    for key in (
        "QWEN_MODEL",
        "QWEN_ACTOR_IMAGE",
        "QWEN_ROLLOUT_IMAGE",
        "QWEN_RESERVATION",
        "QWEN_NODELIST",
        "QWEN_PARTITION",
        "QWEN_CONTROLLER_NODE",
        "QWEN_MOUNTS",
        "QWEN_CONTROLLER_MOUNTS",
        "MCORE_BRIDGE_ROOT",
        "MEGATRON_ROOT",
        "QWEN_GSM8K_DATA",
    ):
        env[key] = str(tmp_path / key)
    for key in (
        "QWEN_PRIVATE_ENV",
        "QWEN_RECOVER_SOURCE",
        "QWEN_REPLAY64_ACCEPTANCE",
        "QWEN_CC_PROTOCOL_ACCEPTANCE",
    ):
        env[key] = str(fixture)
    env["QWEN_OUTPUT_ROOT"] = str(tmp_path / "output with spaces")
    env["QWEN_REPO"] = str(ROOT)
    env.pop("QWEN_LAUNCH_ENV", None)
    env.pop("BASH_ENV", None)
    subprocess.run(
        ["bash", str(RECIPE / "submit_rl.sh"), profile, "trial_name=trial with spaces"],
        env=env,
        check=True,
    )
    result = json.loads(capture.read_text())
    assert result["args"][-2:] == [profile, "trial_name=trial with spaces"]
    assert result["actor"].split(os.pathsep)[0] == str(RECIPE / "runtime")
    assert env["MEGATRON_ROOT"] in result["rollout"].split(os.pathsep)


def test_rl_profiles_preserve_sampling_and_memory_settings(monkeypatch):
    import re

    from examples.swe.utils import SWEPPOConfig

    from areal.api.cli_args import GRPOConfig, to_structured_cfg

    for name in ("swe_rl_256k.yaml", "rlvr_gsm8k_256k.yaml"):
        source = (RECIPE / name).read_text()
        for key in re.findall(r"\$\{oc.env:([A-Za-z_][A-Za-z_0-9]*)(?:[,}])", source):
            monkeypatch.setenv(key, "fixture")
        monkeypatch.setenv("TOTAL_TRAIN_STEPS", "500")
        schema = SWEPPOConfig if name.startswith("swe") else GRPOConfig
        structured = OmegaConf.to_object(
            to_structured_cfg(OmegaConf.load(RECIPE / name), schema)
        )
        assert structured.actor.min_usable_group_size == 8
        assert structured.actor.reward_norm is None
        assert structured.gconfig.reward_normalization
        assert not structured.gconfig.reward_normalization_use_std
        assert (
            "reward_normalization_use_std"
            not in structured.gconfig.to_openai_args_dict()
        )
        assert structured.total_train_steps == 500
        config = OmegaConf.to_container(OmegaConf.load(RECIPE / name), resolve=True)
        assert (
            config["gconfig"]["max_tokens"]
            == config["sglang"]["context_length"]
            == 262144
        )
        assert config["gconfig"]["max_new_tokens"] == 65536
        assert config["gconfig"]["temperature"] == 1
        assert (
            config["gconfig"]["n_samples"] * config["train_dataset"]["batch_size"]
            == 128
        )
        assert config["sglang"]["mem_fraction_static"] == 0.65
        assert config["actor"]["megatron"]["freeze_ple_table"] is True


def test_swe_workflow_kwargs_preserve_sampling_arguments():
    from examples.swe.utils import SWEPPOConfig

    config = SWEPPOConfig()
    config.gconfig.temperature = 0.7
    config.gconfig.top_p = 0.8
    config.gconfig.top_k = 42
    config.gconfig.max_new_tokens = 65536
    config.econfig.timeout = 1800
    kwargs = load_entry().build_workflow_kwargs(config)
    assert kwargs["gen_args"] == dict(
        temperature=0.7, top_p=0.8, top_k=42, max_completion_tokens=65536
    )
    assert kwargs["timeout"] == kwargs["econfig"]["timeout"] == 1800


def test_thinking_defaults_respect_explicit_switch_without_mutation():
    spec = importlib.util.spec_from_file_location(
        "qwen_template_defaults", RECIPE / "runtime/qwen_template_defaults.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.with_template_defaults()["chat_template_kwargs"] == dict(
        enable_thinking=True, reasoning_effort="medium", thinking_option=None
    )
    body = {"chat_template_kwargs": {"thinking_option": "off"}, "other": 42}
    merged = module.with_template_defaults(body)
    assert merged["chat_template_kwargs"] == dict(
        thinking_option="off", reasoning_effort="medium"
    )
    assert merged["other"] == 42
    assert body == {"chat_template_kwargs": {"thinking_option": "off"}, "other": 42}


def test_swe_identity_reward_preserves_partial_scores_and_rejects_invalid_values():
    from examples.swe.reward_transforms import identity_reward

    assert identity_reward(0.4, {}, reward_threshold=0.98) == 0.4
    for value in (float("nan"), float("inf"), -0.1, 1.1):
        with pytest.raises(ValueError, match="within"):
            identity_reward(value, {})
