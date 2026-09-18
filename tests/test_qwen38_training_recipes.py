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
        "QWEN_ARENA_STREAMS_FILE",
        "QWEN_AWEX_FROZEN_CONTRACT",
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
    assert str(RECIPE / "runtime") not in result["actor"].split(os.pathsep)
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
        assert config["sglang"]["mem_fraction_static"] == (
            0.70 if name.startswith("swe") else 0.65
        )
        assert config["actor"]["megatron"]["freeze_ple_table"] is True


def test_thinking_defaults_respect_explicit_switch_without_mutation():
    spec = importlib.util.spec_from_file_location(
        "qwen_template_defaults", RECIPE / "template_defaults.py"
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


@pytest.mark.parametrize("split_mode", ["pair", "trajectory"])
def test_sft_structured_schema_accepts_supported_split_modes(split_mode):
    from examples.swe.config import SweSFTConfig

    from areal.api.cli_args import to_structured_cfg

    config = OmegaConf.to_object(
        to_structured_cfg(
            OmegaConf.create({"swe": {"split_mode": split_mode}}), SweSFTConfig
        ).swe
    )
    assert config.split_mode == split_mode


def test_sft_structured_schema_rejects_invalid_split_mode():
    from examples.swe.config import SweSFTConfig

    from areal.api.cli_args import to_structured_cfg

    with pytest.raises(ValueError, match="split_mode must be either"):
        OmegaConf.to_object(
            to_structured_cfg(
                OmegaConf.create({"swe": {"split_mode": "typo"}}), SweSFTConfig
            ).swe
        )


def test_external_task_selection_preserves_order_and_rejects_drift():
    from examples.swe.qwen38_flash_next.train_rl import select_task_indices

    assert select_task_indices(["a", "b", "c"], ["c", "a"]) == [2, 0]
    for selected in ([], ["a", "a"], ["missing"], "a", [None]):
        with pytest.raises(ValueError):
            select_task_indices(["a", "b"], selected)


def test_cache_isolation_preserves_original_request():
    from types import SimpleNamespace

    from examples.swe.qwen38_flash_next.proxy import wrap_isolated_cache

    original = SimpleNamespace(payload={"input_ids": [1, 2]})
    build = wrap_isolated_cache(lambda _: original)
    first, second = build(None), build(None)
    assert first.payload["cache_salt"] != second.payload["cache_salt"]
    assert first.payload["input_ids"] == original.payload["input_ids"]
    assert "cache_salt" not in original.payload
