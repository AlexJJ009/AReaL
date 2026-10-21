# SPDX-License-Identifier: Apache-2.0
"""Explicit policy server ports without starting a model or a network client."""

import json
import sys
from dataclasses import asdict
from unittest.mock import Mock

import pytest
import yaml

from scripts.tau2 import checkpoint_eval as runner
from tests.test_tau2_evaluate import tau2_evaluate
from tests.test_tau2_ppo import _config


def test_policy_port_flags_must_be_supplied_together(tmp_path):
    """An incomplete explicit server port pair fails before loading a config."""
    argv = [
        "--config",
        "unused",
        "--model-path",
        "unused",
        "--output-root",
        str(tmp_path),
    ]
    with pytest.raises(SystemExit):
        tau2_evaluate.parse_args([*argv, "--policy-port", "21000"])
    args = tau2_evaluate.parse_args(
        [*argv, "--policy-port", "21000", "--policy-nccl-port", "21001"]
    )
    assert args.policy_port == 21000 and args.policy_nccl_port == 21001


def test_policy_server_fixed_ports_cleanup_on_initialization_failure(monkeypatch):
    """An evaluator-owned policy is stopped even if controller startup raises."""
    process = Mock()
    process.poll.return_value = None
    launch = Mock(return_value=process)
    monkeypatch.setattr(tau2_evaluate.SGLangBackend, "launch_server", launch)
    monkeypatch.setattr(tau2_evaluate, "wait_http", Mock())
    with pytest.raises(RuntimeError, match="controller failed"):
        with tau2_evaluate.policy_server(
            {"model_path": "local"}, port=21000, nccl_port=21001, timeout=10
        ) as infos:
            assert infos[0].port == 21000
            raise RuntimeError("controller failed")
    kwargs = launch.call_args.args[0]
    assert kwargs["port"] == 21000 and kwargs["nccl_port"] == 21001
    process.terminate.assert_called_once()
    process.wait.assert_called()


def test_exact_runner_overrides_load_synthetic_config_without_source_write(
    monkeypatch, tmp_path
):
    """Exercise real Hydra only on newly constructed test config, never a real run."""
    config = _config(monkeypatch, tmp_path / "template")
    run_dir = tmp_path / config.trial_name
    config_path = run_dir / "logs/root/tau2-ppo" / config.trial_name / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(asdict(config)))
    original = config_path.read_bytes()
    runtime = tmp_path / "runtime.json"
    runtime.write_text(
        json.dumps(
            {
                key: str(tmp_path / key)
                for key in (
                    "hf_hub_cache",
                    "sao_artifact_root",
                    "tau2_data_dir",
                    "ark_plan_env_file",
                )
            }
            | {"python": sys.executable}
        )
    )
    args = runner.parse_args([str(run_dir), "--runtime", str(runtime), "--steps", "0"])
    job = runner.build_plan(args)["jobs"][0]
    command = job["command"]
    start = command.index(str(runner.REPO / "scripts/tau2/evaluate.py")) + 1
    eval_args = tau2_evaluate.parse_args(command[start:] + ["--check-config"])
    loaded, _ = tau2_evaluate.load_eval_config(eval_args)
    assert config_path.read_bytes() == original
    assert loaded.cluster.fileroot == str(
        run_dir / "offline-checkpoint-eval/step-000/attempt-001"
    )
    assert loaded.stats_logger.fileroot == loaded.cluster.fileroot
    assert loaded.eval_episode_timeout_seconds == 900
    assert loaded.evaluation_rollout.max_concurrent_rollouts == 1
    assert loaded.econfig.user_llm == "qwen35-4b-user"
    assert (
        loaded.econfig.user_llm_args["extra_body"]["chat_template_kwargs"][
            "enable_thinking"
        ]
        is False
    )
    assert loaded.econfig.nl_judge_llm == "deepseek-v4-1-flash-261017"
