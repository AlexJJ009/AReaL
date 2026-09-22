"""The launch audit follows the resolved recipe, not historical N4 constants."""

import importlib.util
import json
from pathlib import Path

import pytest


def _launcher(monkeypatch, tmp_path):
    monkeypatch.setenv("SAO_LAUNCH_DIR", str(tmp_path))
    path = Path(__file__).resolve().parents[1] / "scripts/sao/launch_grpo_run.py"
    spec = importlib.util.spec_from_file_location("grpo_launch_recipe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_launch_audit_accepts_n8_and_rejects_incomplete_group(monkeypatch, tmp_path):
    launcher = _launcher(monkeypatch, tmp_path)
    evidence = tmp_path / "evidence"
    _write(
        evidence / "step-counts.json",
        {
            "expected_optimizer_steps": 1,
            "train_prompts_per_step": 2,
            "samples_per_prompt": 8,
            "resolved_warmup_steps": 0,
        },
    )
    _write(
        evidence / "resolved-config.json",
        {
            "gconfig": {"n_samples": 8},
            "actor": {"optimizer": {"lr": 1e-6}},
        },
    )
    _write(evidence / "epoch-finished.json", {})
    metrics = {
        "ppo_actor/update/update_successful": 1,
        "ppo_actor/update/lr": 1e-6,
        "timeperf/recompute_logp": 1,
        "ppo_actor/update/behave_imp_weight/avg": 1,
    }
    _write(evidence / "steps/1.json", {"published_version": 1, "metrics": metrics})
    rows = [
        {"audit_source_key": p, "audit_sample_idx": n}
        for p in range(2)
        for n in range(8)
    ]
    _write(evidence / "consumed/1.json", rows)
    assert launcher.verify_steps(tmp_path)["n_samples"] == 8
    _write(evidence / "consumed/1.json", rows[:-1])
    with pytest.raises(RuntimeError, match="Broken N8"):
        launcher.verify_steps(tmp_path)
    _write(evidence / "consumed/1.json", rows)
    metrics["ppo_actor/update/lr"] = 0.0
    _write(evidence / "steps/1.json", {"published_version": 1, "metrics": metrics})
    with pytest.raises(RuntimeError, match="learning-rate"):
        launcher.verify_steps(tmp_path)


def test_launch_eval_schedule_includes_new_epoch_end(monkeypatch, tmp_path):
    launcher = _launcher(monkeypatch, tmp_path)
    _write(tmp_path / "evidence/step-counts.json", {"expected_optimizer_steps": 536})
    _write(
        tmp_path / "evidence/resolved-config.json",
        {
            "evaluator": {
                "freq_steps": 20,
                "freq_epochs": 1,
                "eval_before_train": True,
            }
        },
    )
    assert launcher.evaluation_versions(tmp_path) == (0, *range(20, 537, 20), 536)
