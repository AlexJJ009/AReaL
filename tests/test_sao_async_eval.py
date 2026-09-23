# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the SAO async evaluation adapter."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from scripts.sao.async_eval import AsyncEvalGRPOTrainer, SaoGRPOConfig

from areal.api.cli_args import parse_cli_args, to_structured_cfg

REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeController:
    def __init__(self, fail_wait: bool = False, wait_event=None, entered_event=None):
        self.config = SimpleNamespace(
            experiment_name="exp", trial_name="trial-dedicated-eval"
        )
        self.fail_wait = fail_wait
        self.wait_event = wait_event
        self.entered_event = entered_event
        self.initialized = []
        self.collective = []
        self.versions = []
        self.submits = []
        self.destroyed = False

    def initialize(self, **kwargs):
        self.initialized.append(kwargs)

    def _collective_rpc(self, method, **kwargs):
        self.collective.append((method, kwargs))

    def update_weights_from_disk(self, meta):  # pragma: no cover - must not be used
        raise AssertionError(
            "adapter must not call controller.update_weights_from_disk"
        )

    def get_version(self):
        return self.versions[-1] if self.versions else 0

    def set_version(self, version):
        self.versions.append(version)

    def submit(self, item, workflow, workflow_kwargs, **kwargs):
        self.submits.append(
            {
                "item": item,
                "workflow": workflow,
                "workflow_kwargs": workflow_kwargs,
                "kwargs": kwargs,
                "version": self.versions[-1],
            }
        )

    def wait(self, count, timeout=None):
        if self.entered_event is not None:
            self.entered_event.set()
        if self.wait_event is not None:
            self.wait_event.wait(timeout=5)
        if self.fail_wait:
            raise RuntimeError("eval failed")
        return [None] * count

    def destroy(self):
        self.destroyed = True


def _worker(role: str, index: int, gpus: list[int]):
    return SimpleNamespace(
        worker=SimpleNamespace(id=f"{role}/{index}"),
        role=role,
        gpu_devices=gpus,
    )


def _trainer(tmp_path: Path, controller: _FakeController | None = None):
    trainer = object.__new__(AsyncEvalGRPOTrainer)
    trainer.config = SimpleNamespace(
        actor=SimpleNamespace(path=str(tmp_path / "base-model")),
        cluster=SimpleNamespace(fileroot=str(tmp_path)),
        eval_gconfig=SimpleNamespace(n_samples=4),
        experiment_name="exp",
        trial_name="trial",
        train_dataset=SimpleNamespace(path=str(tmp_path / "dataset")),
    )
    trainer.valid_dataloader = [[{"source_id": "a"}, {"source_id": "b"}]]
    trainer._async_eval_rollout = controller or _FakeController()
    trainer._async_eval_executor = None
    trainer._async_eval_futures = []
    return trainer


def test_sao_grpo_config_parses_evaluation_rollout(monkeypatch, tmp_path):
    """The SAO YAML can be structured with the adapter config dataclass."""
    monkeypatch.setenv("SAO_TRIAL_NAME", "unit-test")
    monkeypatch.setenv("SAO_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.setenv("SAO_MODEL_PATH", "Qwen/Qwen3.5-4B-Base")
    monkeypatch.setenv("SAO_DATA_PATH", str(tmp_path / "dataset"))

    cfg, _ = parse_cli_args(
        ["--config", str(REPO_ROOT / "examples/math/sao_grpo.yaml")]
    )
    cfg = to_structured_cfg(cfg, SaoGRPOConfig)
    config = OmegaConf.to_object(cfg)

    assert isinstance(config, SaoGRPOConfig)
    assert config.evaluation_rollout.backend == "sglang:d1p1t1"
    assert config.evaluation_rollout.scheduling_strategy.type == "separation"


def test_dedicated_eval_controller_uses_separate_role_and_records_gpus(
    monkeypatch, tmp_path
):
    """The dedicated controller is launched as its own role after 4+3 GPUs."""
    fake = _FakeController()
    trainer = _trainer(tmp_path, fake)
    trainer.config.evaluation_rollout = SimpleNamespace(backend="sglang:d1p1t1")
    trainer.config.sglang = SimpleNamespace()
    trainer.scheduler = SimpleNamespace(
        gpu_devices=list(range(8)),
        _workers={
            "actor": [_worker("actor", i, [i]) for i in range(4)],
            "rollout": [_worker("rollout", i, [i + 4]) for i in range(3)],
        },
    )

    monkeypatch.setattr(
        "scripts.sao.async_eval.SGLangConfig.build_args",
        lambda **kwargs: {"tp_size": kwargs["tp_size"], "pp_size": kwargs["pp_size"]},
    )

    def fake_as_controller(config, scheduler):
        fake.config = config
        return fake

    monkeypatch.setattr(
        "scripts.sao.async_eval.RemoteSGLangEngine.as_controller",
        fake_as_controller,
    )

    trainer._init_dedicated_eval_rollout()
    trainer.scheduler._workers["dedicated-eval"] = [_worker("dedicated-eval", 0, [7])]
    trainer._write_gpu_allocation()

    assert fake.initialized == [
        {
            "role": "dedicated-eval",
            "server_args": {"tp_size": 1, "pp_size": 1},
        }
    ]
    assert fake.config.experiment_name == "exp"
    assert fake.config.trial_name == "trial-dedicated-eval"
    payload = json.loads((tmp_path / "evidence/gpu-allocation.json").read_text())
    assert payload["roles"]["actor"][0]["gpu_devices"] == [0]
    assert payload["roles"]["dedicated-eval"][0]["gpu_devices"] == [7]


def test_dedicated_eval_controller_is_owned_before_initialize(monkeypatch, tmp_path):
    """A launch failure still leaves the controller reachable for cleanup."""
    fake = _FakeController()
    trainer = _trainer(tmp_path, fake)
    trainer.config.evaluation_rollout = SimpleNamespace(backend="sglang:d1p1t1")
    trainer.config.sglang = SimpleNamespace()
    trainer.scheduler = SimpleNamespace(
        gpu_devices=list(range(8)),
        _workers={
            "actor": [_worker("actor", i, [i]) for i in range(4)],
            "rollout": [_worker("rollout", i, [i + 4]) for i in range(3)],
        },
    )

    def fail_initialize(**kwargs):
        raise RuntimeError("launch failed")

    fake.initialize = fail_initialize
    monkeypatch.setattr(
        "scripts.sao.async_eval.SGLangConfig.build_args",
        lambda **kwargs: {"tp_size": kwargs["tp_size"], "pp_size": kwargs["pp_size"]},
    )

    def fake_as_controller(config, scheduler):
        fake.config = config
        return fake

    monkeypatch.setattr(
        "scripts.sao.async_eval.RemoteSGLangEngine.as_controller",
        fake_as_controller,
    )

    with pytest.raises(RuntimeError, match="launch failed"):
        trainer._init_dedicated_eval_rollout()
    assert trainer._async_eval_rollout is fake


def test_native_eval_rollout_hook_only_disables_base_eval(monkeypatch, tmp_path):
    """The base eval rollout hook no longer launches the dedicated controller."""
    trainer = _trainer(tmp_path)
    monkeypatch.setattr(
        trainer,
        "_init_dedicated_eval_rollout",
        lambda: pytest.fail("dedicated eval should launch after base init"),
    )

    assert trainer._init_rollout(SimpleNamespace(), is_eval=True) is None


def test_dedicated_init_failure_happens_after_base_close_is_available(monkeypatch):
    """If dedicated eval launch fails, PPOTrainer.__init__ sees a closable base."""
    close_saw_saver = []
    config = SimpleNamespace(
        saver=SimpleNamespace(
            mode="sync", freq_steps=1, freq_epochs=None, freq_secs=None
        ),
        evaluator=SimpleNamespace(freq_steps=1, freq_epochs=None, freq_secs=None),
    )

    def fake_base_init(self, config, train_dataset=None, valid_dataset=None):
        self.config = config
        self._online_mode = False
        self.saver = object()

    def fail_dedicated_init(self):
        raise RuntimeError("dedicated launch failed")

    def record_close(self):
        close_saw_saver.append(hasattr(self, "saver"))

    monkeypatch.setattr("scripts.sao.async_eval.PPOTrainer._init_impl", fake_base_init)
    monkeypatch.setattr(
        AsyncEvalGRPOTrainer, "_init_dedicated_eval_rollout", fail_dedicated_init
    )
    monkeypatch.setattr(AsyncEvalGRPOTrainer, "close", record_close)

    with pytest.raises(RuntimeError, match="dedicated launch failed"):
        AsyncEvalGRPOTrainer(config)
    assert close_saw_saver == [True]


def test_dedicated_eval_rejects_wrapped_gpu_allocation(tmp_path):
    """The adapter fails before launching if actor+rollout did not consume 7 GPUs."""
    trainer = _trainer(tmp_path)
    trainer.scheduler = SimpleNamespace(
        gpu_devices=list(range(8)),
        _workers={
            "actor": [_worker("actor", i, [i]) for i in range(4)],
            "rollout": [_worker("rollout", i, [i + 4]) for i in range(2)],
        },
    )

    with pytest.raises(RuntimeError, match="consume 7 GPU slots"):
        trainer._assert_dedicated_eval_preconditions()


def _record_checkpoint_ready(monkeypatch):
    calls = []

    def fake_delete(name):
        calls.append(("delete", name, None))

    def fake_add(name, value, keepalive_ttl=None):
        calls.append(("add", name, keepalive_ttl))

    monkeypatch.setattr("scripts.sao.async_eval.name_resolve.delete", fake_delete)
    monkeypatch.setattr("scripts.sao.async_eval.name_resolve.add", fake_add)
    return calls


def test_eval_jobs_load_checkpoints_without_deleting_and_pin_versions(
    monkeypatch, tmp_path
):
    """Baseline and checkpoint evals use isolated eval readiness keys."""
    controller = _FakeController()
    trainer = _trainer(tmp_path, controller)
    ready_calls = _record_checkpoint_ready(monkeypatch)
    monkeypatch.setenv("SAO_PREFLIGHT", "1")

    trainer._run_eval_job(0, trainer.config.actor.path, "workflow", {"x": 1}, False)
    checkpoint = tmp_path / "checkpoints/exp/trial/default/epoch0epochstep0globalstep19"
    checkpoint.mkdir(parents=True)
    trainer._run_eval_job(20, str(checkpoint), "workflow", {"x": 1}, False)

    assert checkpoint.exists()
    assert [call[0] for call in controller.collective] == [
        "update_weights_from_disk",
        "update_weights_from_disk",
    ]
    metas = [call[1]["meta"] for call in controller.collective]
    assert [meta.version for meta in metas] == [0, 20]
    assert all(meta.clear_checkpoint_after_load is False for meta in metas)
    assert controller.versions == [0, 20]
    assert [call[0] for call in ready_calls] == ["delete", "add", "delete", "add"]
    assert [call[1] for call in ready_calls] == [
        "root/exp/trial-dedicated-eval/update_weights_from_disk/0",
        "root/exp/trial-dedicated-eval/update_weights_from_disk/0",
        "root/exp/trial-dedicated-eval/update_weights_from_disk/0",
        "root/exp/trial-dedicated-eval/update_weights_from_disk/0",
    ]
    assert [call[2] for call in ready_calls if call[0] == "add"] == [120, 120]
    assert {row["version"] for row in controller.submits[:2]} == {0}
    assert {row["version"] for row in controller.submits[2:]} == {20}
    assert (
        json.loads((tmp_path / "evidence/async-eval/20.json").read_text())["status"]
        == "completed"
    )


@pytest.mark.parametrize("n_samples", [2, 4])
def test_snapshot_uses_exact_version_whitelist(monkeypatch, tmp_path, n_samples):
    """Snapshot validation accepts baseline/final versions queued by the adapter."""
    import scripts.sao.snapshot_eval as snapshot_module

    calls = []
    trainer = _trainer(tmp_path, _FakeController())
    trainer.config.eval_gconfig.n_samples = n_samples
    _record_checkpoint_ready(monkeypatch)
    monkeypatch.setattr(
        snapshot_module,
        "snapshot_eval",
        lambda evidence, dataset, version, *, allowed_versions, n_samples: calls.append(
            (evidence, dataset, version, allowed_versions, n_samples)
        ),
    )

    trainer._run_eval_job(0, trainer.config.actor.path, "workflow", {}, True)

    assert calls[0][2:] == (0, (0,), n_samples)


def test_async_queue_does_not_change_version_during_running_eval(monkeypatch, tmp_path):
    """The single worker queue prevents later evals from mutating an active version."""
    release = threading.Event()
    entered = threading.Event()
    controller = _FakeController(wait_event=release, entered_event=entered)
    trainer = _trainer(tmp_path, controller)
    _record_checkpoint_ready(monkeypatch)
    trainer._async_eval_executor = ThreadPoolExecutor(max_workers=1)

    trainer._enqueue_eval(
        version=1,
        checkpoint_path=trainer.config.actor.path,
        eval_workflow="workflow",
        eval_workflow_kwargs={},
        snapshot=False,
    )
    assert entered.wait(timeout=5)
    trainer._enqueue_eval(
        version=2,
        checkpoint_path=trainer.config.actor.path,
        eval_workflow="workflow",
        eval_workflow_kwargs={},
        snapshot=False,
    )

    assert controller.versions == [1]
    release.set()
    trainer._drain_evaluations()
    trainer._async_eval_executor.shutdown(wait=True, cancel_futures=True)
    assert controller.versions == [1, 2]


def test_eval_failure_is_recorded_and_propagated(monkeypatch, tmp_path):
    """A failed async job writes failure evidence and re-raises through futures."""
    controller = _FakeController(fail_wait=True)
    trainer = _trainer(tmp_path, controller)
    _record_checkpoint_ready(monkeypatch)

    with pytest.raises(RuntimeError, match="eval failed"):
        trainer._run_eval_job(3, trainer.config.actor.path, "workflow", {}, False)

    payload = json.loads((tmp_path / "evidence/async-eval/3.json").read_text())
    assert payload["status"] == "failed"
    assert payload["error_type"] == "RuntimeError"
