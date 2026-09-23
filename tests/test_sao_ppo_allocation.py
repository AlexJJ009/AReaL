# SPDX-License-Identifier: Apache-2.0
"""CPU-only allocation tests for the SAO PPO 3/4+critic/1 layout."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.sao.async_eval import AsyncEvalPPOTrainer

from areal.api import Job, Worker
from areal.api.cli_args import SchedulingSpec, SchedulingStrategy
from areal.infra.scheduler.local import LocalScheduler, WorkerInfo


class _FakeController:
    def __init__(self):
        self.config = None
        self.initialized = []

    def initialize(self, **kwargs):
        self.initialized.append(kwargs)


class _FakeProcess:
    pid = 12345
    returncode = None

    def poll(self):
        return None


def _trainer(tmp_path: Path):
    trainer = object.__new__(AsyncEvalPPOTrainer)
    trainer.config = SimpleNamespace(
        cluster=SimpleNamespace(fileroot=str(tmp_path)),
        evaluation_rollout=SimpleNamespace(backend="sglang:d1p1t1"),
        experiment_name="exp",
        sglang=SimpleNamespace(),
        trial_name="ppo431",
    )
    trainer._async_eval_rollout = None
    trainer._async_eval_executor = None
    return trainer


def _install_cpu_process_stubs(monkeypatch):
    monkeypatch.setattr(
        "areal.infra.scheduler.local.run_with_streaming_logs",
        lambda *args, **kwargs: _FakeProcess(),
    )
    monkeypatch.setattr("areal.infra.scheduler.local.time.sleep", lambda _: None)


def _install_cpu_fork_workers(monkeypatch):
    def fake_fork_workers(self, role: str, target_role: str, command=None):
        workers = []
        for index, target in enumerate(self._workers[target_role]):
            worker = Worker(
                id=f"{role}/{index}",
                ip=target.worker.ip,
                worker_ports=[str(31000 + index)],
                engine_ports=[],
            )
            workers.append(
                WorkerInfo(
                    worker=worker,
                    process=None,
                    role=role,
                    gpu_devices=list(target.gpu_devices),
                    created_at=time.time(),
                    log_file=str(self.log_dir / f"{role}.log"),
                    env_vars=target.env_vars.copy(),
                )
            )
        self._workers[role] = workers
        self._colocated_roles[role] = target_role
        return [info.worker.id for info in workers]

    monkeypatch.setattr(LocalScheduler, "fork_workers", fake_fork_workers)


def _job(role: str, replicas: int, *, target: str | None = None) -> Job:
    strategy = (
        SchedulingStrategy(type="colocation", target=target)
        if target is not None
        else SchedulingStrategy(type="separation")
    )
    return Job(
        role=role,
        replicas=replicas,
        tasks=[
            SchedulingSpec(
                task_type="worker",
                gpu=1,
                port_count=2,
                cmd="python -m areal.infra.rpc.rpc_server",
            )
        ],
        scheduling_strategy=strategy,
    )


def _create_local_scheduler(monkeypatch, tmp_path: Path, visible_gpus: list[int]):
    _install_cpu_process_stubs(monkeypatch)
    _install_cpu_fork_workers(monkeypatch)
    return LocalScheduler(gpu_devices=visible_gpus, log_dir=str(tmp_path / "logs"))


def _create_ppo_workers(
    scheduler: LocalScheduler,
    *,
    rollout_replicas: int = 3,
    create_dedicated: bool = False,
) -> None:
    scheduler.create_workers(_job("actor", 4))
    scheduler.create_workers(_job("critic", 4, target="actor"))
    scheduler.create_workers(_job("rollout", rollout_replicas))
    if create_dedicated:
        scheduler.create_workers(_job("dedicated-eval", 1))


def test_ppo_dedicated_eval_records_three_four_colocated_one_layout(
    monkeypatch, tmp_path
):
    trainer = _trainer(tmp_path)
    trainer.scheduler = _create_local_scheduler(monkeypatch, tmp_path, list(range(8)))
    _create_ppo_workers(trainer.scheduler)
    fake = _FakeController()

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
    trainer.scheduler.create_workers(_job("dedicated-eval", 1))
    trainer._write_gpu_allocation()

    payload = json.loads((tmp_path / "evidence/gpu-allocation.json").read_text())
    roles = payload["roles"]
    actor_gpus = [row["gpu_devices"][0] for row in roles["actor"]]
    critic_gpus = [row["gpu_devices"][0] for row in roles["critic"]]
    rollout_gpus = [row["gpu_devices"][0] for row in roles["rollout"]]
    eval_gpus = [row["gpu_devices"][0] for row in roles["dedicated-eval"]]

    assert fake.initialized == [
        {
            "role": "dedicated-eval",
            "server_args": {"tp_size": 1, "pp_size": 1},
        }
    ]
    assert fake.config.trial_name == "ppo431-dedicated-eval"
    assert actor_gpus == [0, 1, 2, 3]
    assert critic_gpus == actor_gpus
    assert rollout_gpus == [4, 5, 6]
    assert eval_gpus == [7]
    assert sorted(set(actor_gpus + rollout_gpus + eval_gpus)) == list(range(8))


@pytest.mark.parametrize(
    ("visible_gpus", "rollout_replicas", "match"),
    [
        (
            list(range(7)),
            3,
            "requires >=8 visible GPUs",
        ),
        (
            list(range(8)),
            4,
            "consume 7 GPU slots",
        ),
    ],
)
def test_ppo_dedicated_eval_rejects_invalid_scheduler_allocation(
    monkeypatch, tmp_path, visible_gpus, rollout_replicas, match
):
    trainer = _trainer(tmp_path)
    trainer.scheduler = _create_local_scheduler(monkeypatch, tmp_path, visible_gpus)
    _create_ppo_workers(trainer.scheduler, rollout_replicas=rollout_replicas)

    with pytest.raises(RuntimeError, match=match):
        trainer._assert_dedicated_eval_preconditions()
