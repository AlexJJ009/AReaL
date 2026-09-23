# SPDX-License-Identifier: Apache-2.0
"""Dedicated async evaluation controller for SAO math training."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import os
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from areal import PPOTrainer
from areal.api import WeightUpdateMeta
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    GRPOConfig,
    InferenceEngineConfig,
    PPOConfig,
    SGLangConfig,
)
from areal.engine import RemoteSGLangEngine
from areal.utils import name_resolve, names
from areal.utils.saver import Saver


@dataclasses.dataclass
class SaoGRPOConfig(GRPOConfig):
    """SAO GRPO config with a separate rollout engine for eval."""

    evaluation_rollout: InferenceEngineConfig = dataclasses.field(
        default_factory=lambda: InferenceEngineConfig(backend="sglang:d1p1t1")
    )


@dataclasses.dataclass
class SaoPPOConfig(PPOConfig):
    """SAO PPO config with a separate rollout engine for eval."""

    evaluation_rollout: InferenceEngineConfig = dataclasses.field(
        default_factory=lambda: InferenceEngineConfig(backend="sglang:d1p1t1")
    )


class AsyncEvalGRPOTrainer(PPOTrainer):
    """PPO trainer variant that queues eval on a dedicated SGLang controller."""

    _DEDICATED_ROLE = "dedicated-eval"

    def _init_impl(self, config, train_dataset=None, valid_dataset=None):
        self._async_eval_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._async_eval_futures: list[concurrent.futures.Future] = []
        self._async_eval_rollout = None
        self._validate_save_eval_sync(config)
        super()._init_impl(config, train_dataset, valid_dataset)
        if not self._online_mode:
            self._init_dedicated_eval_rollout()
            self._write_gpu_allocation()

    @staticmethod
    def _validate_save_eval_sync(config: SaoGRPOConfig | SaoPPOConfig) -> None:
        saver = config.saver
        evaluator = config.evaluator
        fields = ("freq_steps", "freq_epochs", "freq_secs")
        mismatches = {
            field: (getattr(saver, field), getattr(evaluator, field))
            for field in fields
            if getattr(saver, field) != getattr(evaluator, field)
        }
        if mismatches:
            raise ValueError(
                "SAO async eval requires saver and evaluator frequencies to match "
                f"so checkpoints exist before queueing eval: {mismatches}"
            )
        if config.saver.mode != "sync":
            raise ValueError("SAO async eval requires saver.mode='sync'")

    def _init_scheduler(self):
        scheduler = super()._init_scheduler()
        devices = getattr(scheduler, "gpu_devices", None)
        if devices is not None and len(devices) < 8:
            raise RuntimeError(
                "SAO async evaluation requires at least 8 visible GPUs before "
                f"worker launch; scheduler sees {len(devices)}: {devices}"
            )
        return scheduler

    def _init_rollout(
        self,
        rollout_config: InferenceEngineConfig,
        is_eval: bool = False,
        lora_path: str | None = None,
    ):
        if not is_eval:
            controller = super()._init_rollout(
                rollout_config, is_eval=False, lora_path=lora_path
            )
            self._write_gpu_allocation()
            return controller

        return None

    def _init_dedicated_eval_rollout(self) -> None:
        self._assert_dedicated_eval_preconditions()
        config = deepcopy(self.config.evaluation_rollout)
        config.experiment_name = self.config.experiment_name
        config.trial_name = f"{self.config.trial_name}-dedicated-eval"
        config.max_head_offpolicyness = int(1e12)
        alloc = ModelAllocation.from_str(config.backend, name=self._DEDICATED_ROLE)
        server_args = SGLangConfig.build_args(
            sglang_config=self.config.sglang,
            tp_size=alloc.parallel.tp_size,
            pp_size=alloc.parallel.pp_size,
            base_gpu_id=0,
        )
        controller = RemoteSGLangEngine.as_controller(config, self.scheduler)
        self._async_eval_rollout = controller
        controller.initialize(
            role=self._DEDICATED_ROLE,
            server_args=server_args,
        )
        self._async_eval_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="sao-async-eval",
        )

    def _assert_dedicated_eval_preconditions(self) -> None:
        scheduler = self.scheduler
        if scheduler is None:
            raise RuntimeError("Dedicated async eval requires a controller scheduler")
        workers = getattr(scheduler, "_workers", {})
        devices = getattr(scheduler, "gpu_devices", [])
        if len(devices) < 8:
            raise RuntimeError(
                "Dedicated async eval requires >=8 visible GPUs, got "
                f"{len(devices)}: {devices}"
            )
        consumed = sum(
            len(info.gpu_devices)
            for role in ("actor", "rollout")
            for info in workers.get(role, [])
        )
        if consumed != 7:
            raise RuntimeError(
                "Dedicated async eval must launch after actor+rollout consume 7 "
                f"GPU slots; observed {consumed} from roles {sorted(workers)}"
            )

    def _write_gpu_allocation(self) -> None:
        scheduler = getattr(self, "scheduler", None)
        if scheduler is None or not hasattr(self, "config"):
            return
        workers = getattr(scheduler, "_workers", None)
        if workers is None:
            return
        payload = {
            "recorded_ns": time.time_ns(),
            "visible_gpus": list(getattr(scheduler, "gpu_devices", [])),
            "roles": {
                role: [
                    {
                        "worker_id": info.worker.id,
                        "role": info.role,
                        "gpu_devices": list(info.gpu_devices),
                    }
                    for info in infos
                ]
                for role, infos in sorted(workers.items())
            },
        }
        path = Path(self.config.cluster.fileroot) / "evidence" / "gpu-allocation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _evaluate_fn(self, eval_workflow, eval_workflow_kwargs):
        self._enqueue_eval(
            version=0,
            checkpoint_path=self.config.actor.path,
            eval_workflow=eval_workflow,
            eval_workflow_kwargs=eval_workflow_kwargs,
            snapshot=not self._is_preflight(),
        )

    def _evaluate(
        self,
        eval_workflow,
        eval_workflow_kwargs,
        epoch: int,
        epoch_step: int,
        global_step: int,
    ):
        self.check_evaluation()
        if self.valid_dataloader is None or eval_workflow is None:
            return

        def queue_checkpoint_eval() -> None:
            checkpoint_path = Saver.get_model_save_path(
                self.config.experiment_name,
                self.config.trial_name,
                self.config.cluster.fileroot,
                epoch,
                epoch_step,
                global_step,
            )
            self._enqueue_eval(
                version=global_step + 1,
                checkpoint_path=checkpoint_path,
                eval_workflow=eval_workflow,
                eval_workflow_kwargs=eval_workflow_kwargs,
                snapshot=not self._is_preflight(),
            )

        self.evaluator.evaluate(queue_checkpoint_eval, epoch, epoch_step, global_step)
        self.check_evaluation()

    def _enqueue_eval(
        self,
        *,
        version: int,
        checkpoint_path: str,
        eval_workflow,
        eval_workflow_kwargs,
        snapshot: bool,
    ) -> None:
        self.check_evaluation()
        if self._async_eval_executor is None or self._async_eval_rollout is None:
            raise RuntimeError("Dedicated async eval controller is not initialized")
        future = self._async_eval_executor.submit(
            self._run_eval_job,
            version,
            checkpoint_path,
            eval_workflow,
            eval_workflow_kwargs,
            snapshot,
        )
        self._async_eval_futures.append(future)

    def _run_eval_job(
        self,
        version: int,
        checkpoint_path: str,
        eval_workflow,
        eval_workflow_kwargs,
        snapshot: bool,
    ) -> None:
        try:
            self._load_eval_checkpoint(checkpoint_path, version)
            count = 0
            for data in self.valid_dataloader:
                for item in data:
                    self._async_eval_rollout.submit(
                        item,
                        eval_workflow,
                        eval_workflow_kwargs,
                        group_size=self.config.eval_gconfig.n_samples,
                        is_eval=True,
                        reward_normalization=False,
                        drop_incomplete_group=False,
                    )
                    count += 1
            self._async_eval_rollout.wait(count, timeout=None)
            if snapshot:
                from scripts.sao.snapshot_eval import snapshot_eval

                snapshot_eval(
                    Path(self.config.cluster.fileroot) / "evidence",
                    Path(self.config.train_dataset.path),
                    version,
                    allowed_versions=(version,),
                    n_samples=self.config.eval_gconfig.n_samples,
                )
            self._write_eval_status(
                version,
                {
                    "status": "completed",
                    "version": version,
                    "checkpoint_path": str(checkpoint_path),
                    "completed_ns": time.time_ns(),
                    "submitted_prompts": count,
                },
            )
        except BaseException as exc:
            self._write_eval_status(
                version,
                {
                    "status": "failed",
                    "version": version,
                    "checkpoint_path": str(checkpoint_path),
                    "failed_ns": time.time_ns(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise

    def _publish_eval_checkpoint_ready(self, wait_version: int) -> str:
        """Publish the disk checkpoint readiness key expected by rollout workers.

        Remote inference engines wait on the current rollout version's
        ``update_weights_from_disk`` name before sending the HTTP load request.
        Normal training updates publish this key from the training engine after
        saving the checkpoint. Dedicated async eval bypasses that training engine,
        so the eval trainer must publish the same readiness signal itself once it
        is about to load an already-saved checkpoint.
        """

        rollout_config = self._async_eval_rollout.config
        update_name = names.update_weights_from_disk(
            rollout_config.experiment_name,
            rollout_config.trial_name,
            wait_version,
        )
        try:
            name_resolve.delete(update_name)
        except Exception:
            pass
        name_resolve.add(
            update_name, str(datetime.now().timestamp()), keepalive_ttl=120
        )
        return update_name

    def _load_eval_checkpoint(self, checkpoint_path: str, version: int) -> None:
        meta = WeightUpdateMeta(
            type="disk",
            path=str(checkpoint_path),
            version=version,
            clear_checkpoint_after_load=False,
        )
        wait_version = self._async_eval_rollout.get_version()
        self._publish_eval_checkpoint_ready(wait_version)
        self._async_eval_rollout._collective_rpc(
            "update_weights_from_disk",
            meta=meta,
        )
        self._async_eval_rollout.set_version(version)

    def _write_eval_status(self, version: int, payload: dict[str, Any]) -> None:
        path = (
            Path(self.config.cluster.fileroot)
            / "evidence"
            / "async-eval"
            / f"{version}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _is_preflight(self) -> bool:
        return os.environ.get("SAO_PREFLIGHT", "0") == "1"

    def check_evaluation(self) -> None:
        pending = []
        for future in self._async_eval_futures:
            if future.done():
                future.result()
            else:
                pending.append(future)
        self._async_eval_futures = pending

    def _cancel_queued_evaluations(self) -> None:
        self._async_eval_futures = [
            future
            for future in self._async_eval_futures
            if future.done() or future.running() or not future.cancel()
        ]

    def _drain_evaluations(self) -> None:
        first_error: BaseException | None = None
        for future in self._async_eval_futures:
            try:
                future.result()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                for queued in self._async_eval_futures:
                    if not queued.done():
                        queued.cancel()
        self._async_eval_futures = []
        if first_error is not None:
            raise first_error

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None and hasattr(self, "_async_eval_futures"):
            self._cancel_queued_evaluations()
        return super().__exit__(exc_type, exc_value, traceback)

    def close(self):
        first_error: BaseException | None = None
        try:
            if hasattr(self, "_async_eval_executor"):
                self._drain_evaluations()
        except BaseException as exc:
            first_error = exc
        finally:
            executor = getattr(self, "_async_eval_executor", None)
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
                self._async_eval_executor = None
            controller = getattr(self, "_async_eval_rollout", None)
            if controller is not None:
                try:
                    controller.destroy()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                self._async_eval_rollout = None
            try:
                if hasattr(self, "saver"):
                    super().close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


# The dedicated evaluator only uses actor snapshots; PPO retains its critic.
AsyncEvalPPOTrainer = AsyncEvalGRPOTrainer
