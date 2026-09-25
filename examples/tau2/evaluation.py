# SPDX-License-Identifier: Apache-2.0
"""Official tau2 evaluation on the existing dedicated checkpoint evaluator."""

import time
from collections import defaultdict

from scripts.sao.async_eval import AsyncEvalPPOTrainer

from areal.infra.rpc.rtensor import RTensor


class Tau2AsyncEvalTrainer(AsyncEvalPPOTrainer):
    """Reuse GPU allocation/checkpoint loading; summarize official domain rewards."""

    _snapshot_evidence = False

    def _init_dedicated_eval_rollout(self):
        super()._init_dedicated_eval_rollout()
        self._async_eval_rollout.start_proxy()

    def _run_eval_job(
        self, version, checkpoint_path, eval_workflow, eval_workflow_kwargs, snapshot
    ):
        try:
            self._load_eval_checkpoint(checkpoint_path, version)
            by_domain = defaultdict(list)
            for batch in self.valid_dataloader:
                for row in batch:
                    by_domain[row["domain"]].append(row)
            metrics = {}
            for domain, rows in by_domain.items():
                for row in rows:
                    self._async_eval_rollout.submit(
                        row,
                        eval_workflow,
                        eval_workflow_kwargs,
                        group_size=self.config.eval_gconfig.n_samples,
                        is_eval=True,
                        reward_normalization=False,
                        drop_incomplete_group=False,
                    )
                results = self._async_eval_rollout.wait(len(rows), timeout=None)
                if len(results) != len(rows) or any(row is None for row in results):
                    raise RuntimeError(
                        f"Incomplete {domain} evaluation; infrastructure failures are not reward zero"
                    )
                rewards = []
                for row in results:
                    reward = RTensor.localize({"rewards": row["rewards"]})["rewards"]
                    rewards.extend(reward.reshape(-1).tolist())
                expected = len(rows) * self.config.eval_gconfig.n_samples
                if len(rewards) != expected:
                    raise RuntimeError(
                        f"Evaluation coverage mismatch: {len(rewards)} != {expected}"
                    )
                metrics[domain] = {
                    "episodes": expected,
                    "reward_mean": sum(rewards) / expected,
                }
            self._write_eval_status(
                version,
                {
                    "status": "completed",
                    "version": version,
                    "checkpoint_path": str(checkpoint_path),
                    "split": "test"
                    if self.config.experiment_mode == "formal"
                    else "dev",
                    "domains": metrics,
                    "completed_ns": time.time_ns(),
                    "checkpoint_selection": False,
                },
            )
        except BaseException as exc:
            self._write_eval_status(
                version,
                {
                    "status": "failed",
                    "version": version,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise
