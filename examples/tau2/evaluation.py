# SPDX-License-Identifier: Apache-2.0
"""Official tau2 evaluation on the existing dedicated checkpoint evaluator."""

import time
from collections import defaultdict
from functools import partial
from math import gcd

from examples.tau2.utils import FiniteEpochBatcher
from scripts.sao.async_eval import AsyncEvalPPOTrainer

from areal.infra.rpc.rtensor import RTensor


def repeat_groups_for_dispatch(
    groups: list[dict], dp_size: int
) -> tuple[list[dict], int]:
    """Uniformly replicate complete tail groups; token-mean gradients stay equal.

    This is physical DP dispatch only, not new rollout samples. Never replicate
    just some groups: that would change their relative optimization weights.
    """
    if not groups:
        raise ValueError("Cannot dispatch an empty training batch")
    replicas = dp_size // gcd(len(groups), dp_size)
    return [dict(group) for _ in range(replicas) for group in groups], replicas


class Tau2AsyncEvalTrainer(AsyncEvalPPOTrainer):
    """Reuse GPU allocation/checkpoint loading; summarize official domain rewards."""

    _snapshot_evidence = False

    def _init_impl(self, config, *args, **kwargs):
        if (
            config.actor.loss_reduction != "token_mean"
            or config.actor.ppo_n_minibatches != 1
            or not config.actor.disable_dropout
        ):
            raise ValueError(
                "Tau2 finite-tail dispatch requires token_mean, one PPO minibatch, "
                "and disabled dropout"
            )
        super()._init_impl(config, *args, **kwargs)
        epoch = self.recover_info.last_step_info.epoch if self.recover_info else 0
        if self.recover_info is None:
            self.train_dataloader.sampler.seed = config.seed
        self.rollout.prepare_batch = partial(
            self._prepare_training_batch, FiniteEpochBatcher(self.rollout, epoch)
        )

    def _prepare_training_batch(self, prepare, *args, **kwargs):
        kwargs.update(finite_epoch=True, fail_on_rejection=True)
        groups = prepare(*args, **kwargs)
        physical, replicas = repeat_groups_for_dispatch(
            groups, self.actor.parallel_strategy.dp_size
        )
        self._batch_counts = {
            "tau2_batch/real_prompts": len(groups),
            "tau2_batch/real_episodes": len(groups) * self.config.gconfig.n_samples,
            "tau2_batch/physical_prompt_groups": len(physical),
            "tau2_batch/dispatch_replication_factor": replicas,
        }
        return physical

    def _export_and_commit_stats(self, epoch, epoch_step, global_step):
        # Worker token/sample counters describe physical compute. These explicit
        # real counts are the coverage counters for a uniformly replicated tail.
        stats = self.actor.export_stats()
        stats.update(self.rollout.export_stats())
        stats.update(self._batch_counts)
        self.stats_logger.commit(epoch, epoch_step, global_step, stats)

    def train(self, *args, eval_workflow=None, eval_workflow_kwargs=None, **kwargs):
        # Match the math entrypoint: consume the initial evaluation trigger before
        # training, using the backbone rather than a not-yet-saved checkpoint.
        if (
            self.config.evaluator.eval_before_train
            and self.recover_info is None
            and self.valid_dataloader is not None
            and eval_workflow is not None
        ):
            self.evaluator.freq_ctl.check(epochs=0, steps=0)
            self._evaluate_fn(eval_workflow, eval_workflow_kwargs)
        return super().train(
            *args,
            eval_workflow=eval_workflow,
            eval_workflow_kwargs=eval_workflow_kwargs,
            **kwargs,
        )

    def _save_training_state(self, *, epoch, epoch_step, global_step, force=False):
        super()._save_training_state(
            epoch=epoch,
            epoch_step=epoch_step,
            global_step=global_step,
            force=force or global_step + 1 == self.config.total_train_steps,
        )

    def _evaluate(
        self, eval_workflow, eval_workflow_kwargs, epoch, epoch_step, global_step
    ):
        if (
            global_step + 1 == self.config.total_train_steps
            and self.valid_dataloader is not None
        ):
            from areal.utils.saver import Saver

            self._enqueue_eval(
                version=global_step + 1,
                checkpoint_path=Saver.get_model_save_path(
                    self.config.experiment_name,
                    self.config.trial_name,
                    self.config.cluster.fileroot,
                    epoch,
                    epoch_step,
                    global_step,
                ),
                eval_workflow=eval_workflow,
                eval_workflow_kwargs=eval_workflow_kwargs,
                snapshot=False,
            )
        else:
            super()._evaluate(
                eval_workflow, eval_workflow_kwargs, epoch, epoch_step, global_step
            )

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
