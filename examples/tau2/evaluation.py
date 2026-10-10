# SPDX-License-Identifier: Apache-2.0
"""Official tau2 evaluation on the existing dedicated checkpoint evaluator."""

import time
from functools import partial
from pathlib import Path

from examples.tau2.official_runner import (
    INFRASTRUCTURE_ERROR,
    collect_episodes,
    expand_trials,
    official_metrics,
)

from areal.trainer.async_eval import AsyncEvalPPOTrainer
from areal.trainer.rollout_batch import (
    FiniteEpochBatcher,
    group_episode_count,
    repeat_groups_for_dispatch,
)
from areal.utils.stats_logger import StatsLogger


class Tau2AsyncEvalTrainer(AsyncEvalPPOTrainer):
    """Reuse GPU allocation/checkpoint loading; summarize official domain rewards."""

    _snapshot_evidence = False
    _recover_rollout_inputs = True

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
        if config.critic is not None and (
            config.critic.loss_reduction != "token_mean"
            or config.critic.ppo_n_minibatches != 1
            or not config.critic.disable_dropout
        ):
            raise ValueError(
                "Finite-tail critic dispatch requires token_mean, one minibatch and disabled dropout"
            )
        super()._init_impl(config, *args, **kwargs)
        epoch = self.recover_info.last_step_info.epoch if self.recover_info else 0
        if self.recover_info is None:
            self.train_dataloader.sampler.seed = config.seed
        self.rollout.prepare_batch = partial(
            self._prepare_training_batch, FiniteEpochBatcher(self.rollout, epoch)
        )

    def _prepare_training_batch(self, prepare, *args, **kwargs):
        kwargs.update(
            finite_epoch=True,
            fail_on_rejection=True,
            min_usable_group_size=self.config.resolve_min_usable_group_size(),
            max_skipped_inputs=self.config.max_skipped_tasks_per_epoch,
        )
        groups = prepare(*args, **kwargs)
        physical, replicas = repeat_groups_for_dispatch(
            groups, self.actor.parallel_strategy.dp_size
        )
        generator = getattr(self.rollout, "data_generator", None)
        self._batch_counts = {
            "tau2_batch/real_prompts": len(groups),
            # Groups may have lost members after retries (ADR 0011).
            "tau2_batch/real_episodes": sum(map(group_episode_count, groups)),
            "tau2_batch/physical_prompt_groups": len(physical),
            "tau2_batch/dispatch_replication_factor": replicas,
            "tau2_batch/epoch_deferred_tasks": getattr(generator, "deferred_count", 0),
            "tau2_batch/epoch_skipped_tasks": len(getattr(generator, "skipped", ())),
        }
        return physical

    def _additional_stats(self) -> dict[str, float]:
        # Real coverage is distinct from physical DP tail replication.
        return dict(self._batch_counts)

    def _init_dedicated_eval_rollout(self):
        super()._init_dedicated_eval_rollout()
        self._async_eval_rollout.start_proxy()

    def _evaluate_dataset(self, eval_workflow, eval_workflow_kwargs) -> dict:
        """Score the validation split with the official `tau2 run` semantics."""

        tasks = [row for batch in self.valid_dataloader for row in batch]
        repeats = self.config.eval_gconfig.n_samples
        rollout = self.config.evaluation_rollout
        cap = rollout.max_concurrent_rollouts or rollout.consumer_batch_size
        log_path = StatsLogger.get_log_path(
            experiment_name=self.config.experiment_name,
            trial_name=self.config.trial_name,
            fileroot=self.config.cluster.fileroot,
        )
        trajectory_root = (
            Path(log_path) / "tau2-trajectories" / f"eval-{time.time_ns()}"
        )
        records = collect_episodes(
            self._async_eval_rollout,
            expand_trials(tasks, repeats),
            workflow=eval_workflow,
            workflow_kwargs=eval_workflow_kwargs,
            # The cap counts prompt groups; each group was `repeats` episodes.
            max_inflight=cap * repeats,
            trajectory_root=trajectory_root,
        )
        official = official_metrics(records, repeats=repeats)
        domains = {
            domain: {
                "episodes": result["evaluated_episodes"],
                "reward_mean": result["avg_reward"],
                "infra_excluded_episodes": result["infra_excluded_episodes"],
                "policy_free_episodes": result["policy_free_episodes"],
                "pass_hat_k": result["pass_hat_k"],
                "mean_at_k": result["mean_at_k"],
            }
            for domain, result in official["by_domain"].items()
        }
        metrics = {}
        for domain, result in domains.items():
            for key in (
                "episodes",
                "reward_mean",
                "infra_excluded_episodes",
                "policy_free_episodes",
            ):
                metrics[f"eval/{domain}/{key}"] = result[key]
            metrics[f"eval/{domain}/mean@{repeats}"] = result["mean_at_k"]
            for k, value in result["pass_hat_k"].items():
                metrics[f"eval/{domain}/pass^{k}"] = value
        overall = official["overall"]
        metrics["eval/episodes"] = overall["evaluated_episodes"]
        metrics["eval/reward_mean"] = overall["avg_reward"]
        metrics["eval/infra_excluded_episodes"] = overall["infra_excluded_episodes"]
        metrics[f"eval/mean@{repeats}"] = overall["mean_at_k"]
        for k, value in overall["pass_hat_k"].items():
            metrics[f"eval/pass^{k}"] = value
        for k, tasks in overall["pass_hat_k_tasks"].items():
            if tasks < overall["tasks"]:
                metrics[f"eval/pass^{k}_tasks"] = tasks
        return {
            "split": "test" if self.config.experiment_mode == "formal" else "dev",
            "domains": domains,
            "infra_excluded": [
                {k: r.get(k) for k in ("source_id", "attempts", "error_type", "error")}
                for r in records
                if r["status"] == INFRASTRUCTURE_ERROR
            ],
            "trajectory_dir": str(trajectory_root),
            "checkpoint_selection": False,
            "metrics": metrics,
        }
