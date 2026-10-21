# SPDX-License-Identifier: Apache-2.0
"""Official tau2 evaluation on the existing dedicated checkpoint evaluator."""

import json
import time
from copy import deepcopy
from functools import partial
from pathlib import Path

from examples.tau2.api_reliability import Tau2ApiStatsMixin
from examples.tau2.official_runner import (
    INFRASTRUCTURE_ERROR,
    collect_episodes,
    expand_trials,
    official_metrics,
)

from areal import PPOTrainer
from areal.api.cli_args import SGLangConfig
from areal.trainer.async_eval import AsyncEvalPPOTrainer
from areal.trainer.rollout_batch import (
    FiniteEpochBatcher,
    group_episode_count,
    pad_groups_for_dispatch,
)
from areal.trainer.rollout_batch import (
    repeat_groups_for_dispatch as repeat_groups_for_dispatch,
)
from areal.utils.stats_logger import StatsLogger


def checkpoint_eval_stats(official: dict, repeats: int) -> dict:
    """Shared online/offline checkpoint metrics from official episode results."""
    domains = {
        domain: {
            "episodes": result["evaluated_episodes"],
            "reward_mean": result["avg_reward"],
            "infra_excluded_episodes": result["infra_excluded_episodes"],
            "context_limit_episodes": result["context_limit_episodes"],
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
            "context_limit_episodes",
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
    metrics["eval/context_limit_episodes"] = overall["context_limit_episodes"]
    metrics[f"eval/mean@{repeats}"] = overall["mean_at_k"]
    for k, value in overall["pass_hat_k"].items():
        metrics[f"eval/pass^{k}"] = value
    for k, tasks in overall["pass_hat_k_tasks"].items():
        if tasks < overall["tasks"]:
            metrics[f"eval/pass^{k}_tasks"] = tasks
    return {"domains": domains, "metrics": metrics}


class Tau2SyncTrainer(Tau2ApiStatsMixin, PPOTrainer):
    """Synchronous recipe with the same run-wide API failure metrics."""


class Tau2AsyncEvalTrainer(Tau2ApiStatsMixin, AsyncEvalPPOTrainer):
    """Reuse GPU allocation/checkpoint loading; summarize official domain rewards."""

    _snapshot_evidence = False
    _recover_rollout_inputs = True

    @property
    def _offline_eval_only(self) -> bool:
        return getattr(self.config, "offline_eval_only", False)

    @staticmethod
    def _validate_save_eval_sync(config):
        if not getattr(config, "offline_eval_only", False):
            return AsyncEvalPPOTrainer._validate_save_eval_sync(config)
        saver = config.saver
        if (
            saver.mode != "sync"
            or saver.freq_steps is None
            or saver.freq_steps <= 0
            or saver.freq_epochs is not None
            or saver.freq_secs is not None
        ):
            raise ValueError(
                "Offline evaluation requires saver.mode='sync', positive "
                "saver.freq_steps and no epoch/time checkpoint cadence"
            )

    def train(self, *args, eval_workflow=None, eval_workflow_kwargs=None, **kwargs):
        if self._offline_eval_only:
            # Also bypass step-zero and recovered-eval schedule handling.
            eval_workflow = eval_workflow_kwargs = None
        return super().train(
            *args,
            eval_workflow=eval_workflow,
            eval_workflow_kwargs=eval_workflow_kwargs,
            **kwargs,
        )

    def _evaluate_fn(self, eval_workflow, eval_workflow_kwargs):
        if not self._offline_eval_only:
            return super()._evaluate_fn(eval_workflow, eval_workflow_kwargs)

    def _evaluate(
        self, eval_workflow, eval_workflow_kwargs, epoch, epoch_step, global_step
    ):
        if not self._offline_eval_only:
            return super()._evaluate(
                eval_workflow, eval_workflow_kwargs, epoch, epoch_step, global_step
            )

    def _enqueue_eval(self, **kwargs):
        if not self._offline_eval_only:
            return super()._enqueue_eval(**kwargs)

    def _drain_evaluations(self):
        if not self._offline_eval_only:
            return super()._drain_evaluations()

    def _write_gpu_allocation(self):
        super()._write_gpu_allocation()
        if self._offline_eval_only:
            path = Path(self.config.cluster.fileroot) / "evidence/gpu-allocation.json"
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload.update(offline_eval_only=True, online_eval_enabled=False)
                path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # The PPO recipe evaluates every global step interval, including warmup.
    @property
    def _evaluate_during_critic_warmup(self):
        return self.config.algorithm == "ppo"

    @classmethod
    def validate_config(cls, config):
        """Shared CPU preflight and runtime checks, including checkpoint cadence."""
        super().validate_config(config)
        cls._validate_save_eval_sync(config)
        if config.scheduler.type != "local":
            raise NotImplementedError(
                "Dedicated checkpoint evaluation requires LocalScheduler"
            )
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

    def _init_impl(self, config, *args, **kwargs):
        self.validate_config(config)
        super()._init_impl(config, *args, **kwargs)
        epoch = self.recover_info.last_step_info.epoch if self.recover_info else 0
        # StatefulDataLoader restores the cursor, but DistributedSampler does
        # not persist its seed or epoch. Restore those before creating iterators.
        self.train_dataloader.sampler.seed = config.seed
        if self.recover_info is not None:
            last = self.recover_info.last_step_info
            if last.epoch_step == last.steps_per_epoch - 1:
                inputs = self.recover_info.rollout_input_state or {}
                if inputs.get("outstanding") or inputs.get("buffer"):
                    raise ValueError(
                        "Completed epoch recovery has pending rollout inputs"
                    )
                # At EOF torchdata may restart automatically, without the empty
                # batch that normally advances FiniteEpochBatcher. Start the
                # next epoch explicitly; its dataloader cursor must be fresh.
                epoch = last.next().epoch
                # Materialize a queued torchdata restore before resetting it:
                # load_state_dict({}) alone leaves next_iter_state queued.
                self.train_dataloader.state_dict()
                self.train_dataloader.load_state_dict({})
            self.train_dataloader.sampler.set_epoch(epoch)
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
        if getattr(self.config, "user_reliability_weight", False):
            self._prepare_reliability(groups, prepare.epoch + 1)
        physical, padding = pad_groups_for_dispatch(
            groups, self.actor.parallel_strategy.dp_size
        )
        generator = getattr(self.rollout, "data_generator", None)
        self._batch_counts = {
            "tau2_batch/real_prompts": len(groups),
            # Groups may have lost members after retries (ADR 0011).
            "tau2_batch/real_episodes": sum(map(group_episode_count, groups)),
            "tau2_batch/physical_prompt_groups": len(physical),
            "tau2_batch/padding_groups": padding,
            "tau2_batch/epoch_deferred_tasks": getattr(generator, "deferred_count", 0),
            "tau2_batch/epoch_skipped_tasks": len(getattr(generator, "skipped", ())),
        }
        return physical

    def _additional_stats(self) -> dict[str, float]:
        # Real coverage is distinct from physical DP tail padding.
        stats = super()._additional_stats()
        stats.update(self._batch_counts)
        if getattr(self.config, "user_reliability_weight", False):
            stats.update(self._reliability_stats)
        return stats

    def _dedicated_eval_sglang_config(self) -> SGLangConfig:
        config = deepcopy(super()._dedicated_eval_sglang_config())
        if self.config.eval_enable_radix_cache:
            config.disable_radix_cache = False
        return config

    def _init_dedicated_eval_rollout(self):
        if self._offline_eval_only:
            return
        super()._init_dedicated_eval_rollout()
        self._async_eval_rollout.start_proxy()

    def _evaluate_dataset(self, eval_workflow, eval_workflow_kwargs) -> dict:
        """Score validation with official metrics and zero-score context failures."""

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
        stats = checkpoint_eval_stats(official, repeats)
        return {
            "split": "test" if self.config.experiment_mode == "formal" else "dev",
            "domains": stats["domains"],
            "infra_excluded": [
                {k: r.get(k) for k in ("source_id", "attempts", "error_type", "error")}
                for r in records
                if r["status"] == INFRASTRUCTURE_ERROR
            ],
            "trajectory_dir": str(trajectory_root),
            "checkpoint_selection": False,
            "metrics": stats["metrics"],
        }


class Tau2PPOTrainer(Tau2AsyncEvalTrainer):
    """Initialise only backbone critics; checkpoint overrides use the plain loader."""

    def _expected_recovery_trainer_state(self, config):
        expected = super()._expected_recovery_trainer_state(config)
        if getattr(config, "user_reliability_weight", False):
            expected["user_reliability_settings"] = [
                config.user_reliability_shrink_k,
                config.user_reliability_warmup_epochs,
            ]
        return expected

    def _init_impl(self, config, *args, **kwargs):
        super()._init_impl(config, *args, **kwargs)
        if getattr(config, "user_reliability_weight", False):
            from examples.tau2.user_reliability import UserReliability

            self._user_reliability = UserReliability(
                self.train_dataloader.dataset,
                config.user_reliability_shrink_k,
                config.user_reliability_warmup_epochs,
            )
            if self.recover_info is not None:
                state = self.recover_info.trainer_state.get("user_reliability")
                if state is None:
                    raise ValueError("Missing reliability recover state")
                self._user_reliability.load_state_dict(state)

    def _prepare_reliability(self, groups, epoch):
        log_dir = StatsLogger.get_log_path(
            experiment_name=self.config.experiment_name,
            trial_name=self.config.trial_name,
            fileroot=self.config.cluster.fileroot,
        )
        self._user_reliability.begin_epoch(epoch, log_dir)
        self._reliability_stats = self._user_reliability.apply(groups)

    def _process_rollout_batch(self, rollout_batch):
        super()._process_rollout_batch(rollout_batch)
        if getattr(self.config, "user_reliability_weight", False):
            self._user_reliability.observe(rollout_batch)

    def _extra_recovery_trainer_state(self):
        state = super()._extra_recovery_trainer_state()
        if getattr(self.config, "user_reliability_weight", False):
            reliability = self._user_reliability
            state.update(
                user_reliability=reliability.state_dict(),
                user_reliability_settings=[
                    reliability.shrink_k,
                    reliability.warmup_epochs,
                ],
            )
        return state

    def _create_critic(self, critic_config, alloc):
        if critic_config.path == self.config.actor.path:
            from examples.tau2.critic_head import create_head_critic

            return create_head_critic(self, critic_config, alloc)
        return super()._create_critic(critic_config, alloc)
