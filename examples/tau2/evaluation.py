# SPDX-License-Identifier: Apache-2.0
"""Official tau2 evaluation on the existing dedicated checkpoint evaluator."""

from collections import defaultdict
from functools import partial

from areal.infra.rpc.rtensor import RTensor
from areal.trainer.async_eval import AsyncEvalPPOTrainer
from areal.trainer.rollout_batch import FiniteEpochBatcher, repeat_groups_for_dispatch


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

    def _additional_stats(self) -> dict[str, float]:
        # Real coverage is distinct from physical DP tail replication.
        return dict(self._batch_counts)

    def _init_dedicated_eval_rollout(self):
        super()._init_dedicated_eval_rollout()
        self._async_eval_rollout.start_proxy()

    def _evaluate_dataset(self, eval_workflow, eval_workflow_kwargs) -> dict:
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
        flat_metrics = {
            f"eval/{domain}/{key}": value
            for domain, result in metrics.items()
            for key, value in result.items()
        }
        episodes = sum(result["episodes"] for result in metrics.values())
        flat_metrics["eval/episodes"] = episodes
        flat_metrics["eval/reward_mean"] = (
            sum(
                result["reward_mean"] * result["episodes"]
                for result in metrics.values()
            )
            / episodes
        )
        return {
            "split": "test" if self.config.experiment_mode == "formal" else "dev",
            "domains": metrics,
            "checkpoint_selection": False,
            "metrics": flat_metrics,
        }
