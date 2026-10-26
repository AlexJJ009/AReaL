"""Training script for Tau2 benchmark with AReaL proxy mode."""

import argparse
import math
import os
import random
import sys
import warnings
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from datasets import Dataset
from omegaconf import OmegaConf
from tau2.registry import registry

from examples.tau2 import benchmark_variant
from examples.tau2.api_reliability import quota_exit_guard, train_with_quota_exit
from examples.tau2.contracts import (
    MAX_ASSISTANT_TOKENS,
    build_sampling_schedule,
    build_task_rows,
    keep_rollout_group,
    normalize_domains,
    resolve_pinned_hf_snapshot,
    split_critic_tasks,
    validate_episode_batch,
    validate_installed_tau2_revision,
)
from examples.tau2.utils import Tau2PPOConfig

from areal.api.cli_args import load_expr_config
from areal.utils import logging

logger = logging.getLogger("Tau2Train")


def get_tau2_dataset(
    domain: str | Iterable[str],
    type: str = "rl",
    split: str = "train",
    provenance: Mapping[str, str] | None = None,
    domain_effective_episodes: Mapping[str, int] | None = None,
    algorithm: str = "grpo",
    seed: int = 0,
    experiment_mode: str = "qualification",
    critic_dev_fraction: float = 0.2,
    critic_tasks_per_domain: int | None = None,
    critic_rollouts_per_task: int = 1,
    variant: str = "official",
) -> Dataset:
    """Create a HuggingFace Dataset from tau2 task IDs.

    Args:
        domain: One or more τ² domains, or ``mixed``.
        split: Dataset split (e.g., 'train', 'test', 'small')
        type: Dataset type (e.g., 'rl', 'sft'), only 'rl' is supported for now
        variant: τ² benchmark variant; ``fixed`` leaves out the train tasks
            listed in ``benchmark_fixes.json``.

    Returns:
        Dataset: HuggingFace Dataset containing task_id entries
    """
    assert type == "rl", "Only RL dataset is supported for now"

    validate_installed_tau2_revision()
    domains = normalize_domains(domain)
    splits_by_domain: dict[str, dict[str, list[str]]] = {}
    for selected_domain in domains:
        splits_loader_fn = registry.get_task_splits_loader(selected_domain)
        if splits_loader_fn is None:
            raise ValueError(
                f"No task splits loader found for domain {selected_domain}"
            )
        splits_by_domain[selected_domain] = splits_loader_fn()
    dataset_items = build_task_rows(
        splits_by_domain,
        domains=domains,
        split="train" if split == "dev" else split,
    )
    if split in ("train", "dev"):
        excluded = {
            (d, task_id)
            for d in domains
            for task_id in benchmark_variant.train_excluded_task_ids(variant, d)
        }
        dataset_items = [
            row
            for row in dataset_items
            if (row["domain"], row["task_id"]) not in excluded
        ]
    if provenance is not None:
        dataset_items = [{**row, **provenance} for row in dataset_items]
    if critic_rollouts_per_task != 1:
        raise ValueError("Online critic sampling uses one episode per task per epoch")
    if split == "dev" or (experiment_mode == "tune" and split == "train"):
        dataset_items = split_critic_tasks(
            dataset_items,
            seed=seed,
            dev_fraction=critic_dev_fraction,
            tasks_per_domain=critic_tasks_per_domain,
        )
        selected_split = "dev" if split == "dev" else "train"
        dataset_items = [
            row for row in dataset_items if row["critic_split"] == selected_split
        ]
    if domain_effective_episodes:
        dataset_items = build_sampling_schedule(
            dataset_items,
            domain_effective_episodes=domain_effective_episodes,
            algorithm=algorithm,
            seed=seed,
        )
    if split == "train":
        random.Random(seed).shuffle(dataset_items)
    dataset = Dataset.from_list(dataset_items)
    logger.info(
        f"Created dataset with {len(dataset)} scheduled items for domains "
        f"{list(domains)}, split {split}"
    )
    return dataset


def group_filter(x: dict[str, Any]):
    rewards = x["rewards"].reshape(-1).tolist()
    algorithm = "sao" if len(rewards) == 1 else "grpo"
    return keep_rollout_group(algorithm, rewards, dynamic_filter=True)


def _dataset_split(dataset_config: Any, default: str) -> str:
    split = getattr(dataset_config, "split", None)
    if split in ("train", "test"):
        return split
    path_split = str(dataset_config.path).rstrip("/").split("/")[-1]
    return path_split if path_split in ("train", "test") else default


def validate_tau2_recipe(
    config: Tau2PPOConfig, *, evaluation: bool = False
) -> tuple[str, ...]:
    """Fail before worker/API creation when the tool-use contract is inconsistent.

    ``evaluation`` admits thinking policies, which only offline evaluation runs:
    their reasoning is dropped from the history, so the server renders each
    request with the full chat template instead of concatenating tokens.
    """

    config.validate_user_td_split()
    if config.train_user_boundary_values:
        actor = config.actor
        if not (
            actor.gae_timestep_unit == "token"
            and actor.discount == 1.0
            and actor.critic_gae_lambda == 1.0
            and actor.kl_ctl == 0
            and actor.reward_scaling == 1.0
            and actor.reward_bias == 0.0
            and actor.reward_norm is None
            and not actor.overlong_reward_penalty
            and not config.gconfig.reward_normalization
        ):
            raise ValueError(
                "user_td requires token GAE, gamma=1, critic lambda=1 and raw outcome rewards without KL/shaping"
            )
    if config.algorithm == "critic":
        raise ValueError("Use scripts/tau2/train_critic.sh for critic-only training")

    domains = normalize_domains(config.domains)
    expected_domain = domains[0] if len(domains) == 1 else "mixed"
    if config.econfig.domain != expected_domain:
        raise ValueError(
            f"econfig.domain={config.econfig.domain} conflicts with domains={domains}"
        )
    if config.train_batch_episodes is None:
        raise ValueError("train_batch_episodes must be frozen explicitly")
    flexible_grpo = config.algorithm == "grpo" and bool(config.train_task_packs)
    if flexible_grpo:
        samples = config.gconfig.n_samples
        if (
            samples < 2
            or config.train_batch_episodes <= 0
            or config.train_batch_episodes % samples
        ):
            raise ValueError(
                "Multi-pack GRPO requires n_samples >= 2 dividing train_batch_episodes"
            )
        prompt_groups = config.train_batch_episodes // samples
    else:
        prompt_groups = validate_episode_batch(
            config.algorithm, config.train_batch_episodes
        )
    if config.train_dataset.batch_size != prompt_groups:
        raise ValueError(
            "train_dataset.batch_size is the prompt-group count and must equal "
            f"train_batch_episodes / n_samples = {prompt_groups}"
        )
    minimum_queue_size = 2 * config.train_dataset.batch_size
    if config.rollout.queue_size < minimum_queue_size:
        raise ValueError(
            "rollout.queue_size must cover the active prompt batch plus one "
            "reserved consumer batch: "
            f"{config.rollout.queue_size} < {minimum_queue_size}"
        )
    if config.domain_effective_episodes and set(
        config.domain_effective_episodes
    ) != set(domains):
        raise ValueError(
            "domain_effective_episodes must cover selected domains exactly"
        )
    expected_samples = 8 if config.algorithm == "grpo" else 1
    if not flexible_grpo and config.gconfig.n_samples != expected_samples:
        raise ValueError(
            f"{config.algorithm} requires gconfig.n_samples={expected_samples}"
        )
    if config.gconfig.max_new_tokens != MAX_ASSISTANT_TOKENS:
        raise ValueError("τ² requires gconfig.max_new_tokens=4096")
    if config.gconfig.max_tokens != config.econfig.context_window_tokens:
        raise ValueError("τ² requires gconfig.max_tokens=32768")
    if (
        config.gconfig.greedy
        or config.gconfig.temperature <= 0
        or config.gconfig.top_p != 1.0
        or config.gconfig.top_k != int(1e8)
        or config.gconfig.frequency_penalty != 0.0
    ):
        raise ValueError(
            "The first τ² RL contract uses non-greedy temperature sampling with "
            "top_p=1, disabled top_k, and no token penalty so behavior/current "
            "logprob distributions stay comparable"
        )
    agent_config = config.rollout.agent
    if agent_config is None:
        raise ValueError("τ² requires rollout.agent configuration")
    if config.econfig.enable_thinking and not evaluation:
        raise ValueError("τ² training requires enable_thinking=false")
    template = (
        ("hf", "individual") if config.econfig.enable_thinking else ("concat", "concat")
    )
    if (
        agent_config.engine_max_tokens != config.econfig.context_window_tokens
        or (
            agent_config.chat_template_type,
            agent_config.export_style,
        )
        != template
    ):
        raise ValueError(
            "τ² requires engine_max_tokens=32768 and concat template/export "
            "(hf template with individual export when thinking)"
        )
    if config.dynamic_group_filter:
        raise ValueError("Dynamic group filtering is not part of the first τ² baseline")
    if config.econfig.add_thinking_tool:
        raise ValueError("The first non-thinking τ² baseline forbids add_thinking_tool")

    actor = config.actor
    if actor.temperature != config.gconfig.temperature:
        raise ValueError(
            "Actor current-logprob temperature must match rollout temperature"
        )
    if config.algorithm == "sao":
        if config.critic is None or not config.critic.is_critic:
            raise ValueError("SAO requires a qualified scalar critic")
        required = (
            actor.use_direct_dis_loss
            and config.critic_updates_before_actor == 2
            and actor.critic_gae_lambda == 1.0
            and actor.gae_lambda
            == "areal.trainer.ppo.lambda_fn.sao_length_adaptive_gae"
            and actor.gae_lambda_kwargs == {"alpha": 1.5}
            and actor.dis_epsilon_low == 0.3
            and actor.dis_epsilon_high == 5.0
            and actor.discount == 1.0
            and actor.reward_scaling == 1.0
            and actor.reward_bias == 0.0
            and actor.reward_norm is None
            and actor.adv_norm is None
            and (
                config.critic.eps_clip is None
                or (bool(config.train_task_packs) and config.critic.eps_clip == 0.5)
            )
            and actor.ppo_n_minibatches == 1
            and config.critic.ppo_n_minibatches == 1
        )
        if not required:
            raise ValueError(
                "SAO requires the paper Direct DIS bounds, raw gamma=1 rewards, "
                "dual lambdas, unclipped critic MSE (or PPO-matched value clip=0.5), one minibatch, and two "
                "critic updates before the actor"
            )
    elif config.algorithm == "ppo":
        validate_ppo_contract(config, evaluation=evaluation)
    elif config.critic is not None or actor.use_direct_dis_loss:
        raise ValueError("Non-SAO paths must not silently enable the SAO critic/DIS")
    if config.algorithm == "grpo":
        normalization = actor.reward_norm
        if normalization is None or (
            normalization.mean_level,
            normalization.std_level,
            normalization.group_size,
        ) != ("group", "group", config.gconfig.n_samples):
            raise ValueError(
                "GRPO requires group reward normalization matching n_samples"
            )
        if (
            actor.adv_norm is not None
            or config.ref is not None
            or config.teacher is not None
            or actor.kl_ctl != 0
        ):
            raise ValueError(
                "GRPO recipe uses no extra advantage normalization, ref, teacher or KL"
            )
        if not actor.use_decoupled_loss or not actor.recompute_logprob:
            raise ValueError(
                "Async GRPO requires decoupled loss and recomputed logprobs"
            )
        rejection = actor.rejection_sampling
        if rejection is None or (
            rejection.level,
            rejection.action,
            rejection.metric,
            rejection.upper,
            rejection.lower,
        ) != ("token", "mask", "ratio", 5.0, None):
            raise ValueError("Async GRPO requires token ratio rejection above 5")
    if config.train_task_packs and config.algorithm == "sao":
        from scripts.tau2.train_critic_head import validate_head_config

        if config.critic.path == config.actor.path:
            validate_head_config(config)
    return domains


SAO_ACTOR_LAMBDA = "areal.trainer.ppo.lambda_fn.sao_length_adaptive_gae"


def _ppo_actor_lambda_allowed(actor) -> bool:
    """Actor GAE lambda: 1, or SAO's length-adaptive lambda with alpha=1.5.

    The critic target keeps lambda=1 either way (checked by the caller), as in
    SAO and VC-PPO's decoupled GAE; only the actor advantage changes.
    """
    if actor.gae_lambda == 1.0:
        return not actor.gae_lambda_kwargs
    return actor.gae_lambda == SAO_ACTOR_LAMBDA and dict(
        actor.gae_lambda_kwargs or {}
    ) == {"alpha": 1.5}


def validate_ppo_contract(config: Tau2PPOConfig, *, evaluation: bool = False) -> None:
    """Standard clipped GAE PPO with the GRPO recipe's async correction."""
    actor, critic = config.actor, config.critic
    # Offline evaluation selects its actor via --model-path; resolved training
    # configs may already contain a local snapshot instead of the pinned ID.
    if not evaluation and (
        actor.path
        != "Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554"
    ):
        raise ValueError(
            "This PPO recipe requires the pinned Qwen3-4B-Instruct-2507 actor"
        )
    if critic is None or not critic.is_critic:
        raise ValueError("PPO requires a learned scalar critic")
    if not (
        not actor.use_direct_dis_loss
        and not actor.use_sapo_loss
        and not actor.use_cispo_loss
        and actor.eps_clip == 0.2
        and actor.eps_clip_higher in (None, 0.2)
        and actor.c_clip is None
        and actor.m2_threshold is None
        and actor.discount == 1.0
        and _ppo_actor_lambda_allowed(actor)
        and actor.critic_gae_lambda == 1.0
        and actor.reward_scaling == 1.0
        and actor.reward_bias == 0.0
        and actor.reward_norm is None
        and not actor.overlong_reward_penalty
        and not config.gconfig.reward_normalization
        and config.critic_updates_before_actor == 0
        and not critic.freeze_critic_attention
        and not critic.use_lora
        and not critic.init_from_scratch
        and critic.eps_clip == 0.5
        and not critic.mask_no_eos_with_zero
        and config.ref is None
        and config.teacher is None
        and actor.kl_ctl == 0
    ):
        raise ValueError(
            "PPO requires clipped surrogate eps=0.2, gamma=1 GAE with critic "
            "lambda=1 and actor lambda 1 or SAO length-adaptive (alpha=1.5), "
            "no DIS/reward normalization/KL, and a full-parameter critic "
            "with value clip=0.5"
        )
    normalization = actor.adv_norm
    if normalization is None or (
        normalization.mean_level,
        normalization.std_level,
        normalization.mean_leave1out,
        normalization.std_unbiased,
        normalization.eps,
    ) != ("batch", "batch", False, True, 1e-8):
        raise ValueError("PPO requires batch action-token advantage whitening")
    if critic.path == actor.path:
        from scripts.tau2.train_critic_head import validate_head_config

        validate_head_config(config)
    if config.critic_head_lr is not None:
        raise ValueError("PPO uses the same learning rate for critic head and backbone")
    rejection = actor.rejection_sampling
    if not (
        actor.use_decoupled_loss
        and actor.recompute_logprob
        and actor.prox_logp_method == "recompute"
        and actor.importance_sampling_level == "token"
        and config.rollout.max_head_offpolicyness == 2
        and rejection is not None
        and (
            rejection.level,
            rejection.action,
            rejection.metric,
            rejection.upper,
            rejection.lower,
        )
        == ("token", "mask", "ratio", 5.0, None)
    ):
        raise ValueError(
            "Async PPO requires GRPO's recomputed decoupled loss, token ratio mask >5 and staleness=2"
        )
    if (
        config.rollout.agent.tool_call_parser != "qwen25"
        or config.evaluation_rollout.agent.tool_call_parser != "qwen25"
    ):
        raise ValueError("Qwen3 Instruct PPO requires qwen25 tool parser")
    if not (
        config.econfig.benchmark_variant == "fixed"
        and config.experiment_mode in ("formal", "qualification")
        and config.valid_dataset is not None
        and config.eval_gconfig.n_samples == 4
        and config.eval_gconfig.temperature == 0
        and (
            config.offline_eval_only
            or (
                config.evaluator.eval_before_train
                == (config.experiment_mode == "formal")
                and config.evaluator.freq_steps is not None
                and config.evaluator.freq_steps > 0
                and config.evaluator.freq_epochs is None
                and config.evaluator.freq_secs is None
            )
        )
    ):
        raise ValueError(
            "PPO requires fixed test + official Telecom, 4 repeats at temperature 0, positive step cadence, step-0 eval only in formal"
        )


def resolve_actor_snapshot(config: Tau2PPOConfig) -> str:
    """Bind every actor/tokenizer consumer to one immutable HF snapshot."""

    source = config.actor.path
    snapshot = resolve_pinned_hf_snapshot(source)
    config.actor.path = snapshot
    config.tokenizer_path = snapshot
    config.rollout.tokenizer_path = snapshot
    config.evaluation_rollout.tokenizer_path = snapshot
    config.sglang.model_path = snapshot
    config.vllm.model = snapshot
    if config.ref is not None and config.ref.path == source:
        config.ref.path = snapshot
    if config.critic is not None and (
        config.algorithm == "ppo" or bool(config.train_task_packs)
    ):
        config.critic.path = (
            snapshot
            if config.critic.path == source
            else (
                str(Path(config.critic.path).resolve())
                if Path(config.critic.path).is_dir()
                else resolve_pinned_hf_snapshot(config.critic.path)
            )
        )
    return snapshot


def collection_provenance(
    config: Tau2PPOConfig, *, actor_source: str
) -> dict[str, str]:
    """Attach minimal row provenance derived from the resolved runtime config."""

    try:
        policy_id, policy_revision = actor_source.rsplit("@", 1)
    except ValueError as exc:
        raise ValueError("Actor source must be pinned as org/repo@revision") from exc
    if policy_id.count("/") != 1 or not policy_revision:
        raise ValueError("Actor source must be pinned as org/repo@revision")
    return {
        "policy_id": policy_id,
        "policy_revision": policy_revision,
        "simulator_id": config.econfig.user_llm,
        "benchmark_variant": config.econfig.benchmark_variant,
    }


def resolve_effective_episodes(config: Tau2PPOConfig, train_items: int) -> int:
    """Bind effective episodes to the actual finite schedule coverage."""

    if config.total_train_steps is None:
        raise ValueError("total_train_steps must be resolved before episode binding")
    if train_items <= 0:
        raise ValueError("training schedule is empty")
    if config.train_dataset.drop_last:
        batch_sizes = [config.train_dataset.batch_size] * (
            train_items // config.train_dataset.batch_size
        )
    else:
        batch_sizes = [
            min(config.train_dataset.batch_size, train_items - start)
            for start in range(0, train_items, config.train_dataset.batch_size)
        ]
    if not batch_sizes:
        raise ValueError("training schedule has no complete update batch")
    consumed_prompts = 0
    remaining_steps = int(config.total_train_steps)
    for _epoch in range(int(config.total_train_epochs)):
        for batch_size in batch_sizes:
            if remaining_steps <= 0:
                break
            consumed_prompts += batch_size
            remaining_steps -= 1
        if remaining_steps <= 0:
            break
    actual = consumed_prompts * int(config.gconfig.n_samples)
    if config.effective_episodes is None:
        config.effective_episodes = actual
    elif int(config.effective_episodes) != actual:
        raise ValueError(
            "effective_episodes must equal actual scheduled coverage after "
            "domain pruning, task_limit, epochs and total_train_steps: "
            f"{actual} != {config.effective_episodes}"
        )
    return actual


def main(
    args,
    train_rows_hook: Callable[[Tau2PPOConfig, dict[str, Any]], Dataset] | None = None,
):
    # Suppress pydantic UserWarning
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--check-config", action="store_true")
    options, config_args = parser.parse_known_args(args)
    config, _ = load_expr_config(config_args, Tau2PPOConfig)
    if config.train_task_pack is not None and train_rows_hook is None:
        raise ValueError("train_task_pack requires scripts/tau2/train_sao_pack.py")
    actor_source = config.actor.path
    provenance = collection_provenance(config, actor_source=actor_source)
    domains = validate_tau2_recipe(config)
    econfig = config.econfig
    benchmark_variant.validate_task_files(econfig.benchmark_variant, domains)
    dataset_kwargs = dict(
        domain=domains,
        provenance=provenance,
        seed=config.seed,
        experiment_mode=config.experiment_mode,
        critic_dev_fraction=config.critic_dev_fraction,
        critic_tasks_per_domain=config.critic_tasks_per_domain,
        critic_rollouts_per_task=config.critic_rollouts_per_task,
        variant=econfig.benchmark_variant,
    )
    if _dataset_split(config.train_dataset, "train") != "train":
        raise ValueError("Gradient training must use official train tasks")
    matched_pool = config.algorithm == "ppo" or bool(config.train_task_packs)
    if matched_pool:
        from examples.tau2.ppo_data import build_ppo_train_dataset

        if train_rows_hook is not None or config.train_task_pack is not None:
            raise ValueError(
                "Multi-pack recipes use train_task_packs through the standard entry"
            )
        train_dataset = build_ppo_train_dataset(config, dataset_kwargs)
    elif train_rows_hook is None:
        train_dataset = get_tau2_dataset(
            **dataset_kwargs,
            split="train",
            algorithm=config.algorithm,
            domain_effective_episodes=config.domain_effective_episodes,
        )
    else:
        train_dataset = train_rows_hook(config, dataset_kwargs)
    if config.task_limit is not None:
        if config.task_limit < 1:
            raise ValueError("task_limit must be positive")
        train_dataset = train_dataset.select(
            range(min(config.task_limit, len(train_dataset)))
        )
    per_epoch = (
        len(train_dataset) // config.train_dataset.batch_size
        if config.train_dataset.drop_last
        else math.ceil(len(train_dataset) / config.train_dataset.batch_size)
    )
    available_steps = per_epoch * config.total_train_epochs
    if config.total_train_steps is None:
        config.total_train_steps = available_steps
    if not 0 < config.total_train_steps <= available_steps:
        raise ValueError("total_train_steps exceeds the configured dataset and epochs")
    if matched_pool:
        if (
            config.experiment_mode == "formal"
            and config.total_train_steps != available_steps
        ):
            raise ValueError(
                "Formal multi-pack policy must consume complete epochs for uniform task coverage"
            )
        if (
            config.experiment_mode == "qualification"
            and config.evaluator.freq_steps < config.total_train_steps
        ):
            raise ValueError(
                "PPO smoke evaluates only at final step; cadence must cover all smoke steps"
            )
        if config.num_critic_only_steps >= config.total_train_steps:
            raise ValueError("PPO warmup must leave at least one actor update")
    resolve_effective_episodes(config, len(train_dataset))
    valid_dataset = None
    if config.valid_dataset is not None:
        if matched_pool:
            from examples.tau2.ppo_data import build_ppo_eval_dataset

            valid_dataset = build_ppo_eval_dataset(config, dataset_kwargs)
        else:
            valid_split = "test" if config.experiment_mode == "formal" else "dev"
            valid_dataset = get_tau2_dataset(**dataset_kwargs, split=valid_split)
    if (
        config.algorithm == "sao"
        and config.experiment_mode != "qualification"
        and valid_dataset is None
    ):
        raise ValueError("Formal/tune τ² SAO requires a validation dataset")
    if (train_rows_hook is not None or matched_pool) and valid_dataset is not None:
        train_ids = {
            (row["domain"], row["task_id"], row.get("task_pack"))
            for row in train_dataset
        }
        validation_ids = {
            (row["domain"], row["task_id"], row.get("task_pack"))
            for row in valid_dataset
        }
        if train_ids & validation_ids:
            raise ValueError("Tau2 policy train/validation task split is not disjoint")
    from examples.tau2.evaluation import (
        Tau2AsyncEvalTrainer,
        Tau2PPOTrainer,
        Tau2SyncTrainer,
    )

    use_native_async_eval = config.algorithm in ("grpo", "ppo") or (
        config.algorithm == "sao"
        and config.experiment_mode != "qualification"
        and valid_dataset is not None
    )
    trainer_cls = (
        Tau2PPOTrainer
        if config.algorithm == "ppo" or (matched_pool and config.algorithm == "sao")
        else Tau2AsyncEvalTrainer
        if use_native_async_eval
        else Tau2SyncTrainer
    )
    trainer_cls.validate_config(config)
    if options.check_config:
        if config.algorithm == "ppo":
            from examples.tau2.ppo_data import ppo_config_summary

            logger.info(
                "PPO resolved summary:\n%s",
                OmegaConf.to_yaml(
                    ppo_config_summary(config, train_dataset, valid_dataset)
                ),
            )
        logger.info(
            "Resolved config (no workers/API calls):\n%s",
            OmegaConf.to_yaml(OmegaConf.structured(config)),
        )
        logger.info(
            "Train prompts=%d, validation prompts=%d, steps=%d",
            len(train_dataset),
            len(valid_dataset) if valid_dataset is not None else 0,
            config.total_train_steps,
        )
        return
    resolve_actor_snapshot(config)
    # Convert econfig to dict for workflow kwargs
    from dataclasses import asdict

    # Pass one shared path to every training/evaluation worker, independent of
    # inherited environment. Recovery gets a fresh breaker, not a stale trip.
    run_root = Path(os.environ.get("TAU2_RUN_ROOT", config.cluster.fileroot))
    econfig._quota_breaker_state_path = str(
        run_root.resolve() / f"quota-breaker-{uuid4().hex}.json"
    )
    econfig_dict = asdict(econfig)

    # Build workflow kwargs
    workflow_kwargs = dict(
        econfig=econfig_dict,
        gen_args=dict(
            temperature=config.gconfig.temperature,
            top_p=config.gconfig.top_p,
            frequency_penalty=config.gconfig.frequency_penalty,
            seed=config.gconfig.seed,
            max_completion_tokens=MAX_ASSISTANT_TOKENS,
            max_total_tokens=econfig.context_window_tokens,
        ),
        timeout=config.episode_timeout_seconds,
        infra_retries=config.infra_retries,
    )
    if config.train_user_boundary_values:
        workflow_kwargs.update(
            train_user_boundary_values=True,
            user_td_beta=config.user_td_beta,
            user_boundary_detector=config.user_boundary_detector,
            user_td_beta_flagged=config.user_td_beta_flagged,
        )
    if config.user_reliability_weight:
        workflow_kwargs["user_reliability_weight"] = True
    if config.critic_user_scenario:
        workflow_kwargs.update(
            critic_user_scenario=True,
            critic_tokenizer_path=config.tokenizer_path,
        )
    from areal.utils.stats_logger import StatsLogger

    log_path = StatsLogger.get_log_path(
        experiment_name=config.experiment_name,
        trial_name=config.trial_name,
        fileroot=config.cluster.fileroot,
    )
    # Training rollouts keep the official message trajectory of every attempt.
    workflow_kwargs["trajectory_dir"] = os.path.join(
        log_path, "tau2-trajectories", "rollout"
    )

    # Evaluation sampling is explicit and shared across dev/test.
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs.pop("critic_user_scenario", None)
    eval_workflow_kwargs.pop("user_reliability_weight", None)
    eval_workflow_kwargs.pop("critic_tokenizer_path", None)
    eval_workflow_kwargs.pop("train_user_boundary_values", None)
    eval_workflow_kwargs.pop("user_td_beta", None)
    eval_workflow_kwargs.pop("user_boundary_detector", None)
    eval_workflow_kwargs.pop("user_td_beta_flagged", None)
    eval_workflow_kwargs["timeout"] = config.eval_episode_timeout_seconds
    # The evaluator binds evaluation=True and a per-run attempt-record directory.
    eval_workflow_kwargs.pop("trajectory_dir")
    eval_workflow_kwargs["gen_args"] = dict(
        top_p=config.eval_gconfig.top_p,
        frequency_penalty=config.eval_gconfig.frequency_penalty,
        seed=config.eval_gconfig.seed,
        max_total_tokens=econfig.context_window_tokens,
        temperature=config.eval_gconfig.temperature,
        max_completion_tokens=config.eval_gconfig.max_new_tokens,
    )

    with (
        quota_exit_guard(econfig),
        trainer_cls(
            config,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
        ) as trainer,
    ):
        # Record the controller only after dataset/snapshot resolution and engine
        # initialization. Worker records independently describe loaded objects.
        from areal.utils.runtime_audit import write_runtime_audit

        write_runtime_audit(
            "tau2-controller-initialized",
            {
                "config": trainer.config,
                "workflow_kwargs": workflow_kwargs,
                "eval_workflow_kwargs": eval_workflow_kwargs,
                "train_prompts": len(train_dataset),
                "validation_prompts": len(valid_dataset)
                if valid_dataset is not None
                else 0,
                "recovered": trainer.recover_info is not None,
            },
        )
        dynamic_filter_fn = (
            "examples.tau2.train.group_filter" if config.dynamic_group_filter else None
        )
        train_with_quota_exit(
            trainer,
            workflow="examples.tau2.agent.Tau2AgentWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="examples.tau2.agent.Tau2AgentWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
            dynamic_filter_fn=dynamic_filter_fn,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
