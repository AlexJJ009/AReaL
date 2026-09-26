"""Training script for Tau2 benchmark with AReaL proxy mode."""

import argparse
import math
import random
import sys
import warnings
from collections.abc import Iterable, Mapping
from typing import Any

from datasets import Dataset
from omegaconf import OmegaConf
from tau2.registry import registry

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

from areal import PPOTrainer
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
) -> Dataset:
    """Create a HuggingFace Dataset from tau2 task IDs.

    Args:
        domain: One or more τ² domains, or ``mixed``.
        split: Dataset split (e.g., 'train', 'test', 'small')
        type: Dataset type (e.g., 'rl', 'sft'), only 'rl' is supported for now

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
    if provenance is not None:
        dataset_items = [{**row, **provenance} for row in dataset_items]
    if (
        algorithm == "collect"
        or split == "dev"
        or (experiment_mode == "tune" and split == "train")
    ):
        if (
            algorithm == "collect"
            and experiment_mode == "qualification"
            and critic_tasks_per_domain is None
        ):
            critic_tasks_per_domain = 2
        dataset_items = split_critic_tasks(
            dataset_items,
            seed=seed,
            dev_fraction=critic_dev_fraction,
            tasks_per_domain=critic_tasks_per_domain,
        )
        if algorithm != "collect":
            selected_split = "dev" if split == "dev" else "train"
            dataset_items = [
                row for row in dataset_items if row["critic_split"] == selected_split
            ]
    if algorithm == "collect":
        dataset_items = [
            {**row, "sample_round": repeat}
            for row in dataset_items
            for repeat in range(critic_rollouts_per_task)
        ]
    elif domain_effective_episodes:
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


def validate_tau2_recipe(config: Tau2PPOConfig) -> tuple[str, ...]:
    """Fail before worker/API creation when the tool-use contract is inconsistent."""

    domains = normalize_domains(config.domains)
    expected_domain = domains[0] if len(domains) == 1 else "mixed"
    if config.econfig.domain != expected_domain:
        raise ValueError(
            f"econfig.domain={config.econfig.domain} conflicts with domains={domains}"
        )
    if config.train_batch_episodes is None:
        raise ValueError("train_batch_episodes must be frozen explicitly")
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
    if config.gconfig.n_samples != expected_samples:
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
    if (
        agent_config.engine_max_tokens != config.econfig.context_window_tokens
        or agent_config.chat_template_type != "concat"
        or agent_config.export_style != "concat"
    ):
        raise ValueError(
            "τ² requires engine_max_tokens=32768 and concat template/export"
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
            and config.critic.eps_clip is None
            and actor.ppo_n_minibatches == 1
            and config.critic.ppo_n_minibatches == 1
        )
        if not required:
            raise ValueError(
                "SAO requires the paper Direct DIS bounds, raw gamma=1 rewards, "
                "dual lambdas, unclipped critic MSE, one minibatch, and two "
                "critic updates before the actor"
            )
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
    if config.algorithm == "collect":
        optimizer = actor.optimizer
        if optimizer is None or optimizer.lr != 0.0 or optimizer.weight_decay != 0.0:
            raise ValueError(
                "Critic collection requires a fixed policy with lr=0, weight_decay=0"
            )
        if config.train_dataset.drop_last:
            raise ValueError("Critic collection must retain its final partial batch")
    return domains


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
    }


def main(args):
    # Suppress pydantic UserWarning
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--check-config", action="store_true")
    options, config_args = parser.parse_known_args(args)
    config, _ = load_expr_config(config_args, Tau2PPOConfig)
    actor_source = config.actor.path
    provenance = collection_provenance(config, actor_source=actor_source)
    domains = validate_tau2_recipe(config)
    econfig = config.econfig
    dataset_kwargs = dict(
        domain=domains,
        provenance=provenance,
        seed=config.seed,
        experiment_mode=config.experiment_mode,
        critic_dev_fraction=config.critic_dev_fraction,
        critic_tasks_per_domain=config.critic_tasks_per_domain,
        critic_rollouts_per_task=config.critic_rollouts_per_task,
    )
    if _dataset_split(config.train_dataset, "train") != "train":
        raise ValueError("Gradient training must use official train tasks")
    train_dataset = get_tau2_dataset(
        **dataset_kwargs,
        split="train",
        algorithm=config.algorithm,
        domain_effective_episodes=config.domain_effective_episodes,
    )
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
    valid_dataset = None
    if config.valid_dataset is not None:
        valid_split = "test" if config.experiment_mode == "formal" else "dev"
        valid_dataset = get_tau2_dataset(**dataset_kwargs, split=valid_split)
    if options.check_config:
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

    # Evaluation sampling is explicit and shared across dev/test.
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gen_args"] = dict(
        top_p=config.eval_gconfig.top_p,
        frequency_penalty=config.eval_gconfig.frequency_penalty,
        seed=config.eval_gconfig.seed,
        max_total_tokens=econfig.context_window_tokens,
        temperature=config.eval_gconfig.temperature,
        max_completion_tokens=config.eval_gconfig.max_new_tokens,
    )

    from examples.tau2.evaluation import Tau2AsyncEvalTrainer

    trainer_cls = Tau2AsyncEvalTrainer if config.algorithm == "grpo" else PPOTrainer
    with trainer_cls(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="examples.tau2.agent.Tau2AgentWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="examples.tau2.agent.Tau2AgentWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
            dynamic_filter_fn=(
                "examples.tau2.train.group_filter"
                if config.dynamic_group_filter
                else None
            ),
        )


if __name__ == "__main__":
    main(sys.argv[1:])
