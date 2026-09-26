"""CPU-only contracts for the τ² SAO/GRPO implementation."""

import json
import logging
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.tau2.contracts import (
    OFFICIAL_SPLIT_SNAPSHOTS,
    bind_policy_request,
    build_sampling_schedule,
    build_task_rows,
    clean_openai_messages,
    keep_rollout_group,
    normalize_domains,
    remaining_generation_budget,
    resolve_pinned_hf_snapshot,
    tau2_qwen_concat_prefix_matcher,
    validate_episode_batch,
    validate_installed_tau2_revision,
)
from examples.tau2.train import collection_provenance, validate_tau2_recipe
from examples.tau2.utils import Tau2PPOConfig
from scripts.tau2.eval_matrix import build_eval_plan, verify_eval_ledger
from scripts.tau2.train_critic import (
    resolve_training_snapshots,
)

from areal.api.cli_args import load_expr_config
from areal.experimental.openai.types import (
    AgentWorkflowResult,
    InteractionWithTokenLogpReward,
)
from areal.infra.workflow_executor import WorkflowExecutor
from areal.utils.data import drop_non_model_forward_metadata
from areal.utils.offload import get_tms_env_vars, normalize_tms_worker_preload


def _interaction(
    interaction_id: str,
    input_tokens: list[int],
    output_tokens: list[int],
    *,
    parent: InteractionWithTokenLogpReward | None = None,
) -> InteractionWithTokenLogpReward:
    response = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_len=len(input_tokens),
        output_len=len(output_tokens),
        output_logprobs=[-0.2] * len(output_tokens),
        output_versions=[3] * len(output_tokens),
    )
    return InteractionWithTokenLogpReward(
        model_response=response,
        reward=1.0,
        parent=parent,
        chat_template_type="concat",
        completion=SimpleNamespace(id=interaction_id, created=0),
        output_message_list=[{"role": "assistant", "content": interaction_id}],
    )


def test_official_snapshot_counts_bind_178_train_and_100_test():
    """The pinned three-domain identity has the PRD's exact task counts."""

    assert sum(int(item["train"]) for item in OFFICIAL_SPLIT_SNAPSHOTS.values()) == 178
    assert sum(int(item["test"]) for item in OFFICIAL_SPLIT_SNAPSHOTS.values()) == 100


def test_installed_tau2_revision_rejects_relabelled_package():
    class FakeDistribution:
        metadata = {"Name": "tau2"}
        version = "1.0.1"

        @staticmethod
        def read_text(name):
            assert name == "direct_url.json"
            return json.dumps(
                {
                    "url": "https://github.com/sierra-research/tau2-bench.git",
                    "vcs_info": {"commit_id": "0" * 40},
                }
            )

    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_installed_tau2_revision(
            distribution_loader=lambda _: FakeDistribution()
        )


def test_task_rows_keep_domain_identity_without_dataset_duplication():
    """Equal task IDs in different domains remain distinct unique tasks."""

    splits = {
        "airline": {"train": ["0", "1"], "test": ["2"]},
        "retail": {"train": ["0"], "test": ["1"]},
    }

    rows = build_task_rows(
        splits,
        domains=("airline", "retail"),
        split="train",
        require_official_snapshot=False,
    )

    assert rows == [
        {
            "domain": "airline",
            "task_id": "0",
            "split": "train",
            "source_id": "tau2:airline:0",
        },
        {
            "domain": "airline",
            "task_id": "1",
            "split": "train",
            "source_id": "tau2:airline:1",
        },
        {
            "domain": "retail",
            "task_id": "0",
            "split": "train",
            "source_id": "tau2:retail:0",
        },
    ]


@pytest.mark.parametrize(
    "prompt_tokens,expected",
    [(0, 4096), (28671, 4096), (28672, 4095), (32766, 1)],
)
def test_generation_budget_enforces_32k_total_and_4k_response(
    prompt_tokens: int, expected: int
):
    """The final consumer leaves SGLang's reserved slot inside the 32K window."""

    assert remaining_generation_budget(prompt_tokens) == expected


@pytest.mark.parametrize("prompt_tokens", [32767, 32768, 32769])
def test_generation_budget_rejects_context_boundary(prompt_tokens: int):
    """No request is sent once the usable context has been exhausted."""

    with pytest.raises(ValueError, match="context_limit"):
        remaining_generation_budget(prompt_tokens)


def _transfer_tool_call_message(
    summary: str,
    *,
    call_id: str = "call_parent",
    ensure_ascii: bool = False,
    include_top_level_name: bool = False,
) -> dict:
    tool_call = {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "transfer_to_human_agents",
            "arguments": json.dumps({"summary": summary}, ensure_ascii=ensure_ascii),
        },
    }
    if include_top_level_name:
        tool_call["name"] = "transfer_to_human_agents"
    return {
        "role": "assistant",
        "content": "<think>Need transfer.</think>\n\n",
        "tool_calls": [tool_call],
    }


def test_tau2_prefix_matcher_accepts_actual_toolcall_continuation_shape():
    """The task-44 transfer tool call remains one parent-child trajectory."""

    summary = "Transfer S61CZX; only H8Q05L has duration ≤3 hours — upgrade done."
    base_messages = [
        {"role": "system", "content": "You are an airline agent."},
        {"role": "user", "content": "I need help changing my booking."},
    ]
    parent_data = [
        *base_messages,
        _transfer_tool_call_message(summary, ensure_ascii=False),
    ]
    child_messages = [
        *base_messages,
        _transfer_tool_call_message(
            summary,
            ensure_ascii=True,
            include_top_level_name=True,
        ),
        {
            "role": "tool",
            "tool_call_id": "call_parent",
            "content": "Transfer successful",
        },
        {
            "role": "assistant",
            "content": "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.",
        },
    ]

    assert tau2_qwen_concat_prefix_matcher(parent_data, child_messages)


def test_tau2_prefix_matcher_rejects_different_tool_arguments():
    """Tool-call normalization does not merge distinct calls or summaries."""

    base_messages = [{"role": "system", "content": "You are an airline agent."}]
    parent_data = [
        *base_messages,
        _transfer_tool_call_message("Transfer customer S61CZX."),
    ]
    child_messages = [
        *base_messages,
        _transfer_tool_call_message(
            "Transfer customer ABC123.",
            ensure_ascii=True,
            include_top_level_name=True,
        ),
    ]

    assert not tau2_qwen_concat_prefix_matcher(parent_data, child_messages)


def test_tau2_prefix_matcher_rejects_different_tool_call_ids():
    parent_data = [_transfer_tool_call_message("same summary", call_id="call_parent")]
    child_messages = [
        _transfer_tool_call_message(
            "same summary",
            call_id="call_child",
            ensure_ascii=True,
            include_top_level_name=True,
        )
    ]

    assert not tau2_qwen_concat_prefix_matcher(parent_data, child_messages)


def test_tau2_prefix_matcher_keeps_non_tool_content_exact():
    parent_data = [{"role": "assistant", "content": "A"}]
    child_messages = [{"role": "assistant", "content": "B"}]

    assert not tau2_qwen_concat_prefix_matcher(parent_data, child_messages)


def test_policy_request_binds_non_thinking_and_preserves_telecom_user_tools():
    """The consumer sees non-thinking while user-side tool history remains intact."""

    messages = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "changed setting"},
        {"role": "user", "content": "continue", "tool_calls": None},
    ]

    bound = bind_policy_request(
        {
            "messages": messages,
            "extra_body": {"chat_template_kwargs": {"custom": "kept"}},
        },
        enable_thinking=False,
    )

    assert bound["extra_body"]["chat_template_kwargs"] == {
        "custom": "kept",
        "enable_thinking": False,
    }
    assert bound["messages"][0]["tool_calls"] == [{"id": "call-1"}]
    assert bound["messages"][1]["role"] == "tool"
    assert "tool_calls" not in bound["messages"][2]
    assert clean_openai_messages(messages)[1] == messages[1]


def test_group_contract_keeps_sao_success_and_rejects_incomplete_grpo():
    """SAO n=1 reward one survives while a seven-sample GRPO group cannot update."""

    assert keep_rollout_group("sao", [1.0], dynamic_filter=True)
    assert validate_episode_batch("grpo", 48) == 6
    assert validate_episode_batch("sao", 48) == 48
    with pytest.raises(ValueError, match="Incomplete grpo"):
        keep_rollout_group("grpo", [0.0] * 7)


def test_sampling_schedule_consumes_exact_domain_episode_budgets():
    rows = [
        {"domain": "airline", "task_id": str(index), "split": "train"}
        for index in range(3)
    ] + [
        {"domain": "retail", "task_id": str(index), "split": "train"}
        for index in range(2)
    ]

    schedule = build_sampling_schedule(
        rows,
        domain_effective_episodes={"airline": 16, "retail": 8},
        algorithm="grpo",
        seed=1,
    )

    assert sorted((row["domain"], row["sample_slot"]) for row in schedule) == [
        ("airline", 0),
        ("airline", 1),
        ("retail", 0),
    ]
    assert all("task_id" in row for row in schedule)


def test_concat_export_rejects_parent_token_prefix_drift():
    """A re-rendered child may not silently discard its real parent actions."""

    parent = _interaction("parent", [1, 2], [3, 4])
    child = _interaction("child", [1, 9, 3, 4, 5], [6], parent=parent)

    with pytest.raises(ValueError, match="complete parent trajectory"):
        child.to_tensor_dict()


def test_episode_result_materializes_explicit_training_metadata():
    """Proxy outcomes reach the production PPO tensor names and shapes."""

    interaction = _interaction("single", [1, 2], [3, 4])
    result = AgentWorkflowResult(
        reward=0.0,
        terminated=False,
        truncated=True,
        bootstrap_mask=False,
        stop_reason="episode_timeout",
        metadata={"official_score": 1.0, "failure_class": "task_budget"},
    )

    interaction.apply_episode_result(result, episode_id=17)
    tensors = interaction.to_tensor_dict()

    torch.testing.assert_close(
        tensors["terminated"], torch.tensor([False]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        tensors["truncated"], torch.tensor([True]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        tensors["bootstrap_mask"], torch.tensor([False]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        tensors["episode_ids"],
        torch.full((1, 4), 17, dtype=torch.int64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        tensors["action_origin_mask"],
        torch.tensor([[False, False, True, True]]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        tensors["official_scores"], torch.tensor([1.0]), rtol=0, atol=0
    )
    assert tensors["task_budget_failure"].item() is True


def test_rollout_audit_metadata_is_removed_before_model_forward():
    inputs = {
        "input_ids": torch.tensor([[1, 2]]),
        "loss_mask": torch.tensor([[0, 1]]),
        "action_origin_mask": torch.tensor([[0, 1]]),
        "episode_ids": torch.tensor([[17, 17]]),
        "terminated": torch.tensor([True]),
        "truncated": torch.tensor([False]),
        "bootstrap_mask": torch.tensor([False]),
        "official_scores": torch.tensor([1.0]),
        "task_budget_failure": torch.tensor([False]),
        "turn_ids": torch.tensor([[-1, 0]]),
    }

    drop_non_model_forward_metadata(inputs)

    assert set(inputs) == {"input_ids", "loss_mask"}


def test_agent_workflow_result_rejects_ambiguous_episode_end():
    """Every accepted episode is exactly terminal or truncated."""

    with pytest.raises(ValueError, match="XOR"):
        AgentWorkflowResult(reward=0.0, terminated=False, truncated=False)


def test_domain_selection_rejects_mixed_plus_specialist():
    """The mixed alias cannot silently override an explicit specialist."""

    with pytest.raises(ValueError, match="cannot be combined"):
        normalize_domains(["mixed", "airline"])


def test_thin_entrypoints_replace_manifest_and_custom_launcher():
    script_dir = Path("scripts/tau2")
    entrypoints = (
        "run.sh",
        "train_critic.sh",
        "train_mixed.sh",
        "train_airline.sh",
        "train_retail.sh",
        "train_telecom.sh",
    )
    for name in entrypoints:
        path = script_dir / name
        assert path.is_file()
        assert os.access(path, os.X_OK)
        subprocess.run(["bash", "-n", str(path)], check=True)
    runner = (script_dir / "run.sh").read_text()
    assert "scripts/sao/runtime_env.sh" in runner
    assert "AREAL_ALLOW_DEFAULT_ADMIN_KEY=1" in runner
    assert "TAU2_DEEPSEEK_ENV_FILE" in runner
    assert "DEEPSEEK_API_KEY" in runner
    assert 'mktemp -d "${TAU2_RUN_ROOT}.XXXXXX"' in runner
    assert "launch-manifest" not in runner
    assert "nvidia/cuda_runtime/lib" in runner
    assert "libcudart.so.12" in runner
    assert "agent-workflow" not in runner


def test_tms_worker_env_preserves_existing_runtime_preload(monkeypatch):
    system_cxx = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    monkeypatch.setenv("LD_PRELOAD", system_cxx)

    env = get_tms_env_vars()

    preloads = env["LD_PRELOAD"].split(":")
    assert preloads[0] == system_cxx
    assert preloads[1].endswith("torch_memory_saver_hook_mode_preload.abi3.so")

    monkeypatch.setenv("LD_PRELOAD", env["LD_PRELOAD"])
    monkeypatch.setenv("TMS_INIT_ENABLE", "1")
    normalize_tms_worker_preload()

    assert os.environ["LD_PRELOAD"] == preloads[1]


def test_tms_worker_env_rejects_missing_hook(monkeypatch):
    monkeypatch.setenv("LD_PRELOAD", "/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
    monkeypatch.setenv("TMS_INIT_ENABLE", "1")

    with pytest.raises(RuntimeError, match="TMS preload hook is missing"):
        normalize_tms_worker_preload()


def test_qualification_configs_resolve_for_all_entrypoints(monkeypatch, tmp_path):
    actor = "Qwen/Qwen3.5-4B@" + "a" * 40
    monkeypatch.setenv("TAU2_ACTOR_PATH", actor)
    monkeypatch.setenv("TAU2_CRITIC_PATH", str(tmp_path / "critic"))
    monkeypatch.setenv("TAU2_CRITIC_INIT_PATH", actor)
    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path / "run"))

    for config_path in ("examples/tau2/config_sao_qualification.yaml",):
        config, _ = load_expr_config(
            ["--config", config_path],
            Tau2PPOConfig,
        )
        validate_tau2_recipe(config)
        assert config.rollout.queue_size >= 2 * config.train_dataset.batch_size
        if config.algorithm == "sao":
            assert config.enable_offload is True
            assert config.actor.fsdp.offload_params is False
            assert config.actor.fsdp.per_layer_optim_step is False
            assert config.critic is not None
            assert config.critic.offload is True
            assert config.critic.fsdp.offload_params is False
            assert config.critic.fsdp.per_layer_optim_step is False

        config.rollout.queue_size = 2 * config.train_dataset.batch_size - 1
        with pytest.raises(ValueError, match="reserved consumer batch"):
            validate_tau2_recipe(config)

    other_domains = {
        "airline": ("retail", "telecom"),
        "retail": ("airline", "telecom"),
        "telecom": ("airline", "retail"),
    }
    for domain, pruned in other_domains.items():
        overrides = [
            f"domains=[{domain}]",
            f"econfig.domain={domain}",
            "train_batch_episodes=1",
            "effective_episodes=1",
            f"domain_effective_episodes.{domain}=1",
            *(f"~domain_effective_episodes.{other}" for other in pruned),
            "train_dataset.batch_size=1",
        ]
        config, _ = load_expr_config(
            [
                "--config",
                "examples/tau2/config_sao_qualification.yaml",
                *overrides,
            ],
            Tau2PPOConfig,
        )
        assert validate_tau2_recipe(config) == (domain,)


def _official_test_rows() -> list[dict[str, str]]:
    return [
        {"domain": domain, "task_id": f"test-{index}", "split": "test"}
        for domain, count in {"airline": 20, "retail": 40, "telecom": 40}.items()
        for index in range(count)
    ]


def _critic_row(task_index: int, critic_split: str) -> dict:
    episode_id = 1000 + task_index
    domain = (
        "airline" if task_index < 30 else "retail" if task_index < 104 else "telecom"
    )
    return {
        "domain": domain,
        "task_id": f"train-{task_index}",
        "split": "train",
        "critic_split": critic_split,
        "episode_id": episode_id,
        "episode_tensor_id": episode_id,
        "attempt_id": f"attempt-{task_index}",
        "input_ids": [1, 2, 3, 4],
        "attention_mask": [1, 1, 1, 1],
        "loss_mask": [0, 1, 0, 1],
        "action_origin_mask": [0, 1, 0, 1],
        "behavior_logprobs": [0.0, -0.1, 0.0, -0.2],
        "versions": [-1, 3, -1, 3],
        "turn_ids": [-1, 0, -1, 1],
        "token_roles": ["prompt", "assistant", "tool", "assistant"],
        "reward": float(task_index % 2),
        "official_score": float(task_index % 2),
        "terminated": True,
        "truncated": False,
        "bootstrap_mask": False,
        "policy_id": "Qwen/Qwen3.5-4B",
        "policy_revision": "policy0-revision",
        "simulator_id": "deepseek-pinned",
    }


def test_runtime_collection_provenance_is_config_derived_not_manifest_bound():
    config = SimpleNamespace(econfig=SimpleNamespace(user_llm="deepseek-pinned"))

    provenance = collection_provenance(
        config, actor_source=f"Qwen/Qwen3.5-4B@{'b' * 40}"
    )

    assert provenance == {
        "policy_id": "Qwen/Qwen3.5-4B",
        "policy_revision": "b" * 40,
        "simulator_id": "deepseek-pinned",
    }


def test_pinned_hf_source_resolves_the_exact_revision():
    calls = []

    def resolver(**kwargs):
        calls.append(kwargs)
        return "/cache/snapshots/abc"

    observed = resolve_pinned_hf_snapshot(
        f"Qwen/Qwen3.5-4B@{'a' * 40}", snapshot_resolver=resolver
    )

    assert observed == "/cache/snapshots/abc"
    assert calls == [{"repo_id": "Qwen/Qwen3.5-4B", "revision": "a" * 40}]


def test_critic_training_resolves_all_actor_backbone_consumers(monkeypatch, tmp_path):
    source = f"Qwen/Qwen3.5-4B@{'a' * 40}"
    monkeypatch.setenv("TAU2_ACTOR_PATH", source)
    monkeypatch.setenv("TAU2_CRITIC_INIT_PATH", source)
    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path / "run"))
    config, _ = load_expr_config(
        ["--config", "examples/tau2/config_critic_production.yaml"],
        Tau2PPOConfig,
    )

    snapshot = resolve_training_snapshots(
        config,
        snapshot_resolver=lambda **_: "/cache/snapshots/pinned",
    )

    assert snapshot == "/cache/snapshots/pinned"
    assert config.actor.path == snapshot
    assert config.critic is not None and config.critic.path == snapshot
    assert config.tokenizer_path == snapshot
    assert config.rollout.tokenizer_path == snapshot
    assert config.sglang.model_path == snapshot
    assert config.vllm.model == snapshot


@pytest.mark.asyncio
async def test_proxy_tensor_dump_seals_same_episode_identity(tmp_path):
    """Production tensor export and dump retain independent mask and one ID."""

    interaction = _interaction("episode", [1, 2], [3, 4])
    interaction.apply_episode_result(
        AgentWorkflowResult(
            reward=1.0,
            terminated=True,
            truncated=False,
            metadata={"official_score": 1.0},
        ),
        episode_id=23,
    )
    executor = object.__new__(WorkflowExecutor)
    executor._get_dump_dir = lambda is_eval: str(tmp_path / "dump")
    executor._get_tokenizer = lambda: SimpleNamespace(
        decode=lambda ids, **kwargs: f"[{len(ids)} tokens]"
    )
    executor.inference_engine = SimpleNamespace(get_version=lambda: 3)
    executor.logger = logging.getLogger("Tau2DumpTest")
    provenance = _critic_row(0, "train")
    success, reason = await executor._dump_trajectory(
        interaction.to_tensor_dict(),
        task_id=7,
        is_eval=False,
        source_data={
            "domain": "airline",
            "task_id": "train-0",
            "split": "train",
            "critic_split": "train",
            "attempt_id": "attempt-0",
            **{
                key: provenance[key]
                for key in (
                    "policy_id",
                    "policy_revision",
                    "simulator_id",
                )
            },
        },
    )
    assert success, reason
    dumped_path = tmp_path / "dump" / "3" / "7.jsonl"
    dumped = json.loads(dumped_path.read_text())
    assert dumped["episode_id"] == dumped["episode_tensor_id"] == 23
    assert dumped["action_origin_mask"] == [0, 0, 1, 1]
    assert dumped["token_roles"] == ["prompt", "prompt", "assistant", "assistant"]


def test_eval_plan_covers_policy0_and_four_by_three_actor_matrix():
    """The ledger plans every official task/repeat for all required checkpoints."""

    checkpoints = {
        branch: f"checkpoint-{branch}"
        for branch in ("policy0", "mixed", "airline", "retail", "telecom")
    }

    plan = build_eval_plan(
        _official_test_rows(),
        checkpoints=checkpoints,
        repeats=4,
        simulator_id="deepseek-pinned",
    )

    assert plan["actor_matrix_units"] == 12
    assert plan["planned_cells"] == 5 * 100 * 4
    assert plan["side_effects"] == {
        "api_calls": 0,
        "gpu_processes": 0,
        "queue_tasks": 0,
    }


def test_eval_ledger_marks_missing_or_infra_cells_partial():
    """Infra failures stay visible and never silently shrink the denominator."""

    checkpoints = {
        branch: f"checkpoint-{branch}"
        for branch in ("policy0", "mixed", "airline", "retail", "telecom")
    }
    plan = build_eval_plan(
        _official_test_rows(),
        checkpoints=checkpoints,
        repeats=1,
        simulator_id="deepseek-pinned",
    )
    first = {**plan["cells"][0], "status": "infra_failed"}

    report = verify_eval_ledger(plan, [first])

    assert report["status"] == "partial"
    assert report["coverage"] == 0.0
    assert report["infra_failed"] == [first["cell_id"]]
    assert len(report["missing"]) == plan["planned_cells"] - 1
