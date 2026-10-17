"""Offline user-boundary contracts using recorded policy spans and official roles."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.tau2.user_boundary import classify_user_boundaries
from examples.tau2.utils import Tau2PPOConfig

from areal.experimental.openai.types import (
    AgentWorkflowResult,
    InteractionWithTokenLogpReward,
)
from areal.utils.data import batched_call

EXPECTED = {
    "202": {5011: 1, 5076: 2, 5856: 3},
    "97": {4974: 2, 5359: 1, 5461: 2, 5837: 3},
    "70": {
        5033: 1,
        5116: 2,
        5170: 2,
        5437: 2,
        5974: 1,
        6145: 2,
        6580: 2,
        7023: 1,
        7299: 3,
    },
}


def _fixture(number):
    return json.loads(
        (
            Path(__file__).parent / "fixtures" / f"tau2_user_boundaries_{number}.json"
        ).read_text()
    )


def _interaction(inputs, outputs, parent=None):
    return InteractionWithTokenLogpReward(
        model_response=SimpleNamespace(
            input_tokens=inputs,
            output_tokens=outputs,
            input_len=len(inputs),
            output_len=len(outputs),
            output_logprobs=[-0.2] * len(outputs),
            output_versions=[1] * len(outputs),
        ),
        parent=parent,
        chat_template_type="concat",
    )


@pytest.mark.parametrize("number", ["202", "70", "97"])
@pytest.mark.parametrize("warm_cache", [False, True])
def test_recorded_official_roles_align_to_exact_generated_endpoints(number, warm_cache):
    """Tools rendered as user and malformed tool tags retain official semantics."""
    fixture = _fixture(number)
    kinds = classify_user_boundaries(fixture["messages"])
    assert kinds == list(EXPECTED[number].values())
    parent = None
    tokens = []
    for a, b in fixture["spans"]:
        inputs = tokens + [0] * (a - len(tokens))
        outputs = [1] * (b - a + 1)
        parent = _interaction(inputs, outputs, parent)
        tokens = inputs + outputs
    if warm_cache:
        parent.to_tensor_dict()
    parent.apply_episode_result(
        AgentWorkflowResult(
            reward=0.0,
            terminated=True,
            truncated=False,
            metadata={"user_boundary_types": kinds, "user_td_beta": 0.25},
        ),
        episode_id=123,
    )
    result = parent.to_tensor_dict()
    expected = torch.zeros((1, fixture["seqlen"]), dtype=torch.int32)
    for position, kind in EXPECTED[number].items():
        expected[0, position] = kind
    torch.testing.assert_close(result["user_boundary_types"], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        result["user_td_beta"], torch.tensor([0.25]), rtol=0, atol=0
    )
    assert not any(
        m.get("raw_data") is None for m in fixture["messages"] if m["role"] == "user"
    )


def test_disabled_export_preserves_existing_tensor_keys():
    """Absent opt-in metadata adds no new payload fields."""
    interaction = _interaction([0, 0], [1, 1])
    interaction.apply_episode_result(
        AgentWorkflowResult(reward=0.0, terminated=True, truncated=False), episode_id=1
    )
    assert "user_boundary_types" not in interaction.to_tensor_dict()
    assert "user_td_beta" not in interaction.to_tensor_dict()


def test_boundary_count_mismatch_refuses_export():
    """A missing policy message cannot silently shift boundary labels."""
    interaction = _interaction([0], [1])
    with pytest.raises(ValueError, match="boundary.*count|count.*boundary"):
        interaction.apply_episode_result(
            AgentWorkflowResult(
                reward=0.0,
                terminated=True,
                truncated=False,
                metadata={"user_boundary_types": [1, 3], "user_td_beta": 0.5},
            ),
            episode_id=1,
        )
        interaction.to_tensor_dict()


def test_terminal_without_user_and_multiple_tool_results_preserve_semantics():
    """Only actual user messages create user boundaries, including user tools."""
    generated = {"role": "assistant", "raw_data": {"id": "policy"}}
    assert classify_user_boundaries([generated]) == [4]
    assert classify_user_boundaries(
        [
            generated,
            {"role": "tool", "requestor": "assistant"},
            {"role": "tool", "requestor": "assistant"},
            generated,
            {"role": "user", "tool_calls": [{"name": "user_tool"}]},
            {"role": "tool", "requestor": "user"},
        ]
    ) == [2, 3]
    assert (
        classify_user_boundaries(
            [{"role": "assistant", "raw_data": None}, {"role": "user"}]
        )
        == []
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_user_boundary_values": 1},
        {"user_td_beta": True},
        {"user_td_beta": float("nan")},
        {"user_td_beta": float("inf")},
        {"user_td_beta": -0.1},
        {"user_td_beta": 1.1},
        {"user_td_beta": 0.5},
        {"algorithm": "grpo", "train_user_boundary_values": True},
        {"algorithm": "critic", "train_user_boundary_values": True},
    ],
)
def test_invalid_user_td_configuration_refuses_early(kwargs):
    """Invalid types, range and algorithm combinations fail before workers."""
    with pytest.raises(ValueError, match="user_td"):
        Tau2PPOConfig(**kwargs)


def test_beta_sequence_scalar_survives_batched_call():
    """Per-episode beta stays scalar through variable-length concatenation."""
    rows = [
        {
            "attention_mask": torch.ones((1, width), dtype=torch.bool),
            "user_td_beta": torch.tensor([beta]),
        }
        for width, beta in [(3, 0.25), (7, 0.5)]
    ]
    observed = batched_call(lambda data: data, rows)
    for result, row in zip(observed, rows, strict=True):
        torch.testing.assert_close(
            result["user_td_beta"], row["user_td_beta"], rtol=0, atol=0
        )


def test_corrupt_official_role_changes_expected_boundary():
    """An adverse fixture mutation proves role assertions detect a misclassification."""
    fixture = _fixture("70")
    fixture["messages"][17] = {"role": "tool", "requestor": "assistant"}
    assert classify_user_boundaries(fixture["messages"])[7] == 2
    assert classify_user_boundaries(fixture["messages"]) != list(
        EXPECTED["70"].values()
    )


@pytest.mark.parametrize("algorithm", ["ppo", "sao"])
def test_enabled_recipe_accepts_valid_and_refuses_mutated_assumptions(
    algorithm, monkeypatch, tmp_path
):
    """Structured loading and recipe revalidation both enforce the feature contract."""
    from examples.tau2.train import validate_tau2_recipe

    from areal.api.cli_args import load_expr_config

    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path))
    monkeypatch.setenv("TAU2_TRIAL_NAME", "user-boundary-test")
    monkeypatch.delenv("TAU2_ACTOR_PATH", raising=False)
    if algorithm == "sao":
        monkeypatch.setenv("TAU2_CRITIC_PATH", str(tmp_path / "critic"))
    else:
        monkeypatch.delenv("TAU2_CRITIC_PATH", raising=False)
    config, _ = load_expr_config(
        ["--config", f"examples/tau2/config_{algorithm}.yaml"], Tau2PPOConfig
    )
    config.train_user_boundary_values = True
    config.user_td_beta = 0.5
    validate_tau2_recipe(config)
    for field, value in [
        ("gae_timestep_unit", "turn"),
        ("discount", 0.9),
        ("critic_gae_lambda", 0.9),
        ("kl_ctl", 0.1),
        ("reward_scaling", 2.0),
        ("reward_bias", 1.0),
        ("overlong_reward_penalty", True),
    ]:
        previous = getattr(config.actor, field)
        setattr(config.actor, field, value)
        with pytest.raises(ValueError, match="user_td"):
            validate_tau2_recipe(config)
        setattr(config.actor, field, previous)
    config.user_td_beta = float("nan")
    with pytest.raises(ValueError, match="user_td"):
        validate_tau2_recipe(config)


def test_boundary_metadata_is_replayable_and_excluded_from_model_forward():
    """Dumps preserve boundary evidence while model inputs omit training metadata."""
    from areal.infra.workflow_executor import WorkflowExecutor
    from areal.utils.data import drop_non_model_forward_metadata

    interaction = _interaction([0, 0], [1, 1])
    interaction.apply_episode_result(
        AgentWorkflowResult(
            reward=0,
            terminated=True,
            truncated=False,
            metadata={"user_boundary_types": [3], "user_td_beta": 0.5},
        ),
        episode_id=1,
    )
    payload = interaction.to_tensor_dict()
    fields = WorkflowExecutor._trajectory_replay_fields(
        payload,
        row=0,
        seqlen=4,
        ids=[0, 0, 1, 1],
        mask=[0, 0, 1, 1],
    )
    assert fields["user_boundary_types"] == [0, 0, 0, 3]
    assert fields["user_td_beta"] == 0.5
    model_inputs = dict(payload)
    drop_non_model_forward_metadata(model_inputs)
    assert "user_boundary_types" not in model_inputs
    assert "user_td_beta" not in model_inputs


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_workflow_only_opt_in_exports_full_official_boundary_metadata(
    monkeypatch, enabled
):
    """The workflow carries terminal evidence without making any policy requests."""
    import examples.tau2.agent as module
    from examples.tau2.utils import Tau2EnvConfig

    messages = _fixture("202")["messages"]

    class Runner:
        def __init__(self, **kwargs):
            self.messages = messages

        def llm_route_evidence(self):
            return {}

        async def run(self, task):
            return SimpleNamespace(
                messages=messages,
                reward_info=None,
                reward=0.0,
                terminated=True,
                truncated=False,
                stop_reason="user_stop",
                error_type=None,
            )

    monkeypatch.setattr(module, "Tau2Runner", Runner)
    monkeypatch.setattr(module, "_get_task", lambda **kwargs: object())
    monkeypatch.delenv("TAU2_USAGE_LOG", raising=False)
    monkeypatch.setenv("TAU2_TEST_BOUNDARY_KEY", "test-user")
    workflow = module.Tau2AgentWorkflow(
        econfig=Tau2EnvConfig(
            user_llm="test-user",
            user_llm_base_url="http://user.invalid/v1",
            user_llm_api_key_env="TAU2_TEST_BOUNDARY_KEY",
        ),
        train_user_boundary_values=enabled,
        user_td_beta=0.5 if enabled else 1.0,
    )
    result = await workflow.run(
        {"task_id": "unused"},
        base_url="http://policy.invalid/v1",
        api_key="test-policy",
    )
    if enabled:
        assert result.metadata["user_boundary_types"] == [1, 2, 3]
        assert result.metadata["user_td_beta"] == 0.5
    else:
        assert "user_boundary_types" not in result.metadata
        assert "user_td_beta" not in result.metadata
