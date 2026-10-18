"""CPU oracles for detector-driven, per-boundary actor TD splitting."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.tau2.utils import Tau2PPOConfig
from tests.test_tau2_user_boundaries import _interaction
from tests.test_user_td_split import ACTIVE, ADAPTIVE, actor, batch

from areal.experimental.openai.types import AgentWorkflowResult
from areal.trainer.ppo.critic import ppo_loss_fn
from areal.utils import stats_tracker

FIXTURES = Path(__file__).parent / "fixtures"


def flagged_batch(points=(14,), beta=0.0):
    data = batch(enabled=True)
    del data["user_td_beta"]
    flags = torch.zeros_like(data["user_boundary_types"], dtype=torch.bool)
    flags[0, list(points)] = True
    data["user_boundary_flags"] = flags
    data["user_boundary_detector_ids"] = flags.to(torch.int32)
    data["user_td_beta_flagged"] = torch.tensor([beta])
    return data


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("lam", [1.0, ADAPTIVE])
@pytest.mark.parametrize("points", [(14,), (2,), (2, 14), ()])
def test_flagged_boundaries_match_residual_oracle_and_keep_critic_targets(
    beta, lam, points
):
    data = flagged_batch(points, beta)
    # Mix a flagged and an entirely unflagged episode in one production batch.
    other = flagged_batch((), beta)
    mixed = {key: torch.cat([value, other[key]]) for key, value in data.items()}
    result = actor(lam)._compute_advantages(mixed)
    reference = actor(lam)._compute_advantages(batch(enabled=True))
    deltas = torch.tensor([0.125, 0.25, 0.125, 0.125, 0.125, 0.25, 0.125, 0.75])
    for boundary, slot, gap in [(2, 1, 0.125), (10, 5, 0.125), (14, 7, 0.625)]:
        if boundary in points:
            deltas[slot] -= (1 - beta) * gap
    decay = 1.0 if lam == 1.0 else float(torch.tensor(1 - 1 / (1.5 * 8)))
    expected = torch.tensor(
        [sum(decay ** (j - i) * deltas[j] for j in range(i, 8)) for i in range(8)]
    )
    torch.testing.assert_close(
        result["advantages"][0, ACTIVE], expected, rtol=1e-6, atol=1e-7
    )
    torch.testing.assert_close(
        result["advantages"][1], reference["advantages"][0], rtol=0, atol=0
    )
    for row in range(2):
        torch.testing.assert_close(
            result["returns"][row], reference["returns"][0], rtol=0, atol=0
        )
        torch.testing.assert_close(
            result["critic_loss_mask"][row],
            reference["critic_loss_mask"][0],
            rtol=0,
            atol=0,
        )
    if 14 in points and beta == 0 and lam == 1:
        assert (
            result["advantages"][0, 13] == data["values"][0, 14] - data["values"][0, 13]
        )


@pytest.mark.parametrize("lam", [1.0, ADAPTIVE])
@pytest.mark.parametrize("beta", [None, 0.0, 0.25, 0.5, 1.0])
def test_disabled_detector_matches_7c939b08_bit_for_bit(lam, beta):
    golden = json.loads((FIXTURES / "per_boundary_beta_base.json").read_text())
    assert golden["commit"] == "7c939b0883123fd380dcc61c837be4c6cdaa5091"
    expected = golden[f"{lam}/{beta}"]
    result = actor(lam)._compute_advantages(batch(enabled=beta is not None, beta=beta))
    for key in ("advantages", "returns", "loss_mask", "critic_loss_mask"):
        if key in expected:
            torch.testing.assert_close(
                result[key], torch.tensor(expected[key]), rtol=0, atol=0
            )
        else:
            assert key not in result
    for clip in (None, 0.5):
        assert (
            ppo_loss_fn(result["values"] + 0.125, result, eps_clip=clip).item()
            == expected[f"loss_{clip}"]
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "legacy",
        "missing_flags",
        "missing_beta",
        "missing_types",
        "missing_ids",
        "bad_shape",
        "bad_dtype",
        "tool_flag",
        "bad_beta",
        "bad_id",
    ],
)
def test_invalid_boundary_transport_refuses_early(mutation):
    data = flagged_batch()
    if mutation == "legacy":
        data["user_td_beta"] = torch.ones(1)
    elif mutation == "missing_flags":
        del data["user_boundary_flags"]
    elif mutation == "missing_beta":
        del data["user_td_beta_flagged"]
    elif mutation == "missing_types":
        del data["user_boundary_types"]
    elif mutation == "missing_ids":
        del data["user_boundary_detector_ids"]
    elif mutation == "bad_shape":
        data["user_boundary_flags"] = torch.zeros(1, 2, dtype=torch.bool)
    elif mutation == "bad_dtype":
        data["user_boundary_flags"] = data["user_boundary_flags"].float()
    elif mutation == "tool_flag":
        data["user_boundary_flags"][0, 6] = True
    elif mutation == "bad_beta":
        data["user_td_beta_flagged"].fill_(float("nan"))
    elif mutation == "bad_id":
        data["user_boundary_detector_ids"][0, 14] = 999
    with pytest.raises((ValueError, RuntimeError), match="user_|boundar|detector"):
        actor(1.0)._compute_advantages(data)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"user_boundary_detector": "u1"},
        {"train_user_boundary_values": True, "user_boundary_detector": "judge"},
        {
            "train_user_boundary_values": True,
            "user_boundary_detector": "u1",
            "user_td_beta": 1.0,
        },
        {
            "train_user_boundary_values": True,
            "user_boundary_detector": "u1",
            "user_td_beta": 0.5,
        },
        {
            "train_user_boundary_values": True,
            "user_boundary_detector": "u1",
            "user_td_beta_flagged": -1,
        },
        {
            "train_user_boundary_values": True,
            "user_boundary_detector": "u1",
            "user_td_beta_flagged": True,
        },
    ],
)
def test_invalid_detector_config_refuses_early(kwargs):
    with pytest.raises(ValueError, match="user_td|boundary"):
        Tau2PPOConfig(algorithm="ppo", **kwargs)


@pytest.mark.parametrize("algorithm", ["ppo", "sao"])
def test_detector_recipe_structured_overrides_validate(
    monkeypatch, tmp_path, algorithm
):
    from examples.tau2.train import validate_tau2_recipe

    from areal.api.cli_args import load_expr_config

    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path))
    monkeypatch.setenv("TAU2_TRIAL_NAME", "per-boundary-test")
    monkeypatch.delenv("TAU2_ACTOR_PATH", raising=False)
    monkeypatch.delenv("TAU2_CRITIC_PATH", raising=False)
    if algorithm == "sao":
        monkeypatch.setenv("TAU2_CRITIC_PATH", str(tmp_path / "critic"))
    config, _ = load_expr_config(
        [
            "--config",
            f"examples/tau2/config_{algorithm}.yaml",
            "+train_user_boundary_values=true",
            "+user_boundary_detector=u1",
            "+user_td_beta_flagged=0.25",
        ],
        Tau2PPOConfig,
    )
    assert config.user_td_beta is None
    assert config.user_td_beta_flagged == 0.25
    validate_tau2_recipe(config)
    config.user_td_beta = 1.0
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_tau2_recipe(config)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,expected",
    [
        ("u1_positive", "U1"),
        ("u1_negative", None),
        ("u7a_positive", None),
        ("u7a_negative", None),
    ],
)
async def test_recorded_rule_detector_cases(name, expected):
    from examples.tau2.user_boundary import (
        RuleUserBoundaryDetector,
        detect_user_boundaries,
    )

    record = json.loads((FIXTURES / "tau2_boundary_detectors.json").read_text())[name]
    result = await detect_user_boundaries(
        record["messages"], record["stop"], RuleUserBoundaryDetector(), final_reward=0.0
    )
    assert result["user_boundary_flags"][-1] == (expected is not None)
    assert result["user_boundary_detectors"][-1] == expected
    assert len(result["user_boundary_flags"]) == len(result["user_boundary_types"])


@pytest.mark.asyncio
async def test_async_detector_sees_each_user_boundary_and_only_prior_history():
    from examples.tau2.user_boundary import BoundaryDetection, detect_user_boundaries

    messages = json.loads((FIXTURES / "tau2_user_boundaries_202.json").read_text())[
        "messages"
    ]
    seen = []

    async def detector(context):
        seen.append(context)
        assert context.messages[-1]["role"] == "user"
        return BoundaryDetection(flag=True, detector="gold")

    result = await detect_user_boundaries(
        messages, "user_stop", detector, final_reward=0.0
    )
    assert result["user_boundary_flags"] == [True, False, True]
    assert result["user_boundary_detectors"] == ["gold", None, "gold"]
    assert len(seen) == 2
    assert len(seen[0].messages) == 4


@pytest.mark.parametrize("warm_cache", [False, True])
def test_flag_transport_batching_dump_and_forward_filter(warm_cache):
    from areal.infra.workflow_executor import WorkflowExecutor
    from areal.utils.data import batched_call, drop_non_model_forward_metadata

    parent = _interaction([0, 0], [1, 1])
    interaction = _interaction([0, 0, 1, 1, 0, 0], [1, 1], parent)
    if warm_cache:
        interaction.to_tensor_dict()
    interaction.apply_episode_result(
        AgentWorkflowResult(
            reward=2,
            terminated=True,
            truncated=False,
            metadata={
                "user_boundary_types": [1, 3],
                "user_boundary_flags": [True, False],
                "user_boundary_detectors": ["U1", None],
                "user_td_beta_flagged": 0.0,
                "user_boundary_gate_suppressed": False,
            },
        ),
        episode_id=1,
    )
    payload = interaction.to_tensor_dict()
    assert "user_td_beta" not in payload
    assert payload["user_boundary_flags"].tolist() == [
        [False, False, False, True, False, False, False, False]
    ]
    longer = _interaction([0] * 10, [1, 1, 1])
    longer.apply_episode_result(
        AgentWorkflowResult(
            reward=0,
            terminated=True,
            truncated=False,
            metadata={
                "user_boundary_types": [3],
                "user_boundary_flags": [False],
                "user_boundary_detectors": [None],
                "user_td_beta_flagged": 0.5,
                "user_boundary_gate_suppressed": True,
            },
        ),
        episode_id=2,
    )
    other_payload = longer.to_tensor_dict()
    assert payload["user_boundary_gate_suppressed"].tolist() == [False]
    assert other_payload["user_boundary_gate_suppressed"].tolist() == [True]
    rows = batched_call(lambda data: data, [payload, other_payload])
    torch.testing.assert_close(
        rows[0]["user_boundary_flags"], payload["user_boundary_flags"], rtol=0, atol=0
    )
    for key in (
        "user_boundary_flags",
        "user_boundary_detector_ids",
        "user_td_beta_flagged",
        "user_boundary_gate_suppressed",
    ):
        torch.testing.assert_close(rows[1][key], other_payload[key], rtol=0, atol=0)
    fields = WorkflowExecutor._trajectory_replay_fields(
        payload, row=0, seqlen=8, ids=[0] * 8, mask=[0, 0, 1, 1, 0, 0, 1, 1]
    )
    assert fields["user_boundary_flags"] == payload["user_boundary_flags"][0].tolist()
    assert fields["user_boundary_detectors"][3] == "U1"
    assert fields["user_boundary_detectors"][7] is None
    assert fields["user_td_beta_flagged"] == 0
    assert fields["user_boundary_gate_suppressed"] is False
    inputs = dict(payload)
    drop_non_model_forward_metadata(inputs)
    assert not any(k.startswith("user_") for k in inputs)


@pytest.mark.parametrize(
    "mutation",
    [
        "both_modes",
        "missing_name",
        "bad_count",
        "unknown_name",
        "tool_flag",
        "nan_beta",
    ],
)
def test_export_rejects_inconsistent_detector_metadata(mutation):
    metadata = {
        "user_boundary_types": [3],
        "user_boundary_flags": [True],
        "user_boundary_detectors": ["U1"],
        "user_td_beta_flagged": 0.0,
    }
    if mutation == "both_modes":
        metadata["user_td_beta"] = 1.0
    elif mutation == "missing_name":
        del metadata["user_boundary_detectors"]
    elif mutation == "bad_count":
        metadata["user_boundary_flags"] = []
    elif mutation == "unknown_name":
        metadata["user_boundary_detectors"] = ["unknown"]
    elif mutation == "tool_flag":
        metadata["user_boundary_types"] = [2]
    elif mutation == "nan_beta":
        metadata["user_td_beta_flagged"] = float("nan")
    interaction = _interaction([0, 0], [1, 1])
    with pytest.raises(ValueError, match="user_td|boundar"):
        interaction.apply_episode_result(
            AgentWorkflowResult(
                reward=0, terminated=True, truncated=False, metadata=metadata
            ),
            episode_id=1,
        )
        interaction.to_tensor_dict()


def test_flagged_statistics_count_detectors_and_initial_user_segments():
    from areal.trainer.ppo.user_boundary import log_user_boundary_stats

    stats_tracker.export_all()
    data = actor(1.0)._compute_advantages(flagged_batch((2, 14)))
    data["user_boundary_detector_ids"][0, 14] = 2
    log_user_boundary_stats(data)
    stats = stats_tracker.export_all()
    assert stats["user_boundary/flagged_count"] == 2
    assert stats["user_boundary/detector_U1_count"] == 1
    assert stats["user_boundary/detector_U7a_count"] == 1
    assert stats["user_boundary/flagged/user_segment_mean"] == 0.375
    log_user_boundary_stats(actor(1.0)._compute_advantages(flagged_batch(())))
    stats = stats_tracker.export_all()
    assert stats["user_boundary/flagged_count"] == 0
    assert stats["user_boundary/detector_U1_count"] == 0
    assert stats["user_boundary/flagged/user_segment_mean"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("reward", [0.0, 0.5, 1.0, 1.25])
@pytest.mark.parametrize("stop", ["user_stop", "timeout", "context_limit"])
@pytest.mark.parametrize("injected", [False, True])
async def test_workflow_records_detected_boundaries_before_tensor_export(
    monkeypatch, tmp_path, reward, stop, injected
):
    import examples.tau2.agent as module
    from examples.tau2.utils import Tau2EnvConfig

    record = json.loads((FIXTURES / "tau2_boundary_detectors.json").read_text())[
        "u1_positive"
    ]
    messages = record["messages"]
    written = []

    class Runner:
        def __init__(self, **kwargs):
            self.messages = messages

        def llm_route_evidence(self):
            return {}

        async def run(self, task):
            return SimpleNamespace(
                messages=messages,
                reward_info=None,
                reward=reward,
                terminated=stop == "user_stop",
                truncated=stop != "user_stop",
                stop_reason=stop,
                error_type=None,
            )

    monkeypatch.setattr(module, "Tau2Runner", Runner)
    monkeypatch.setattr(module, "_get_task", lambda **kwargs: object())
    monkeypatch.setattr(
        module, "write_trajectory_record", lambda root, record: written.append(record)
    )
    monkeypatch.delenv("TAU2_USAGE_LOG", raising=False)
    monkeypatch.setenv("TAU2_BOUNDARY_TEST_KEY", "test-user")
    from examples.tau2.user_boundary import BoundaryDetection

    async def detector(context):
        return BoundaryDetection(True, "judge")

    workflow = module.Tau2AgentWorkflow(
        econfig=Tau2EnvConfig(
            user_llm="test-user",
            user_llm_base_url="http://user.invalid/v1",
            user_llm_api_key_env="TAU2_BOUNDARY_TEST_KEY",
        ),
        train_user_boundary_values=True,
        user_boundary_detector=detector if injected else "u1",
        trajectory_dir=str(tmp_path),
    )

    # Recorded message objects need only implement the dump API used by workflow.
    class Message(dict):
        def model_dump(self, **kwargs):
            return dict(self)

    messages[:] = [Message(m) for m in messages]
    monkeypatch.setattr(module, "policy_turns", lambda messages: [])
    result = await workflow.run(
        {"task_id": "unused"},
        base_url="http://policy.invalid/v1",
        api_key="test-policy",
    )
    hit = injected or stop == "user_stop"
    flagged = hit and reward < 1
    assert result.reward == reward
    assert result.metadata["user_boundary_flags"][-1] == flagged
    assert result.metadata["user_boundary_detectors"][-1] == (
        ("judge" if injected else "U1") if flagged else None
    )
    assert result.metadata["user_boundary_gate_suppressed"] == (hit and reward >= 1)
    assert result.metadata["user_td_beta_flagged"] == 0
    assert "user_td_beta" not in result.metadata
    for key in (
        "user_boundary_types",
        "user_boundary_flags",
        "user_boundary_detectors",
        "user_boundary_gate_suppressed",
    ):
        assert written[0][key] == result.metadata[key]


@pytest.mark.parametrize("name", ["rules", "u7a", "U7a"])
def test_removed_training_detectors_are_rejected(name):
    with pytest.raises(ValueError, match="user_boundary_detector"):
        Tau2PPOConfig(
            algorithm="ppo",
            train_user_boundary_values=True,
            user_boundary_detector=name,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("reward", [0.0, 0.999, 1.0, 1.5])
@pytest.mark.parametrize("name", ["u1_positive", "u7a_positive", "u1_negative"])
async def test_u1_failure_gate_records_only_suppressed_hits(reward, name):
    from examples.tau2.user_boundary import (
        RuleUserBoundaryDetector,
        detect_user_boundaries,
    )

    record = json.loads((FIXTURES / "tau2_boundary_detectors.json").read_text())[name]
    result = await detect_user_boundaries(
        record["messages"],
        record["stop"],
        RuleUserBoundaryDetector(),
        final_reward=reward,
    )
    hit = name == "u1_positive"
    assert result["user_boundary_flags"][-1] == (hit and reward < 1)
    assert result["user_boundary_detectors"][-1] == (
        "U1" if hit and reward < 1 else None
    )
    assert result["user_boundary_gate_suppressed"] == (hit and reward >= 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("reward", [0.0, 1.0])
@pytest.mark.parametrize("is_async", [False, True])
async def test_injected_detector_is_gated_at_every_user_boundary(reward, is_async):
    from examples.tau2.user_boundary import BoundaryDetection, detect_user_boundaries

    messages = json.loads((FIXTURES / "tau2_user_boundaries_202.json").read_text())[
        "messages"
    ]
    seen = []

    def detector(context):
        seen.append(context.boundary_type)
        return BoundaryDetection(True, "gold")

    async def async_detector(context):
        return detector(context)

    result = await detect_user_boundaries(
        messages,
        "timeout",
        async_detector if is_async else detector,
        final_reward=reward,
    )
    assert seen == [1, 3]
    assert result["user_boundary_flags"] == [reward < 1, False, reward < 1]
    assert result["user_boundary_detectors"] == (
        ["gold", None, "gold"] if reward < 1 else [None, None, None]
    )
    assert result["user_boundary_gate_suppressed"] == (reward >= 1)


def test_gate_suppression_stats_count_episodes_not_boundaries():
    from areal.trainer.ppo.user_boundary import log_user_boundary_stats

    stats_tracker.export_all()
    data = actor(1.0)._compute_advantages(flagged_batch(()))
    data = {
        key: value.repeat(3, *([1] * (value.ndim - 1))) for key, value in data.items()
    }
    data["user_boundary_gate_suppressed"] = torch.tensor([True, False, True])
    log_user_boundary_stats(data)
    stats = stats_tracker.export_all()
    assert stats["user_boundary/gate_suppressed_count"] == 2
    assert stats["user_boundary/flagged_count"] == 0
    assert stats["user_boundary/detector_U1_count"] == 0
    data["user_boundary_gate_suppressed"].fill_(False)
    log_user_boundary_stats(data)
    assert stats_tracker.export_all()["user_boundary/gate_suppressed_count"] == 0


def test_detector_transport_ids_remain_stable():
    from areal.trainer.ppo.user_boundary import USER_BOUNDARY_DETECTOR_NAMES

    assert USER_BOUNDARY_DETECTOR_NAMES == (None, "U1", "U7a", "gold", "judge")


@pytest.mark.asyncio
@pytest.mark.parametrize("requires_transfer", [False, True])
async def test_workflow_gate_uses_reward_after_fixed_transfer_rule(
    monkeypatch, tmp_path, requires_transfer
):
    from contextlib import nullcontext

    from tau2.data_model.simulation import RewardInfo
    from tau2.data_model.tasks import Task

    import examples.tau2.agent as module
    from examples.tau2.utils import Tau2EnvConfig

    class Message(SimpleNamespace):
        def model_dump(self, **kwargs):
            return vars(self)

    messages = [
        Message(
            role="assistant", content="Shall I proceed?", tool_calls=[], raw_data={}
        ),
        Message(
            role="user", content="Yes, please. ###STOP###", tool_calls=[], raw_data=None
        ),
    ]
    task = Task(
        id="test",
        user_scenario={"instructions": "Please help me."},
        evaluation_criteria={
            "actions": [
                {
                    "action_id": "transfer",
                    "name": "transfer_to_human_agents",
                    "arguments": {},
                }
            ]
            if requires_transfer
            else []
        },
    )
    monkeypatch.setattr(module, "_get_task", lambda **kwargs: task)
    monkeypatch.setattr(module.Tau2Runner, "_get_environment", lambda self: object())
    monkeypatch.setattr(
        module.Tau2Runner, "_get_agent_and_user", lambda self, **kwargs: (None, None)
    )
    monkeypatch.setattr(
        module.Tau2Runner, "_nl_assertion_evaluation", lambda self: nullcontext()
    )
    monkeypatch.setattr(
        module.Tau2Runner,
        "_get_orchestrator",
        lambda self, **kwargs: SimpleNamespace(
            run=lambda: SimpleNamespace(
                messages=messages, termination_reason="user_stop"
            )
        ),
    )
    monkeypatch.setattr(
        module, "evaluate_simulation", lambda **kwargs: RewardInfo(reward=1.0)
    )
    monkeypatch.setenv("TAU2_BOUNDARY_TEST_KEY", "test-user")
    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path))
    monkeypatch.delenv("TAU2_USAGE_LOG", raising=False)
    workflow = module.Tau2AgentWorkflow(
        econfig=Tau2EnvConfig(
            domain="telecom",
            benchmark_variant="fixed",
            user_llm="test-user",
            user_llm_base_url="http://user.invalid/v1",
            user_llm_api_key_env="TAU2_BOUNDARY_TEST_KEY",
        ),
        train_user_boundary_values=True,
        user_boundary_detector="u1",
    )
    result = await workflow.run(
        {"task_id": "test"}, base_url="http://policy.invalid/v1", api_key="test"
    )
    assert result.reward == (0.0 if requires_transfer else 1.0)
    assert result.metadata["user_boundary_flags"] == [requires_transfer]
    assert result.metadata["user_boundary_gate_suppressed"] is (not requires_transfer)
    if requires_transfer:
        assert result.metadata["reward_info"]["info"]["official_reward"] == 1.0
