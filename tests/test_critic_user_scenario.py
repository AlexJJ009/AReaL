"""CPU contracts for scenario-only privileged critic transport and alignment."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from tau2.data_model.tasks import Task

from examples.tau2.utils import Tau2CriticConfig, Tau2PPOConfig
from tests.test_sao_dual_lambda import TinyTrainEngine, make_actor, rollout
from tests.test_tau2_user_boundaries import _interaction

from areal.api.cli_args import MicroBatchSpec, PPOCriticConfig
from areal.experimental.openai.types import AgentWorkflowResult
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.trainer.ppo.critic import PPOCritic, ppo_loss_fn
from areal.trainer.ppo.user_boundary import critic_loss_mask
from areal.utils import stats_tracker
from areal.utils.data import (
    batched_call,
    concat_padded_tensors,
    drop_non_model_forward_metadata,
    pack_tensor_dict,
    split_padded_tensor_dict_into_mb_list,
)
from areal.utils.functional import loss_reduction_weight

EXPECTED_PROFILE_HEADER = (
    "Privileged context for value estimation. The text below is the hidden scenario "
    "given to the simulated user of this conversation: why the user is contacting "
    "the assistant, what the user knows and does not know, and how the user has "
    "been instructed to behave. The assistant never sees this text. Use it to "
    "estimate how likely the conversation is to end with the task completed "
    "correctly, and to tell whether later user messages follow or deviate from "
    "this scenario."
)


@pytest.fixture(autouse=True)
def isolate_statistics():
    stats_tracker.export_all()
    yield
    stats_tracker.export_all()


def task():
    return Task.model_validate(
        {
            "id": "scenario-test",
            "user_scenario": {
                "persona": "Patient caller",
                "instructions": {
                    "domain": "retail",
                    "reason_for_call": "Return one item",
                    "known_info": "Order ABC",
                    "unknown_info": "Refund timing",
                    "task_instructions": "Ask for a refund",
                },
            },
            "evaluation_criteria": {
                "actions": [
                    {
                        "action_id": "GOLD_ACTION_771",
                        "name": "gold_write_771",
                        "arguments": {"secret_key_771": "GOLD_ARGUMENT_771"},
                    }
                ],
                "nl_assertions": ["GOLD_ASSERTION_771"],
                "reward_basis": ["NL_ASSERTION"],
            },
        }
    )


def with_prefix(data, prefixes=((11, 12), (13, 14, 15))):
    data = copy.deepcopy(data)
    width = max(map(len, prefixes))
    data["critic_prefix_input_ids"] = torch.tensor(
        [list(p) + [0] * (width - len(p)) for p in prefixes]
    )
    data["critic_prefix_lengths"] = torch.tensor(list(map(len, prefixes)))
    return data


def test_profile_is_exact_simulator_text_and_excludes_evaluation_fields():
    from examples.tau2.critic_scenario import render_critic_profile

    source = task()
    rendered = render_critic_profile(source)
    assert rendered == (
        "<|im_start|>system\n<user_profile>\n"
        + EXPECTED_PROFILE_HEADER
        + "\n\n"
        + str(source.user_scenario)
        + "\n</user_profile><|im_end|>\n"
    )
    for marker in (
        "evaluation_criteria",
        "actions",
        "nl_assertions",
        "reward_basis",
        "GOLD_ASSERTION_771",
        "NL_ASSERTION",
        "GOLD_ACTION_771",
        "gold_write_771",
        "secret_key_771",
        "GOLD_ARGUMENT_771",
    ):
        assert marker not in rendered
    changed = source.model_copy(deep=True)
    changed.evaluation_criteria.nl_assertions = ["OTHER_GOLD_999"]
    assert render_critic_profile(changed) == rendered


def test_profile_header_is_identical_for_airline_retail_and_pack(tmp_path):
    import hashlib

    from examples.tau2 import task_pack
    from examples.tau2.critic_scenario import render_critic_profile

    sources = []
    for domain, task_id in (
        ("airline", "airline-test"),
        ("retail", "retail-test"),
    ):
        source = task().model_copy(deep=True)
        source.id = task_id
        source.user_scenario.instructions.domain = domain
        source.user_scenario.instructions.reason_for_call = task_id
        sources.append(source)
    pack_source = task().model_copy(deep=True)
    pack_source.id = "pack-test"
    pack_source.user_scenario.instructions = "Return my pack order."
    domain_dir = tmp_path / "retail"
    domain_dir.mkdir()
    files = {
        "retail/tasks.json": json.dumps([pack_source.model_dump()]),
        "retail/task_db.json": json.dumps({"pack-test": "retail/db.json"}),
        "retail/db.json": "{}",
    }
    for name, content in files.items():
        (tmp_path / name).write_text(content)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "name": "header-test",
                "source": "unit-test",
                "files": {
                    name: hashlib.sha256(content.encode()).hexdigest()
                    for name, content in files.items()
                },
            }
        )
    )
    sources.append(task_pack.get_task(str(tmp_path), "retail", "pack-test"))
    headers = [
        render_critic_profile(source).split(str(source.user_scenario), 1)[0]
        for source in sources
    ]
    assert (
        headers
        == ["<|im_start|>system\n<user_profile>\n" + EXPECTED_PROFILE_HEADER + "\n\n"]
        * 3
    )


def test_missing_or_unexpected_scenario_fails_before_rendering():
    from examples.tau2.critic_scenario import render_critic_profile

    with pytest.raises(ValueError, match="scenario"):
        render_critic_profile(SimpleNamespace(user_scenario=None))
    poisoned = task().user_scenario.model_copy(
        update={"evaluation_criteria": "DO_NOT_RENDER"}
    )
    with pytest.raises(ValueError, match="scenario"):
        render_critic_profile(SimpleNamespace(user_scenario=poisoned))


def test_missing_scenario_counts_failure_without_tokenizing():
    from examples.tau2.critic_scenario import critic_scenario_metadata

    class Tokenizer:
        def encode(self, *args, **kwargs):
            pytest.fail("Missing scenario must fail before tokenization")

    with pytest.raises(ValueError, match="scenario"):
        critic_scenario_metadata(SimpleNamespace(user_scenario=None), Tokenizer())
    assert stats_tracker.export_all()["critic_scenario/missing_scenario"] == 1


def test_disabled_helpers_preserve_original_object_and_inputs():
    from areal.trainer.ppo.critic_input import actor_input, prepare_critic_input

    data = rollout()
    assert actor_input(data) is data
    assert prepare_critic_input(data) is data


@pytest.mark.parametrize("warm_cache", [False, True])
def test_metadata_export_batch_split_pack_and_rpc_keep_per_sample_prefix(warm_cache):
    samples = []
    prefixes = [[11, 12, 13, 14, 15, 11, 12], [14, 15]]
    for prefix in prefixes:
        interaction = _interaction([1, 2], [3, 4])
        if warm_cache:
            interaction.to_tensor_dict()
        interaction.apply_episode_result(
            AgentWorkflowResult(
                reward=1,
                terminated=True,
                truncated=False,
                metadata={"critic_prefix_input_ids": prefix},
            ),
            episode_id=len(samples),
        )
        samples.append(interaction.to_tensor_dict())
    for sample, prefix in zip(samples, prefixes):
        assert sample["critic_prefix_lengths"].tolist() == [len(prefix)]
    combined = concat_padded_tensors(samples)
    decoded = deserialize_value(json.loads(json.dumps(serialize_value(combined))))
    # Advantage computation uses this same concatenate/unpad round trip. Prefix
    # lengths may exceed the original sequence width and must not be trimmed.
    restored = batched_call(lambda x: x, samples)
    for sample, prefix in zip(restored, prefixes):
        assert sample["critic_prefix_input_ids"][0, : len(prefix)].tolist() == prefix
    mbs = split_padded_tensor_dict_into_mb_list(decoded, MicroBatchSpec(n_mbs=2))
    for mb, row in zip(mbs.mbs, mbs.forward_indices):
        prefix = prefixes[row]
        assert mb["critic_prefix_lengths"].tolist() == [len(prefix)]
        assert mb["critic_prefix_input_ids"][0, : len(prefix)].tolist() == prefix
        packed = pack_tensor_dict(mb)
        assert torch.equal(
            packed["critic_prefix_input_ids"], mb["critic_prefix_input_ids"]
        )
    model_inputs = dict(decoded)
    drop_non_model_forward_metadata(model_inputs)
    assert "critic_prefix_input_ids" not in model_inputs
    assert "critic_prefix_lengths" not in model_inputs


def test_actor_actual_forward_inputs_are_bit_identical():
    actor = make_actor(1.0)
    captured = []

    def forward(input_, aggregate_fn):
        captured.append(copy.deepcopy(input_))
        return torch.zeros_like(input_["input_ids"], dtype=torch.float32)

    actor.engine.forward = forward
    original = rollout()
    actor._compute_logp(copy.deepcopy(original))
    actor._compute_logp(with_prefix(original))
    assert captured[0].keys() == captured[1].keys()
    for key in captured[0]:
        assert torch.equal(captured[0][key], captured[1][key]), key


def test_actor_update_engine_inputs_and_loss_are_bit_identical():
    actor = make_actor(1.0)
    data = actor._compute_advantages(rollout())
    calls = []

    def train_batch(data, loss_fn, loss_weight_fn):
        calls.append({key: value.detach().clone() for key, value in data.items()})
        return {"update_successful": 1.0, "grad_norm": 0.0, "lr": 0.01}

    actor.engine.train_batch = train_batch
    actor._ppo_update(copy.deepcopy(data))
    actor._ppo_update(with_prefix(data))
    assert calls[0].keys() == calls[1].keys()
    for key in calls[0]:
        assert torch.equal(calls[0][key], calls[1][key]), key


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_workflow_attaches_scenario_ids_only_when_enabled(
    monkeypatch, tmp_path, enabled
):
    import examples.tau2.agent as agent_mod
    from examples.tau2.critic_scenario import render_critic_profile
    from examples.tau2.utils import Tau2RunInfo
    from tests.test_tau2_official_runtime import _workflow_with_fake_runner

    import areal.utils.hf_utils as hf_utils

    source = task()

    async def run(runner, actual_task):
        assert actual_task is source
        return Tau2RunInfo.model_construct(
            reward=1.0,
            messages=[],
            reward_info=None,
            error_type=None,
            stop_reason="user_stop",
        )

    workflow = _workflow_with_fake_runner(monkeypatch, tmp_path, run)
    workflow.critic_user_scenario = enabled
    workflow.critic_tokenizer_path = "pinned-actor-tokenizer"
    monkeypatch.setattr(agent_mod, "_get_task", lambda **kwargs: source)
    calls = []

    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert not add_special_tokens
            calls.append(text)
            return [11, 12, 13]

    def load(path):
        assert path == "pinned-actor-tokenizer"
        return Tokenizer()

    monkeypatch.setattr(hf_utils, "load_hf_tokenizer", load)
    result = await workflow.run(
        {"task_id": source.id},
        base_url="http://policy.invalid/v1",
        api_key="test-session",
    )
    assert ("critic_prefix_input_ids" in result.metadata) == enabled
    assert calls == ([render_critic_profile(source)] if enabled else [])


def test_replay_dump_keeps_prefix_separate_from_actor_tokens():
    from areal.infra.workflow_executor import WorkflowExecutor

    data = with_prefix(rollout())
    result = WorkflowExecutor._trajectory_replay_fields(
        data, row=1, seqlen=4, ids=[5, 6, 7, 8], mask=[0, 1, 1, 1]
    )
    assert result["input_ids"] == [5, 6, 7, 8]
    assert result["critic_prefix_input_ids"] == [13, 14, 15]
    assert result["critic_prefix_lengths"] == 3


def test_distinct_prefix_lengths_align_values_to_hand_computed_positions():
    class PositionCritic:
        def eval(self):
            pass

        def forward(self, input_, aggregate_fn):
            ids = input_["input_ids"]
            positions = torch.arange(ids.shape[1]).expand_as(ids)
            assert torch.equal(input_["position_ids"], positions)
            assert input_["attention_mask"].sum(-1).tolist() == [6, 7]
            # FSDP reconstructs only max(valid sequence length), omitting the
            # extra right padding supplied in input_.
            return aggregate_fn([(100 * positions[:, :7] + ids[:, :7]).unsqueeze(-1)])

    source = with_prefix(rollout())
    result = PPOCritic(PPOCriticConfig(), PositionCritic())._compute_values(source)
    assert result.shape == source["input_ids"].shape
    assert result[:, :4].tolist() == [[201, 302, 403, 504], [305, 406, 507, 608]]
    assert result[:, 4].tolist() == [0, 0]
    assert source["input_ids"].tolist() == rollout()["input_ids"].tolist()


@pytest.mark.parametrize("eps_clip", [None, 0.2])
@pytest.mark.parametrize("reduction", ["token_mean", "sequence_mean"])
@pytest.mark.parametrize("packed", [False, True])
def test_prefix_loss_gradients_and_weights_equal_original(eps_clip, reduction, packed):
    from areal.trainer.ppo.critic_input import prepare_critic_input

    source = rollout()
    source["returns"] = torch.ones_like(source["values"])
    source["critic_loss_mask"] = source["loss_mask"].clone()
    extended = prepare_critic_input(with_prefix(source))
    if packed:
        source, extended = pack_tensor_dict(source), pack_tensor_dict(extended)
    base = torch.nn.Embedding(16, 1)
    with torch.no_grad():
        base.weight.copy_(torch.arange(16).float()[:, None] / 16)
    prefixed = copy.deepcopy(base)
    for data, model in ((source, base), (extended, prefixed)):
        data["test_loss"] = ppo_loss_fn(
            model(data["input_ids"]), data, eps_clip, reduction
        )
        data["test_loss"].backward()
    torch.testing.assert_close(
        extended["test_loss"], source["test_loss"], rtol=0, atol=0
    )
    torch.testing.assert_close(prefixed.weight.grad, base.weight.grad, rtol=0, atol=0)
    for data in (source, extended):
        data["test_weight"] = loss_reduction_weight(
            critic_loss_mask(data), reduction, data.get("cu_seqlens")
        )
    assert source["test_weight"] == extended["test_weight"]


def test_production_critic_update_masks_prefix_without_changing_actor_batch():
    actor = make_actor(1.0)
    data = actor._compute_advantages(rollout())
    config = PPOCriticConfig(ppo_n_minibatches=1, eps_clip=None)
    base_engine, prefix_engine = (
        TinyTrainEngine(critic=True),
        TinyTrainEngine(critic=True),
    )
    original = copy.deepcopy(data)
    PPOCritic(config, base_engine)._ppo_update(copy.deepcopy(data))
    base_stats = stats_tracker.export_all()
    privileged = with_prefix(data)
    PPOCritic(config, prefix_engine)._ppo_update(privileged)
    prefix_stats = stats_tracker.export_all()
    assert prefix_stats["critic_scenario/prefix_len/avg"] == 2.5
    assert prefix_stats["critic_scenario/prefix_len/max"] == 3
    assert prefix_stats["critic_scenario/missing_scenario"] == 0
    for key in (
        "n_valid_tokens",
        "critic_target",
        "critic_target_second_moment",
        "critic_mse",
        "critic_residual",
    ):
        assert base_stats[key] == prefix_stats[key], key
    assert privileged["input_ids"].shape == original["input_ids"].shape
    torch.testing.assert_close(
        base_engine.model.weight, prefix_engine.model.weight, rtol=0, atol=0
    )
    assert base_engine.losses == prefix_engine.losses
    gradient, mask = prefix_engine.output_gradients[0]
    assert gradient.shape[1] == original["input_ids"].shape[1] + 3
    for row, length in enumerate((2, 3)):
        assert not mask[row, :length].any()
        assert not gradient[row, :length].any()


@pytest.mark.parametrize("algorithm", ["grpo", "sao", "critic"])
def test_scenario_flag_requires_ppo_with_critic(algorithm):
    with pytest.raises(ValueError, match="critic_user_scenario"):
        Tau2PPOConfig(
            algorithm=algorithm, critic=Tau2CriticConfig(), critic_user_scenario=True
        )


def test_scenario_config_rejects_missing_critic_and_detector_but_accepts_beta():
    with pytest.raises(ValueError, match="critic_user_scenario"):
        Tau2PPOConfig(algorithm="ppo", critic_user_scenario=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        Tau2PPOConfig(
            algorithm="ppo",
            critic=Tau2CriticConfig(),
            critic_user_scenario=True,
            train_user_boundary_values=True,
            user_boundary_detector="u1",
        )
    cfg = Tau2PPOConfig(
        algorithm="ppo",
        critic=Tau2CriticConfig(),
        critic_user_scenario=True,
        train_user_boundary_values=True,
        user_td_beta=0.5,
    )
    cfg.validate_user_td_split()
    assert Tau2PPOConfig().critic_user_scenario is False
