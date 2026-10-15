"""CPU numeric oracles for fresh-batch and repeated-fit critic EV logging."""

import copy
import statistics
from contextlib import nullcontext

import pytest
import torch

from tests.test_sao_critic_updates import setup_batch

from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.trainer.ppo.critic import PPOCriticController, PPOCriticControllerV2
from areal.utils import stats_tracker


@pytest.fixture(autouse=True)
def clean_stats():
    stats_tracker.export_all()
    yield
    stats_tracker.export_all()


def _oracle(prediction, target):
    """Compute centered population variances independently of moment reduction."""
    residual = [p - t for p, t in zip(prediction, target)]
    target_variance = statistics.pvariance(target)
    return {
        "critic_target": statistics.mean(target),
        "critic_target_second_moment": statistics.mean(t * t for t in target),
        "critic_residual": statistics.mean(residual),
        "critic_mse": statistics.mean(r * r for r in residual),
        "critic_target_variance": target_variance,
        "critic_residual_variance": statistics.pvariance(residual),
        "critic_explained_variance": (
            1 - statistics.pvariance(residual) / target_variance
            if target_variance > 0
            else None
        ),
        "critic_explained_variance_defined": float(target_variance > 0),
    }


def _observe_training(critic, monkeypatch):
    forwards = []
    train_batch = critic.engine.train_batch

    def observed(data, **kwargs):
        mask = data["loss_mask"].bool()
        with torch.no_grad():
            before = critic.engine.model(data["input_ids"]).squeeze(-1)[mask]
        result = train_batch(data, **kwargs)
        with torch.no_grad():
            after = critic.engine.model(data["input_ids"]).squeeze(-1)[mask]
        forwards.append(
            (before.tolist(), data["returns"][mask].tolist(), after.tolist())
        )
        return result

    monkeypatch.setattr(critic.engine, "train_batch", observed)
    # Amplify the same-batch fitting effect so the merged/pre-update distinction
    # cannot pass merely because the toy optimizer made a negligible change.
    critic.engine.optimizer.param_groups[0]["lr"] = 0.5
    return forwards


def _transport(critic, transport, domain):
    if transport == "local":
        return critic
    cls = PPOCriticController if transport == "rpc" else PPOCriticControllerV2
    controller = cls.__new__(cls)

    def dispatch(method, *args, **kwargs):
        kwargs.pop("rpc_meta", None)
        args, kwargs = deserialize_value(serialize_value([list(args), kwargs]))
        # A real worker has no controller-local stats scope. Only serialized
        # kwargs can select the update-specific worker metrics.
        with stats_tracker.DEFAULT_TRACKER.disable_scope():
            with stats_tracker.scope(domain) if domain else nullcontext():
                result = getattr(critic, method)(*args, **kwargs)
        if method == "ppo_update":
            result = [result]  # Controller repeats each DP report per row.
        return deserialize_value(serialize_value(result))

    def gateway(path, payload=None):
        if path == "/step_lr_scheduler":
            dispatch("step_lr_scheduler")
            return {}
        method = path.rsplit("/", 1)[-1]
        if method == "update":
            method = "ppo_update"
        result = dispatch(
            method,
            *deserialize_value(payload["args"]),
            **deserialize_value(payload["kwargs"]),
        )
        return {"result": serialize_value(result)}

    controller._custom_function_call = dispatch
    controller._replicated_function_call = dispatch
    controller._gateway_post = gateway
    return controller


def _assert_metrics(stats, scope, expected):
    for name, value in expected.items():
        actual = stats[f"{scope}/{name}"]
        if value is None:
            assert actual is None
        else:
            assert actual == pytest.approx(value, rel=2e-5, abs=2e-6)


@pytest.mark.parametrize("transport", ["local", "rpc", "gateway"])
@pytest.mark.parametrize("domain", ["", "domain/math"])
@pytest.mark.parametrize("constant_target", [False, True])
def test_sao_each_update_logs_its_own_forward_and_primary_is_first(
    monkeypatch, transport, domain, constant_target
):
    """Real SGD fits twice; independently check both forwards and reject pooling."""
    from areal.trainer.ppo.update import update_critic_before_actor

    actor, critic, raw, fixed, _ = setup_batch()
    if constant_target:
        fixed[0]["returns"].fill_(1.0)
    forwards = _observe_training(critic, monkeypatch)
    stats_tracker.export_all()
    with stats_tracker.scope(domain) if domain else nullcontext():
        update_critic_before_actor(
            actor, _transport(critic, transport, domain), raw, fixed, 2
        )
    stats = stats_tracker.export_all()
    scope = f"{domain}/ppo_critic" if domain else "ppo_critic"
    assert len(forwards) == 2
    expected = [_oracle(before, target) for before, target, _ in forwards]
    for index, metrics in enumerate(expected, 1):
        _assert_metrics(stats, f"{scope}/critic_update_{index}", metrics)
    primary = f"{scope}/critic_explained_variance_pre_update"
    assert stats[primary] == stats[f"{scope}/critic_update_1/critic_explained_variance"]
    assert stats[f"{primary}_defined"] == float(not constant_target)
    # The old SAO key meant a pooled forward; it must disappear, not be reused.
    for name in expected[0]:
        assert f"{scope}/{name}" not in stats
    if not constant_target:
        first_ev, second_ev = [m["critic_explained_variance"] for m in expected]
        merged = _oracle(
            forwards[0][0] + forwards[1][0], forwards[0][1] + forwards[1][1]
        )["critic_explained_variance"]
        assert abs(first_ev - second_ev) > 1e-3
        assert abs(stats[primary] - merged) > 1e-3
        print(f"{transport=} {domain=} EV1={first_ev} EV2={second_ev} merged={merged}")


@pytest.mark.parametrize("critic_only", [False, True])
def test_ppo_single_minibatch_preserves_legacy_metrics_and_pre_update_ev(
    monkeypatch, critic_only
):
    """One loss forward precedes SGD in both PPO and critic-only consumers."""
    actor, critic, _, fixed, _ = setup_batch()
    forwards = _observe_training(critic, monkeypatch)
    stats_tracker.export_all()
    if not critic_only:
        actor.ppo_update(copy.deepcopy(fixed))
    critic.ppo_update(copy.deepcopy(fixed))
    stats = stats_tracker.export_all()
    assert len(forwards) == 1
    before, target, after = forwards[0]
    expected = _oracle(before, target)
    _assert_metrics(stats, "ppo_critic", expected)
    assert (
        stats["ppo_critic/critic_explained_variance_pre_update"]
        == stats["ppo_critic/critic_explained_variance"]
    )
    assert stats["ppo_critic/critic_explained_variance_pre_update_defined"] == 1.0
    assert not any("critic_update_" in key for key in stats)
    post_ev = _oracle(after, target)["critic_explained_variance"]
    assert abs(stats["ppo_critic/critic_explained_variance"] - post_ev) > 1e-3
    assert critic.optimizer_steps == 1
    assert actor.optimizer_steps == int(not critic_only)


def test_ppo_multiple_minibatches_keeps_legacy_ev_without_pre_update_label(monkeypatch):
    """Several optimizer minibatches cannot claim a full-batch pre-update EV."""
    _, critic, _, fixed, _ = setup_batch()
    critic.component.config.ppo_n_minibatches = 2
    forwards = _observe_training(critic, monkeypatch)
    stats_tracker.export_all()
    critic.ppo_update(copy.deepcopy(fixed * 2))
    stats = stats_tracker.export_all()
    assert len(forwards) == 2
    expected = _oracle(forwards[0][0] + forwards[1][0], forwards[0][1] + forwards[1][1])
    _assert_metrics(stats, "ppo_critic", expected)
    assert "ppo_critic/critic_explained_variance_pre_update" not in stats
