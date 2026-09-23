# SPDX-License-Identifier: Apache-2.0
"""Optional rollout profiling must not block the default training path."""

from types import SimpleNamespace
from unittest.mock import Mock

from examples.math import sao_ppo


def test_default_actor_update_never_calls_rollout_profiler(monkeypatch, tmp_path):
    monkeypatch.delenv("SAO_PROFILE_ROLLOUT", raising=False)
    actor_update = Mock(return_value="updated")
    trainer = SimpleNamespace(
        recover_info=None,
        actor=SimpleNamespace(
            prepare_batch=Mock(), compute_advantages=Mock(), ppo_update=actor_update
        ),
        critic=SimpleNamespace(ppo_update=Mock()),
        rollout=SimpleNamespace(
            server_infos=[SimpleNamespace(host="localhost", port=1)]
        ),
        stats_logger=SimpleNamespace(commit=Mock()),
        _save_perf_tracer=Mock(),
    )
    post = Mock(side_effect=AssertionError("default training must not call profiler"))
    monkeypatch.setattr(sao_ppo.requests, "post", post)
    sao_ppo.install_audit_hooks(trainer, tmp_path)

    assert trainer.actor.ppo_update({"batch": "fixture"}) == "updated"
    actor_update.assert_called_once_with({"batch": "fixture"})
    post.assert_not_called()
