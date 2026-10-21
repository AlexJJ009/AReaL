# SPDX-License-Identifier: Apache-2.0
"""Dedicated eval cache isolation and checkpoint flush regression tests (CPU)."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from examples.tau2.evaluation import Tau2AsyncEvalTrainer
from examples.tau2.utils import Tau2PPOConfig

from areal.api.alloc_mode import ModelAllocation
from areal.engine.sglang_remote import SGLangBackend


def test_eval_radix_config_default_and_structured_override():
    config = OmegaConf.structured(Tau2PPOConfig)
    assert config.eval_enable_radix_cache is False
    overridden = OmegaConf.merge(config, {"eval_enable_radix_cache": True})
    assert overridden.eval_enable_radix_cache is True


@pytest.mark.parametrize("enabled", [False, True])
def test_eval_radix_switch_only_changes_dedicated_server(monkeypatch, enabled):
    trainer = object.__new__(Tau2AsyncEvalTrainer)
    trainer.config = Tau2PPOConfig()
    # Set dynamically so the pre-implementation test reaches server construction.
    trainer.config.eval_enable_radix_cache = enabled
    trainer.config.rollout.backend = "sglang:d1p1t1"
    trainer.config.rollout._version = "v1"
    trainer.scheduler = SimpleNamespace(gpu_devices=[0], _workers={})
    trainer.rollout_alloc = ModelAllocation.from_str("sglang:d1p1t1", name="rollout")
    controller = Mock()
    monkeypatch.setattr("areal.trainer.rl_trainer.is_single_controller", lambda: True)
    monkeypatch.setattr(trainer, "_is_v1_awex_colocate", lambda _: False)
    monkeypatch.setattr(trainer, "_write_gpu_allocation", lambda: None)
    monkeypatch.setattr(
        "areal.trainer.async_eval.RemoteSGLangEngine.as_controller",
        lambda *_: controller,
    )

    trainer._init_rollout(trainer.config.rollout)
    trainer._init_dedicated_eval_rollout()
    try:
        trainer._init_rollout(trainer.config.rollout)
        launches = controller.initialize.call_args_list
        assert [call.kwargs["role"] for call in launches] == [
            "rollout",
            "dedicated-eval",
            "rollout",
        ]
        assert [
            call.kwargs["server_args"]["disable_radix_cache"] for call in launches
        ] == [True, not enabled, True]
        assert trainer.config.sglang.disable_radix_cache is True
        controller.start_proxy.assert_called_once()
    finally:
        trainer._async_eval_executor.shutdown()


def test_eval_checkpoint_disk_request_preserves_flush_default(monkeypatch):
    trainer = object.__new__(Tau2AsyncEvalTrainer)
    controller = Mock()
    controller.get_version.return_value = 3
    trainer._async_eval_rollout = controller
    monkeypatch.setattr(trainer, "_publish_eval_checkpoint_ready", lambda _: None)
    requests = []

    def dispatch(method, *, meta):
        assert method == "update_weights_from_disk"
        assert meta.path == "checkpoint-4"
        assert meta.clear_checkpoint_after_load is False
        requests.extend(
            SGLangBackend().build_disk_weight_update_requests(meta).requests
        )

    controller._collective_rpc.side_effect = dispatch
    trainer._load_eval_checkpoint("checkpoint-4", version=4)

    assert len(requests) == 1
    assert requests[0].endpoint == "/update_weights_from_disk"
    assert requests[0].payload["model_path"] == "checkpoint-4"
    assert "flush_cache" not in requests[0].payload
    controller.set_version.assert_called_once_with(4)
