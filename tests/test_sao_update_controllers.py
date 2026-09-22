"""M4 controller envelopes with real HTTP RTensors and trainable consumers.

The compute transport runs locally; RTensor storage/fetch uses the actual RPC
server. This covers controller result shapes, not distributed worker scheduling.
"""

import copy

import orjson
import pytest
import requests
import torch

from tests.test_sao_critic_updates import setup_batch

from areal.infra.controller.train_controller import _merge_tensors
from areal.infra.rpc.rtensor import RTensor, fetch
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.trainer.ppo.actor import PPOActorController, PPOActorControllerV2
from areal.trainer.ppo.critic import PPOCriticController, PPOCriticControllerV2
from areal.trainer.ppo.update import update_critic_before_actor

pytest_plugins = ["tests.test_rtensor"]


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_critic_first_controller_keeps_remote_targets_and_refreshes_values(
    rpc_server, version
):
    actor, critic, raw, fixed, events = setup_batch()

    def remote(value):
        result = RTensor.remotize(value, node_addr=rpc_server)
        handles = []
        RTensor._collect_all(result, handles)
        for handle in handles:
            response = requests.put(
                f"http://{rpc_server}/data/{handle.shard.shard_id}",
                data=orjson.dumps(serialize_value(fetch(handle.shard.shard_id))),
                timeout=10,
            )
            response.raise_for_status()
        return result

    def controller(role, cls):
        instance = cls.__new__(cls)

        def dispatch(method, *args, **kwargs):
            kwargs.pop("rpc_meta", None)
            local_args = RTensor.localize(
                deserialize_value(serialize_value(list(args)))
            )
            result = getattr(role, method)(*local_args, **kwargs)
            result = remote(result)
            return (
                _merge_tensors([result], [list(range(len(local_args[0])))])
                if args
                else result
            )

        def gateway(path, payload=None):
            if path == "/step_lr_scheduler":
                role.step_lr_scheduler()
                return {}
            method = path.rsplit("/", 1)[-1]
            if method == "update":
                method = "ppo_update"
            args = deserialize_value(payload["args"])
            kwargs = deserialize_value(payload["kwargs"])
            result = dispatch(method, *args, **kwargs)
            return {"result": serialize_value(result)}

        instance._custom_function_call = dispatch
        instance._gateway_post = gateway
        return instance

    actor_cls, critic_cls = (
        (PPOActorController, PPOCriticController)
        if version == "v1"
        else (PPOActorControllerV2, PPOCriticControllerV2)
    )
    actor_controller, critic_controller = (
        controller(actor, actor_cls),
        controller(critic, critic_cls),
    )
    raw_remote, fixed_remote = remote(raw), remote(fixed)
    target = fixed_remote[0]["returns"]
    target_id = target.shard.shard_id
    result = update_critic_before_actor(
        actor_controller, critic_controller, raw_remote, fixed_remote, 2
    )
    assert target.shard.shard_id == target_id and target.data.is_meta
    assert critic.optimizer_steps == 2 and actor.optimizer_steps == 1
    assert result["actor"][0]["successful"] == 1.0
    target_local = RTensor.localize(target)
    torch.testing.assert_close(
        target_local, fixed[0]["returns"][:, : target_local.shape[1]], rtol=0, atol=0
    )
    refreshed = RTensor.localize(copy.deepcopy(raw_remote))
    refreshed[0]["values"] = critic.predictions[-1][0]
    expected = actor.component.compute_advantages(refreshed)[0]["advantages"]
    torch.testing.assert_close(actor.advantages[-1][0], expected, rtol=0, atol=0)
    assert events.index("critic.forward.2") < events.index("actor.optimizer.1")
