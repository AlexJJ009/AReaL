# SPDX-License-Identifier: Apache-2.0
"""Pinned Arena inventory with the native AReaL PPO/AWEX training loop."""

import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path


def build_workflow_kwargs(config):
    """Keep the validated sampling arguments independent of the SWE CLI."""
    return dict(
        econfig=asdict(config.econfig),
        gen_args=dict(
            temperature=config.gconfig.temperature,
            top_p=config.gconfig.top_p,
            top_k=config.gconfig.top_k,
            max_completion_tokens=config.gconfig.max_new_tokens,
        ),
        timeout=config.econfig.timeout,
    )


def select_training_rows(rows, split, scope="acceptance"):
    ids = [row["data_id"] for row in rows]
    if sorted(ids) != split["all_data_ids"]:
        raise ValueError("Live inventory differs from pinned benchmark")
    heldout = set(split["heldout"])
    selected = [row for row in rows if row["data_id"] not in heldout]
    if set(row["data_id"] for row in selected) != set(split["training_pool"]):
        raise ValueError("Training/heldout partition mismatch")
    if scope == "training_pool":
        return selected
    if scope != "acceptance":
        raise ValueError("SWE task scope must be acceptance or training_pool")
    acceptance_ids = split["rl_acceptance_ids"]
    if len(acceptance_ids) != 16 or len(set(acceptance_ids)) != 16:
        raise ValueError("RL acceptance requires sixteen unique tasks")
    if not set(acceptance_ids) <= set(split["training_pool"]):
        raise ValueError("RL acceptance tasks must belong to the training pool")
    by_id = {row["data_id"]: row for row in selected}
    return [by_id[key] for key in acceptance_ids]


def decorate_controller(controller):
    initialize = controller.initialize

    def init(**kwargs):
        args = dict(kwargs["server_args"])
        if args.get("tp_size") != 4 or args.get("ep_size") != 4:
            raise ValueError("Requires validated SGLang TP4/EP4")
        args.update(
            linear_attn_prefill_backend="flashinfer",
            linear_attn_decode_backend="flashinfer",
            ple_offload_embedding=True,
        )
        return initialize(**{**kwargs, "server_args": args})

    controller.initialize = init
    start_proxy = controller.start_proxy

    def start():
        scheduler = controller.scheduler
        original = scheduler.fork_workers

        def fork(*args, **kwargs):
            if (
                kwargs.get("command")
                == "areal.experimental.openai.proxy.proxy_rollout_server"
            ):
                kwargs["command"] = "qwen_isolated_cache_proxy"
            return original(*args, **kwargs)

        scheduler.fork_workers = fork
        try:
            return start_proxy()
        finally:
            scheduler.fork_workers = original

    controller.start_proxy = start
    return controller


def read_start_source():
    """Keep fresh-model training separate from explicit checkpoint recovery."""
    mode = os.environ.get("QWEN_SWE_START_MODE", "recover")
    if mode == "fresh":
        return None
    if mode != "recover":
        raise ValueError("QWEN_SWE_START_MODE must be fresh or recover")
    return json.loads(Path(os.environ["QWEN_RECOVER_SOURCE"]).read_text())


def validate_start_state(recover_info, weight_version, source):
    if source is None:
        if recover_info is not None or weight_version != 0:
            raise ValueError(
                "Fresh SWE training must start without checkpoint at version 0"
            )
        return None
    if recover_info is None:
        raise ValueError("Required source checkpoint was not restored")
    actual_step = recover_info.last_step_info.global_step
    if actual_step != source["expected_saved_global_step"]:
        raise ValueError("Restored checkpoint step differs from audited source")
    if weight_version != source["expected_restored_weight_version"]:
        raise ValueError("Restored rollout weight version differs from source")
    return actual_step


def main(argv):
    import examples.swe.train_swe_rl as entry

    from areal.engine.sglang_remote import RemoteSGLangEngine
    from areal.trainer import PPOTrainer

    root = Path(__file__).parent
    sys.path.insert(0, str(root.parent))
    acceptance = json.loads(Path(os.environ["QWEN_REPLAY64_ACCEPTANCE"]).read_text())
    if acceptance.get("status") != "passed":
        raise ValueError("64K optimizer/AWEX replay has not passed")
    config, _ = entry.load_expr_config(argv, entry.SWEPPOConfig)
    expected_harness = "claude-code-with-skills@5.0.2"
    configured_stream = config.econfig.arena_streams[0]
    if (
        configured_stream.harness != expected_harness
        or configured_stream.llm_protocol != "anthropic"
    ):
        raise ValueError(
            "Claude Code acceptance requires the pinned harness and Anthropic protocol"
        )
    protocol = json.loads(Path(os.environ["QWEN_CC_PROTOCOL_ACCEPTANCE"]).read_text())
    if (
        protocol.get("status") != "passed"
        or protocol.get("harness") != expected_harness
    ):
        raise ValueError("A fresh Claude Code execution acceptance is required")
    if config.enable_offload:
        raise ValueError("AWEX actor must not globally activate TMS")
    if config.actor.weight_update_mode != "awex":
        raise ValueError("This acceptance run requires AWEX")
    if config.actor.backend != "megatron:(attn:d1p8t8|ffn:d1p8t1e8)":
        raise ValueError("Actor topology differs from accepted topology")
    if config.should_accept_fn is not None or config.gconfig.n_samples != 8:
        raise ValueError("Keep all eight-sample groups for acceptance")
    if (
        config.train_dataset.batch_size != 16
        or config.rollout.consumer_batch_size != 16
    ):
        raise ValueError("RL acceptance requires sixteen groups per update")
    if (
        config.rollout.queue_size is not None
        and config.rollout.queue_size <= config.train_dataset.batch_size
    ):
        raise ValueError(
            "Rollout queue must exceed the batch size to allow task submission"
        )
    split = json.loads((root.parent / "fixtures/split.json").read_text())
    split["rl_acceptance_ids"] = json.loads(
        (root.parent / "fixtures/parallel-canary-split.json").read_text()
    )["selected"]
    task_scope = os.environ.get("QWEN_SWE_TASK_SCOPE", "acceptance")
    original_resolver = entry._resolve_arena_stream

    def resolve(client, configured):
        stream, rows = original_resolver(client, configured)
        return stream, select_training_rows(rows, split, task_scope)

    entry._resolve_arena_stream = resolve
    dataset, streams = entry.get_arena_mixture_dataset(
        config.econfig, size_multiple=config.train_dataset.batch_size
    )
    config.econfig.arena_streams = streams
    config.econfig.arena_streams_yaml_b64 = ""
    config.econfig.arena_streams_file = ""
    ids = list(dataset["data_id"])
    audit = {
        "benchmark": split["benchmark"],
        "training_count": len(ids),
        "training_pool_count": len(split["training_pool"]),
        "selection": task_scope,
        "unique_training_count": len(set(ids)),
        "heldout_count": len(split["heldout"]),
        "heldout_overlap": 0,
        "ordered_training_ids": ids,
        "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
    }
    Path(os.environ["QWEN_ARENA_OUTPUT"], "training-inventory.json").write_text(
        json.dumps(audit, indent=2)
    )
    factory = RemoteSGLangEngine.as_controller
    RemoteSGLangEngine.as_controller = staticmethod(
        lambda *a, **kw: decorate_controller(factory(*a, **kw))
    )
    source = read_start_source()
    destination_recover = copy.deepcopy(config.recover)
    if source is not None:
        for key in ("fileroot", "experiment_name", "trial_name"):
            setattr(config.recover, key, source[key])
    from areal.infra.scheduler.slurm import SlurmScheduler

    class SWERecoveryTrainer(PPOTrainer):
        def _init_scheduler(self):
            return SlurmScheduler(
                exp_config=self.config, container_mounts=os.environ["QWEN_MOUNTS"]
            )

    with SWERecoveryTrainer(
        config, train_dataset=dataset, valid_dataset=None
    ) as trainer:
        actual_step = validate_start_state(
            trainer.recover_info, trainer.rollout.get_version(), source
        )
        # Loading uses the existing checkpoint; subsequent saves use the new
        # trial. Preserve the original checkpoint and its evidence in place.
        if trainer.recover_handler.config is not config.recover:
            raise ValueError("Unexpected recovery configuration ownership")
        for key in ("fileroot", "experiment_name", "trial_name"):
            setattr(config.recover, key, getattr(destination_recover, key))
        Path(os.environ["QWEN_ARENA_OUTPUT"], "recovery-lineage.json").write_text(
            json.dumps(
                {
                    "start_mode": "fresh" if source is None else "recover",
                    "source": source,
                    "restored_global_step": actual_step,
                    "restored_weight_version": trainer.rollout.get_version(),
                    "destination": {
                        key: getattr(config.recover, key)
                        for key in ("fileroot", "experiment_name", "trial_name")
                    },
                    "optimizer_load_enabled": source is not None
                    and not config.recover.no_load_optim,
                },
                indent=2,
            )
        )
        from weight_probe_8node import install

        install(
            trainer,
            root.parent / "fixtures/weight-probe-fixture.json",
            Path(os.environ["QWEN_ARENA_OUTPUT"]) / "weight-probes",
        )
        replay_fixtures = os.environ.get("QWEN_INITIAL_PREFILL_FIXTURES")
        if replay_fixtures:
            # Before train(), no Arena workflows have been submitted and the
            # endpoint still holds its initial HF weights. Never replay this
            # evidence against a recovered or updated model.
            if trainer.rollout.get_version() != 0:
                raise ValueError("Initial-weight replay cannot run after recovery")
            stats = trainer.rollout.dispatcher.staleness_manager.get_stats()
            if stats.running != 0:
                raise ValueError("Initial-weight replay requires idle inference")
            server = trainer.rollout.server_infos[0]
            endpoint = f"http://{server.host}:{server.port}"
            output = Path(os.environ["QWEN_ARENA_OUTPUT"])
            provenance = output / "initial-prefill-provenance.json"
            provenance.write_text(
                json.dumps(
                    {
                        "weight_version": 0,
                        "weight_source": "initial_hf_loaded",
                        "endpoint": endpoint,
                        "dedicated_idle": True,
                        "model_path": config.tokenizer_path,
                        "running_workflows": stats.running,
                    },
                    indent=2,
                )
            )
            subprocess.run(
                [
                    sys.executable,
                    str(root / "replay_initial_prefill.py"),
                    "--fixtures",
                    replay_fixtures,
                    "--endpoint",
                    endpoint,
                    "--initial-weight-evidence",
                    str(provenance),
                    "--output",
                    str(output / "initial-prefill-replay"),
                ],
                check=True,
            )
        trainer.train(
            workflow="qwen_claude_audit.ClaudeArenaWorkflow",
            workflow_kwargs=build_workflow_kwargs(config),
            dynamic_filter_fn=None,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
