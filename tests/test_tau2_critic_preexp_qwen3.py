# SPDX-License-Identifier: Apache-2.0
"""Qwen3 profile, exact replay/head loading and resident-cache regressions."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pytest
import torch

from scripts.tau2 import critic_preexp as pre
from scripts.tau2 import critic_preexp_data as data
from scripts.tau2 import critic_preexp_extract as extraction
from scripts.tau2 import critic_preexp_qwen3 as q3
from scripts.tau2.critic_features import extract_score_inputs
from scripts.tau2.critic_preexp_stats import TaskBootstrap


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


@pytest.fixture
def qwen3_pool(tmp_path):
    """Real task/batch counts with short, deterministic token sequences."""
    train, validation, batches = [], [], []
    validation_path = tmp_path / "validation.pt"
    val_rows = []
    for i in range(60):
        val_rows.append(
            dict(
                domain="airline" if i < 20 else "retail",
                task_id=f"val{i}",
                source_id=f"val:{i}",
                rewards=torch.tensor([float(i % 2)]),
                input_ids=torch.arange(8),
                attention_mask=torch.ones(8),
                loss_mask=torch.tensor([0, 0, 1, 1, 1, 1, 1, 1]),
            )
        )
    torch.save(val_rows, validation_path)
    val_hash = data.sha256(validation_path)
    for index, row in enumerate(val_rows):
        validation.append(
            dict(
                run="r480",
                pt_file=str(validation_path),
                pt_sha256=val_hash,
                row_index=index,
                domain=row["domain"],
                task_id=row["task_id"],
                source_id=row["source_id"],
                reward=float(index % 2),
                seqlen=8,
                loss_tokens=6,
                input_ids_sha256=data.array_hash(row["input_ids"]),
                loss_mask_sha256=data.array_hash(row["loss_mask"]),
            )
        )
    for step in range(94):
        count = 11 if step % 47 == 46 else 16
        start = step * 16
        orders = list(range(start, start + count))
        tasks, sources, rewards = [], [], []
        for order in orders:
            task = (step % 47) * 16 + order - start
            row = dict(
                domain="airline" if task < 410 else "retail",
                task_id=f"train{task}",
                source_id=f"train:{task}",
                reward=float(task % 2),
                input_ids=list(range(8)),
                attention_mask=[1] * 8,
                loss_mask=[0, 0, 1, 1, 1, 1, 1, 1],
                rollout_task_id=order,
                split="train",
            )
            path = tmp_path / "rollout" / f"{order}.jsonl"
            write_json(path, row)
            raw = path.read_bytes().splitlines()[0]
            train.append(
                dict(
                    run="r480",
                    rollout_file=str(path),
                    file_sha256=data.sha256(path),
                    line_sha256=hashlib.sha256(raw).hexdigest(),
                    line=1,
                    domain=row["domain"],
                    task_id=row["task_id"],
                    source_id=row["source_id"],
                    reward=row["reward"],
                    seqlen=8,
                    loss_tokens=6,
                    input_ids_sha256=data.array_hash(row["input_ids"]),
                    loss_mask_sha256=data.array_hash(row["loss_mask"]),
                )
            )
            tasks.append(row["task_id"])
            sources.append(row["source_id"])
            rewards.append(row["reward"])
        mean = sum(rewards) / count
        metrics = dict(
            global_step=step,
            completed_step=step + 1,
            epoch=step // 47,
            epoch_step=step % 47,
            metrics={
                "tau2_batch/task_id_start": start,
                "tau2_batch/real_episodes": count,
                "rollout/reward": mean,
            },
        )
        path = tmp_path / "metrics" / f"{step}.json"
        write_json(path, metrics)
        batches.append(
            dict(
                global_step=step,
                completed_step=step + 1,
                epoch=step // 47,
                epoch_step=step % 47,
                episode_order=orders,
                task_ids=tasks,
                source_ids=sources,
                reward_mean=mean,
                metrics_file=str(path),
                metrics_sha256=data.sha256(path),
            )
        )
    manifest = dict(
        schema_version="offline-critic-pool/1",
        protocol=data.QWEN3_PROTOCOL,
        synthetic=True,
        training=train,
        validation=validation,
        replay_batches=batches,
        counts={"training": {"rows": 1494}, "validation": {"rows": 60}},
    )
    path = tmp_path / "pool-manifest.json"
    write_json(path, manifest)
    path.with_name(path.name + ".sha256").write_text(data.sha256(path))
    return path


def test_qwen3_pool_metrics_bind_short_epoch_batches(qwen3_pool):
    """Both epoch tails contain eleven rows; second epoch starts at logical752."""
    manifest, rows = data.load_pool(qwen3_pool, synthetic=True)
    train = [row for row in rows if row["_split"] == "train"]
    assert len(train) == 1494
    assert [
        row["_episode_order"] for row in train if row["_trainer_step"] == 46
    ] == list(range(736, 747))
    assert [
        row["_episode_order"] for row in train if row["_trainer_step"] == 47
    ] == list(range(752, 768))
    assert len(manifest["replay_batches"]) == 94
    # Source metrics are immutable evidence, not just an unauthenticated order.
    source = Path(manifest["replay_batches"][47]["metrics_file"])
    source.write_text(source.read_text() + " ")
    with pytest.raises(ValueError, match="Replay metrics SHA256"):
        data.load_pool(qwen3_pool, synthetic=True)


def test_qwen3_replay_membership_tampering_rejected(qwen3_pool):
    """Re-signing a changed manifest cannot substitute a different source batch."""
    payload = json.loads(qwen3_pool.read_text())
    payload["replay_batches"][0]["task_ids"][0] = "wrong-task"
    write_json(qwen3_pool, payload)
    qwen3_pool.with_name(qwen3_pool.name + ".sha256").write_text(
        data.sha256(qwen3_pool)
    )
    with pytest.raises(ValueError, match="source membership"):
        data.load_pool(qwen3_pool, synthetic=True)


@pytest.mark.parametrize("trained", [False, True])
def test_qwen3_layers_equal_actual_score_input_and_hidden_states(tmp_path, trained):
    """Use HF's real AutoModelForTokenClassification, with a nonzero biased head."""
    from transformers import AutoModelForTokenClassification, Qwen3Config

    torch.manual_seed(12)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=36,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_labels=1,
    )
    config._attn_implementation = "sdpa"
    model = AutoModelForTokenClassification.from_config(config).eval()
    if trained:
        with torch.no_grad():
            model.model.layers[0].mlp.down_proj.weight.add_(0.03125)
            model.score.bias.fill_(0.25)
        model.save_pretrained(tmp_path / "critic")
        model = AutoModelForTokenClassification.from_pretrained(
            tmp_path / "critic", local_files_only=True
        ).eval()
    row = dict(
        input_ids=list(range(8)),
        attention_mask=[1] * 8,
        loss_mask=[0, 0, 1, 1, 1, 1, 1, 1],
    )
    positions = torch.tensor([1, 3, 6])
    captured = []
    hook = model.score.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone())
    )
    result = extraction.extract_layers(
        model, row, positions, ["final", "9", "18", "27"]
    )
    hook.remove()
    direct = extract_score_inputs(model, row, positions)
    torch.testing.assert_close(
        result["final"], captured[0][0, positions], rtol=0, atol=0
    )
    torch.testing.assert_close(direct, result["final"], rtol=0, atol=0)
    with torch.inference_mode():
        output = model.model(
            torch.tensor([row["input_ids"]]), output_hidden_states=True, use_cache=False
        )
    assert extraction.layer_indices(36) == [9, 18, 27]
    for index in (9, 18, 27):
        torch.testing.assert_close(
            result[str(index)],
            output.hidden_states[index][0, positions],
            rtol=0,
            atol=0,
        )
    assert model.score.bias is not None


def test_qwen3_initial_head_saved_weight_and_bias_exact(tmp_path):
    """A saved random head and nonzero bias must survive without reinitialization."""
    from safetensors.torch import save_file

    path = tmp_path / "critic-initial" / "epoch0epochstep-1globalstep-1"
    path.mkdir(parents=True)
    expected = {
        "weight": torch.arange(2560).reshape(1, -1).bfloat16() / 2560,
        "bias": torch.tensor([0.375], dtype=torch.bfloat16),
    }
    save_file(
        {f"score.{k}": v for k, v in expected.items()}, path / "model.safetensors"
    )
    write_json(path / "config.json", {"model_type": "qwen3"})
    actual, provenance = q3.read_initial_head(path)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        assert actual[key].dtype == expected[key].dtype
    assert provenance["checkpoint_files_sha256"] == {
        "model.safetensors": data.sha256(path / "model.safetensors")
    }
    with pytest.raises(ValueError, match="critic-initial"):
        q3.read_initial_head(path.parent)


@pytest.fixture
def cache(tmp_path):
    """Two heterogeneous shards, one mixed split; catches reordered metadata too."""
    generator = torch.Generator().manual_seed(4)
    records = []
    for shard_index in range(2):
        n = 18
        part = dict(features=torch.randn(n, 3, generator=generator).bfloat16())
        for key in (
            "trajectory_id",
            "domain",
            "task_id",
            "source_id",
            "run",
            "split",
            "reward",
            "sampling_weight",
            "original_action_count",
            "step",
            "order",
            "progress",
            "turn",
            "planted_label",
        ):
            part[key] = []
        for i in range(n):
            episode = i // 3 + 6 * shard_index
            split = "train" if episode % 2 == 0 else "validation"
            values = dict(
                trajectory_id=f"ep{episode}",
                domain="airline" if episode % 3 else "retail",
                task_id=str(episode),
                source_id=f"source:{episode}",
                run="r480",
                split=split,
                reward=float(episode % 3 != 0),
                sampling_weight=2.0 if split == "train" else 1.0,
                original_action_count=6 if split == "train" else 3,
                step=episode // 4,
                order=episode % 4,
                progress=(i % 3) / 2,
                turn=0,
                planted_label=0.0,
            )
            for key, value in values.items():
                part[key].append(value)
        p = tmp_path / f"features-{shard_index}.pt"
        torch.save(part, p)
        records.append(dict(path=p.name, rows=n, sha256=data.sha256(p)))
    path = tmp_path / "manifest.json"
    write_json(
        path,
        dict(
            schema="tau2-critic-features-v1",
            hidden_size=3,
            feature_semantics="score_input_final_norm_shifted_action_prefix",
            shards=records,
        ),
    )
    return path


def mini_experiment(store):
    experiment = pre.Preexperiment.__new__(pre.Preexperiment)
    experiment.store = store
    experiment.x, experiment.y, experiment.mass, experiment.identities = (
        store.training()
    )
    experiment.indices = defaultdict(list)
    for i, identity in enumerate(experiment.identities):
        experiment.indices[identity].append(i)
    experiment.episode_map = {r["trajectory_id"]: r for r in store.episodes}
    experiment.bootstrap = TaskBootstrap(
        store.evaluate_targets({"shared": (torch.zeros(3), torch.tensor(0.78))})[
            "rows"
        ],
        require_60=False,
    )
    return experiment


def test_cached_features_repeat_simulation_without_rereads(cache, monkeypatch):
    """Repeated runs have identical gradients, heads and EV rows without shard I/O."""
    experiment = mini_experiment(pre.PreexpStore(cache))
    expected, expected_head = experiment.simulate(
        lr=1e-3, steps=6, batch=2, diagnostics=True
    )

    def no_load(*args, **kwargs):
        pytest.fail("Simulation reread cached feature shards")

    monkeypatch.setattr(torch, "load", no_load)
    actual, head = experiment.simulate(lr=1e-3, steps=6, batch=2, diagnostics=True)
    assert actual == expected
    for a, b in zip(head, expected_head, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    # The optimizer must consume saved initial weight *and* bias.
    initial = {
        "weight": torch.tensor([[0.25, -0.5, 0.125]]),
        "bias": torch.tensor([0.625]),
    }
    baseline, _ = experiment.simulate(lr=0.0, steps=1, batch=2, initial_head=initial)
    assert baseline["evaluations"][0]["weight_norm"] == pytest.approx(
        float(initial["weight"].norm())
    )
    assert baseline["evaluations"][0]["bias"] == 0.625


def test_bootstrap_qwen3_seed_uses_registered_draw_matrix(cache):
    """Qwen3's seed reaches both actual resampling and reported provenance."""
    store = pre.PreexpStore(cache)
    rows = store.evaluate_targets({"shared": (torch.zeros(3), torch.tensor(0.5))})[
        "rows"
    ]
    bootstrap = TaskBootstrap(rows, require_60=False, seed=20261015)
    g = torch.Generator().manual_seed(20261015)
    draws = torch.randint(
        len(bootstrap.keys), (20000, len(bootstrap.keys)), generator=g
    )
    counts = torch.zeros_like(bootstrap.counts).scatter_add_(
        1, draws, torch.ones_like(draws, dtype=torch.float64)
    )
    torch.testing.assert_close(bootstrap.counts, counts, rtol=0, atol=0)
    assert bootstrap.report(rows)["seed"] == 20261015


def test_qwen3_batches_replay_and_whole_epoch_shuffle(qwen3_pool):
    """Shuffled arms visit both epochs exactly once without replacing short tails."""
    manifest, rows = data.load_pool(qwen3_pool, synthetic=True)
    experiment = q3.Qwen3Preexperiment.__new__(q3.Qwen3Preexperiment)
    experiment.pool, experiment.pool_metadata = rows, manifest
    experiment.episode_map = {
        r["_trajectory_id"]: dict(
            trajectory_id=r["_trajectory_id"], step=r["_trainer_step"]
        )
        for r in rows
        if r["_split"] == "train"
    }
    experiment.store = type(
        "Store", (), {"episodes": list(experiment.episode_map.values())}
    )()
    replay = experiment.batches(16, 94, 20261015, replay=True)
    shuffled = experiment.batches(16, 94, 20261015)
    assert (
        list(map(len, replay))
        == list(map(len, shuffled))
        == [16] * 46 + [11] + [16] * 46 + [11]
    )
    for epoch in range(2):
        original = [
            r["trajectory_id"] for b in replay[epoch * 47 : (epoch + 1) * 47] for r in b
        ]
        changed = [
            r["trajectory_id"]
            for b in shuffled[epoch * 47 : (epoch + 1) * 47]
            for r in b
        ]
        assert set(original) == set(changed) and original != changed
    assert shuffled == experiment.batches(16, 94, 20261015)
    assert shuffled != experiment.batches(16, 94, 20261016)


def test_qwen3_cache_provenance_and_backbone_alias_guards(tmp_path):
    """Qwen3 base and r480-final cannot silently accept the Qwen3.5 identities."""
    base = dict(
        protocol=data.QWEN3_PROTOCOL,
        depth=36,
        hidden_size=2560,
        layer="final",
        feature_semantics="score_input_final_norm_shifted_action_prefix",
        checkpoint=str(tmp_path / pre.QWEN3_REVISION),
        checkpoint_config_sha256="a" * 64,
        checkpoint_files_sha256={"model.safetensors": "b" * 64},
    )
    pre.validate_primary_cache(base)
    trained = {
        **base,
        "checkpoint": str(
            tmp_path / "critic-instruct-formal-v1/critic/epoch1epochstep46globalstep93"
        ),
        "checkpoint_files_sha256": {"model.safetensors": "c" * 64},
    }
    pre.validate_e7_cache_alias("r480-final", trained, base)
    with pytest.raises(ValueError, match="pinned base"):
        pre.validate_primary_cache(
            {**base, "checkpoint": str(tmp_path / pre.PINNED_BASE_REVISION)}
        )
    with pytest.raises(ValueError, match="registered.*checkpoint"):
        pre.validate_e7_cache_alias(
            "r480-final",
            {
                **trained,
                "checkpoint": str(
                    tmp_path
                    / "critic-instruct-formal-v1/default/epoch1epochstep46globalstep93"
                ),
            },
            base,
        )


def test_q2_checks_both_domains_at_registered_tolerance():
    """Retail disagreement alone fails Q2 even if Airline and head norm look good."""
    real = {d: {"explained_variance": 0.0} for d in ("airline", "retail")}
    good = {d: {"explained_variance": 0.1} for d in real}
    assert q3.q2_status(good, real)["status"] == "成立"
    good["retail"]["explained_variance"] = 0.1001
    assert q3.q2_status(good, real)["status"] == "不成立"


def test_qwen3_extraction_parts_merge_without_planted_rule(qwen3_pool, tmp_path):
    """Qwen3 extraction/merge uses seed20261015 and has no out-of-scope E8 gate."""
    from transformers import AutoModelForTokenClassification, Qwen3Config

    pool, all_rows = data.load_pool(qwen3_pool, synthetic=True)
    rows = [all_rows[0], all_rows[-1]]
    rows[1]["_pool_index"] = 1
    config = Qwen3Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=36,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        num_labels=1,
    )
    config._attn_implementation = "sdpa"
    model = AutoModelForTokenClassification.from_config(config).eval()
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint)
    rule = {"status": "not_applicable", "rule": None}
    parts = [tmp_path / f"part-{i}" for i in range(2)]
    for index, output in enumerate(parts):
        extraction.write_part(
            model,
            rows,
            output,
            pool_manifest=pool,
            tokenizer=None,
            rule=rule,
            layers=["final", "9", "18", "27"],
            checkpoint=checkpoint,
            shard_index=index,
            shard_count=2,
        )
    for layer in ("final", "9", "18", "27"):
        merged = tmp_path / f"merged-{layer}"
        result = extraction.merge_parts([p / layer for p in parts], merged, rows, pool)
        assert result["trajectories"] == 2 and result["feature_rows"] == 12
        manifest = json.loads((merged / "manifest.json").read_text())
        assert (
            manifest["seed"] == 20261015 and manifest["protocol"] == data.QWEN3_PROTOCOL
        )
        assert manifest["run_ids"] == ["r480"] and manifest["planted_rule"] == rule
        for record in manifest["shards"]:
            payload = torch.load(merged / record["path"], weights_only=True)
            row = next(
                r for r in rows if r["_trajectory_id"] == payload["trajectory_id"][0]
            )
            expected = extraction.extract_layers(
                model, row, torch.tensor(payload["position"]), [layer]
            )[layer]
            torch.testing.assert_close(
                payload["features"], expected.bfloat16(), rtol=0, atol=0
            )


def test_qwen3_e3_runner_exact_replication_then_five_rates_three_seeds(
    tmp_path, monkeypatch
):
    """Q2 first uses stored head/replay, then fifteen K2 arms use train-selected lr."""
    experiment = q3.Qwen3Preexperiment.__new__(q3.Qwen3Preexperiment)
    experiment.out = tmp_path
    experiment.provenance = {}
    experiment.seeds = (20261015, 20261016, 20261017)
    experiment.initial_checkpoint = tmp_path / "initial"
    experiment.reference_path = tmp_path / "reference.json"
    domain_metrics = {d: {"explained_variance": 0.2} for d in ("airline", "retail")}
    row = dict(
        domain="airline", task_id="test", source_id="source", n_tokens=2, target=1.0
    )
    write_json(
        experiment.reference_path,
        dict(completed_step=94, rows=[row], by_domain=domain_metrics),
    )
    experiment.results = {
        "E2": {
            "probes": {"ridge_shared": {"validation": {"by_domain": domain_metrics}}}
        }
    }
    initial = {"weight": torch.ones(1, 3), "bias": torch.tensor([0.375])}
    monkeypatch.setattr(
        q3, "read_initial_head", lambda path: (initial, {"source": str(path)})
    )
    calls = []

    def simulate(**kwargs):
        calls.append(kwargs)
        return dict(
            lr=kwargs["lr"],
            seed=kwargs.get("seed", 20261015),
            evaluations=[
                dict(step=94, validation=dict(by_domain=domain_metrics, rows=[row]))
            ],
            maximum_adjacent_domain_bias_jump={d: 0.1 for d in domain_metrics},
        ), None

    experiment.simulate = simulate
    cv_calls = []
    experiment.select_lrs = lambda **kwargs: cv_calls.append(kwargs) or [{"lr": 1e-4}]
    result = experiment.run_E3()
    assert calls[0] == {"lr": 5e-6, "replay": True, "initial_head": initial}
    assert cv_calls == [{"steps": 94}]
    assert [(r["lr"], r["seed"]) for r in calls[1:]] == [
        (lr, seed) for lr in pre.LRS for seed in experiment.seeds
    ]
    assert result["selected_lr"] == 1e-4 and result["Q2"]["status"] == "成立"
    assert (tmp_path / "E3-Q2.json").is_file() and (tmp_path / "E3.json").is_file()


@pytest.mark.parametrize("failed_worker", [False, True])
def test_qwen3_launcher_eight_shards_and_worker_failure(tmp_path, failed_worker):
    """The E1 shell propagates worker errors and extracts precisely two backbones."""
    import os
    import subprocess

    interpreter = tmp_path / "fake-python"
    interpreter.write_text("""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args=sys.argv[1:]
def value(flag):return args[args.index(flag)+1]
mode=value('--mode')
out=Path(value('--output'));out.parent.mkdir(parents=True,exist_ok=True)
if mode=='estimate':out.write_text('{}')
if mode=='extract':
 out.mkdir(parents=True,exist_ok=True)
 (out/'arguments.json').write_text(json.dumps(dict(model=value('--model-name'), shard=value('--shard-index'), gpu=os.environ['CUDA_VISIBLE_DEVICES'])))
 if os.environ.get('FAIL_WORKER')=='1' and value('--model-name')=='base' and value('--shard-index')=='3':sys.exit(7)
if mode=='merge':
 out.mkdir(exist_ok=True)
 (out/(value('--model-name')+'.merged')).write_text('ok')
""")
    interpreter.chmod(0o755)
    output = tmp_path / "e1"
    script = Path(pre.__file__).with_name("launch_critic_preexp_qwen3.sh")
    result = subprocess.run(
        [
            "bash",
            str(script),
            str(interpreter),
            str(tmp_path),
            "pool",
            "base",
            "trained",
            str(output),
        ],
        env={**os.environ, "FAIL_WORKER": str(int(failed_worker))},
        capture_output=True,
        text=True,
    )
    assert (result.returncode != 0) == failed_worker, result.stderr
    if failed_worker:
        assert not (output / "caches/base.merged").exists()
    else:
        for model in ("base", "r480"):
            assert (output / f"caches/{model}.merged").is_file()
            records = [
                json.loads(p.read_text())
                for p in output.glob(f"parts-{model}/part-*/arguments.json")
            ]
            assert len(records) == 8
            assert sorted(int(r["shard"]) for r in records) == list(range(8))
            assert all(r["gpu"] == r["shard"] and r["model"] == model for r in records)


def test_q3_train_selected_rate_positive_probe_and_q2_prerequisite():
    """Ratio claims require positive ridge EV, matching Q2 and a stable offset."""
    probe = {d: {"explained_variance": 0.2} for d in ("airline", "retail")}

    def arm(ev, jump=0.1):
        return dict(
            lr=1e-4,
            evaluations=[
                dict(
                    validation=dict(
                        by_domain={d: {"explained_variance": ev} for d in probe}
                    )
                )
            ],
            maximum_adjacent_domain_bias_jump={d: jump for d in probe},
        )

    q2 = {"status": "成立"}
    assert q3.q3_status([arm(0.15)], probe, q2)["retail"]["status"] == "成立"
    assert (
        q3.q3_status([arm(0.15)], probe, {"status": "不成立"})["retail"]["status"]
        == "不确定"
    )
    assert q3.q3_status([arm(0.15, 0.21)], probe, q2)["retail"]["status"] == "不确定"
    assert q3.q3_status([arm(0.01)], probe, q2)["retail"]["status"] == "不成立"
    probe["retail"]["explained_variance"] = 0.0
    assert q3.q3_status([arm(0.15)], probe, q2)["retail"]["status"] == "不确定"


@pytest.mark.parametrize("invalid", [False, True])
def test_qwen3_pool_preserves_analysis_thread_count(qwen3_pool, invalid):
    """Integrity checks must not silently force later fits onto one CPU thread."""
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        if invalid:
            qwen3_pool.write_text(qwen3_pool.read_text() + " ")
            with pytest.raises(ValueError, match="Pool manifest SHA256"):
                data.load_pool(qwen3_pool, synthetic=True)
        else:
            data.load_pool(qwen3_pool, synthetic=True)
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(previous)


def test_qwen3_cli_rejects_stale_e2_provenance(tmp_path, monkeypatch):
    """Separate E3/E7 launchers cannot consume E2 from a different feature pool."""
    from types import SimpleNamespace

    pool, features = tmp_path / "pool.json", tmp_path / "features.json"
    pool.write_text("pool")
    features.write_text("features")
    write_json(
        tmp_path / "E2.json",
        dict(
            protocol=data.QWEN3_PROTOCOL,
            pool_manifest_sha256="wrong",
            feature_manifest_sha256=data.sha256(features),
            synthetic=False,
        ),
    )
    fake = SimpleNamespace(
        results={}, run_E7=lambda caches: pytest.fail("stale E2 reached E7")
    )
    monkeypatch.setattr(q3, "Qwen3Preexperiment", lambda *args, **kwargs: fake)
    with pytest.raises(ValueError, match="Saved E2 provenance"):
        q3.main(
            [
                "--pool-manifest",
                str(pool),
                "--features",
                str(features),
                "--out",
                str(tmp_path),
                "--experiments",
                "E7",
            ]
        )


def test_q3_checks_reaching_signal_within_budget_before_final_step():
    """A train-selected arm can meet Q3 at step50 even if step94 falls back."""
    probe = {d: {"explained_variance": 0.2} for d in ("airline", "retail")}
    evaluations = [
        dict(
            step=step,
            validation=dict(by_domain={d: {"explained_variance": ev} for d in probe}),
        )
        for step, ev in ((0, 0.0), (50, 0.15), (94, 0.01))
    ]
    arm = dict(
        lr=1e-4,
        evaluations=evaluations,
        maximum_adjacent_domain_bias_jump={d: 0.1 for d in probe},
    )
    result = q3.q3_status([arm], probe, {"status": "成立"})
    assert result["airline"]["status"] == "成立"
    assert result["airline"]["first_step_reaching_70pct"] == 50
    assert not result["airline"]["all_lr_seed_means_below_40pct"]
