# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for the SAO task-pack policy wrapper."""

import os
import subprocess
from collections import Counter
from pathlib import Path

import pytest
from datasets import Dataset

from examples.tau2 import task_pack, train
from examples.tau2.utils import Tau2PPOConfig
from scripts.tau2 import train_sao_pack

from areal.api.cli_args import load_expr_config


@pytest.fixture
def pack_setup(monkeypatch, tmp_path):
    """Use two domain pools and official validation rows without any API calls."""

    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.setenv("TAU2_TRIAL_NAME", "pack-test")
    monkeypatch.setenv("TAU2_CRITIC_PATH", str(tmp_path))
    pack = str(tmp_path / "pack")
    pack_ids = {"airline": ["p0", "p1"], "retail": ["p0"]}
    monkeypatch.setattr(task_pack, "domains", lambda _: list(pack_ids))
    monkeypatch.setattr(task_pack, "task_ids", lambda _, domain: pack_ids[domain])
    monkeypatch.setattr(train.benchmark_variant, "validate_task_files", lambda *a: None)
    calls = []

    def official_dataset(**kwargs):
        calls.append(kwargs.copy())
        split = kwargs["split"]
        ids = {"airline": ["o0"], "retail": ["o0", "o1"]}
        rows = [
            {
                "domain": domain,
                "task_id": task_id if split == "train" else "v0",
                "split": split,
                "source_id": f"tau2:{domain}:{task_id}",
                **kwargs["provenance"],
            }
            for domain in kwargs["domain"]
            for task_id in (ids[domain] if split == "train" else ["v0"])
        ]
        return Dataset.from_list(rows)

    monkeypatch.setattr(train, "get_tau2_dataset", official_dataset)
    monkeypatch.setattr(train_sao_pack, "get_tau2_dataset", official_dataset)
    args = [
        "--config",
        "examples/tau2/config_sao.yaml",
        "econfig.benchmark_variant=fixed",
        "effective_episodes=null",
        "domain_effective_episodes.airline=12",
        "domain_effective_episodes.retail=12",
        "train_dataset.batch_size=4",
        "train_batch_episodes=4",
    ]
    config, _ = load_expr_config(args + [f"train_task_pack={pack}"], Tau2PPOConfig)
    kwargs = {
        "domain": tuple(config.domains),
        "provenance": train.collection_provenance(
            config, actor_source=config.actor.path
        ),
        "seed": config.seed,
        "experiment_mode": config.experiment_mode,
        "variant": config.econfig.benchmark_variant,
    }
    return config, kwargs, args, pack, calls


def _identity(row):
    return row["domain"], row["task_id"], row.get("task_pack")


def test_combined_rows_pack_and_official_keep_source_and_provenance(pack_setup):
    config, kwargs, _, pack, calls = pack_setup
    config.domain_effective_episodes = {}

    rows = list(train_sao_pack.build_train_dataset(config, kwargs))

    assert len(rows) == 6
    assert {_identity(r) for r in rows} == {
        ("airline", "p0", pack),
        ("airline", "p1", pack),
        ("retail", "p0", pack),
        ("airline", "o0", None),
        ("retail", "o0", None),
        ("retail", "o1", None),
    }
    for row in rows:
        assert row["split"] == "train"
        prefix = "pack" if row["task_pack"] else "tau2"
        assert row["source_id"] == f"{prefix}:{row['domain']}:{row['task_id']}"
        assert all(row[key] == value for key, value in kwargs["provenance"].items())
    assert "domain_effective_episodes" not in calls[0]


def test_schedule_four_times_pool_repeats_every_row_four_times(pack_setup):
    config, kwargs, *_ = pack_setup

    rows = list(train_sao_pack.build_train_dataset(config, kwargs))

    counts = Counter(_identity(row) for row in rows)
    assert len(rows) == 24
    assert len(counts) == 6
    assert set(counts.values()) == {4}
    assert rows == list(train_sao_pack.build_train_dataset(config, kwargs))


@pytest.mark.parametrize("airline,retail", [(13, 12), (2, 12), (9, 12)])
def test_schedule_nonuniform_budget_fails(pack_setup, airline, retail):
    config, kwargs, *_ = pack_setup
    config.domain_effective_episodes = {"airline": airline, "retail": retail}

    with pytest.raises(ValueError, match="same positive number"):
        train_sao_pack.build_train_dataset(config, kwargs)


def test_schedule_missing_domain_budget_fails(pack_setup):
    config, kwargs, *_ = pack_setup
    config.domain_effective_episodes = {"airline": 12}

    with pytest.raises(ValueError, match="cover selected domains exactly"):
        train_sao_pack.build_train_dataset(config, kwargs)


def test_plain_policy_with_pack_fails_before_loading_rows(pack_setup):
    _, _, args, pack, calls = pack_setup

    with pytest.raises(ValueError, match="train_task_pack requires"):
        train.main(args + [f"train_task_pack={pack}", "--check-config"])
    assert calls == []


def test_pack_entry_without_pack_fails(pack_setup):
    _, _, args, _, calls = pack_setup

    with pytest.raises(ValueError, match="requires train_task_pack"):
        train_sao_pack.main(args + ["--check-config"])
    assert calls == []


def test_pack_entry_nonfixed_variant_fails(pack_setup):
    _, _, args, pack, calls = pack_setup

    with pytest.raises(ValueError, match="requires benchmark_variant=fixed"):
        train_sao_pack.main(
            args
            + [
                f"train_task_pack={pack}",
                "econfig.benchmark_variant=official",
                "--check-config",
            ]
        )
    assert calls == []


def test_pack_entry_nonsao_algorithm_fails(pack_setup):
    config, kwargs, *_ = pack_setup
    config.algorithm = "grpo"

    with pytest.raises(ValueError, match="requires algorithm=sao"):
        train_sao_pack.build_train_dataset(config, kwargs)


def test_pack_entry_task_limit_truncating_schedule_fails(pack_setup):
    config, kwargs, *_ = pack_setup
    config.task_limit = 23

    with pytest.raises(ValueError, match="must not truncate"):
        train_sao_pack.build_train_dataset(config, kwargs)


def test_check_config_pack_validation_unchanged_and_no_workers(pack_setup, monkeypatch):
    _, _, args, pack, calls = pack_setup
    logs = []
    monkeypatch.setattr(train.logger, "info", lambda *a: logs.append(a))

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "check-config must not create workers or resolve snapshots"
        )

    monkeypatch.setattr(train, "resolve_actor_snapshot", forbidden)
    monkeypatch.setattr(train, "PPOTrainer", forbidden)

    train.main(args + ["--check-config"])
    plain_validation_args = calls[-1]
    plain_validation = list(train.get_tau2_dataset(**plain_validation_args))
    train_sao_pack.main(args + [f"train_task_pack={pack}", "--check-config"])
    pack_validation_args = calls[-1]
    pack_validation = list(train.get_tau2_dataset(**pack_validation_args))

    assert pack_validation_args == plain_validation_args
    assert pack_validation_args["split"] == "test"
    assert pack_validation == plain_validation
    assert all("task_pack" not in row for row in pack_validation)
    assert logs[-1] == ("Train prompts=%d, validation prompts=%d, steps=%d", 24, 2, 12)


def test_pack_and_official_same_task_id_are_distinct(pack_setup, monkeypatch):
    config, kwargs, *_ = pack_setup
    monkeypatch.setattr(task_pack, "task_ids", lambda _, domain: ["o0"])
    config.domain_effective_episodes = {}

    rows = list(train_sao_pack.build_train_dataset(config, kwargs))

    assert len({_identity(row) for row in rows}) == 5


def test_policy_official_train_validation_overlap_fails(pack_setup, monkeypatch):
    _, _, args, pack, _ = pack_setup
    original = train.get_tau2_dataset

    def overlapping(**kwargs):
        rows = list(original(**kwargs))
        if kwargs["split"] == "test":
            rows[0]["task_id"] = "o0"
        return Dataset.from_list(rows)

    monkeypatch.setattr(train, "get_tau2_dataset", overlapping)
    with pytest.raises(ValueError, match="not disjoint"):
        train_sao_pack.main(args + [f"train_task_pack={pack}", "--check-config"])


def test_policy_pack_task_id_matching_validation_is_disjoint(pack_setup, monkeypatch):
    _, _, args, pack, _ = pack_setup
    monkeypatch.setattr(task_pack, "task_ids", lambda _, domain: ["v0"])
    train_sao_pack.main(
        args
        + [
            f"train_task_pack={pack}",
            "domain_effective_episodes.airline=8",
            "domain_effective_episodes.retail=12",
            "--check-config",
        ]
    )


def test_plain_policy_legacy_overlap_behavior_is_preserved(pack_setup, monkeypatch):
    """The wrapper guard does not change the no-pack policy validation path."""
    _, _, args, _, _ = pack_setup
    original = train.get_tau2_dataset

    def overlapping(**kwargs):
        rows = list(original(**kwargs))
        if kwargs["split"] == "dev":
            rows[0]["task_id"] = "o0"
        return Dataset.from_list(rows)

    monkeypatch.setattr(train, "get_tau2_dataset", overlapping)
    train.main(args + ["experiment_mode=qualification", "--check-config"])


def test_run_sh_pack_nonsao_algorithm_is_rejected(tmp_path):
    """Reject GRPO before loading runtime keys or invoking Python."""
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_python.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "scripts/tau2/run.sh",
            "policy-pack",
            "mixed",
            "grpo",
            "--check-config",
        ],
        env={
            **os.environ,
            "AREAL_PYTHON": str(fake_python),
            "TAU2_RUN_ROOT": str(tmp_path),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Policy-pack mode requires sao" in result.stderr


@pytest.mark.parametrize("domain", ["mixed", "airline", "retail"])
def test_run_sh_pack_sao_routes_wrapper_and_original_config(tmp_path, domain):
    fake_python = tmp_path / "python"
    fake_python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    fake_python.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "scripts/tau2/run.sh",
            "policy-pack",
            domain,
            "sao",
            "--check-config",
            "train_task_pack=/pack",
            "effective_episodes=null",
        ],
        env={
            **os.environ,
            "AREAL_PYTHON": str(fake_python),
            "TAU2_RUN_ROOT": str(tmp_path),
            "TAU2_CRITIC_PATH": str(tmp_path),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    args = result.stdout.splitlines()
    root = Path(__file__).resolve().parents[1]
    assert args[:3] == [
        str(root / "scripts/tau2/train_sao_pack.py"),
        "--config",
        str(root / "examples/tau2/config_sao.yaml"),
    ]
    assert "--check-config" in args
    assert "train_task_pack=/pack" in args
    if domain != "mixed":
        assert f"domains=[{domain}]" in args
        assert f"econfig.domain={domain}" in args
        other = "retail" if domain == "airline" else "airline"
        assert f"~domain_effective_episodes.{other}" in args
