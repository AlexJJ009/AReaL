# SPDX-License-Identifier: Apache-2.0
"""Reject incomplete or stale final DCP reload evidence without allocating GPUs."""

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.sao.verify_completion import verify_reload


def _fixture(tmp_path: Path, role: str = "actor"):
    config = {"experiment_name": "test", "trial_name": "formal"}
    frozen = {"model_path": str(tmp_path / "base")}
    root = (
        tmp_path
        / "checkpoints/root/test/formal"
        / ("default" if role == "actor" else "critic")
    )
    paths = {
        "model_path": frozen["model_path"],
        "dcp_path": str(root / "recover_checkpoint"),
        "hf_path": str(root / "epoch0epochstep134globalstep134"),
    }
    metadata = {}
    for kind, name in (("dcp", ".metadata"), ("hf", "config.json")):
        path = Path(paths[f"{kind}_path"]) / name
        path.parent.mkdir(parents=True)
        path.write_bytes(b"fixture")
        metadata[kind] = {
            "files": {
                name: {
                    "size": 7,
                    "mtime_ns": path.stat().st_mtime_ns,
                    "ctime_ns": path.stat().st_ctime_ns,
                    "sha256": hashlib.sha256(b"fixture").hexdigest(),
                }
            }
        }
    count = 4 if role == "critic" else 3
    ranks = [
        {
            "rank": rank,
            "world_size": 4,
            "role": role,
            "status": "ok",
            "passed": True,
            "paths": paths,
            "metadata": metadata,
            "optimizer_step_status": "ok",
            "optimizer_steps": {f"p{i}": 135 for i in range(count)},
            "comparisons": {f"p{i}": {"allclose": True} for i in range(count)},
            "noop_checks": {"p0": {"changed_from_base": True}},
            "forward": {"finite_all_ranks": True},
            "checkpoint_files_stable_during_reload": True,
        }
        for rank in range(4)
    ]
    payload = {"status": "ok", "passed": True, "ranks": copy.deepcopy(ranks)}
    path = tmp_path / "evidence" / f"final-reload-{role}.json"
    path.parent.mkdir()
    path.write_text(json.dumps(payload))
    return config, frozen, path, payload


@pytest.mark.parametrize("role", ["actor", "critic"])
def test_complete_reload_receipt_matches_live_files(tmp_path, role):
    config, frozen, _, _ = _fixture(tmp_path, role)
    assert verify_reload(tmp_path, role, frozen, config)["passed"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_rank",
        "duplicate_rank",
        "wrong_role",
        "wrong_paths",
        "missing_optimizer",
        "wrong_optimizer",
        "missing_comparison",
        "noop",
        "nonfinite",
        "stale_metadata",
    ],
)
def test_final_reload_rejects_incomplete_or_stale_evidence(tmp_path, mutation):
    config, frozen, path, payload = _fixture(tmp_path)
    rank = payload["ranks"][0]
    if mutation == "missing_rank":
        payload["ranks"].pop()
    elif mutation == "duplicate_rank":
        rank["rank"] = 1
    elif mutation == "wrong_role":
        rank["role"] = "critic"
    elif mutation == "wrong_paths":
        rank["paths"]["hf_path"] = str(tmp_path / "old")
    elif mutation == "missing_optimizer":
        rank["optimizer_steps"].pop("p0")
    elif mutation == "wrong_optimizer":
        rank["optimizer_steps"]["p0"] = 134
    elif mutation == "missing_comparison":
        rank["comparisons"].pop("p0")
    elif mutation == "noop":
        rank["noop_checks"]["p0"]["changed_from_base"] = False
    elif mutation == "nonfinite":
        rank["forward"]["finite_all_ranks"] = False
    else:
        (Path(rank["paths"]["dcp_path"]) / ".metadata").write_bytes(b"changed")
    path.write_text(json.dumps(payload))
    assert not verify_reload(tmp_path, "actor", frozen, config)["passed"]


def test_same_size_unhashed_shard_overwrite_invalidates_reload(tmp_path):
    config, frozen, path, payload = _fixture(tmp_path)
    shard = Path(payload["ranks"][0]["paths"]["dcp_path"]) / "large.distcp"
    with shard.open("wb") as stream:
        stream.truncate(17 * 1024 * 1024)
    stat = shard.stat()
    binding = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "sha256": None,
    }
    for rank in payload["ranks"]:
        rank["metadata"]["dcp"]["files"][shard.name] = dict(binding)
    path.write_text(json.dumps(payload))
    assert verify_reload(tmp_path, "actor", frozen, config)["passed"]
    with shard.open("r+b") as stream:
        stream.write(b"changed")
    os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert shard.stat().st_size == stat.st_size
    assert not verify_reload(tmp_path, "actor", frozen, config)["passed"]
