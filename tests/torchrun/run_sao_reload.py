# SPDX-License-Identifier: Apache-2.0
"""Final SAO DCP reload probe.

This is a GPU torchrun probe to run after the full 135-step epoch exists. It
does not train or call optimizer.step(); it reloads native FSDP from DCP and
checks selected parameters against the role-matched HF checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.tensor import DTensor

from tests.torchrun.run_sao_fsdp_preflight import (
    _make_batch,
    _make_engine,
    _setup_distributed,
)

TEXT_REQUIRED_SUFFIXES: tuple[str, ...] = (
    "language_model.layers.0.linear_attn.A_log",
    "language_model.layers.0.linear_attn.dt_bias",
    "language_model.layers.0.linear_attn.norm.weight",
)
CRITIC_REQUIRED_SUFFIXES: tuple[str, ...] = ("score.weight",)


def _sha256_small(path: Path, *, max_bytes: int = 16 * 1024 * 1024) -> str | None:
    if not path.is_file() or path.stat().st_size > max_bytes:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_path_metadata(path: str) -> dict[str, Any]:
    root = Path(path)
    files: dict[str, dict[str, Any]] = {}
    for child in sorted(root.iterdir()) if root.is_dir() else []:
        if not child.is_file():
            continue
        stat = child.stat()
        files[child.name] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "sha256": _sha256_small(child),
        }
    return {"path": str(root), "exists": root.exists(), "files": files}


def inspect_dcp_metadata(path: str) -> dict[str, Any]:
    out = inspect_path_metadata(path)
    if not Path(path, ".metadata").is_file():
        out["metadata_status"] = "missing"
        return out
    try:
        metadata = FileSystemReader(path).read_metadata()
    except Exception as exc:
        out["metadata_status"] = "unreadable"
        out["metadata_error"] = f"{type(exc).__name__}: {exc}"
        return out
    state_keys = sorted(str(key) for key in metadata.state_dict_metadata.keys())
    out["metadata_status"] = "read"
    out["state_dict_key_count"] = len(state_keys)
    out["state_dict_key_sample"] = state_keys[:20]
    return out


def inspect_hf_metadata(path: str) -> dict[str, Any]:
    out = inspect_path_metadata(path)
    index_path = Path(path) / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        out["weight_map_count"] = len(index.get("weight_map", {}))
        out["metadata"] = index.get("metadata", {})
    return out


def _hf_weight_map(path: str) -> dict[str, str]:
    index_path = Path(path) / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as f:
            return json.load(f)["weight_map"]
    return {
        key: "model.safetensors"
        for key in safe_open(Path(path) / "model.safetensors", framework="pt").keys()
    }


def resolve_required_suffixes(
    available: Any, required_suffixes: tuple[str, ...]
) -> dict[str, str]:
    keys = list(available)
    resolved: dict[str, str] = {}
    for suffix in required_suffixes:
        matches = [key for key in keys if key == suffix or key.endswith(f".{suffix}")]
        if not matches:
            raise KeyError(f"required suffix not found: {suffix}")
        if len(matches) > 1:
            raise KeyError(f"required suffix is ambiguous: {suffix} -> {matches}")
        resolved[suffix] = matches[0]
    return resolved


def load_hf_tensor(path: str, key: str) -> torch.Tensor:
    weight_map = _hf_weight_map(path)
    if key not in weight_map:
        raise KeyError(f"HF key not found: {key}")
    shard = Path(path) / weight_map[key]
    with safe_open(shard, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def _full_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        return tensor.full_tensor().detach().cpu()
    return tensor.detach().cpu()


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    flat = tensor.detach().float().flatten()
    sample = flat[:: max(flat.numel() // 4096, 1)][:4096] if flat.numel() else flat
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sample_sum": float(sample.sum().item()) if sample.numel() else 0.0,
    }


def _comparison_tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype is torch.float32:
        return 1.0e-6, 1.0e-6
    return 0.0, 0.0


def _compare_tensors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual_cast = actual.to(dtype=expected.dtype)
    diff = (actual_cast.float() - expected.float()).abs()
    rtol, atol = _comparison_tolerance(expected.dtype)
    return {
        "actual": _tensor_summary(actual_cast),
        "expected": _tensor_summary(expected),
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "exact_equal_after_dtype_cast": bool(torch.equal(actual_cast, expected)),
        "rtol": rtol,
        "atol": atol,
        "allclose": bool(torch.allclose(actual_cast, expected, rtol=rtol, atol=atol)),
    }


def _required_suffixes(role: str) -> tuple[str, ...]:
    return TEXT_REQUIRED_SUFFIXES + (
        CRITIC_REQUIRED_SUFFIXES if role == "critic" else ()
    )


def _resolve_selected_pairs(
    *,
    role: str,
    model_param_names: Any,
    hf_weight_keys: Any,
) -> list[tuple[str, str, str]]:
    required = _required_suffixes(role)
    model_keys = resolve_required_suffixes(model_param_names, required)
    hf_keys = resolve_required_suffixes(hf_weight_keys, required)
    return [(suffix, model_keys[suffix], hf_keys[suffix]) for suffix in required]


def _optimizer_step_for_param(
    optimizer: torch.optim.Optimizer, param: torch.Tensor
) -> Any:
    state = optimizer.state.get(param, {})
    step = state.get("step")
    if step is None:
        return None
    if isinstance(step, torch.Tensor):
        if step.numel() == 1:
            return step.detach().cpu().item()
        return {"shape": list(step.shape), "dtype": str(step.dtype)}
    if isinstance(step, (int, float)):
        return step
    return str(step)


def optimizer_step_status(steps: dict, expected: int, required_count: int) -> str:
    if len(steps) != required_count or any(
        not isinstance(value, (int, float)) or isinstance(value, bool)
        for value in steps.values()
    ):
        return "unknown_absent"
    if not steps or any(value != expected for value in steps.values()):
        return "mismatch"
    return "ok"


def _finite_forward(engine: Any, role: str) -> dict[str, Any]:
    seq_len = 64
    prompt_len = 16
    batch = _make_batch(
        role=role,
        device=engine.device,
        seq_len=seq_len,
        prompt_len=prompt_len,
        vocab_size=int(getattr(engine.model_config, "vocab_size", 151936)),
        rank=dist.get_rank(),
    )
    batch = {
        key: value
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask", "loss_mask"}
    }
    out = engine.forward_batch(batch)
    tensor = out[0] if isinstance(out, list) else out
    finite = torch.isfinite(tensor.detach()).all().to(torch.int32)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite_all_ranks": bool(finite.cpu().item()),
    }


def _write_json(path: str, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def aggregate_rank_statuses(gathered: list[dict[str, Any] | None]) -> str:
    rank_statuses = [item.get("status") if item else "error" for item in gathered]
    if all(status == "ok" for status in rank_statuses):
        return "ok"
    if any(status == "unknown" for status in rank_statuses) and not any(
        status == "error" for status in rank_statuses
    ):
        return "unknown"
    return "error"


def _run(args: argparse.Namespace) -> dict[str, Any]:
    _setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    engine = None
    try:
        if world_size != 4:
            raise ValueError(
                f"Final reload qualification requires four ranks, got {world_size}"
            )
        if args.expected_steps != 135:
            raise ValueError(
                f"expected final epoch steps must be 135, got {args.expected_steps}"
            )
        if not Path(args.dcp_path).is_dir():
            raise FileNotFoundError(args.dcp_path)
        if not Path(args.hf_path).is_dir():
            raise FileNotFoundError(args.hf_path)

        engine = _make_engine(args)
        engine.eval()
        named = dict(engine.model.named_parameters())
        weight_map = _hf_weight_map(args.hf_path)
        pairs = _resolve_selected_pairs(
            role=args.role,
            model_param_names=named.keys(),
            hf_weight_keys=weight_map.keys(),
        )

        base = {name: _full_cpu_tensor(named[name]) for _, name, _ in pairs}
        base_summary = {name: _tensor_summary(tensor) for name, tensor in base.items()}
        checkpoint_before = {
            "hf": inspect_hf_metadata(args.hf_path),
            "dcp": inspect_dcp_metadata(args.dcp_path),
        }

        engine._load_from_dcp(args.dcp_path, with_optim=True)
        loaded = {name: _full_cpu_tensor(named[name]) for _, name, _ in pairs}

        comparisons: dict[str, Any] = {}
        noop_checks: dict[str, Any] = {}
        for _, name, hf_key in pairs:
            hf_tensor = load_hf_tensor(args.hf_path, hf_key)
            comparisons[name] = _compare_tensors(loaded[name], hf_tensor)
            noop_checks[name] = _compare_tensors(base[name], loaded[name])
            noop_checks[name]["changed_from_base"] = not torch.equal(
                base[name].to(loaded[name].dtype), loaded[name]
            )

        if not any(item["changed_from_base"] for item in noop_checks.values()):
            raise RuntimeError(
                "selected parameters did not change from Base after DCP load"
            )
        if not all(item["allclose"] for item in comparisons.values()):
            raise RuntimeError(
                "one or more selected parameters differ from HF checkpoint"
            )

        optimizer_steps = {
            name: _optimizer_step_for_param(engine.optimizer, named[name])
            for _, name, _ in pairs
            if engine.optimizer is not None
        }
        optimizer_step_state = optimizer_step_status(
            optimizer_steps, args.expected_steps, len(pairs)
        )
        if optimizer_step_state == "mismatch":
            raise RuntimeError(f"optimizer step mismatch: {optimizer_steps}")

        forward = _finite_forward(engine, args.role)
        if not forward["finite_all_ranks"]:
            raise RuntimeError(f"non-finite forward result: {forward}")
        checkpoint_after = {
            "hf": inspect_hf_metadata(args.hf_path),
            "dcp": inspect_dcp_metadata(args.dcp_path),
        }
        if checkpoint_before != checkpoint_after:
            raise RuntimeError("checkpoint files changed during reload probe")

        return {
            "status": "ok" if optimizer_step_state == "ok" else "unknown",
            "passed": optimizer_step_state == "ok",
            "role": args.role,
            "rank": rank,
            "world_size": world_size,
            "paths": {
                "model_path": args.model_path,
                "dcp_path": args.dcp_path,
                "hf_path": args.hf_path,
            },
            "runtime": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": str(engine.device),
                "model_type": getattr(engine.model_config, "model_type", None),
                "architectures": getattr(engine.model_config, "architectures", None),
                "backend": engine.config.backend,
            },
            "metadata": {
                "base": inspect_hf_metadata(args.model_path),
                **checkpoint_after,
            },
            "checkpoint_files_stable_during_reload": True,
            "selected_pairs": [
                {"required_suffix": suffix, "model_key": name, "hf_key": hf_key}
                for suffix, name, hf_key in pairs
            ],
            "base_summary": base_summary,
            "comparisons": comparisons,
            "noop_checks": noop_checks,
            "optimizer_step_status": optimizer_step_state,
            "optimizer_steps": optimizer_steps,
            "forward": forward,
        }
    finally:
        if engine is not None:
            engine.destroy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["actor", "critic"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dcp-path", required=True)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--expected-steps", type=int, default=135)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5.0e-6)
    parser.add_argument("--logprobs-chunk-size", type=int, default=64)
    args = parser.parse_args()

    payload: dict[str, Any]
    try:
        payload = _run(args)
    except Exception as exc:
        payload = {
            "status": "error",
            "passed": False,
            "rank": dist.get_rank() if dist.is_initialized() else None,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    gathered: list[Any] | None = None
    if dist.is_initialized():
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, payload)
        rank = dist.get_rank()
    else:
        gathered = [payload]
        rank = 0

    rank_path = f"{args.output_json}.rank{rank}.json"
    _write_json(rank_path, payload)
    if rank == 0:
        aggregate_status = aggregate_rank_statuses(gathered)
        aggregate = {
            "status": aggregate_status,
            "passed": aggregate_status == "ok",
            "ranks": gathered,
            "rank_files": [
                f"{args.output_json}.rank{i}.json" for i in range(len(gathered))
            ],
        }
        _write_json(args.output_json, aggregate)
        print(json.dumps(aggregate, indent=2, sort_keys=True, default=str), flush=True)

    if dist.is_initialized():
        dist.destroy_process_group()
    if payload.get("status") != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
