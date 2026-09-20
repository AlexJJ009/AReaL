#!/usr/bin/env python3
"""Bounded SGLang preflight for SAO math qualification.

This script has two deliberately separate modes:

* ``audit`` performs static checks and writes a JSON record without touching GPUs.
* ``run`` starts SGLang, requests generation logprobs, profiles inference kernels,
  and writes a JSON record that fails when mandatory runtime evidence is missing.

The runtime mode is intentionally small.  It is a preflight probe for C04/C18/C20,
not a blanket acceptance gate for the full training run.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gzip
import hashlib
import importlib
import inspect
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXPECTED_SGLANG_VERSION = "0.5.10.post1"
EXPECTED_LAYER_COUNTS = {"linear_attention": 24, "full_attention": 8}
DEFAULT_CONFIG_PATH = Path("examples/math/sao_ppo.yaml")
DEFAULT_PROMPTS = (
    "Question: What is 2+3? Answer:",
    "Question: If x=4, what is x squared? Answer:",
)
MANDATORY_PROFILE_PATHS = {
    "gdn_or_linear_attention": (
        "gated_delta",
        "gdn",
        "linear_attention",
        "chunk_gated_delta",
    ),
    "causal_convolution": ("causal_conv", "conv1d"),
    "full_attention": (
        "flash_attn",
        "flashattention",
        "batchprefillwith",
        "batchdecodewith",
        "trtllm_mha",
    ),
    "cuda_graph": ("cudagraph", "cudagraphlaunch", "cuda_graph", "cuda graph"),
    "overlap": ("event_loop_overlap", "two_batch_overlap", "overlap_schedule"),
    "gpu_kernel_timing": ("__runtime_populated__",),
    "cuda_graph_runtime_event": ("__runtime_populated__",),
}


@dataclasses.dataclass
class Check:
    name: str
    ok: bool
    details: dict[str, Any]


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def clock_anchor(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "wall_utc": utc_now(),
        "time_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "perf_counter_ns": time.perf_counter_ns(),
        "pid": os.getpid(),
    }


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path, *, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fin:
        for chunk in iter(lambda: fin.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        from importlib import metadata

        return metadata.version(name)
    except Exception:
        return None


def run_text(cmd: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception:
        return None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fin:
        return json.load(fin)


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as fin:
        data = yaml.safe_load(fin)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return data


def resolve_path(raw: str | None, env_name: str, description: str) -> Path:
    value = raw or os.getenv(env_name)
    if not value:
        flag = env_name.lower().replace("_", "-")
        raise SystemExit(f"Missing {description}: pass --{flag} or set {env_name}")
    return Path(value).expanduser().resolve()


def model_manifest(model_path: Path, *, hash_shards: bool) -> dict[str, Any]:
    files = [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    missing = [name for name in files if not (model_path / name).is_file()]
    shard_paths = sorted(model_path.glob("model.safetensors-*.safetensors"))
    if not shard_paths:
        missing.append("model.safetensors-*.safetensors")
    if missing:
        raise FileNotFoundError(f"Model path is missing mandatory files: {missing}")

    manifest: dict[str, Any] = {
        "path": str(model_path),
        "config_sha256": sha256_file(model_path / "config.json"),
        "index_sha256": sha256_file(model_path / "model.safetensors.index.json"),
        "files": {},
        "shards": [],
    }
    for path in sorted(model_path.iterdir()):
        if path.is_file():
            manifest["files"][path.name] = {"size": path.stat().st_size}
    for path in shard_paths:
        item: dict[str, Any] = {"name": path.name, "size": path.stat().st_size}
        if hash_shards:
            item["sha256"] = sha256_file(path)
        manifest["shards"].append(item)
    return manifest


def layer_counts(config_json: dict[str, Any]) -> dict[str, int]:
    text_config = config_json.get("text_config", {})
    layer_types = text_config.get("layer_types", [])
    counts: dict[str, int] = {}
    for layer_type in layer_types:
        counts[layer_type] = counts.get(layer_type, 0) + 1
    return counts


def intended_sglang_config(config_path: Path) -> dict[str, Any]:
    data = read_yaml_mapping(config_path)
    raw = data.get("sglang")
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} does not contain an sglang mapping")
    return raw


def check_intended_flags(raw: dict[str, Any]) -> Check:
    expected = {
        "attention_backend": "flashinfer",
        "dtype": "bfloat16",
        "disable_cuda_graph": False,
        "disable_overlap_schedule": False,
        "disable_radix_cache": True,
        "chunked_prefill_size": -1,
        "skip_tokenizer_init": True,
    }
    observed = {key: raw.get(key) for key in expected}
    return Check(
        name="intended_sglang_flags",
        ok=observed == expected,
        details={"expected": expected, "observed": observed},
    )


def static_sglang_api_audit() -> dict[str, Any]:
    modules = [
        "sglang",
        "sglang.srt.entrypoints.engine",
        "sglang.srt.managers.io_struct",
        "sglang.srt.server_args",
    ]
    result: dict[str, Any] = {"ok": True, "modules": {}}
    try:
        loaded = {name: importlib.import_module(name) for name in modules}
        sglang = loaded["sglang"]
        engine_module = loaded["sglang.srt.entrypoints.engine"]
        io_struct = loaded["sglang.srt.managers.io_struct"]
        generate_params = set(
            inspect.signature(engine_module.Engine.generate).parameters
        )
        required_generate_params = {
            "input_ids",
            "return_logprob",
            "logprob_start_len",
            "top_logprobs_num",
            "token_ids_logprob",
        }
        req_fields = set(io_struct.GenerateReqInput.__dataclass_fields__)
        missing_generate_params = sorted(required_generate_params - generate_params)
        result.update(
            {
                "version": getattr(sglang, "__version__", None)
                or package_version("sglang"),
                "has_engine": hasattr(sglang, "Engine"),
                "engine_signature": str(inspect.signature(sglang.Engine)),
                "generate_signature": str(
                    inspect.signature(engine_module.Engine.generate)
                ),
                "start_profile_signature": str(
                    inspect.signature(engine_module.Engine.start_profile)
                ),
                "stop_profile_signature": str(
                    inspect.signature(engine_module.Engine.stop_profile)
                ),
                "generate_required_params": sorted(required_generate_params),
                "generate_missing_params": missing_generate_params,
                "generate_req_has_required_fields": all(
                    field in req_fields for field in required_generate_params
                ),
            }
        )
        for name, module in loaded.items():
            result["modules"][name] = {"file": getattr(module, "__file__", None)}
        result["ok"] = (
            result["version"] == EXPECTED_SGLANG_VERSION
            and result["has_engine"]
            and not missing_generate_params
            and result["generate_req_has_required_fields"]
        )
    except Exception as exc:
        result.update(
            {
                "ok": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
    return result


def collect_static_evidence(
    *,
    model_path: Path,
    config_path: Path,
    artifact_root: Path,
    hash_model_shards: bool,
) -> dict[str, Any]:
    root = repo_root()
    model = model_manifest(model_path, hash_shards=hash_model_shards)
    config_json = load_json(model_path / "config.json")
    counts = layer_counts(config_json)
    layer_check = Check(
        name="qwen35_layer_mix",
        ok=counts == EXPECTED_LAYER_COUNTS,
        details={"expected": EXPECTED_LAYER_COUNTS, "observed": counts},
    )
    raw_sglang = intended_sglang_config(config_path)
    flag_check = check_intended_flags(raw_sglang)
    versions = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "sglang": package_version("sglang"),
        "torch": package_version("torch"),
        "flashinfer-python": package_version("flashinfer-python"),
        "flashinfer-cubin": package_version("flashinfer-cubin"),
        "sglang-kernel": package_version("sglang-kernel"),
        "transformers": package_version("transformers"),
    }
    return {
        "timestamp_utc": utc_now(),
        "clock_anchor": clock_anchor("static_evidence"),
        "repo": {
            "root": str(root),
            "git_sha": run_text(["git", "rev-parse", "HEAD"], root),
            "git_branch": run_text(["git", "branch", "--show-current"], root),
            "git_status_short": run_text(["git", "status", "--short"], root),
            "pyproject_sha256": sha256_file(root / "pyproject.toml"),
            "uv_lock_sha256": sha256_file(root / "uv.lock"),
        },
        "artifact_root": str(artifact_root),
        "model": model,
        "qwen35_config": {
            "architecture": config_json.get("architectures"),
            "model_type": config_json.get("model_type"),
            "text_model_type": config_json.get("text_config", {}).get("model_type"),
            "num_hidden_layers": config_json.get("text_config", {}).get(
                "num_hidden_layers"
            ),
            "layer_counts": counts,
            "linear_conv_kernel_dim": config_json.get("text_config", {}).get(
                "linear_conv_kernel_dim"
            ),
        },
        "intended_sglang": {
            "config_path": str(config_path),
            "raw": raw_sglang,
        },
        "versions": versions,
        "sglang_api": static_sglang_api_audit(),
        "checks": [dataclasses.asdict(layer_check), dataclasses.asdict(flag_check)],
    }


def import_sglang_engine() -> Any:
    module = importlib.import_module("sglang")
    version = getattr(module, "__version__", None) or package_version("sglang")
    if version != EXPECTED_SGLANG_VERSION:
        raise RuntimeError(
            f"Expected sglang {EXPECTED_SGLANG_VERSION}, found {version!r}"
        )
    return module.Engine


def build_areal_server_args(
    *,
    model_path: Path,
    raw_sglang: dict[str, Any],
    tp_size: int,
    base_gpu_id: int,
) -> dict[str, Any]:
    from areal.api.cli_args import SGLangConfig

    cfg = SGLangConfig(**{**raw_sglang, "model_path": str(model_path)})
    return SGLangConfig.build_args(
        sglang_config=cfg,
        tp_size=tp_size,
        base_gpu_id=base_gpu_id,
    )


def compact_server_args(args: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "model_path",
        "tokenizer_path",
        "trust_remote_code",
        "dtype",
        "tp_size",
        "base_gpu_id",
        "attention_backend",
        "disable_cuda_graph",
        "disable_overlap_schedule",
        "disable_radix_cache",
        "chunked_prefill_size",
        "context_length",
        "mem_fraction_static",
        "max_running_requests",
        "skip_tokenizer_init",
        "log_level",
    )
    return {key: args.get(key) for key in keys if key in args}


def encode_prompts(model_path: Path, prompts: Iterable[str]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
    )
    items = []
    input_ids = []
    for prompt in prompts:
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not token_ids:
            raise RuntimeError(f"Tokenizer returned empty input_ids for {prompt!r}")
        input_ids.append(token_ids)
        items.append(
            {
                "prompt": prompt,
                "input_ids": token_ids,
                "length": len(token_ids),
            }
        )
    return {
        "tokenizer_class": tokenizer.__class__.__name__,
        "items": items,
        "input_ids": input_ids,
    }


def generation_request(
    encoded_prompts: dict[str, Any], max_new_tokens: int
) -> dict[str, Any]:
    return {
        "input_ids": encoded_prompts["input_ids"],
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
        "top_logprobs_num": 1,
    }


def warmup_request(
    encoded_prompts: dict[str, Any], max_new_tokens: int
) -> dict[str, Any]:
    first_ids = encoded_prompts["input_ids"][0]
    return {
        "input_ids": first_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max(1, min(2, max_new_tokens)),
            "ignore_eos": True,
        },
        "return_logprob": False,
    }


def require_logprob_response(response: dict[str, Any]) -> dict[str, Any]:
    meta = response.get("meta_info")
    if not isinstance(meta, dict):
        raise RuntimeError("SGLang response is missing meta_info")
    output_ids = response.get("output_ids")
    output_logprobs = meta.get("output_token_logprobs")
    input_logprobs = meta.get("input_token_logprobs")
    if not isinstance(output_ids, list) or not output_ids:
        raise RuntimeError("SGLang response is missing non-empty output_ids")
    if not isinstance(output_logprobs, list) or len(output_logprobs) != len(output_ids):
        raise RuntimeError("SGLang response output logprobs do not match output_ids")
    if not isinstance(input_logprobs, list) or not input_logprobs:
        raise RuntimeError("SGLang response is missing input token logprobs")
    return {
        "text": response.get("text"),
        "output_ids": output_ids,
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": meta.get("completion_tokens"),
        "finish_reason": meta.get("finish_reason"),
        "input_token_logprobs": input_logprobs,
        "output_token_logprobs": output_logprobs,
        "output_top_logprobs": meta.get("output_top_logprobs"),
        "weight_version": meta.get("weight_version"),
    }


def summarize_generation(responses: Any, elapsed_s: float) -> dict[str, Any]:
    items = responses if isinstance(responses, list) else [responses]
    parsed = [require_logprob_response(item) for item in items]
    completion_tokens = sum(int(item.get("completion_tokens") or 0) for item in parsed)
    prompt_tokens = sum(int(item.get("prompt_tokens") or 0) for item in parsed)
    return {
        "elapsed_s": elapsed_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens_per_s": completion_tokens / elapsed_s if elapsed_s > 0 else None,
        "responses": parsed,
    }


def collect_torch_runtime() -> dict[str, Any]:
    import torch

    runtime: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        runtime.update(
            {
                "current_device": device,
                "device_name": props.name,
                "device_capability": (
                    list(props.major_minor)
                    if hasattr(props, "major_minor")
                    else [props.major, props.minor]
                ),
                "memory_allocated_bytes": torch.cuda.memory_allocated(device),
                "memory_reserved_bytes": torch.cuda.memory_reserved(device),
                "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
        )
    return runtime


def collect_child_pids(parent_pid: int) -> list[int]:
    output = run_text(["ps", "-eo", "pid=,ppid=,cmd="], repo_root())
    if not output:
        return []
    children: dict[int, list[int]] = {}
    for line in output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 2:
            continue
        with contextlib.suppress(ValueError):
            pid = int(parts[0])
            ppid = int(parts[1])
            children.setdefault(ppid, []).append(pid)
    result: list[int] = []
    stack = list(children.get(parent_pid, []))
    while stack:
        pid = stack.pop()
        result.append(pid)
        stack.extend(children.get(pid, []))
    return sorted(result)


def collect_nvidia_smi_for_pids(pids: Iterable[int]) -> dict[str, Any]:
    wanted = {str(pid) for pid in pids}
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,gpu_uuid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    output = run_text(cmd, repo_root())
    rows = []
    if output:
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4 or parts[0] not in wanted:
                continue
            rows.append(
                {
                    "pid": int(parts[0]),
                    "process_name": parts[1],
                    "gpu_uuid": parts[2],
                    "used_memory_mib": int(parts[3]),
                }
            )
    return {
        "source": "nvidia-smi compute-apps",
        "queried_pids": sorted(int(pid) for pid in wanted),
        "rows": rows,
        "note": "This is server/subprocess GPU memory readback, distinct from parent-process torch allocator stats.",
    }


def read_trace_json(path: Path) -> dict[str, Any] | None:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as fin:
                return json.load(fin)
        with path.open("r", encoding="utf-8") as fin:
            return json.load(fin)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def event_text(event: dict[str, Any]) -> str:
    fields: list[str] = []
    for key in ("name", "cat", "ph"):
        value = event.get(key)
        if value is not None:
            fields.append(str(value))
    args = event.get("args")
    if isinstance(args, dict):
        for key, value in args.items():
            fields.append(str(key))
            if value is not None:
                fields.append(str(value))
    return " ".join(fields).lower()


def is_gpu_timed_event(event: dict[str, Any]) -> bool:
    has_duration = isinstance(event.get("dur"), int | float) and event["dur"] > 0
    return has_duration and event.get("cat") in ("kernel", "gpu")


def scan_profile_dir(profile_dir: Path) -> dict[str, Any]:
    files = [path for path in profile_dir.rglob("*") if path.is_file()]
    matched: dict[str, list[str]] = {key: [] for key in MANDATORY_PROFILE_PATHS}
    events_by_path: dict[str, int] = {}
    kernel_events: list[dict[str, Any]] = []
    cuda_graph_events: list[dict[str, Any]] = []
    for path in files:
        if path.stat().st_size == 0:
            continue
        if not (
            path.name.endswith(".json")
            or path.name.endswith(".trace")
            or path.name.endswith(".json.gz")
            or path.name.endswith(".trace.gz")
        ):
            continue
        data = read_trace_json(path)
        if not isinstance(data, dict):
            continue
        trace_events = data.get("traceEvents")
        if not isinstance(trace_events, list):
            continue
        events_by_path[str(path)] = len(trace_events)
        for event in trace_events:
            if not isinstance(event, dict):
                continue
            haystack = event_text(event)
            if (
                event.get("cat") in ("cuda_runtime", "cuda_driver", "cuda")
                and "cudagraphlaunch" in str(event.get("name", "")).lower()
            ):
                cuda_graph_events.append(
                    {
                        "file": str(path),
                        "name": event.get("name"),
                        "ts": event.get("ts"),
                        "dur_us": event.get("dur"),
                    }
                )
            if is_gpu_timed_event(event):
                kernel_events.append(
                    {
                        "file": str(path),
                        "name": event.get("name"),
                        "cat": event.get("cat"),
                        "ts": event.get("ts"),
                        "dur_us": event.get("dur"),
                    }
                )
            for key, patterns in MANDATORY_PROFILE_PATHS.items():
                if key in (
                    "gdn_or_linear_attention",
                    "causal_convolution",
                    "full_attention",
                ) and not is_gpu_timed_event(event):
                    continue
                if any(pattern in haystack for pattern in patterns):
                    matched[key].append(str(path))
    matched = {key: sorted(set(paths)) for key, paths in matched.items()}
    missing_kernel_timing = not kernel_events
    missing_cuda_graph_runtime = not cuda_graph_events
    matched["gpu_kernel_timing"] = (
        [event["file"] for event in kernel_events] if kernel_events else []
    )
    matched["cuda_graph_runtime_event"] = (
        [event["file"] for event in cuda_graph_events] if cuda_graph_events else []
    )
    missing = [key for key, paths in matched.items() if not paths]
    if missing_kernel_timing:
        missing.append("gpu_kernel_timing")
    if missing_cuda_graph_runtime:
        missing.append("cuda_graph_runtime_event")
    missing = sorted(set(missing))
    return {
        "profile_dir": str(profile_dir),
        "files": [str(path) for path in files],
        "trace_event_counts": events_by_path,
        "matched_paths": matched,
        "kernel_events_sample": kernel_events[:200],
        "cuda_graph_events_sample": cuda_graph_events[:50],
        "missing_mandatory_paths": missing,
        "ok": not missing,
        "limits": (
            "Kernel names are collected only from parsed profiler event payloads, "
            "not filenames or arbitrary raw text. This proves the named paths appeared "
            "in the profiled run with event timestamps/durations where available, but it does not prove "
            "training-side numerical parity or an end-to-end speedup."
        ),
    }


def write_record(record: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        json.dump(record, fout, ensure_ascii=False, indent=2, sort_keys=True)
        fout.write("\n")


def run_preflight(args: argparse.Namespace) -> int:
    model_path = resolve_path(args.model_path, "SAO_MODEL_PATH", "model path")
    artifact_root = resolve_path(
        args.artifact_root, "SAO_ARTIFACT_ROOT", "artifact root"
    )
    config_path = Path(args.config).resolve()
    static = collect_static_evidence(
        model_path=model_path,
        config_path=config_path,
        artifact_root=artifact_root,
        hash_model_shards=args.hash_model_shards,
    )
    raw_sglang = static["intended_sglang"]["raw"]
    server_args = build_areal_server_args(
        model_path=model_path,
        raw_sglang=raw_sglang,
        tp_size=args.tp_size,
        base_gpu_id=args.base_gpu_id,
    )

    engine_kwargs = {
        key: value
        for key, value in server_args.items()
        if value is not None
        and key not in {"host", "port", "dist_init_addr", "nnodes", "node_rank"}
    }
    engine_kwargs.update(
        {
            "skip_tokenizer_init": True,
            "log_level": args.sglang_log_level,
            "log_requests": True,
            "show_time_cost": True,
        }
    )
    encoded_prompts = encode_prompts(model_path, DEFAULT_PROMPTS)
    request = generation_request(encoded_prompts, args.max_new_tokens)
    warmup = warmup_request(encoded_prompts, args.max_new_tokens)
    run_id = utc_now().replace(":", "").replace("-", "")
    profile_dir = artifact_root / "sglang-preflight" / run_id / "profile"
    record = {
        **static,
        "mode": "run",
        "clock_anchors": [clock_anchor("run_record_created")],
        "run_contract": {
            "claims": [
                "actual SGLang generation with token logprobs",
                "actual SGLang profiler trace scanned for mandatory paths",
                "same local model path in AReaL server args and runtime engine args",
                "skip_tokenizer_init=true engine path with local tokenizer input_ids",
            ],
            "non_claims": [
                "training-side logprob parity unless a separate HF comparison record is present",
                "C19 actor update acceleration",
                "full async PPO acceptance",
            ],
        },
        "areal_server_args": compact_server_args(server_args),
        "runtime": {},
        "tokenizer_fixture": encoded_prompts,
        "warmup_request": warmup,
        "generation_request": request,
    }
    engine = None
    try:
        Engine = import_sglang_engine()
        import torch

        record["clock_anchors"].append(clock_anchor("torch_imported"))
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing to run GPU preflight")
        device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
        capability = [device_props.major, device_props.minor]
        if capability != [8, 0]:
            raise RuntimeError(
                f"Expected SM80/A100 for this preflight, found capability {capability}"
            )
        record["runtime"]["torch_prelaunch"] = collect_torch_runtime()
        record["clock_anchors"].append(clock_anchor("before_engine_init"))
        from sglang.srt.server_args import ServerArgs

        unsupported = set(engine_kwargs) - set(ServerArgs.__dataclass_fields__)
        # The native AReaL CLI builder omits disabled boolean flags. Do the
        # same for obsolete options when exercising the Python Engine API.
        unsupported_active = {
            k: engine_kwargs[k] for k in unsupported if engine_kwargs[k] is not False
        }
        if unsupported_active:
            raise ValueError(f"Unsupported active SGLang options: {unsupported_active}")
        record["runtime"]["omitted_disabled_cli_flags"] = sorted(unsupported)
        for key in unsupported:
            engine_kwargs.pop(key)
        engine = Engine(**engine_kwargs)
        record["clock_anchors"].append(clock_anchor("after_engine_init"))
        record["runtime"]["engine_server_args"] = compact_server_args(
            dataclasses.asdict(engine.server_args)
            if dataclasses.is_dataclass(engine.server_args)
            else vars(engine.server_args)
        )
        if (
            record["runtime"]["engine_server_args"].get("skip_tokenizer_init")
            is not True
        ):
            raise RuntimeError(
                "Engine server args did not retain skip_tokenizer_init=true"
            )
        server_pids = collect_child_pids(os.getpid())
        record["runtime"]["server_pids_after_init"] = server_pids
        record["runtime"]["nvidia_smi_after_init"] = collect_nvidia_smi_for_pids(
            [os.getpid(), *server_pids]
        )

        warmup_start = clock_anchor("before_warmup_generation")
        warmup_perf_start = time.perf_counter()
        warmup_response = engine.generate(**warmup)
        warmup_elapsed_s = time.perf_counter() - warmup_perf_start
        warmup_end = clock_anchor("after_warmup_generation")
        record["warmup"] = {
            "clock_span": {
                "start": warmup_start,
                "end": warmup_end,
                "elapsed_perf_counter_s": warmup_elapsed_s,
                "elapsed_monotonic_ns": (
                    warmup_end["monotonic_ns"] - warmup_start["monotonic_ns"]
                ),
            },
            "response_shape": {
                "has_output_ids": bool(warmup_response.get("output_ids"))
                if isinstance(warmup_response, dict)
                else None,
            },
        }
        record["clock_anchors"].extend([warmup_start, warmup_end])
        record["clock_anchors"].append(clock_anchor("before_start_profile"))
        engine.start_profile(
            output_dir=str(profile_dir),
            num_steps=args.profile_steps,
            activities=["CPU", "GPU"],
            profile_by_stage=True,
            profile_prefix="sao_sglang_preflight",
        )
        record["clock_anchors"].append(clock_anchor("after_start_profile"))
        generation_start = clock_anchor("before_measured_generation")
        start = time.perf_counter()
        responses = engine.generate(**request)
        elapsed_s = time.perf_counter() - start
        generation_end = clock_anchor("after_measured_generation")
        record["clock_anchors"].extend([generation_start, generation_end])
        # num_steps makes SGLang stop and export automatically. Calling stop a
        # second time raises even after a successful profiled generation.
        record["profile_completion"] = {
            "mode": "automatic",
            "steps": args.profile_steps,
        }
        record["generation"] = summarize_generation(responses, elapsed_s)
        record["generation"]["clock_span"] = {
            "start": generation_start,
            "end": generation_end,
            "elapsed_perf_counter_s": elapsed_s,
            "elapsed_monotonic_ns": (
                generation_end["monotonic_ns"] - generation_start["monotonic_ns"]
            ),
        }
        record["runtime"]["torch_post_generation"] = collect_torch_runtime()
        record["runtime"]["server_pids_post_generation"] = collect_child_pids(
            os.getpid()
        )
        record["runtime"]["nvidia_smi_post_generation"] = collect_nvidia_smi_for_pids(
            [os.getpid(), *record["runtime"]["server_pids_post_generation"]]
        )
        record["profile"] = scan_profile_dir(profile_dir)
        record["logprob_fixture"] = {
            "purpose": "Small deterministic input_id/token/logprob fixture for trainer-side matching.",
            "comparison_status": "sglang_only",
            "numerical_parity_limits": (
                "The fixture binds SGLang token ids and selected-token logprobs. "
                "It does not prove parity with the training engine until a separate "
                "same-input trainer/HF logprob comparison consumes these ids."
            ),
            "items": record["generation"]["responses"],
        }
        record["ok"] = bool(record["profile"]["ok"])
    except Exception as exc:
        record["ok"] = False
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if engine is not None:
            with contextlib.suppress(Exception):
                engine.shutdown()

    output_path = (
        Path(args.output).resolve()
        if args.output
        else artifact_root / "sglang-preflight" / "latest.json"
    )
    write_record(record, output_path)
    print(str(output_path))
    return 0 if record["ok"] else 1


def audit_preflight(args: argparse.Namespace) -> int:
    model_path = resolve_path(args.model_path, "SAO_MODEL_PATH", "model path")
    artifact_root = resolve_path(
        args.artifact_root, "SAO_ARTIFACT_ROOT", "artifact root"
    )
    config_path = Path(args.config).resolve()
    record = collect_static_evidence(
        model_path=model_path,
        config_path=config_path,
        artifact_root=artifact_root,
        hash_model_shards=args.hash_model_shards,
    )
    checks = record["checks"]
    record.update(
        {
            "mode": "audit",
            "runtime_evidence": {
                "status": "not_run",
                "reason": (
                    "audit mode may import SGLang for API/signature readback, but "
                    "does not start an engine or allocate GPUs"
                ),
            },
            "ok": all(item["ok"] for item in checks) and record["sglang_api"]["ok"],
        }
    )
    output_path = (
        Path(args.output).resolve()
        if args.output
        else artifact_root / "sglang-preflight" / "audit.json"
    )
    write_record(record, output_path)
    print(str(output_path))
    return 0 if record["ok"] else 1


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=str(repo_root() / DEFAULT_CONFIG_PATH),
        help="AReaL YAML config containing the sglang block.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local Qwen3.5-4B-Base snapshot path. Defaults to SAO_MODEL_PATH.",
    )
    parser.add_argument(
        "--artifact-root",
        default=None,
        help="Evidence root. Defaults to SAO_ARTIFACT_ROOT.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON evidence path. Defaults under artifact-root/sglang-preflight.",
    )
    parser.add_argument(
        "--hash-model-shards",
        action="store_true",
        help="Also hash large safetensors shards. This is slower but binds C04 stronger.",
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    add_common_args(common)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    audit = subparsers.add_parser(
        "audit",
        parents=[common],
        help="Static checks only; no GPU allocation.",
    )
    audit.set_defaults(func=audit_preflight)

    run = subparsers.add_parser(
        "run",
        parents=[common],
        help="Run SGLang generation and profiler.",
    )
    run.add_argument("--tp-size", type=int, default=1)
    run.add_argument("--base-gpu-id", type=int, default=0)
    run.add_argument("--max-new-tokens", type=int, default=8)
    run.add_argument("--profile-steps", type=int, default=4)
    run.add_argument("--sglang-log-level", default="info")
    run.set_defaults(func=run_preflight)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
