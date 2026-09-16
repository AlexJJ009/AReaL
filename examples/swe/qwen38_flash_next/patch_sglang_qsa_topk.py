# SPDX-License-Identifier: Apache-2.0
"""Patch only the disposable canary filesystem: QSA top-k ordering/recording."""

import argparse
import ast
import hashlib
import importlib.util
import inspect
from pathlib import Path

EXPECTED_SHA256 = "5482e38d30bfaf1624ec0625b4896cbb395a1637f75c183c8ca723c9f6055ff8"
TARGET_RELATIVE = Path("srt/layers/attention/qsa/kernel.py")


def stable_qsa_topk(logits, row_starts, row_ends, topk):
    """Reference selection: score descending, lower relative ID wins exact ties."""
    import torch

    if (
        logits.ndim != 2
        or logits.dtype != torch.float32
        or type(topk) is not int
        or topk <= 0
    ):
        raise ValueError("Requires FP32 [rows,keys] scores and positive integer topk")
    rows, keys = logits.shape
    if row_starts.shape != (rows,) or row_ends.shape != (rows,):
        raise ValueError("Row bounds must match score rows")
    if row_starts.dtype not in (torch.int32, torch.int64) or row_ends.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Row bounds must be integer tensors")
    starts = row_starts.to(device=logits.device, dtype=torch.long)
    ends = row_ends.to(device=logits.device, dtype=torch.long)
    if not ((starts >= 0) & (starts <= ends) & (ends <= keys)).all():
        raise ValueError("Invalid row bounds")
    positions = torch.arange(keys, device=logits.device)[None, :]
    valid = (positions >= starts[:, None]) & (positions < ends[:, None])
    if not torch.isfinite(logits[valid]).all():
        raise ValueError("Nonfinite valid scores")
    # Invalid positions cannot beat any finite candidate. Stable sort retains
    # ascending absolute (and hence relative) IDs when scores are exactly equal.
    scores = logits.masked_fill(~valid, -float("inf"))
    width = min(topk, keys)
    chosen = scores.argsort(dim=-1, descending=True, stable=True)[:, :width]
    chosen_valid = valid.gather(1, chosen)
    relative = chosen - starts[:, None]
    sentinel = torch.iinfo(torch.int32).max
    relative = relative.masked_fill(~chosen_valid, sentinel).sort(dim=-1).values
    result = torch.full((rows, topk), -1, dtype=torch.int32, device=logits.device)
    result[:, :width] = torch.where(relative == sentinel, -1, relative).to(torch.int32)
    return result


WRAPPER = """
# Diagnostic-only: canonical mode preserves native sets; stable mode resolves ties.
import inspect as _qsa_inspect
import json as _qsa_json
import os as _qsa_os
import uuid as _qsa_uuid
from pathlib import Path as _QsaPath
_qsa_original_fast_topk = qsa_fast_topk
_qsa_long_calls = 0

def qsa_fast_topk(logits, row_starts, row_ends, topk):
    global _qsa_long_calls
    raw = _qsa_original_fast_topk(logits, row_starts, row_ends, topk)
    canonical = _qsa_os.environ.get("QWEN_QSA_CANONICAL_TOPK") == "1"
    stable = _qsa_os.environ.get("QWEN_QSA_STABLE_TOPK") == "1"
    returned = raw
    if stable:
        returned = stable_qsa_topk(logits, row_starts, row_ends, topk)
    elif canonical:
        # Sorting the slots, rather than replacing IDs, preserves the exact set.
        key = torch.where(raw < 0, torch.iinfo(raw.dtype).max, raw)
        returned = raw.gather(-1, key.argsort(dim=-1, stable=True))
    directory = _qsa_os.environ.get("QWEN_QSA_TOPK_DUMP_DIR")
    if directory:
        from sglang.srt.distributed import get_tensor_model_parallel_rank
        if get_tensor_model_parallel_rank() == 0:
            frame = _qsa_inspect.currentframe().f_back
            try:
                owner = frame.f_locals.get("self")
                if frame.f_code.co_name == "select_prefill_tokens" and frame.f_locals["q"].shape[0] >= 2177:
                    layer = owner.layer_id
                    # This bounded test requires a single chunk per prefill.
                    rows = frame.f_locals["q"].shape[0]
                    if logits.shape[0] != rows:
                        raise RuntimeError("QSA recording requires one unchunked prefill")
                    expected_layer = 3 + 4 * (_qsa_long_calls % 12)
                    if layer != expected_layer:
                        raise RuntimeError(f"Expected QSA layer {expected_layer}, got {layer}")
                    _qsa_long_calls += 1
                    out = _QsaPath(directory)
                    out.mkdir(parents=True, exist_ok=True)
                    all_layers = _qsa_os.environ.get("QWEN_QSA_DUMP_ALL_LAYERS") == "1"
                    meta = {"layer_id": layer, "long_prefill_call_index": _qsa_long_calls - 1,
                            "forward_index": (_qsa_long_calls - 1) // 12, "rows": rows,
                            "canonical_order": canonical or stable, "stable_selection": stable, "pid": _qsa_os.getpid(),
                            "all_layers": all_layers,
                            "expected_qsa_calls_per_forward": 12}
                    if layer == 3 or all_layers:
                        torch.save(dict(meta, logits=logits.detach().cpu(),
                                        row_starts=row_starts.detach().cpu(), row_ends=row_ends.detach().cpu(),
                                        raw_chosen=raw.detach().cpu(), returned_chosen=returned.detach().cpu()),
                                   out / f"topk-{_qsa_os.getpid()}-{_qsa_long_calls:04d}-{_qsa_uuid.uuid4().hex}.pt")
                    if layer == 47:
                        # A separate completion record lets analysis reject partial forwards.
                        (out / f"forward-{_qsa_os.getpid()}-{meta['forward_index']:04d}.json").write_text(_qsa_json.dumps(dict(meta, completed_qsa_calls=12)))
            finally:
                del frame
    return returned
"""


def patched_source(source: str) -> str:
    if hashlib.sha256(source.encode()).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Unexpected QSA kernel source SHA256; refusing patch")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "qsa_fast_topk"
    )
    if [arg.arg for arg in function.args.args] != [
        "logits",
        "row_starts",
        "row_ends",
        "topk",
    ]:
        raise ValueError("Unexpected qsa_fast_topk signature")
    result = source + "\n" + inspect.getsource(stable_qsa_topk) + WRAPPER
    compile(result, str(TARGET_RELATIVE), "exec")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", type=Path, help="Override the installed QSA kernel path"
    )
    args = parser.parse_args()
    target = args.target
    if target is None:
        spec = importlib.util.find_spec("sglang")
        if spec is None or spec.origin is None:
            raise RuntimeError("Cannot locate the installed SGLang package")
        target = Path(spec.origin).parent / TARGET_RELATIVE
    original = target.read_text()
    target.write_text(patched_source(original))
    print(f"QSA wrapper installed; original SHA256={EXPECTED_SHA256}", flush=True)


if __name__ == "__main__":
    main()
