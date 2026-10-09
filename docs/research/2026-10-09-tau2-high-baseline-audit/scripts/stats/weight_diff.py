# ruff: noqa: E731, E741, UP031  # archived audit script, kept as run
# CPU-only: relative L2 change of checkpoints vs base Qwen3.5-4B, parsing safetensors with numpy.
import glob
import json
import struct

import numpy as np


def load(paths):
    out = {}
    for p in paths:
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
            base = 8 + n
            mm = np.memmap(p, dtype=np.uint8, mode="r")
            for k, v in h.items():
                if k == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                out[k] = (v["dtype"], v["shape"], mm[base + a : base + b])
    return out


def f32(t):
    dt, sh, buf = t
    if dt == "BF16":
        return (buf.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    if dt == "F32":
        return buf.view(np.float32)
    if dt == "F16":
        return buf.view(np.float16).astype(np.float32)
    raise ValueError(dt)


if __name__ == "__main__":
    snap = sorted(
        glob.glob(
            "/data_storage/yl_test/lgx/data-1/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e8*/"
        )
    )[0]
    base = load(sorted(glob.glob(snap + "*.safetensors")))
    B = "/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20260927-nonthinking/"
    norm = lambda k: k.replace("model.language_model.", "model.").replace(
        "language_model.model.", "model."
    )
    bmap = {norm(k): k for k in base}
    for run, sub in [
        (
            "grpo",
            "grpo-formal-v2/checkpoints/root/tau2-grpo/tau2-grpo-nonthinking-formal-v2/default/",
        ),
        (
            "sao",
            "sao-mixed-v4-disk/checkpoints/root/tau2-sao/tau2-sao-nonthinking-mixed-v4-disk/default/",
        ),
    ]:
        for ck in ["epoch0epochstep4globalstep4", "epoch1epochstep22globalstep45"]:
            c = load([B + sub + ck + "/model.safetensors"])
            num = den = 0.0
            matched = 0
            maxrel = 0
            unchanged = 0
            tot = 0
            for k, t in c.items():
                bk = bmap.get(norm(k))
                if bk is None or base[bk][1] != t[1]:
                    continue
                x = f32(t)
                y = f32(base[bk])
                d = x - y
                num += float(np.dot(d, d))
                den += float(np.dot(y, y))
                matched += 1
                unchanged += int(np.count_nonzero(d == 0))
                tot += d.size
            print(
                run,
                ck,
                "matched tensors",
                matched,
                "/",
                len(c),
                "rel L2 change %.3e" % (num**0.5 / den**0.5),
                "frac elements bitwise unchanged %.4f" % (unchanged / tot),
                flush=True,
            )
