# SPDX-License-Identifier: Apache-2.0
"""Small fixed-input BF16 forward/backward checks before full-weight probes."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from flash_attn import flash_attn_func
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule


def compare(name, inputs, fast, reference):
    left = [x.clone().requires_grad_(True) for x in inputs]
    right = [x.clone().requires_grad_(True) for x in inputs]
    actual, expected = fast(*left), reference(*right)
    probe = torch.randn_like(actual)
    actual_grad = torch.autograd.grad((actual.float() * probe.float()).sum(), left)
    expected_grad = torch.autograd.grad((expected.float() * probe.float()).sum(), right)
    result = {
        "name": name,
        "output_tolerance": {"atol": 0.02, "rtol": 0.03},
        "gradient_tolerance": {"atol": 0.03, "rtol": 0.05},
        "comparisons": [],
    }
    for label, a, b in [("output", actual, expected)] + [
        (f"gradient_{i}", a, b)
        for i, (a, b) in enumerate(zip(actual_grad, expected_grad, strict=True))
    ]:
        difference = a.float() - b.float()
        result["comparisons"].append(
            {
                "tensor": label,
                "max_abs": difference.abs().max().item(),
                "rms_error": difference.square().mean().sqrt().item(),
                "reference_rms": b.float().square().mean().sqrt().item(),
                "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            }
        )
        tolerance = result[
            "output_tolerance" if label == "output" else "gradient_tolerance"
        ]
        torch.testing.assert_close(a.float(), b.float(), **tolerance)
    result["passed"] = True
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False

    def random(shape, dtype=torch.bfloat16):
        return torch.randn(shape, dtype=dtype, device="cuda")

    results = []
    qkv = [random((2, 128, 4, 64)) for _ in range(3)]
    results.append(
        compare(
            "fa2_vs_sdpa_fp32",
            qkv,
            lambda q, k, v: flash_attn_func(q, k, v, causal=True),
            lambda q, k, v: F.scaled_dot_product_attention(
                q.transpose(1, 2).float(),
                k.transpose(1, 2).float(),
                v.transpose(1, 2).float(),
                is_causal=True,
            ).transpose(1, 2),
        )
    )
    conv = [random((2, 256, 128)), random((256, 4)), random((256,))]
    results.append(
        compare(
            "causal_conv1d_vs_torch_fp32",
            conv,
            lambda x, w, b: causal_conv1d_fn(x, w, b, activation="silu"),
            lambda x, w, b: F.silu(
                F.conv1d(
                    x.float(), w.float().unsqueeze(1), b.float(), padding=3, groups=256
                )[..., : x.shape[-1]]
            ),
        )
    )
    delta = qkv + [
        -random((2, 128, 4), torch.float32).abs() * 0.1,
        random((2, 128, 4)).sigmoid(),
    ]
    results.append(
        compare(
            "fla_chunk_vs_transformers_reference",
            delta,
            lambda q, k, v, g, beta: chunk_gated_delta_rule(
                q, k, v, g, beta, use_qk_l2norm_in_kernel=True
            )[0],
            lambda q, k, v, g, beta: torch_chunk_gated_delta_rule(
                q, k, v, g, beta, use_qk_l2norm_in_kernel=True
            )[0],
        )
    )
    record = {
        "passed": True,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "checks": results,
    }
    Path(args.output).write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
