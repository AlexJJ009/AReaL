# SPDX-License-Identifier: Apache-2.0
"""Compare SGLang selected-token logprobs with the native FSDP actor at Base."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from run_sao_fsdp_preflight import _make_engine, _setup_distributed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text())
    prompts = fixture["tokenizer_fixture"]["input_ids"]
    responses = fixture["generation"]["responses"]
    _setup_distributed()
    engine = _make_engine(
        SimpleNamespace(
            role="actor",
            seq_len=9216,
            model_path=os.environ["SAO_MODEL_PATH"],
            lr=1e-6,
            logprobs_chunk_size=1024,
        )
    )
    engine.eval()
    rows = []
    sequences = []
    independent = []
    try:
        for prompt, response in zip(prompts, responses, strict=True):
            sequence = prompt + response["output_ids"]
            sequences.append(torch.tensor(sequence))
            data = {
                "input_ids": torch.tensor([sequence], dtype=torch.int64),
                "attention_mask": torch.ones((1, len(sequence)), dtype=torch.bool),
            }
            result = engine.forward_batch(data).reshape(-1)
            independent.append(result[: len(sequence) - 1].float().cpu())
            actual = result[len(prompt) - 1 : len(sequence) - 1].float().cpu()
            expected = torch.tensor([v[0] for v in response["output_token_logprobs"]])
            difference = (actual - expected).abs()
            row = {
                "prompt_tokens": len(prompt),
                "output_tokens": len(expected),
                "max_abs_error": difference.max().item(),
                "mean_abs_error": difference.mean().item(),
                "fsdp_logprobs": actual.tolist(),
                "sglang_logprobs": expected.tolist(),
            }
            rows.append(row)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0.15)
            assert difference.mean() <= 0.03
        batch_ids = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        masks = (
            torch.arange(batch_ids.shape[1])[None, :]
            < torch.tensor([len(s) for s in sequences])[:, None]
        )
        together = engine.forward_batch(
            {"input_ids": batch_ids, "attention_mask": masks}
        )
        for i, reference in enumerate(independent):
            torch.testing.assert_close(
                together[i, : len(reference)].float().cpu(),
                reference,
                rtol=0,
                atol=1e-6,
            )
        if dist.get_rank() == 0:
            args.output.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "batched_sequence_isolation": True,
                        "world_size": dist.get_world_size(),
                        "model_path": os.environ["SAO_MODEL_PATH"],
                        "fixture_sha256": hashlib.sha256(
                            args.fixture.read_bytes()
                        ).hexdigest(),
                        "tolerance": {
                            "max_absolute_logprob": 0.15,
                            "mean_absolute_logprob": 0.03,
                        },
                        "comparisons": rows,
                    },
                    indent=2,
                )
                + "\n"
            )
    finally:
        engine.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
