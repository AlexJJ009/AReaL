import gzip
import json
from pathlib import Path

from scripts.sao import sglang_preflight


def test_layer_counts_binds_qwen35_hybrid_layout():
    config = {
        "text_config": {
            "layer_types": ["linear_attention"] * 24 + ["full_attention"] * 8
        }
    }

    assert sglang_preflight.layer_counts(config) == {
        "linear_attention": 24,
        "full_attention": 8,
    }


def test_intended_flags_rejects_slow_or_unbounded_paths():
    check = sglang_preflight.check_intended_flags(
        {
            "attention_backend": "flashinfer",
            "dtype": "bfloat16",
            "disable_cuda_graph": False,
            "disable_overlap_schedule": False,
            "disable_radix_cache": True,
            "chunked_prefill_size": -1,
            "skip_tokenizer_init": True,
        }
    )
    assert check.ok

    bad = sglang_preflight.check_intended_flags(
        {
            "attention_backend": "fa3",
            "dtype": "bfloat16",
            "disable_cuda_graph": True,
            "disable_overlap_schedule": False,
            "disable_radix_cache": False,
            "chunked_prefill_size": 8192,
            "skip_tokenizer_init": False,
        }
    )
    assert not bad.ok
    assert bad.details["observed"]["attention_backend"] == "fa3"


def test_logprob_response_requires_token_ids_and_logprobs():
    parsed = sglang_preflight.require_logprob_response(
        {
            "text": " 5",
            "output_ids": [220, 20],
            "meta_info": {
                "prompt_tokens": 7,
                "completion_tokens": 2,
                "input_token_logprobs": [[None, 1, "Question"], [-0.1, 2, ":"]],
                "output_token_logprobs": [[-0.2, 220, " "], [-0.3, 20, "5"]],
                "output_top_logprobs": [[[-0.2, 220, " "]], [[-0.3, 20, "5"]]],
                "finish_reason": {"type": "length"},
                "weight_version": None,
            },
        }
    )

    assert parsed["output_ids"] == [220, 20]
    assert parsed["output_token_logprobs"][1][1] == 20


def test_profile_scan_fails_when_mandatory_paths_are_missing(tmp_path: Path):
    trace = {
        "traceEvents": [
            {"name": "flashinfer attention decode", "cat": "gpu", "dur": 10},
            {"name": "cudaMemcpyAsync", "cat": "cuda", "dur": 3},
        ]
    }
    (tmp_path / "chunk_gated_delta_event_loop_overlap_filename.trace.json").write_text(
        json.dumps(trace), encoding="utf-8"
    )

    result = sglang_preflight.scan_profile_dir(tmp_path)

    assert not result["ok"]
    assert "gdn_or_linear_attention" in result["missing_mandatory_paths"]
    assert "overlap" in result["missing_mandatory_paths"]
    assert "cuda_graph_runtime_event" in result["missing_mandatory_paths"]


def test_profile_scan_passes_only_with_all_required_path_markers(tmp_path: Path):
    trace = {
        "traceEvents": [
            {
                "name": "flashinfer::BatchDecodeWithPagedKVCacheKernel",
                "cat": "kernel",
                "dur": 11,
            },
            {"name": "chunk_gated_delta conv1d prefill", "cat": "gpu", "dur": 12},
            {"name": "cudaGraphLaunch", "cat": "cuda", "dur": 2},
            {"name": "event_loop_overlap schedule", "cat": "cpu", "dur": 1},
        ]
    }
    with gzip.open(tmp_path / "rank0.trace.json.gz", "wt", encoding="utf-8") as fout:
        json.dump(trace, fout)

    result = sglang_preflight.scan_profile_dir(tmp_path)

    assert result["ok"]
    assert result["missing_mandatory_paths"] == []
    assert result["kernel_events_sample"]
    assert result["cuda_graph_events_sample"][0]["name"] == "cudaGraphLaunch"


def test_generation_request_uses_input_ids_not_prompt_text():
    encoded = {
        "input_ids": [[1, 2, 3], [4, 5]],
        "items": [],
        "tokenizer_class": "FakeTokenizer",
    }

    request = sglang_preflight.generation_request(encoded, max_new_tokens=7)

    assert request["input_ids"] == [[1, 2, 3], [4, 5]]
    assert "prompt" not in request
    assert "prompts" not in request
    assert request["return_logprob"] is True
    assert request["logprob_start_len"] == 0
