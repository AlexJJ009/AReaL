# SPDX-License-Identifier: Apache-2.0
"""Score a saved actor export in fresh SGLang after native workers have exited."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.sao.sglang_preflight import build_areal_server_args, import_sglang_engine

from areal.engine.sglang_remote import SGLangBackend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    publication = json.loads(args.publication.read_text())
    server = build_areal_server_args(
        model_path=args.checkpoint.resolve(),
        raw_sglang=config["sglang"],
        tp_size=1,
        base_gpu_id=0,
    )
    kwargs = {
        key: value
        for key, value in server.items()
        if value is not None
        and key not in {"host", "port", "dist_init_addr", "nnodes", "node_rank"}
    }
    from sglang.srt.server_args import ServerArgs

    for key in set(kwargs) - set(ServerArgs.__dataclass_fields__):
        if kwargs[key] is not False:
            raise ValueError(f"Unsupported active SGLang setting: {key}")
        del kwargs[key]
    kwargs.update(skip_tokenizer_init=True, log_level="warning")
    engine = import_sglang_engine()(**kwargs)
    backend = SGLangBackend()
    try:
        target_len = len(publication["scored_token_ids"])
        payload = backend.build_score_request(
            publication["token_ids"], target_len, False, publication["version"]
        ).payload
        responses = [engine.generate(**payload) for _ in range(3)]
        scores = [
            backend.parse_score_response(response, target_len) for response in responses
        ]
        reference = publication["replicas"][0]["rollout"]
        actor = publication["replicas"][0]["actor"]
        live_error = max(
            abs(a - b) for row in scores for a, b in zip(row, reference, strict=True)
        )
        actor_error = max(
            abs(a - b) for row in scores for a, b in zip(row, actor, strict=True)
        )
        result = {
            "checkpoint": str(args.checkpoint.resolve()),
            "version": publication["version"],
            "token_ids": publication["token_ids"],
            "fresh_sglang_logprobs": scores,
            "live_sglang_logprobs": reference,
            "actor_logprobs": actor,
            "max_fresh_vs_live_sglang": live_error,
            "max_fresh_vs_actor": actor_error,
            "raw_responses": responses,
            "scope": "diagnostic measurement only; does not raise or change the existing cross-engine tolerance",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {key: value for key, value in result.items() if key != "raw_responses"}
            )
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
