# SPDX-License-Identifier: Apache-2.0
"""Replay exact initial-weight failures on a dedicated, idle SGLang endpoint.

The caller must separately verify the server's initial HF weight provenance.
Do not use the live RL endpoint: it has already received optimizer updates.
"""

import argparse
import json
import math
import urllib.request
from pathlib import Path


def compare(fixture, response):
    rows = response["meta_info"]["input_token_logprobs"]
    ids = fixture["input_ids"]
    if len(rows) != len(ids) or [row[1] for row in rows] != ids:
        raise ValueError("Incomplete or misaligned prefill token identities")
    if not all(math.isfinite(row[0]) for row in rows[1:]):
        raise ValueError("Nonfinite prefill logprobs")
    result = []
    for target in fixture["targets"]:
        position = target["absolute_token_position"]
        if ids[position] != target["token_id"]:
            raise ValueError("Fixture target token mismatch")
        value = rows[position][0]
        result.append(
            dict(
                target,
                replay_prefill_logp=value,
                diff_from_decode=abs(value - target["original_decode_logp"]),
                diff_from_megatron=abs(value - target["megatron_prefill_logp"]),
            )
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--initial-weight-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provenance = json.loads(args.initial_weight_evidence.read_text())
    if (
        provenance.get("weight_version") != 0
        or provenance.get("weight_source") != "initial_hf_loaded"
        or provenance.get("endpoint") != args.endpoint
        or provenance.get("dedicated_idle") is not True
    ):
        raise ValueError("Requires dedicated idle initial-HF endpoint evidence")
    args.output.mkdir(parents=True, exist_ok=False)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, body=None, as_json=True):
        req = urllib.request.Request(
            args.endpoint.rstrip("/") + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with opener.open(req, timeout=600) as response:
            return json.load(response) if as_json else response.read().decode()

    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    observations = []
    for fixture in json.loads(args.fixtures.read_text()):
        for repeat in range(2):
            # Dedicated endpoint only; isolate each repeat from prefix reuse.
            flushed = request("/flush_cache", as_json=False)
            (
                args.output / ("{}-{}-flush.txt".format(fixture["audit_id"], repeat))
            ).write_text(flushed)
            response = request("/generate", fixture["sglang_prefill_request"])
            name = "{}-{}.json".format(fixture["audit_id"], repeat)
            (args.output / name).write_text(json.dumps(response))
            observations.append(
                {
                    "audit_id": fixture["audit_id"],
                    "repeat": repeat,
                    "targets": compare(fixture, response),
                }
            )
    (args.output / "comparison.json").write_text(json.dumps(observations, indent=2))


if __name__ == "__main__":
    main()
