# SPDX-License-Identifier: Apache-2.0
"""Fixed-input evidence at native PPO/AWEX rollout transition boundaries."""

import concurrent.futures
import hashlib
import http.client
import json
import math
import urllib.error
import urllib.request
from pathlib import Path


def request(url, body):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(
        url + "/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with opener.open(req, timeout=180) as response:
        return json.load(response)


def collect(rollout, fixture_path, output, send=request):
    fixture_bytes = Path(fixture_path).read_bytes()
    samples = json.loads(fixture_bytes)["samples"]
    servers = list(rollout.server_infos)
    if len(servers) != 16:
        raise ValueError("Expected all sixteen SGLang replicas")
    version = rollout.get_version()

    def runtime_state():
        stats = rollout.dispatcher.staleness_manager.get_stats()
        return {
            "dispatcher_paused": rollout.dispatcher.is_paused(),
            "running_workflows": stats.running,
            "enqueued_workflows": stats.enqueued,
        }

    before = runtime_state()
    root = Path(output) / f"version-{version}"
    root.mkdir(parents=True, exist_ok=False)

    def engine_probe(index, server):
        observations = []
        for sample in samples:
            ids = sample["input_ids"]
            for repeat in range(2):
                result = send(
                    f"http://{server.host}:{server.port}",
                    {
                        "input_ids": ids,
                        "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                        "return_logprob": True,
                        "logprob_start_len": 0,
                    },
                )
                # Preserve responses before validation, including failures.
                record = {
                    "sample_id": sample["sample_id"],
                    "repeat": repeat,
                    "input_ids": ids,
                    "response": result,
                }
                path = root / f"engine-{index}-observation-{len(observations)}.json"
                path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
                logps = result["meta_info"]["input_token_logprobs"]
                if len(logps) != len(ids):
                    raise ValueError("Incomplete fixed-input logprobs")
                if [row[1] for row in logps] != ids:
                    raise ValueError("Fixed-input logprob token identities differ")
                if not all(math.isfinite(row[0]) for row in logps[1:]):
                    raise ValueError("Nonfinite fixed-input logprobs")
                observations.append(path.name)
        return {
            "engine": index,
            "host": server.host,
            "port": server.port,
            "observations": observations,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        pending = [
            pool.submit(engine_probe, i, server) for i, server in enumerate(servers)
        ]
        engines = []
        unavailable = []
        for index, item in enumerate(pending):
            try:
                engines.append(item.result())
            except (
                TimeoutError,
                urllib.error.URLError,
                ConnectionError,
                http.client.RemoteDisconnected,
                http.client.IncompleteRead,
            ) as exc:
                # An observational request can queue behind live generation.
                # Preserve the evidence gap without terminating the RL update.
                # Token identity/nonfinite validation failures still propagate.
                unavailable.append({"engine": index, "error_type": type(exc).__name__})
    end_version = rollout.get_version()
    after = runtime_state()
    if end_version != version:
        raise ValueError("Weight version changed during fixed-input probe")
    report = {
        "status": "incomplete_probe" if unavailable else "collected_not_accepted",
        "unavailable_engines": unavailable,
        "weight_version": version,
        "end_weight_version": end_version,
        "weight_source": "initial_hf_loaded" if version == 0 else "awex_published",
        "runtime_before": before,
        "runtime_after": after,
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "cache_policy": "native cache; no manual flush or memory lifecycle changes",
        "engines": engines,
    }
    (root / "manifest.json").write_text(json.dumps(report, indent=2))
    return report


def install(trainer, fixture, output):
    """Record transport failures; retain strict validation of received probe data."""
    if (
        trainer.config.actor.weight_update_mode != "awex"
        or trainer._should_offload_rollout
    ):
        raise ValueError("Requires the accepted native AWEX memory lifecycle")
    collect(trainer.rollout, fixture, output)
    restore = trainer._restore_awex_rollout_after_stats

    def restore_and_probe():
        result = restore()
        # Native restoration is complete. Existing workflows may still be active;
        # record their count without adding lifecycle or scheduling operations.
        collect(trainer.rollout, fixture, output)
        return result

    trainer._restore_awex_rollout_after_stats = restore_and_probe
