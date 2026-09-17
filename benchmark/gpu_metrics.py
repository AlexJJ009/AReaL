# SPDX-License-Identifier: Apache-2.0

"""Read existing DCGM exporter metrics without opening a new profiling session."""

from __future__ import annotations

import argparse
import json
import math
import re
import signal
import threading
import time
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

METRICS = {
    "DCGM_FI_PROF_SM_ACTIVE": ("ratio", 1),
    "DCGM_FI_PROF_SM_OCCUPANCY": ("ratio", 1),
    "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE": ("ratio", 1),
    "DCGM_FI_PROF_DRAM_ACTIVE": ("ratio", 1),
    "DCGM_FI_DEV_GPU_UTIL": ("percent", 100),
    "DCGM_FI_DEV_MEM_COPY_UTIL": ("percent", 100),
    "DCGM_FI_DEV_FB_USED": ("MiB", None),
    "DCGM_FI_DEV_POWER_USAGE": ("W", None),
    "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION": ("mJ", None),
    "DCGM_FI_DEV_SM_CLOCK": ("MHz", None),
    "DCGM_FI_DEV_MEM_CLOCK": ("MHz", None),
    "DCGM_FI_PROF_PCIE_TX_BYTES": ("bytes_per_second", None),
    "DCGM_FI_PROF_PCIE_RX_BYTES": ("bytes_per_second", None),
}
LINE = re.compile(r"^(\w+)(?:\{(.*)\})?\s+([^\s]+)(?:\s+\S+)?$")
LABEL = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')


def parse_metrics(text: str) -> list[dict]:
    points = []
    for line in text.splitlines():
        match = LINE.match(line)
        if not match or match[1] not in METRICS:
            continue
        labels = {
            key: json.loads('"' + value + '"')
            for key, value in LABEL.findall(match[2] or "")
        }
        if "gpu" not in labels:
            continue
        value = float(match[3])
        unit, maximum = METRICS[match[1]]
        valid = (
            math.isfinite(value)
            and 0 <= value < 1e15
            and (maximum is None or value <= maximum)
        )
        points.append(
            {
                "metric": match[1],
                "gpu": labels["gpu"],
                "uuid": labels.get("UUID"),
                "unit": unit,
                "value": value if valid else None,
                "valid": valid,
                "raw_value": match[3],
            }
        )
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exporter-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    stop = threading.Event()
    for signum in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(signum, lambda *_: stop.set())
    opener = build_opener(ProxyHandler({}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as stream:
        while not stop.is_set():
            start = time.monotonic()
            row = {"scrape_wall_time": time.time(), "scrape_monotonic": start}
            try:
                with opener.open(args.exporter_url, timeout=5) as response:
                    payload = response.read().decode()
                row["points"] = parse_metrics(payload)
                row["source_timestamp_available"] = False
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            stop.wait(max(0, args.interval - (time.monotonic() - start)))


if __name__ == "__main__":
    main()
