# SPDX-License-Identifier: Apache-2.0
"""Report native metadata conflicts; recognize only the release's explicit overrides.

Override recognition is a packaging check, not proof of CUDA/runtime compatibility.
The GPU, scorer and model preflights remain independently required.
"""

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

# AReaL v2.1.0 already overrides these upstream exact pins in pyproject.toml.
# This task pins the chosen values and preserves the raw `uv pip check` output.
OVERRIDES = {
    ("torch", "nvidia-cudnn-cu12"): "9.16.0.29",
    ("sglang", "openai"): "2.30.0",
    ("sglang", "soundfile"): "0.12.1",
    ("sglang", "torchao"): "0.15.0",
}


def audit():
    distributions = {
        canonicalize_name(d.metadata["Name"]): d for d in metadata.distributions()
    }
    conflicts = []
    for consumer, distribution in sorted(distributions.items()):
        for raw in distribution.requires or []:
            req = Requirement(raw)
            if req.marker and not req.marker.evaluate({"extra": ""}):
                continue
            name = canonicalize_name(req.name)
            observed = distributions[name].version if name in distributions else None
            if observed is None or (req.specifier and observed not in req.specifier):
                conflicts.append(
                    {
                        "consumer": consumer,
                        "requirement": raw,
                        "observed": observed,
                        "release_override": observed is not None
                        and OVERRIDES.get((consumer, name)) == observed,
                    }
                )
    return {
        "metadata_clean": not conflicts,
        "conflicts": conflicts,
        "only_documented_release_overrides": all(
            c["release_override"] for c in conflicts
        ),
        "runtime_compatibility": "requires_separate_real_preflights",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit()
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    raise SystemExit(0 if result["only_documented_release_overrides"] else 1)


if __name__ == "__main__":
    main()
