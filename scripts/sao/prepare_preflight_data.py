# SPDX-License-Identifier: Apache-2.0
"""Make a clearly labelled small dataset for native 4+4 infrastructure probes."""

import argparse
import json
from pathlib import Path

from datasets import DatasetDict, load_from_disk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-rows", type=int, default=8)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    source = load_from_disk(args.source)
    selected = {}
    for i, row in enumerate(source["test"]):
        selected.setdefault(row["benchmark"], i)
    probe = DatasetDict(
        train=source["train"].select(range(args.train_rows)),
        test=source["test"].select(list(selected.values())),
    )
    assert len(probe["test"]) == 5
    probe.save_to_disk(args.output)
    (output / "preflight-manifest.json").write_text(
        json.dumps(
            {
                "preflight_only": True,
                "source": args.source,
                "train_source_ids": list(probe["train"]["source_id"]),
                "test_source_ids": list(probe["test"]["source_id"]),
                "formal_dataset_modified": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
