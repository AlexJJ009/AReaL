# SPDX-License-Identifier: Apache-2.0
"""Read back the exact authorized deletion set and its recovery archive."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    args = parser.parse_args()
    assert sha256(args.root / "receipt.json") == args.receipt_sha256
    receipt = json.loads((args.root / "receipt.json").read_text())
    plan = json.loads((args.root / "plan.json").read_text())
    assert receipt["status"] == "completed" and receipt["recoverable"] is True
    assert receipt["removed_images"] == len({t["id"] for t in plan["targets"]}) == 10
    selected_tags = {
        target[key]
        for target in plan["targets"]
        for key in ("tag", "extra_tag")
        if key in target
    }
    assert receipt["removed_tags"] == len(selected_tags) == 11
    assert receipt["reclaimed_filesystem_bytes"] > 0
    archive = args.root / receipt["archive"]
    assert archive.parent == args.root and archive.is_file()
    archive_hash = sha256(archive)
    assert archive_hash == receipt["archive_sha256"]
    tags = set(
        subprocess.check_output(
            ["docker", "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"], text=True
        ).splitlines()
    )
    assert selected_tags.isdisjoint(tags)
    subprocess.run(
        ["docker", "container", "inspect", receipt["protected_container"]],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    print(
        json.dumps(
            {
                "passed": True,
                "archive_sha256": archive_hash,
                "removed_images": receipt["removed_images"],
                "removed_tags": receipt["removed_tags"],
                "reclaimed_filesystem_bytes": receipt["reclaimed_filesystem_bytes"],
                "protected_container_retained": True,
            }
        )
    )


if __name__ == "__main__":
    main()
