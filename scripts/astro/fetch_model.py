#!/usr/bin/env python3
"""Download one pinned HF model revision into <dest>, verifying every file.

Usage: fetch_model.py <manifest.json> <dest dir>

The manifest pins repo, revision and every file with its size and hash: `sha256` for LFS files (HF's own oid)
and `blob_sha1` for the rest (git blob SHA1, which HF gives instead). A file that already matches is skipped,
so a killed run resumes. Download goes to <name>.part and is renamed only after the hash matches, so a
truncated file never looks complete.

When behind a proxy, files above PARALLEL_MIN are fetched as CHUNK-sized ranges over WORKERS connections,
written at their offsets. Set FETCH_PROXY to the proxy URL (empty or unset = direct).

Stops before writing if the filesystem has less than MIN_FREE_GB left.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

PROXY = os.environ.get("FETCH_PROXY", "")
MIN_FREE_GB = 40
WORKERS = int(os.environ.get("FETCH_WORKERS", "6"))
CHUNK = 32 << 20
ATTEMPTS = 5
PARALLEL_MIN = 256 << 20
if PROXY:
    OPENER = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})
    )
else:
    OPENER = urllib.request.build_opener()


def free_gb(path: Path) -> float:
    s = os.statvfs(path)
    return s.f_bavail * s.f_frsize / 2**30


def digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def blob_sha1(path: Path) -> str:
    h = hashlib.sha1()
    h.update(b"blob %d\0" % path.stat().st_size)  # noqa: UP031 - git hashes a byte-string header
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def matches(path: Path, entry: dict) -> bool:
    if not path.exists() or path.stat().st_size != entry["size"]:
        return False
    if "sha256" in entry:
        return digest(path, "sha256") == entry["sha256"]
    return blob_sha1(path) == entry["blob_sha1"]


def get_stream(url: str, part: Path, size: int, dest: Path) -> None:
    with OPENER.open(url, timeout=120) as resp, part.open("wb") as fp:
        done = 0
        next_report = 1 << 30
        while True:
            chunk = resp.read(8 << 20)
            if not chunk:
                break
            fp.write(chunk)
            done += len(chunk)
            if done >= next_report:
                next_report += 1 << 30
                if free_gb(dest) < MIN_FREE_GB:
                    raise SystemExit(f"STOP: free space below {MIN_FREE_GB} GB")
                print(f"  {done / 2**30:.1f}/{size / 2**30:.1f} GB", flush=True)


def get_ranges(url: str, part: Path, size: int, dest: Path) -> None:
    """Fetch [0, size) as CHUNK-sized ranges over WORKERS connections, each written at its offset.

    A mirror behind the proxy regularly cuts a range response short, so a chunk is re-requested from where it
    stopped, up to ATTEMPTS times. Finished chunk indexes are appended to <part>.done, so a killed run resumes
    instead of re-downloading gigabytes; the sidecar is removed once the file passes its hash.
    """
    if free_gb(dest) < MIN_FREE_GB:
        raise SystemExit(f"STOP: free space below {MIN_FREE_GB} GB")
    ranges = [(start, min(start + CHUNK, size) - 1) for start in range(0, size, CHUNK)]
    ledger = part.with_suffix(part.suffix + ".done")
    have = set()
    if part.exists() and part.stat().st_size == size and ledger.exists():
        have = {int(line) for line in ledger.read_text().split() if line.strip()}
    else:
        ledger.unlink(missing_ok=True)
    fd = os.open(part, os.O_WRONLY | os.O_CREAT)
    lock = Lock()
    try:
        os.ftruncate(fd, size)
        todo = [(i, span) for i, span in enumerate(ranges) if i not in have]
        print(f"  {len(todo)} of {len(ranges)} chunks to fetch", flush=True)
        count = [len(have)]

        def fetch(item):
            index, (start, end) = item
            offset = start
            for attempt in range(ATTEMPTS):
                req = urllib.request.Request(
                    url, headers={"Range": f"bytes={offset}-{end}"}
                )
                try:
                    with OPENER.open(req, timeout=180) as resp:
                        if resp.status != 206:
                            raise OSError(
                                f"range request returned {resp.status}, not 206"
                            )
                        while True:
                            buf = resp.read(8 << 20)
                            if not buf:
                                break
                            os.pwrite(fd, buf, offset)
                            offset += len(buf)
                except OSError as exc:
                    if attempt == ATTEMPTS - 1:
                        raise
                    print(
                        f"  chunk {index}: {exc}; retrying from {offset}",
                        flush=True,
                    )
                    time.sleep(2 * (attempt + 1))
                    continue
                if offset > end:
                    break
                if attempt == ATTEMPTS - 1:
                    raise OSError(f"short range {start}-{end}: stopped at {offset}")
                print(f"  chunk {index} short at {offset}; retrying", flush=True)
                time.sleep(2 * (attempt + 1))
            with lock:
                with ledger.open("a") as fp:
                    fp.write(f"{index}\n")
                count[0] += 1
                print(
                    f"  chunk {count[0]}/{len(ranges)} "
                    f"({count[0] * CHUNK / 2**30:.1f}/{size / 2**30:.1f} GB)",
                    flush=True,
                )

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for _ in pool.map(fetch, todo):
                pass
    finally:
        os.close(fd)


def main() -> int:
    manifest = json.loads(Path(sys.argv[1]).read_text())
    dest = Path(sys.argv[2])
    dest.mkdir(parents=True, exist_ok=True)
    # Any mirror may serve the bytes: every file is checked against the hash the manifest pins from HF, so a
    # mirror that differs by one byte fails instead of being trusted. ModelScope is ~25x faster than HF through
    # the platform proxy, which is why the template is an env override.
    template = os.environ.get(
        "FETCH_URL_TEMPLATE", "https://huggingface.co/{repo}/resolve/{revision}/{path}"
    )

    for entry in manifest["files"]:
        out = dest / entry["path"]
        if matches(out, entry):
            print("have", entry["path"], flush=True)
            continue
        if free_gb(dest) < MIN_FREE_GB:
            print(f"STOP: free space below {MIN_FREE_GB} GB", flush=True)
            return 1
        part = out.with_suffix(out.suffix + ".part")
        part.parent.mkdir(parents=True, exist_ok=True)
        url = template.format(path=entry["path"], **manifest)
        size = entry["size"]
        print(
            "get",
            entry["path"],
            size,
            "parallel" if size >= PARALLEL_MIN else "stream",
            flush=True,
        )
        if size >= PARALLEL_MIN:
            get_ranges(url, part, size, dest)
        else:
            get_stream(url, part, size, dest)
        part.with_suffix(part.suffix + ".done").unlink(missing_ok=True)
        part.replace(out)
        if not matches(out, entry):
            print("FAIL hash mismatch", entry["path"], flush=True)
            return 1
        print("ok", entry["path"], flush=True)

    print("ALL OK", len(manifest["files"]), "files", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
