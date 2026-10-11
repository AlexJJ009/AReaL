"""Fetch Qwen3-4B 2507 snapshots: LFS weights from ModelScope, small files from HF.

Every LFS file must match the sha256 recorded on Hugging Face at the pinned revision.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path("/data_storage/yl_test/lgx/data-1/.cache/models")
KEEP = (".json", ".safetensors", ".txt", ".jinja")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


for repo in sys.argv[1:]:
    info = HfApi().model_info(repo, files_metadata=True)
    out = ROOT / f"{repo.split('/')[1]}@{info.sha}"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"repo": repo, "revision": info.sha, "files": {}}
    for s in info.siblings:
        name = s.rfilename
        if not name.endswith(KEEP):
            continue
        target = out / name
        if s.lfs:
            url = f"https://www.modelscope.cn/models/{repo}/resolve/master/{name}"
            want = s.lfs.sha256
        else:
            url = f"https://huggingface.co/{repo}/resolve/{info.sha}/{name}"
            want = None
        for attempt in range(10):
            if not (target.exists() and want and sha256(target) == want):
                subprocess.run(
                    [
                        "curl",
                        "-sSfL",
                        "--retry",
                        "5",
                        "-C",
                        "-",
                        "-o",
                        str(target),
                        url,
                    ],
                    check=False,
                )
            got = sha256(target) if target.exists() else None
            if got and (want is None or got == want):
                break
            print(f"sha mismatch {name} attempt {attempt}: {got} != {want}", flush=True)
            target.unlink(missing_ok=True)
        else:
            raise SystemExit(f"failed {repo}/{name}")
        manifest["files"][name] = {"sha256": got, "hf_lfs_sha256": want}
        print(repo, name, "ok", flush=True)
    (out / "fetch_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("DONE", out, flush=True)
