# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples/swe/qwen38_flash_next/patch_sglang_qsa_compress_gather.py"
)


@pytest.mark.parametrize(
    "source",
    [
        b"# Unknown SGLang release\n",
        b"            source_keys = token_k\n"
        b"            source_rope = metadata.extend_rope_matrix\n",
    ],
)
def test_patch_unknown_source_refuses_without_modifying_files(tmp_path, source):
    target = tmp_path / "qsa_indexer.py"
    target.write_bytes(source)
    audit = tmp_path / "audit.json"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--target", str(target), "--audit", str(audit)],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Unknown or already-patched" in result.stderr
    assert target.read_bytes() == source
    assert not audit.exists()
