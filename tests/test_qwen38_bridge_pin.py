# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("mutation", ["clean", "revision", "tracked", "untracked"])
def test_bridge_pin_rejects_runtime_drift(tmp_path, mutation):
    recipe = Path(__file__).parents[1] / "examples/swe/qwen38_flash_next"
    check = tmp_path / "check"
    check.mkdir()
    shutil.copy(recipe / "verify_bridge.sh", check)
    bridge = tmp_path / "bridge"
    bridge.mkdir()

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(bridge), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    git("init")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    (bridge / "source.py").write_text("value = 1\n")
    git("add", "source.py")
    git("commit", "-m", "test fixture")
    revision = git("rev-parse", "HEAD")
    (check / "runtime.env").write_text(f"QWEN_BRIDGE_COMMIT={revision}\n")
    if mutation == "revision":
        (check / "runtime.env").write_text("QWEN_BRIDGE_COMMIT=wrong\n")
    elif mutation == "tracked":
        (bridge / "source.py").write_text("value = 2\n")
    elif mutation == "untracked":
        (bridge / "shadow.py").write_text("value = 2\n")

    result = subprocess.run(
        ["bash", str(check / "verify_bridge.sh")],
        env={**os.environ, "MCORE_BRIDGE_ROOT": str(bridge)},
        capture_output=True,
        text=True,
    )

    assert (result.returncode == 0) == (mutation == "clean")
    if mutation != "clean":
        assert "mismatch" in result.stderr or "dirty" in result.stderr
