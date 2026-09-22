"""Exercise snapshot dependency imports with the launcher's actual module path."""

import os
import subprocess
import sys
from pathlib import Path


def test_grpo_launcher_snapshot_imports_preserve_stdlib_queue(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    launcher = repo / "scripts/sao/launch_grpo_run.py"
    # A fresh interpreter is essential: pytest may already cache stdlib queue.
    probe = """
import runpy
import sys
from pathlib import Path
launcher = Path(sys.argv[1])
sys.path.insert(0, str(launcher.parent))
sys.path.insert(0, str(launcher.parents[2]))
runpy.run_path(str(launcher), run_name="import_probe")
from queue import Queue
from datasets import load_from_disk
from scripts.sao.snapshot_eval import snapshot_eval
assert Queue.__module__ == "queue"
assert callable(load_from_disk) and callable(snapshot_eval)
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, str(launcher)],
        cwd=repo,
        env={**os.environ, "SAO_LAUNCH_DIR": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir()), "Import probe must not launch a run"
