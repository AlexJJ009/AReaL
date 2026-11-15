"""Run: python3 bin/codex-hooks/test_guard.py (or python3 -m pytest -q bin/codex-hooks/test_guard.py)."""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import guard  # noqa: E402

R = str(guard.ROOT)

ALLOW = [
    "echo hook-probe",
    "ssh -V",
    "uv run --frozen pytest -q tests/test_x.py 2>&1 | tail -5",
    "git commit -m 'x' && git log -1",
    "kill 12345",
    f"rm -rf {R}/_artifacts/main/old-run",
    f"rm -rf {R}/tmp/x",
    "rsync -a ./a/ ./b/",
    "curl -s http://127.0.0.1:8317/v1/models -o /dev/null",
    "nvidia-smi",
]
DENY = [
    (f"cat {R}/cpa/keys/config.yaml", "CPA stack"),
    (f"ls {R}/cpa", "CPA stack"),
    ("cpactl stop", "CPA stack"),
    ("kill $(pgrep cli-proxy-api)", "CPA processes"),
    ("cat ~/.ssh/id_ed25519", "SSH keys"),
    ("ssh astro-a100 true", "no remote hosts"),
    ("scp f host:/tmp/", "no remote hosts"),
    ("rsync -a ./x host:/data/", "no remote hosts"),
    ("git push origin HEAD", "never pushes"),
    ("gh pr list", "GitHub CLI"),
    ("codex login", "log in"),
    ("ls /root/.codex", "not ours"),
    ("pkill -f python", "kill by PID"),
    ("killall python3", "kill by PID"),
    (f"rm -rf {R}/Areal", "workspace roots"),
    (f"rm -r -f {R}/_worktrees", "workspace roots"),
    ("nohup timeout 5 git push", "never pushes"),
]


def test_allow():
    for c in ALLOW:
        assert guard.verdict(c) is None, c


def test_deny():
    for c, why in DENY:
        r = guard.verdict(c)
        assert r and why in r, (c, r)


def test_script_body_is_inspected(tmp_path):
    s = tmp_path / "x.sh"
    s.write_text("echo hi\ngit push origin main\n")
    assert "never pushes" in (guard.verdict(f"bash {s}") or "")


def test_hook_protocol_exit_codes():
    def run(cmd):
        p = subprocess.run(
            [sys.executable, str(HERE / "guard.py")],
            input=json.dumps({"tool_input": {"command": cmd}, "cwd": "."}),
            capture_output=True,
            text=True,
        )
        return p.returncode, p.stderr

    assert run("echo ok") == (0, "")
    rc, err = run("git push")
    assert rc == 2 and "never pushes" in err
    assert (
        run("{not json")[0] == 0
    )  # malformed payloads are allowed (logged), never block


if __name__ == "__main__":
    import tempfile

    test_allow()
    test_deny()
    with tempfile.TemporaryDirectory() as d:
        test_script_body_is_inspected(Path(d))
    test_hook_protocol_exit_codes()
    print(f"guard tests passed ({len(ALLOW)} allow, {len(DENY) + 1} deny)")
