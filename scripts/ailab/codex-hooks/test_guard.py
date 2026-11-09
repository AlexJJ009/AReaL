"""Run: python3 -m pytest -q bin/codex-hooks/test_guard.py (or python3 bin/codex-hooks/test_guard.py)."""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import guard  # noqa: E402

WS = "science-agent.zhangzhixia-p+root.ailab-llmagent.ws@h.pjlab.org.cn"
POD = "lgx-exp-1-abcde.zhangzhixia-p+root.ailab-llmagent.pod@h.pjlab.org.cn"
KEYS = (
    "-o BatchMode=yes -o UserKnownHostsFile=/x/known -i /root/.ssh/phai-service-ed25519"
)

ALLOW = [
    "echo hook-probe",
    "ssh -V",
    f"ssh {KEYS} {WS} 'source /etc/profile.d/ssh-init.sh; rjob get lgx-exp-1 --namespace ailab-llmagent'",
    f"ssh -n {KEYS} {POD} 'docker info'",
    f"setsid ssh -N {KEYS} -o ExitOnForwardFailure=yes -R 127.0.0.1:24647:172.31.23.186:4647 {POD} &",
    f"ssh {KEYS} -R 24647:127.0.0.1:14647 {POD} true",
    f"rsync -a -e 'ssh {KEYS}' ./pieces/ {WS}:/mnt/shared-storage-gpfs2/x/",
    f"ssh {WS} 'rjob delete lgx-exp-20261105-1 --namespace ailab-llmagent'",
    'ssh $WS -n "rjob delete $JOB --namespace ailab-llmagent"',
    "docker rm -f lgx-exp-buildkit && docker rmi lgx-exp-image:1",
    "systemctl status phai-api",
    "uv run --frozen pytest -q tests/control 2>&1 | tail -5",
    "git commit -m 'x' && git log -1",
    "kill 12345",
    "# never touch /root/.codex\nsource lib.sh",
    'wssh "$INIT; rjob delete $JOB --namespace ailab-llmagent" > "$A/rjob-delete.txt" 2>&1',
]
DENY = [
    ("ssh root@172.31.23.186 hostname", "ailab gateway"),
    ("ssh -o BatchMode=yes phai-cpu1 'ls'", "ailab gateway"),
    ("ssh localhost true", "ailab gateway"),
    ("scp file root@172.31.23.187:/tmp/", "ailab gateway"),
    ("rsync -a x/ cpu2:/data/", "ailab gateway"),
    (f"ssh -R 1080 {POD}", "Nomad ports"),
    (f"ssh -R 127.0.0.1:3128:127.0.0.1:3128 {POD}", "Nomad ports"),
    (f"ssh -o RemoteForward=8080 {POD}", "RemoteForward"),
    (
        f"ssh {WS} 'rjob delete msttrl-smoke-20261104 --namespace ailab-llmagent'",
        "lgx-*",
    ),
    ("rjob stop --namespace ailab-llmagent someone-else", "lgx-*"),
    ("rjob delete --all", "shared"),
    ("rjob delete other-job > out.txt", "lgx-*"),
    ("cat /root/.codex/auth.json # read", "someone else"),
    ("rjob submit --name lgx-x --gpu 8 -- sleep 1", "grant"),
    (
        "rjob submit --name lgx-exp-1 --cpu 8 --gpu 0 -- bash -c 'sleep infinity'",
        "grant",
    ),
    ("systemctl restart phai-api", "production services"),
    ("sudo systemctl stop nomad.service", "production services"),
    ("docker system prune -af", "prune"),
    ("docker rm -f 3f2a1b", "lgx-exp-"),
    ("docker rmi lgx-cache:abc", "lgx-exp-"),
    ("git push origin main", "never pushes"),
    ("gh pr create", "GitHub CLI"),
    ("codex login", "log in"),
    ("cat /root/.codex/auth.json", "someone else"),
    ("pkill -f nomad", "pkill -f"),
]


def run(command, cwd="/"):
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": command},
        "cwd": cwd,
    }
    return subprocess.run(
        [sys.executable, str(HERE / "guard.py")],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
    )


def test_allowed_commands_pass():
    for command in ALLOW:
        assert (
            guard.decide({"tool_name": "Bash", "tool_input": {"command": command}})
            is None
        ), command


def test_denied_commands_name_the_rule():
    for command, words in DENY:
        reason = guard.decide({"tool_name": "Bash", "tool_input": {"command": command}})
        assert reason and words in reason, (command, reason)


def test_scripts_run_by_the_command_are_checked(tmp_path):
    (tmp_path / "ok.sh").write_text(f"ssh {KEYS} {WS} true\n")
    (tmp_path / "bad.sh").write_text(
        "#!/bin/bash\nfor h in a; do\n  ssh root@172.31.23.186 uptime\ndone\n"
    )
    for command in (
        "bash bad.sh",
        "./bad.sh",
        "bash -s < bad.sh",
        f"ssh {WS} 'bash -s' < bad.sh",
    ):
        reason = guard.decide(
            {
                "tool_name": "Bash",
                "tool_input": {"command": command},
                "cwd": str(tmp_path),
            }
        )
        assert reason and "bad.sh" in reason, command
    assert (
        guard.decide(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "bash ok.sh"},
                "cwd": str(tmp_path),
            }
        )
        is None
    )


def test_exit_codes_and_fail_open():
    assert run("echo ok").returncode == 0
    denied = run("git push")
    assert denied.returncode == 2 and "never pushes" in denied.stderr
    broken = subprocess.run(
        [sys.executable, str(HERE / "guard.py")],
        input="not json",
        capture_output=True,
        text=True,
    )
    assert broken.returncode == 0


def test_other_tools_are_not_inspected():
    assert (
        guard.decide(
            {"tool_name": "apply_patch", "tool_input": {"command": "git push"}}
        )
        is None
    )


def test_rjob_submit_follows_claudes_grant(tmp_path, monkeypatch):
    import time

    monkeypatch.setattr(guard, "GRANTS", tmp_path)
    wt = "/mnt/shared-storage-gpfs2/gpfs2-agent-deepresearch/lgx/areal-dev/_worktrees/AReaL/ailab-x"

    def decide(command):
        return guard.decide(
            {"tool_name": "Bash", "tool_input": {"command": command}, "cwd": wt}
        )

    ok = "rjob submit --name lgx-exp-1 --cpu 8 --gpu 0 -- sleep 1"
    assert "grant" in decide(ok)
    (tmp_path / "ailab-x.json").write_text(
        json.dumps({"expires": time.time() + 3600, "rjob_submit": {"max_gpu": 0}})
    )
    assert decide(ok) is None
    assert "GPUs" in decide("rjob submit --name lgx-exp-1 --gpu 1 -- sleep 1")
    assert "lgx-*" in decide("rjob submit --name other --gpu 0 -- sleep 1")
    (tmp_path / "ailab-x.json").write_text(
        json.dumps({"expires": time.time() - 1, "rjob_submit": {"max_gpu": 0}})
    )
    assert "grant" in decide(ok)
    other = guard.decide(
        {
            "tool_name": "Bash",
            "tool_input": {"command": ok},
            "cwd": "/mnt/shared-storage-gpfs2/gpfs2-agent-deepresearch/lgx/areal-dev/_worktrees/AReaL/ailab-y",
        }
    )
    assert "grant" in other


if __name__ == "__main__":
    import tempfile

    test_allowed_commands_pass()
    test_denied_commands_name_the_rule()
    with tempfile.TemporaryDirectory() as d:
        test_scripts_run_by_the_command_are_checked(Path(d))
    test_exit_codes_and_fail_open()
    test_other_tools_are_not_inspected()
    print("ok")
