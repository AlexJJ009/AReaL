#!/usr/bin/env python3
"""PreToolUse guard for every Codex job in the AReaL research workspace (registered in $CODEX_HOME/hooks.json).

The sandbox already limits local writes to the worktree, its .git, the uv cache and the artifact directory.
This guard covers what the sandbox cannot see: remote commands and shared services. It reads the Bash command,
plus any local shell script the command runs, and denies:

- SSH/SCP/rsync/SFTP to any host except the ailab gateway (`*@h.pjlab.org.cn`: the workspace and RJob pods);
- `ssh -R` forwards other than to the Nomad server ports on cpu1, and dynamic (SOCKS) reverse forwards:
  no tunnel may give an RJob internet access;
- `rjob delete|stop|patch` of a job whose name does not start with `lgx-` (the namespace is shared), and
  `rjob submit` without Claude's grant for this worktree (`notes/grants/<slug>.json`, outside Codex's writable
  roots), with more GPUs than the grant allows, or with a job name not starting with `lgx-`;
- stopping or restarting production services; Docker deletes of objects not named `lgx-exp-*`; prunes;
- `git push`, `gh`, `codex login`, `/root/.codex`, `pkill -f`.

It is a seatbelt for direct mistakes, not a sandbox: shell variables (`$JOB`) are not resolved and pass.
A denial prints the reason on stderr and exits 2; Codex shows the reason to the model. Internal errors allow
the command and are logged, so a guard bug never stops all Codex work.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
import time
from pathlib import Path

# The workspace root (bin/codex-hooks/guard.py -> root): the AReaL research workspace on the ailab dev machine.
ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "tmp" / "codex-guard.log"
GRANTS = ROOT / "_notes" / "AReaL" / "grants"
ALLOWED_HOST = re.compile(r"^[A-Za-z0-9_.+-]+@h\.pjlab\.org\.cn$")
CPU_HOST = re.compile(
    r"^(root@)?(172\.31\.\d+\.\d+|phai-cpu\d|cpu\d(\.\S*)?|localhost|127\.0\.0\.1)$"
)
REMOTE_TOOLS = {"ssh", "scp", "sftp", "rsync", "autossh", "sshpass"}
SSH_VALUE_OPTS = set("bcDEeFIiJLlmOoPpQRSWw")
# ssh -R targets that are allowed: the Nomad server RPC/HTTP ports on cpu1 (production and dev).
TUNNEL_TARGETS = re.compile(
    r"^(127\.0\.0\.1|172\.31\.23\.186):(4646|4647|14646|14647)$"
)
SERVICES = r"(phai-api\S*|nomad\S*|nexus\S*|phai-bulk-proxy\S*|docker\S*|containerd\S*|cpa\S*|sshd\S*)"
TEXT_RULES = [
    (
        re.compile(
            r"\bsystemctl\s+(?:--\S+\s+)*(stop|restart|kill|disable|mask|reload)\s+"
            + SERVICES
        ),
        "production services on cpu1 are never stopped, restarted or reloaded by Codex",
    ),
    (
        re.compile(
            r"\bdocker\s+(system|image|container|builder|volume|network|buildx)\s+prune\b|\bdocker\s+system\s+prune\b"
        ),
        "docker prune touches every user's objects on cpu1",
    ),
    (re.compile(r"\bgit\s+push\b"), "Codex never pushes"),
    (re.compile(r"(^|[\s;&|(])gh\s"), "Codex never uses the GitHub CLI"),
    (re.compile(r"\bcodex\s+login\b"), "never log in to Codex on cpu1"),
    (re.compile(r"/root/\.codex\b"), "/root/.codex belongs to someone else"),
    (
        re.compile(r"\bpkill\s+(-\S*\s+)*-\S*f"),
        "pkill -f matches unrelated command lines; kill by PID",
    ),
]
DOCKER_DELETE = re.compile(
    r"\bdocker\s+(?:image\s+rm|container\s+rm|rmi|rm|stop|kill|volume\s+rm|network\s+rm)\b(.*)"
)
RJOB_VERB = re.compile(r"\brjob\s+(delete|stop|patch)\b(.*)")
RJOB_VALUE_OPTS = {"--namespace", "-n", "--showname", "--charged-group", "--yaml", "-f"}
SCRIPT_RUNNERS = {"bash", "sh", "zsh", "source", "."}


def log(entry):
    try:
        with LOG.open("a") as f:
            f.write(
                json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}) + "\n"
            )
    except OSError:
        pass


def segments(text):
    """Simple commands of a shell text: split on newlines and control operators, then shlex each part."""
    for line in re.split(r"\n|;|&&|\|\||\||&(?!>)|\$\(|`|\(|\)", text):
        try:
            words = shlex.split(line, comments=True)
        except ValueError:
            words = line.split()
        while words and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0]):
            words = words[1:]  # leading VAR=value assignments
        for prefix in (
            "sudo",
            "nice",
            "nohup",
            "setsid",
            "timeout",
            "exec",
            "env",
            "time",
            "stdbuf",
        ):
            while words and os.path.basename(words[0]) == prefix:
                words = words[1:]
                while words and (
                    words[0].startswith("-")
                    or re.match(r"^\d+[smh]?$", words[0])
                    or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0])
                ):
                    words = words[1:]
        if words:
            yield words


def check_ssh(words):
    tool = os.path.basename(words[0])
    if tool not in REMOTE_TOOLS:
        return None
    if tool in ("ssh", "autossh"):
        i, dest = 1, None
        while i < len(words):
            w = words[i]
            if w.startswith("-") and len(w) > 1 and dest is None:
                opt = w[1:]
                value = None
                for k, c in enumerate(opt):
                    if c in SSH_VALUE_OPTS:
                        value = opt[k + 1 :] or (
                            words[i + 1] if i + 1 < len(words) else ""
                        )
                        if not opt[k + 1 :]:
                            i += 1
                        if c == "R" and not tunnel_ok(value):
                            return f"ssh -R {value}: only forwards to cpu1 Nomad ports are allowed"
                        if c == "J" and not all(host_ok(h) for h in value.split(",")):
                            return (
                                f"ssh -J {value}: jump hosts must be *@h.pjlab.org.cn"
                            )
                        if c == "o" and re.match(
                            r"(?i)^(RemoteForward|ProxyCommand|ProxyJump)", value
                        ):
                            return f"ssh -o {value}: not allowed; use -R/-J forms the guard can check"
                        break
                i += 1
                continue
            dest = w
            break
        if dest is not None and not host_ok(dest):
            return f"ssh to {dest}: Codex may only reach the ailab gateway (*@h.pjlab.org.cn)"
        return None
    for w in words[1:]:  # scp / rsync / sftp: user@host:path or host:path operands
        m = re.match(r"^([^/\s:]+):", w)
        if m and not w.startswith("-") and not host_ok(m.group(1)):
            return f"{tool} to {m.group(1)}: Codex may only reach the ailab gateway (*@h.pjlab.org.cn)"
        if tool == "sftp" and not w.startswith("-") and "@" in w and not host_ok(w):
            return f"sftp to {w}: Codex may only reach the ailab gateway"
    return None


def host_ok(dest):
    if dest.startswith("$"):
        return True  # unresolved variable; see module docstring
    return bool(ALLOWED_HOST.match(dest)) and not CPU_HOST.match(dest)


def tunnel_ok(spec):
    if spec.startswith("$"):
        return True
    parts = spec.split(":")
    # [bind:]port:host:hostport is a fixed forward; [bind:]port alone is a dynamic SOCKS forward.
    if len(parts) < 3:
        return False
    return bool(TUNNEL_TARGETS.match(":".join(parts[-2:])))


def check_rjob(text, grant=None):
    for m in RJOB_VERB.finditer(text):
        verb, rest = m.group(1), m.group(2)
        rest = re.split(r"[;&|\n]", rest, maxsplit=1)[0]
        try:
            words = shlex.split(rest)
        except ValueError:
            words = rest.split()
        i = 0
        while i < len(words):
            w = words[i]
            if w in ("--all", "-a") or w.startswith("--all="):
                return f"rjob {verb} --all: the namespace is shared"
            if re.match(r"\d*[<>]", w):
                break  # a redirection ends the job names
            if w in RJOB_VALUE_OPTS:
                i += 2
                continue
            if w.startswith("-"):
                i += 1
                continue
            if not (
                w.startswith("lgx-")
                or w.startswith("$")
                or w.startswith("'")
                or w.startswith('"')
            ):
                return f"rjob {verb} {w}: only jobs named lgx-* may be changed (the namespace is shared)"
            i += 1
    for m in re.finditer(r"\brjob\s+submit\b([^\n;|&]*)", text):
        allowed = (grant or {}).get("rjob_submit")
        if not allowed:
            return (
                "rjob submit needs Claude's grant for this worktree; say in REPORT what you would submit"
                " and why, and Claude decides"
            )
        g = re.search(r"--gpu(?:=|\s+)(\S+)", m.group(1))
        gpus = g.group(1).strip("'\"") if g else "0"
        if not gpus.isdigit() or int(gpus) > allowed.get("max_gpu", 0):
            return f"rjob submit --gpu {gpus}: the grant allows at most {allowed.get('max_gpu', 0)} GPUs"
        name = re.search(r"--name(?:=|\s+)(\S+)", m.group(1))
        if not name or not name.group(1).strip("'\"").startswith(("lgx-", "$")):
            return "rjob submit: name the job lgx-* (the namespace is shared)"
    return None


def load_grant(cwd):
    """Claude's live-check grant for the worktree the command runs in, if present and unexpired."""
    m = re.search(r"/_worktrees/[^/]+/([^/]+)", str(cwd))
    path = GRANTS / f"{m.group(1)}.json" if m else None
    if path is None or not path.is_file():
        return None
    grant = json.loads(path.read_text())
    return grant if float(grant.get("expires", 0)) > time.time() else None


def check_docker(text):
    for m in DOCKER_DELETE.finditer(text):
        rest = re.split(r"[;&|\n]", m.group(1), maxsplit=1)[0]
        try:
            words = shlex.split(rest)
        except ValueError:
            words = rest.split()
        targets = [w for w in words if not w.startswith("-")]
        bad = [
            w for w in targets if not (w.startswith("lgx-exp-") or w.startswith("$"))
        ]
        if bad or not targets:
            return f"docker delete/stop of {bad or 'unnamed objects'}: only lgx-exp-* objects you created"
    return None


def check_text(text, grant=None):
    # Comment lines document rules (bin/lib.sh names /root/.codex); they do not run.
    text = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    for pattern, reason in TEXT_RULES:
        if pattern.search(text):
            return reason
    for words in segments(text):
        reason = check_ssh(words)
        if reason:
            return reason
    return check_rjob(text, grant) or check_docker(text)


def scripts(text, cwd):
    """Local shell scripts the command runs: `bash x.sh`, `./x.sh`, `bash -s < x.sh`, `source x`."""
    found = set()
    for words in segments(text):
        head = os.path.basename(words[0])
        if head in SCRIPT_RUNNERS:
            found.update(w for w in words[1:] if not w.startswith("-"))
        elif words[0].startswith(("./", "/")) and words[0].endswith(".sh"):
            found.add(words[0])
    found.update(re.findall(r"<\s*([^\s;&|<>]+\.sh)\b", text))
    for name in found:
        path = Path(cwd, name) if not os.path.isabs(name) else Path(name)
        try:
            if path.is_file() and path.stat().st_size < 1_000_000:
                yield path, path.read_text(errors="replace")
        except OSError:
            continue


def decide(payload):
    if payload.get("tool_name") != "Bash":
        return None
    command = (payload.get("tool_input") or {}).get("command") or ""
    cwd = payload.get("cwd") or "."
    grant = load_grant(cwd)
    reason = check_text(command, grant)
    if reason:
        return reason
    for path, body in scripts(command, cwd):
        reason = check_text(body, grant)
        if reason:
            return f"{path}: {reason}"
    return None


def main():
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
        reason = decide(payload)
    except Exception as exc:  # noqa: BLE001 - fail open, but leave a trace
        log({"error": repr(exc), "payload": raw[:2000]})
        return 0
    if reason:
        log(
            {
                "deny": reason,
                "cwd": payload.get("cwd"),
                "command": (payload.get("tool_input") or {}).get("command", "")[:2000],
            }
        )
        print(
            f"Blocked by Codex guard: {reason}. See {ROOT / 'AGENTS.md'}.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
