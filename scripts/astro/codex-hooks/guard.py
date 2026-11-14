#!/usr/bin/env python3
"""PreToolUse guard for every Codex job in the AReaL workspace on astro-a100 ($CODEX_HOME/hooks.json).

Codex runs here WITHOUT a filesystem sandbox: the machine is an unprivileged container that denies user
namespaces, and Codex's Linux sandbox needs bubblewrap (tested 2026-11-14: bwrap "No permissions to create new
namespace"; the legacy landlock path panics). Jobs therefore run as root with danger-full-access, and this guard
is the only seatbelt. It reads the Bash command, plus any local shell script the command runs, and denies:

- anything touching the CPA stack ($LGX/cpa: OAuth credentials, API keys, keeper DB) or its processes;
- SSH keys and remote tools (ssh/scp/sftp/rsync to a host, autossh): Codex jobs on astro have no remote hosts;
- `git push`, `gh`, `codex login`, `/root/.codex`, `pkill -f`/`killall`;
- recursive deletion of the workspace roots.

It is a seatbelt for direct mistakes, not a sandbox: shell variables are not resolved and pass. A denial
prints the reason on stderr and exits 2; Codex shows it to the model. Internal errors allow the command and
are logged, so a guard bug never stops all Codex work.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
import time
from pathlib import Path

# bin/codex-hooks/guard.py -> the workspace root (/root/lgx).
ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "tmp" / "codex-guard.log"
LGX = re.escape(str(ROOT))

TEXT_RULES = [
    (
        re.compile(LGX + r"/cpa\b|\bcpactl\b"),
        "the CPA stack (keys, OAuth credentials, keeper DB) is off limits to Codex",
    ),
    (
        re.compile(r"\b(cli-proxy-api|mihomo|cpa-usage-keeper)\b"),
        "the CPA processes are never touched by Codex",
    ),
    (re.compile(r"(/root|~|\$HOME)/\.ssh\b"), "SSH keys are off limits"),
    (re.compile(r"\bgit\s+push\b"), "Codex never pushes"),
    (re.compile(r"(^|[\s;&|(])gh\s"), "Codex never uses the GitHub CLI"),
    (re.compile(r"\bcodex\s+login\b"), "never log in to Codex here"),
    (re.compile(r"/root/\.codex\b"), "/root/.codex is not ours"),
    (
        re.compile(r"\bpkill\s+(-\S*\s+)*-\S*f|\bkillall\b"),
        "pkill -f / killall match unrelated processes; kill by PID",
    ),
]
REMOTE_TOOLS = {"ssh", "scp", "sftp", "autossh", "sshpass"}
PROTECTED = {
    "/",
    "/root",
    str(ROOT),
    f"{ROOT}/Areal",
    f"{ROOT}/_worktrees",
    f"{ROOT}/_artifacts",
    f"{ROOT}/home",
}
SCRIPT_RUNNERS = {"bash", "sh", "zsh", "source", "."}
PREFIXES = (
    "sudo",
    "nice",
    "nohup",
    "setsid",
    "timeout",
    "exec",
    "env",
    "time",
    "stdbuf",
)


def log(entry):
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
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
            words = words[1:]
        for prefix in PREFIXES:
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


def check_words(words):
    tool = os.path.basename(words[0])
    args = words[1:]
    if tool in REMOTE_TOOLS and not (tool == "ssh" and args in (["-V"], [])):
        return f"{tool}: Codex jobs on astro have no remote hosts"
    if tool == "rsync" and any(
        re.match(r"^[^/\s]+:", a) for a in args if not a.startswith("-")
    ):
        return "rsync to a remote host: Codex jobs on astro have no remote hosts"
    if tool == "rm" and any(a.startswith("-") and ("r" in a or "R" in a) for a in args):
        for a in args:
            if not a.startswith("-") and os.path.normpath(a) in PROTECTED:
                return f"rm -r {a}: workspace roots are never deleted"
    return None


def scripts_run(words, cwd):
    if os.path.basename(words[0]) in SCRIPT_RUNNERS and len(words) > 1:
        cand = words[1]
    elif words[0].endswith(".sh"):
        cand = words[0]
    else:
        return None
    p = Path(cand) if os.path.isabs(cand) else Path(cwd) / cand
    try:
        if p.is_file() and p.stat().st_size < 200_000:
            return p.read_text(errors="replace")
    except OSError:
        pass
    return None


def verdict(command, cwd=".", depth=0):
    for rx, reason in TEXT_RULES:
        if rx.search(command):
            return reason
    for words in segments(command):
        reason = check_words(words)
        if reason:
            return reason
        if depth < 2:
            body = scripts_run(words, cwd)
            if body:
                reason = verdict(body, cwd, depth + 1)
                if reason:
                    return (
                        f"script {words[-1] if len(words) == 1 else words[1]}: {reason}"
                    )
    return None


def main():
    try:
        payload = json.load(sys.stdin)
        command = (payload.get("tool_input") or {}).get("command") or ""
        if isinstance(command, list):
            command = " ".join(command)
        reason = verdict(command, payload.get("cwd") or ".")
    except Exception as exc:  # never block all work on a guard bug
        log({"error": repr(exc)})
        return 0
    if reason:
        log({"deny": command[:500], "reason": reason})
        print(f"Blocked by the AReaL astro guard: {reason}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
