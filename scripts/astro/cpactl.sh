#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Usage: cpactl.sh start|stop|restart|status
# Runs the self-hosted CPA stack on astro-a100 as three plain processes, in place of the AiLab compose project
# phai-cpa: this machine is an unprivileged container (no CAP_SYS_ADMIN, mount and unshare denied, cgroups
# read-only), so dockerd starts but cannot create or even import a container (tested 2026-11-14).
#
#   egress  mihomo, mixed proxy 127.0.0.1:17893 (default US exit) and 127.0.0.1:17894 (Gemini fallback group);
#           same trojan exits as the cpu1/AiLab config, dialled directly (no platform proxy hop here)
#   cpa     CLIProxyAPI v8.0.42, API on 127.0.0.1:8317, upstream traffic through the egress proxy
#   keeper  CPA Usage Keeper v2.0.1, dashboard on :18319 (login required), polls the CPA management API
#
# Each service runs under a restart loop in its own session, so it survives SSH disconnects and crashes. A
# container restart kills everything (there is no init system): run `cpactl.sh start` again; it is idempotent.
# State lives in $CPA_HOME (default /root/lgx/cpa, mode 700): bin/, keys/ (config, OAuth auths, keeper DB),
# mihomo/, run/ (CPA working dir: logs/, static/), state/ (pids and service logs).
set -euo pipefail
C=${CPA_HOME:-${AREAL_WORKSPACE_ROOT:-/root/lgx}/cpa}
ST=$C/state
mkdir -p "$ST" "$C/run/logs"
SERVICES=(egress cpa keeper)

cmd_of() {
  case $1 in
    egress) echo "exec $C/bin/mihomo -d $C/mihomo -f $C/mihomo/config.yaml" ;;
    cpa) echo "cd $C/run && exec $C/bin/cli-proxy-api -config $C/keys/config.yaml" ;;
    keeper) echo "set -a; . $C/keys/cpa-usage-keeper.env; set +a; cd $C/keys/keeper && exec $C/bin/cpa-usage-keeper" ;;
  esac
}

alive() { [ -s "$ST/$1.pid" ] && kill -0 "$(cat "$ST/$1.pid")" 2> /dev/null; }

start_one() {
  if alive "$1"; then echo "$1 already running (loop pid $(cat "$ST/$1.pid"))"; return; fi
  # The loop restarts the service 5 s after any exit; stop kills the loop's whole process group.
  setsid nohup bash -c "while :; do ( $(cmd_of "$1") ); echo \"\$(date -Is) $1 exited rc=\$?, restarting in 5s\"; sleep 5; done" \
    >> "$ST/$1.log" 2>&1 < /dev/null &
  echo $! > "$ST/$1.pid"
  echo "$1 started (loop pid $!)"
}

stop_one() {
  if alive "$1"; then kill -TERM -- "-$(cat "$ST/$1.pid")" 2> /dev/null || true; echo "$1 stopped"; else echo "$1 not running"; fi
  rm -f "$ST/$1.pid"
}

probe() {  # probe <url> -> HTTP code (000 when unreachable)
  curl -s -o /dev/null -m 5 -w '%{http_code}' --noproxy '*' "$1" || true
}

status() {
  rc=0
  for s in "${SERVICES[@]}"; do
    if alive "$s"; then echo "$s: running (loop pid $(cat "$ST/$s.pid"))"; else echo "$s: DOWN"; rc=1; fi
  done
  # 401 without a key means the CPA API is up and still enforcing auth.
  echo "cpa api 127.0.0.1:8317/v1/models (no key): $(probe http://127.0.0.1:8317/v1/models)"
  echo "keeper 127.0.0.1:18319: $(probe http://127.0.0.1:18319/)"
  echo "egress via 17893 -> gstatic: $(curl -s -o /dev/null -m 8 -w '%{http_code}' -x http://127.0.0.1:17893 https://www.gstatic.com/generate_204 || true)"
  return $rc
}

case ${1:-} in
  start)
    for s in "${SERVICES[@]}"; do start_one "$s"; sleep 2; done ;;
  stop)
    for s in keeper cpa egress; do stop_one "$s"; done ;;
  restart)
    "$0" stop; sleep 2; "$0" start ;;
  status)
    status ;;
  *) sed -n 3p "$0" >&2; exit 2 ;;
esac
