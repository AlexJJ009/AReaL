# Round 2: Qwen3.5 XCCL GPU investigation (in progress)

Started 2026-10-14. Baseline 315617b9; original round-1 edits preserved. Eight local
A100 80GB GPUs were empty at start. No crash result yet. Logs and experiment commands
will be recorded under round2/.

The first experiment uses the original 1 actor + 1 TP1 server harness with actor TMS and
offload/onload. No API, pueue scheduling, commits, rebase or cli_args.py edits are
authorized.

## Initial experiment

`old-1a1s` failed before actor startup: Conda libstdc++ lacks GLIBCXX_3.4.31 needed by
SGLang JIT. This is not the requested crash. `old-1a1s-libfix` retries with
`/usr/lib/x86_64-linux-gnu` first in LD_LIBRARY_PATH, matching the production crash
stack system libstdc++. No packages or shared venv were changed.

## Minimal transport controls

`probe-enabled-fresh-envfix` and `probe-enabled-cycles` passed exact-value GPU
validation (GPUs 1,5, TMS sender only, NCCL 2.27.5, CUMEM=0 both peers), respectively
fresh communication and three pause/resume cycles. These small synchronous broadcasts do
not reproduce the full FSDP/SGLang failure; simple TMS-enabled transport is not
sufficient to trigger it. Logs retained under round2/. Environment variable
NCCL_VERSION=2.28.3-1 is stale container metadata; runtime NCCL logs identify
2.27.5+cuda12.9.

## Old-code 1 actor + 1 server direct result

After correcting process-only native library setup, `old-1a1s-runtime` on exact 315617b9
completed direct XCCL without crash. The full served comparison covered 627 served / 723
canonical actor parameters, no uncovered destinations. Only 24 A_log parameters
mismatched, proving the round-1 precision defect on GPU. Recovery remains in flight. The
server chose NCCL P2P/IPC despite no explicit CUMEM flag. Forced mixed CUMEM0/1
mini-probes failed communicator init with or without TMS (invalid argument, distinct
from target watchdog illegal address).

## Production-size baseline started

Old 1a1s finished direct and actual DCP recovery without target crash; both full
comparisons report the same A_log precision defect. All GPU processes were empty before
`round2/run_multi.sh old-4a4s --mode old --case-timeout 1800`. This baseline uses 4 FSDP
ranks GPUs0–3 and 4 TP1 SGLang servers GPUs4–7, TMS on, repeated offload/onload,
original model revision, memory_efficient_load=false, receiver context32768/mem_fraction
.5/max_running_requests4/CUDA graphs on, actor compute bf16/master fp32. SDPA and
optimizer=None are deliberate bounded weight-only simplifications; no critic or API
rollouts. Commands/environment/source hashes are in old-4a4s/provenance.json.

## Receiver TMS inheritance discovered

4a4s actor-TMS-only direct push and all four full readbacks completed (only24 A_log
mismatches each); remaining run stopped intentionally, recovery unverified there. Actual
production uses single-controller LocalScheduler, which sets get_tms_env_vars for all
worker roles when enable_offload=true (local.py:755). SGLangBackend.build_server_env
copies that environment unchanged. Therefore a server configured
enable_memory_saver=false can still inherit globally active TMS preload. Prior harness
disabled receiver TMS and missed this condition. Controlled receiver-TMS run is next.

## Target crash reproduced — 2026-10-14 08:37 UTC

`old-4a4s-receiver-tms` on315617b9 crashes inside unfinished direct push.
server-3.log:146 contains exact production signature
`[PG ID 4 PG GUID update_weight_group Rank 4] Process group watchdog thread terminated with exception: CUDA error: an illegal memory access was encountered`.
Actor marker XCCL_PUSH_BEGIN direct exists, no END; HTTP connection aborts. Receiver TMS
enabled is the controlled change from successful 4a4s direct baseline. All server
effective-env.json prove old build_server_env preserved INIT1 and TMS hook despite
native saverfalse.

Candidate fix in SGLangBackend.build_server_env strips inherited training TMS
flags/preload only, preserving other preloads and native SGLang saver ownership. CPU
exact315617b9:15failed/2passed; candidate17passed. GPU green pending.

## Candidate direct GPU green

Fixed source + exactly same inherited receiverTMS environment completed direct push and
all four full served comparisons: each627 parameters, zero mismatches, zero uncovered.
effective-env readback confirms flags absent and only unrelated libstdc preload retained
in servers; actor TMS remains active. Actual DCP recovery+second push is running.
