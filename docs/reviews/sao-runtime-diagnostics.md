# SAO startup diagnostics

Runtime inspection is opt-in through `AREAL_RUNTIME_AUDIT_DIR`. Without it, normal
training emits no additional snapshots or model reads. The snapshots are written from
initialized FSDP engines, ready inference servers, and constructed Tau2 workflows. The
Tau2 controller snapshot follows dataset and checkpoint path resolution and engine
construction. These stages are separate: controller config alone is not proof of worker
consumption.

FSDP evidence includes parameter trainability, optimizer groups, TMS state, GPU
visibility, and the loaded scalar head digest. A local DTensor shard that cannot be
compared to the full checkpoint is labeled as such. These initialization records do not
assert post-DCP-recovery weights or learning effectiveness.

`scripts/sao/check_critic_head.py` checks the current single-file Qwen3.5 critic export
before launch, rejecting missing, ambiguous, malformed, non-finite, or all-zero scalar
heads. It does not manufacture a value manifest or change the compatibility loader used
to create new critics. Historical DCP resume is a different path and cannot establish
correctness of a new initial HF load.

`scripts/sao/gpu_diagnostics.py --output-dir DIR -- COMMAND` explicitly wraps one job
with NVIDIA dmon/pmon and timestamped process-memory snapshots. It reuses the existing
bounded GPU admission check, retains the pre-admission observation, and owns only its
own monitor subprocesses. It does not read process environments or command-line secrets,
install a daemon, or kill unrelated GPU users.

NVIDIA may report host PIDs invisible in a container's `/proc`; missing process metadata
means unresolved ownership, not an idle GPU. Per-process utilization can also be
unavailable. Runtime worker PID/GPU records provide separate local identity evidence.
Host-level attribution requires a host process view.

To remove scaffolding on the next iteration, omit the diagnostic wrapper and unset
`AREAL_RUNTIME_AUDIT_DIR`. The local frozen Tau2 launch script also supports
`SAO_DIAGNOSTICS=0`. Existing evidence remains available. Full formal SAO capacity and
loaded configuration still require the real run; CPU tests are not a substitute.
