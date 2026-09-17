# Validating sample-level refill

Use three levels of evidence: correctness, isolated scheduling benefit, then real
rollout and training performance. A CPU scheduling speedup is not a GPU throughput
claim.

## 1. Correctness regression

In the development environment:

```bash
python -m pytest -q tests/test_sample_level_refill.py
```

The event-controlled test starts four groups of 12, completes three members in each, and
requires group five to start before any earlier group finishes. It also checks a
48-sample ceiling, intact groups and member order, pause, exhausted group staleness,
duplicate and stale-attempt progress, failure, cancellation cleanup, queue-full
rollback, worker admission, remote timeout and late callbacks. These are correctness
assertions; there are no timing-based speedup assertions in CI.

## 2. Reproducible CPU dispatcher benchmark

```bash
python -m benchmark.sample_level_refill --output sample-refill.json
```

The benchmark uses the real `WorkflowExecutor`, `BatchTaskDispatcher` and
`GroupedRolloutWorkflow`. Async sleeps supply fixed episode durations, so no model, GPU
or external environment is needed. Both arms run the same 32 prompt groups, 12 members
per group, with at most 48 active samples:

| Arm              | `max_concurrent_rollouts`    | `max_concurrent_samples` |
| ---------------- | ---------------------------- | ------------------------ |
| Group admission  | 4                            | `None`                   |
| Sample admission | 4 (superseded for admission) | 48                       |

The staleness budget admits the entire finite cohort in both arms. Defaults compare
uniform 0.5-second episodes with groups containing nine 0.05-second episodes and three
0.5-second episodes. Both paths warm up; three repeats alternate A/B order. Use
`--short-members 3` to reproduce the document's four-groups-each-finish-three pattern,
or change durations and group counts to probe other service-time mixes.

Each run must finish and accept every episode, including all long episodes. The script
checks membership, ordering, zero rejections and peak concurrency. JSON contains every
run, elapsed time, episodes/second, groups/second, mean and peak active samples, and
whether refill preceded the first completed group. Speedup is the ratio of median
elapsed times for the same complete workload.

The expected pattern is reduced idle capacity and shorter elapsed time with long tails.
Uniform durations should show little benefit and may expose scheduling overhead. Sample
refill also cannot help when the group staleness budget prevents another prompt from
starting. The controlled event tests verify this constraint.

This benchmark assumes episode service time is independent of concurrency. Real
inference has shared GPU compute, KV memory, prefill/decode interference and environment
limits. Consequently, this result proves a scheduling opportunity, not an end-to-end
training speedup.

## 3. Real workload A/B

First use a frozen checkpoint and the same finite set of prompt IDs, seeds and rollout
multiplicity. Keep the model, inference configuration, nodes/GPUs, environment
concurrency, timeouts, filtering and staleness settings identical. For group size 12,
compare the two admission settings above. Do not compare 48 groups in the baseline with
48 samples in the treatment.

Run the arms sequentially on the same hardware, alternate order, exclude a common
warmup, and repeat at least three times for a pilot. Increase repetitions when
run-to-run variance is comparable to the observed improvement. Finish the same cohort in
each run; counting only fast completions in a fixed time window can hide long-task
starvation. Preserve unfinished, timed-out and rejected task counts.

Collect:

- **Useful throughput**: accepted, trainable logical episodes per wall-clock second and
  accepted groups per second. Count actual usable members of accepted groups, not tensor
  rows, LLM calls or attempted samples.
- **Latency and resource use**: complete-cohort makespan, group latency p50/p95/p99,
  episode starts/ends in both arms, GPU utilization, generation tokens/second and memory
  peaks. `sample_inflight` is reserved sample capacity, including queued members; it is
  not a GPU utilization measurement.
- **Correctness and bias**: usable-member yield, rejection/timeout rate, and final
  acceptance rates by episode-length bucket. No additional early group acceptance or
  longest-episode dropping should be introduced.
- **Version age**: in training, compare p50/p95/max age against the consuming weight
  version, using generated-token versions and excluding prompt sentinel versions. Track
  `rollout/partial_groups` and peak memory for unfinished-group accumulation.

Finally run short training A/B trials from the same checkpoint and data settings.
Compare reward/evaluation curves at equal numbers of consumed logical episodes, as well
as time to a fixed quality target. Admission changes execution order, so bitwise
equality is not a suitable quality criterion. Repeat with matched seeds before
attributing quality differences to this feature.

A useful result improves accepted training throughput beyond run-to-run noise without
materially worsening quality, rejection, long-task acceptance, version age or memory. A
predeclared engineering target (for example, at least 10% useful throughput gain with
those guardrails) is more informative than selecting a favorable run after seeing the
results.

## Asynchronous batch consumption with GPU counters

`benchmark.swe_sample_refill` also supports a bounded rolling prompt window:

```bash
python -m benchmark.swe_sample_refill \
  --mode group --batch-size 16 --group-size 8 \
  --prefetch-batches 3 --measure-batches 10 --sample-slots 128 \
  --output group-run --manifest cohort.json --config /path/to/swe-config.yaml
```

Run the other arm with `--mode sample`, a fresh output directory and the same
manifest/config. Both arms submit 48 prompts initially, consume any 16 complete accepted
groups as one batch, and replenish exactly 16 prompts per consumption. After ten
measured batches, the remaining 32 groups drain without being counted as additional
measured batches. No slow episode is cancelled to obtain a ready batch. Rejected groups
are recorded and do not count toward a ready batch; each opens one window slot for the
next prompt in the fixed candidate order.

The stream must contain at least 192 distinct prompt identities and should include
additional candidates to replace rejected groups. A fixed seed shuffles the candidate
list before selecting the cohort, and both arms verify the same manifest. The number of
attempted prompts may differ when rejection counts differ; report these counts and
paired outcomes alongside batch-ready throughput. The benchmark keeps weights frozen and
separately enforces the rolling window; its staleness budget covers the full cohort so a
nonadvancing weight version cannot stall the collector.

`batches.jsonl` records first-batch latency and each subsequent batch-ready interval.
`measurement.json` provides the wall-clock alignment point. Report batch 1 separately;
use the predeclared batch 2–10 interval for steady consumption metrics. `summary.json`
separates measured consumption time from final drain time. `sglang-metrics.jsonl.gz`
captures the existing SGLang Prometheus endpoint every five seconds for token throughput
and server queue analysis.

On a node with an existing DCGM exporter, capture counters without opening a second
profiling session:

```bash
python -m benchmark.gpu_metrics \
  --exporter-url "$DCGM_EXPORTER_URL" --output gpu-metrics.jsonl --interval 1
```

SM Active, SM Occupancy, Tensor Active and DRAM Active are recorded as ratios; GPU
utilization is a percentage. The collector keeps raw values and marks
out-of-range/sentinel values invalid rather than clamping them. Scrape timestamps are
retrieval times, not a guarantee of the exporter's source sampling frequency. Compare
the same measured batch windows across all GPUs and report first-batch warmup and final
drain separately. Higher GPU activity alone is insufficient: check usable-group
throughput, generation tokens/second, energy per accepted sample, failures and reward
too.

## Measured inference pilots

These are single paired pilots on eight NVIDIA L20X GPUs, with frozen model weights.
Both modes consume complete groups in completion order from the same shuffled candidate
list. They do not establish training convergence or a workload-independent speedup. The
default remains group-level admission; sample-level refill is opt-in.

### SWE-bench Verified: asynchronous rolling consumption

Qwen3-4B-Instruct-2507, eight TP1 SGLang replicas, B=16 prompts, n=8 episodes, 128
sample slots and a 48-prompt rolling window. Both modes consumed ten batches (160
accepted groups / 1,280 episodes). The steady window spans batch 1 ready to batch 10
ready, excluding startup and final drain.

| Metric                              | Group admission | Sample refill |                Change |
| ----------------------------------- | --------------: | ------------: | --------------------: |
| Ten batches ready                   |      127.29 min |    101.24 min |                -20.5% |
| Steady window duration              |      117.05 min |     90.68 min |                -22.5% |
| Accepted groups/hour, steady window |           73.82 |         95.28 |                +29.1% |
| Median batch-ready interval         |        625.64 s |      566.63 s |                 -9.4% |
| SM Active, steady window            |          35.67% |        50.07% |             +14.40 pp |
| Average active episodes             |           55.62 |        121.67 | Same 128-slot ceiling |

First-batch latency did not improve (10.24 to 10.55 minutes). Completion-order selection
changed the consumed prompt set: 144 of 160 prompts were common to both runs. Before the
measurement ended, 1/161 versus 3/163 groups were rejected and replaced; over the entire
run including drain, 1/193 versus 9/195 were rejected. Final drain took 54.01 versus
53.49 minutes. Consumed episodes included 63 versus 73 harness failures; reward-one
rates were 13.13% versus 14.38%, which are descriptive outcomes, not evidence of quality
improvement. All measured groups had eight members; slow groups were not dropped to
declare a batch ready.

### Boba math: inference with simulated RL pauses

Qwen3-4B thinking, eight TP1 SGLang replicas, B=16, n=8, 128 sample slots and 48-prompt
lookahead. Maximum new tokens was 16,384; temperature 0.6, top-p 0.95, top-k 20, and
SGLang static memory fraction 0.85. Each mode measured six batches, then drained the
remaining candidates: 96 consumed groups and 128 total completed groups (1,024
episodes). Every ready batch pauses real generation for a fixed ten seconds and resumes
unfinished work. No training engine, weight synchronization, or reward-based group
rejection is included.

| Metric                                                   | Group admission |   Sample refill |   Change |
| -------------------------------------------------------- | --------------: | --------------: | -------: |
| Mean rollout interval, batches 2–6                       |        105.20 s |         95.13 s |    -9.6% |
| Mean cycle including pause/resume and simulated training |        116.22 s |        106.15 s |    -8.7% |
| Steady accepted-group throughput                         |        baseline | 1.106x baseline |   +10.6% |
| All-candidate completion time                            |        baseline | 0.958x baseline |    -4.2% |
| Cycle-window SM Active                                   |          55.13% |          55.32% | +0.19 pp |
| Mean output tokens, all candidates                       |       10,729.01 |       10,573.39 |   -1.45% |

Neither run rejected a group. The small output-length difference and single execution
order limit causal precision: this pilot shows a latency benefit but no meaningful SM
Active increase. Boba B=8/n=8 at the same 128-slot limit did not improve steady rollout
time (54.35 to 55.41 seconds). Benefits depend on workload and configuration.

GPU means are time-weighted across all eight cards. SWE uses its steady consumption
window; Boba uses complete simulated cycles after the first batch, including the fixed
pauses. These GPU windows are intentionally different and must not be compared as
identical training measurements. DCGM scrape coverage was approximately complete, but
exporter source update timestamps were unavailable; one-second polling does not imply
one-second hardware-counter updates. Reverse-order repetitions and actual
weight-update/version-age validation remain outstanding.
