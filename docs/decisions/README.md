# Architecture Decisions

ADRs record the durable technical choices of this AReaL fork: the τ²-Bench training and
evaluation protocol, the SAO port, critic training, and the shared runtime and
trajectory contracts. Each record explains why a boundary exists when code and usage
guides alone do not. Usage lives in component guides such as
[examples/tau2/README.md](../../examples/tau2/README.md). Run results, job IDs and
progress belong in reports and PRs, not here.

Start from [template.md](template.md), the **MADR 4.0.0 minimal template** from
[upstream](https://github.com/adr/madr/blob/4.0.0/template/adr-template-minimal.md),
reformatted only by this repository's `mdformat` hook, with its
[license declaration](MADR-LICENSE) (`MIT OR CC0-1.0`) and the
[MIT license text](MADR-LICENSE-MIT). Records add `Date`, `Status` and an explicit
"Applies to" line, and keep the Consequences section.

Write an ADR only for a durable choice: state ownership, component responsibilities,
cross-component contracts, costly-to-reverse dependencies, or a protocol/algorithm
choice whose reason a future maintainer needs. Record actual decisions and only the
options that were actually considered. Use `NNNN-short-title.md` with an English title.
For a genuine reversal, keep the old rationale and link the successor with
`Supersedes`/`Superseded by`. Do not renumber records or reuse numbers.

## Current records

- [0001 Run τ²-Bench through the pinned official harness and AReaL proxy](0001-official-tau2-harness.md)
- [0002 Use non-thinking Qwen3.5-4B with a 32K episode context for τ²-Bench](0002-tau2-policy-generation-protocol.md)
- [0003 Fix τ²-Bench data, domains and evaluation protocol](0003-tau2-data-and-evaluation.md)
- [0004 Implement SAO as Direct DIS, dual-lambda GAE and critic-first updates](0004-sao-implementation-semantics.md)
- [0005 Train the τ² initial critic online against a fixed policy](0005-tau2-online-critic.md)
- [0006 Keep a critic-free asynchronous GRPO recipe for τ²-Bench](0006-tau2-async-grpo-baseline.md)
- [0007 Carry explicit episode termination and bootstrap flags into GAE](0007-explicit-episode-termination.md)
- [0008 Run Qwen3.5 FSDP2/SGLang training from a locked uv environment on A100](0008-qwen35-runtime-environment.md)
- [0009 Schedule τ² episodes with the official runner's retry, exclusion and metric rules](0009-official-tau2-runner-semantics.md)
