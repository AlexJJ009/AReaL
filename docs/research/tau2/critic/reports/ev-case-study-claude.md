# Why tau2 critic-only training barely reaches positive explained variance (Claude case study)

Author: Claude (independent of the Codex study; its output was not read). Date:
2026-10-14. Output dir:
`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-critic-ev-case-study/claude/`
Everything below was computed on CPU from files on disk (no GPU job, no pueue task, no
API call, no repo or run-directory change). Scripts and derived data are in the output
dir; each claim names the script/file. Runs: r479 = critic-qwen35-formal-v1 (random
head), r499 = critic-head-qwen35-v2, r510 = critic-head-qwen35-mix-r2 (still running; I
used steps up to 65, the last validation available when I read it), r480 = Instruct (not
analysed beyond the noise table).

## 1. Conclusion

The observed near-zero EV is not one phenomenon. Three causes stack, with different
weights per domain.

1. **Airline: the critic has learned a real ranking, but its output amplitude is tiny,
   and EV punishes that.** In r499 the trajectory-mean prediction ranks success/failure
   with AUC 0.88 (95% CI \[0.70, 1.00\], 8 failures in 20), yet EV is 0.025, because
   predictions span only about 0.77-0.92 (between-trajectory SD 0.017 against target SD
   0.49). The least-squares slope that would rescale the predictions is 12 (r499, step
   94\) and 2-3 (r510), so a trajectory-level affine recalibration lifts Airline EV from
   0.025 to 0.175 (r499) and 0.17 to 0.27 (r510, step 60). The value-head weight norm is
   0.0015 (r499 step 94; `headnorm.txt`) because Adam moves each weight at most about lr
   per step: 94 steps x 5e-6 caps each component at 4.7e-4, and the weights moved
   coherently for only a fraction of that. This is an optimisation-budget problem (lr x
   steps), not a lack of signal. Training-batch EV on fresh rollouts is about the same
   as validation EV (r510: 0.04 -> 0.12 -> 0.17 -> 0.19 over epochs 1-4 on seen tasks,
   validation Airline 0.02-0.18), so there is no overfitting gap: the critic is
   under-fit, not over-fit.
1. **Airline's reachable level is limited by what a first-token (task-identity)
   predictor can know, and part of what the critic learned is exactly that.** The 7-draw
   task-level ceiling is 0.59 \[0.36, 0.78\]. The critic's trajectory mean correlates
   0.57-0.81 with the 4-draw task success rate of the ceiling experiment (r479 0.70,
   r499 0.57, r510 0.81), and after controlling for that rate the residual correlation
   with the outcome is about 0 for r479 (0.01) and r510 (-0.12), 0.42 for r499. A
   bag-of-words logistic model trained on the same stored rollouts, seeing only the
   first user message, already reaches Airline val AUC 0.77, EV 0.20 \[0.05, 0.37\];
   seeing the whole trajectory it reaches AUC 0.91, EV 0.48 \[0.15, 0.74\]. Averaged
   over five prefix lengths (a pseudo token-level EV) it gets Airline 0.24 \[-0.01,
   0.44\]. So a cheap non-neural model already beats the critic's token EV by 0.07-0.2
   on the same data, which says the signal is learnable from the stored data and the
   critic has not extracted it all. At the token level (section 5) the r499 critic's
   Airline separation peaks mid-trajectory (decile AUC 0.92) and decays to 0.71 at the
   end, and its mean change per turn is identical for successes and failures, i.e. it
   carries task difficulty plus generic position and tool/user effects, not
   outcome-deciding events.
1. **Retail: there is almost nothing to learn at the start, little to learn later, and
   almost no failing training data.** The 7-draw ceiling for the first token is 0.24
   \[0.07, 0.39\] and more than half of Retail tasks have mixed outcomes across draws,
   i.e. the outcome is decided by user-side randomness after the task is known. The same
   bag-of-words model has Retail AUC 0.46 (EV -0.007) from the first message and only
   reaches AUC 0.85 / EV 0.19 on the complete trajectory; averaged over prefixes the
   pseudo token-level EV is 0.062 \[0.007, 0.10\]. The training data contain very few
   Retail failures: 29 of 674 Retail trajectories in r479 and in r499 (pack 11-16/536,
   official 13-18/138) and 46 of 442 in r510. The critic cannot be expected to exceed
   about 0.06 token-level EV in Retail from this data, and EV about 0 is the expected
   outcome, not evidence of a training bug.

**Validation noise decides what is "real".** On 20 Airline / 40 Retail trajectories the
bootstrap SD of EV is 0.01-0.02 for r499 (tiny because predictions are tiny), 0.03-0.10
for r510 Airline, 0.03-0.04 for r510 Retail and up to 0.09 for r479 (random head).
Differences between steps need a paired test (section 3): r499 Airline 50 -> 94 (+0.020,
paired CI \[0.009, 0.039\]) is real but trivial in size; r510 Airline 20 -> 60 (+0.099,
\[-0.045, 0.230\]) is not distinguishable from zero. A perfect task-identity predictor
would score anywhere in \[-0.2, 0.62\] on the 40 Retail validation trajectories
(simulated 95% interval), so Retail validation EV cannot tell 0 from the ceiling.

**Cheapest reproduction and fix (details in sections 6-7).** The poor result is
reproducible offline, with no API, from the 4007 stored training rollouts present when I
extracted them (1494 + 1494 + 1019 from r479, r499 and the still-running r510; token ids
and loss masks included) and the three stored validation sets (180 trajectories on the
same 60 tasks). The first discriminating experiment is a frozen-feature linear probe
(0.3-0.7 GPU-h): if it reaches Airline EV above 0.2 on validation, the representation
holds the signal and the fix is optimisation (head/body learning rate, steps); if it
does not, the base features do not encode it and the lever is data or fine-tuning depth.
Ranked fixes: (1) raise head learning rate (zero code cost, expected Airline EV +0.1 to
+0.2); (2) validate with calibrated EV, AUC and several draws per task (zero training
cost); (3) sigmoid/BCE head; (4) failure-enriched Retail data; (5) lower-variance labels
from prefix re-rollouts (expensive).

## 2. Where EV comes from in the code (so the numbers are interpreted correctly)

- `scripts/tau2/train_critic.py::_metric_row` records, per validation trajectory, only
  sums: `n_tokens`, `error_sum`, `sse`, `target_sum`, `target2_sum`, plus prediction
  mean/min/max. Per-token predictions are not stored.
- `scripts/qualification/critic_pretrain.py::aggregate_validation` computes EV
  **token-pooled**, with the global bias removed:
  `EV = 1 - (mse - bias^2) / var_target`, where `var_target` is the token-weighted
  variance of the 0/1 label. Long trajectories dominate; the label is constant within a
  trajectory. I recomputed EV from `rows` for all four runs and all 20 steps and it
  equals the stored value to all printed digits (`val_noise.txt`), so the formulas below
  are the ones the reports use. (README 16.14.5 mentions bias-removed EV for 499 only;
  r479's stored EV is also bias-removed.)
- Consequence 1: because the target is constant within a trajectory, any prediction
  movement inside a trajectory that is not an exact step to the right value costs EV.
  For r479 (random head) the within-trajectory prediction RMS is 0.10-0.25 and costs
  0.04-0.44 of target variance (`val_decomp.txt`, column `within/tv`); for r499 it is
  0.014-0.03 and costs 0.001-0.004.
- Consequence 2: EV conflates amplitude with ranking. A critic that ranks perfectly but
  outputs 0.8 +/- 0.01 has EV near 0.
- Consequence 3: the validation set is one T=1 rollout per task, collected once per run;
  the three runs' outcomes agree on only 80-83% of the same 60 tasks (`oracle.txt`), so
  the label variance and which tasks fail differ between runs (Airline failures: 10 / 8
  / 8 for r479 / r499 / r510).

## 3. Sampling noise of validation EV (question 2, last part)

Method (`val_noise.py`, output `val_noise.json`, `val_noise.txt`): per run, per
validation step, per domain: (a) bootstrap over trajectories (1000 resamples) using the
stored per-trajectory sums; (b) a permutation null: permute the 0/1 outcomes across
trajectories, keeping each trajectory's predictions (the permuted SSE is recomputed
exactly from the stored sums), giving the EV distribution of a critic whose ranking is
unrelated to outcomes; (c) trajectory-mean AUC. `extra1.py` adds a paired bootstrap for
step differences and the step-to-step jitter.

| run, step | domain  | EV     | bootstrap SD | 95% CI             | permutation-null 95th pct | AUC (traj-mean) | failures / n |
| --------- | ------- | ------ | ------------ | ------------------ | ------------------------- | --------------- | ------------ |
| r499, 94  | Airline | 0.025  | 0.017        | \[0.001, 0.071\]   | 0.018                     | 0.88            | 8/20         |
| r499, 94  | Retail  | -0.001 | 0.009        | \[-0.027, 0.007\]  | 0.010                     | 0.48            | 5/40         |
| r510, 60  | Airline | 0.168  | 0.101        | \[-0.040, 0.339\]  | 0.101                     | 0.83            | 8/20         |
| r510, 60  | Retail  | 0.028  | 0.028        | \[-0.035, 0.063\]  | 0.026                     | 0.60            | 9/40         |
| r479, 94  | Airline | 0.038  | 0.027        | \[-0.020, 0.085\]  | 0.016                     | 0.86            | 10/20        |
| r479, 94  | Retail  | -0.078 | 0.085        | \[-0.330, -0.032\] | -0.022                    | 0.66            | 5/40         |

Findings.

- Noise scales with prediction amplitude. r499's critic is so flat that EV differences
  of 0.01 are resolvable (paired CI), but they are also practically meaningless. r510's
  larger amplitude brings SD 0.06-0.10 on Airline (20 trajectories, 8 failures) and the
  CI of its best EV, 0.18, spans \[0.04, 0.31\].
- Airline EV is above its permutation null in r479/r499/r510 at step 94/94/60 (p =
  0.009, 0.021, 0.013): the Airline ranking is real. Retail is not for r479 (p = 0.24)
  and r499 (p = 0.25); r510 step 60 has p = 0.03, but the same run has p = 0.23 at step
  50 and 0.10 at step 55, so treat it as unconfirmed.
- Paired differences (same trajectories): r499 Airline 50 -> 94 +0.020 \[0.009, 0.039\];
  r499 Airline 25 -> 94 +0.010 \[-0.005, 0.039\]; r510 Airline 20 -> 60 +0.099 \[-0.045,
  0.230\]; r510 Airline 55 -> 60 -0.011 \[-0.089, 0.069\]; r510 Retail 50 -> 60 +0.029
  \[0.014, 0.058\]; r479 Airline 90 -> 94 +0.112 \[0.064, 0.204\] (a recovery from a
  transient dip, README's "unstable +0.15").
- Step-to-step jitter in the second half of training, SD across validation steps: r499
  Airline 0.008, Retail 0.002; r510 Airline 0.059, Retail 0.022; r479 Airline 0.062,
  Retail 0.097. A "peak" such as r510 Airline 0.179 (step 55) next to 0.168 (60) and
  0.137 (50) should be read as 0.14-0.18, and a window average is a better summary than
  the maximum.
- Effect of a different outcome draw: r479/r499/r510 ran different T=1 rollouts of the
  same 60 tasks; the 7-draw per-task statistics show the task-identity oracle itself
  ranges widely. A perfect predictor of the per-task success rate (4-draw estimate from
  experiment 475) scores EV 0.70 / 0.47 / 0.26 on the r479 / r499 / r510 Airline
  validation sets (simulated expectation 0.71-0.78, SD 0.10), and 0.20 / 0.015 / 0.04 on
  Retail (simulated 0.35-0.36, SD 0.18-0.26; 95% interval \[-0.2, 0.62\])
  (`oracle.txt`). So comparing r499 with r510 on Airline mixes a real training
  difference with a validation-set difference of a few tenths. (Caveat: the 4-draw
  estimates are themselves noisy, and 475 used the fixed-v2 data while r499/r510
  validate on fixed-v3; the simulation assumes the noisy estimates are true.)
- Power for Retail: to resolve a Retail EV to +/-0.05 with an oracle-scale predictor you
  need on the order of 2000+ trajectories (SD 0.2 at n=40 scales as 1/sqrt(n)); with 40,
  only large effects are detectable. For a flat critic like r499 the SD is 0.009 but
  then EV is bounded by the amplitude.

## 4. "No learnable signal" or "signal the critic fails to learn"? (questions 1 and 2)

### 4.1 What is on disk

| Item                      | Path (under `.../codex-sao-math/`)                                | Content                                                                                                                                                                       |
| ------------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| validation reports        | `<run>/tau2-critic-fit-evidence/validation-step-*.json`           | per-trajectory sums, no per-token values                                                                                                                                      |
| validation rollouts       | `<run>/tau2-critic-fit-evidence/fixed-validation-rollouts.pt`     | 60 rows: `input_ids`, `loss_mask`, `turn_ids`, `rewards` (full tokens)                                                                                                        |
| training rollouts         | `<run>/logs/root/*/*/rollout/0/*.jsonl`                           | r479: 1494, r499: 1494, r510: 1019 so far; keys include `prompt`, `completion`, `input_ids`, `loss_mask`, `reward`, `source_id`, `domain`, `official_score` (~756 MB per run) |
| validation rollout logs   | `<run>/logs/root/*/*/eval-rollout/0/*.jsonl`                      | the same 60 val trajectories as text                                                                                                                                          |
| train metrics             | `<run>/tau2-critic-fit-evidence/train-metrics/step-*.json`        | per-step pre-update critic mse, residual variance, target variance, old_value                                                                                                 |
| critic checkpoints        | `<run>/checkpoints/root/*/*/critic/epoch*` (every 5 steps, 15 GB) | full HF weights incl. `score.weight`/`score.bias`                                                                                                                             |
| ceiling experiment 475    | `user-sim-20261012/api-user/deepseek-test-u0p1/episodes.jsonl`    | 240 outcomes (60 tasks x 4), outcomes only, no tokens                                                                                                                         |
| ceiling 473-476, 481, 482 | `user-sim-20261012/api-user/*/episodes.jsonl`                     | outcomes only                                                                                                                                                                 |

The 479 training rollouts and the ceiling episodes both exist; only the former have
tokens. I extracted all of them into `trajs.pkl` (`extract.py`).

### 4.2 Task-level signal in the training data (`train_ceiling.py`, `train_ceiling.txt`)

First-token ceiling `(T - W)/T` by training stratum, pooling r479 and r499 (same 747
tasks, same frozen actor, T=1, k=4 draws) and r510 (k=3):

| stratum          | tasks | mean success | ceiling (k=4) | fraction of tasks with mixed outcomes |
| ---------------- | ----- | ------------ | ------------- | ------------------------------------- |
| Airline pack     | 380   | 0.916        | 0.30          | 0.20                                  |
| Airline official | 30    | 0.367        | 0.47          | 0.43                                  |
| Retail pack      | 268   | 0.975        | 0.15          | 0.08                                  |
| Retail official  | 69    | 0.888        | 0.14          | 0.32                                  |

Split-half agreement of task means between r479 and r499: Airline pack 0.45, Airline
official 0.62, Retail pack 0.14, Retail official 0.32. On the test tasks the 7-draw
ceiling (475 outcomes + the three validation draws, `ceil7.py`) is Airline 0.59 \[0.36,
0.78\], Retail 0.24 \[0.07, 0.39\]; consistent with README 16.14.4 (0.61 / 0.20).

Failures available for learning (`extra1.txt`): Airline 73/760 (pack) + 37/60 (official)
in r479; Retail 11/536 + 18/138 in r479, 16/536 + 13/138 in r499, 13/223 + 33/219 in
r510. A critic that has seen about 30 Retail failure trajectories in 94 steps cannot
learn Retail failure signatures.

### 4.3 What a cheap model extracts from the same stored data (`probe_bow.py`, `probe_bow2.py`)

Setup: per domain, hashed unigram+bigram TF-IDF-like features of the text up to a
fraction f of the assistant turns (f=0 means only the first user message), plus four
counters (tool calls, tool errors, turns, length); L2 logistic regression (L-BFGS,
lambda chosen by task-grouped CV on the training set). Training: all r479+r499+r510
training rollouts of that domain (Airline 2217, Retail 1790 trajectories). Test: the
three stored validation sets (Airline 60 = 20 tasks x 3 draws, 26 failures; Retail 120,
19 failures). Metrics are trajectory-level, bias-removed EV as in the validation report
(unweighted by tokens), 95% CI by bootstrap over tasks.

| domain  | prefix f           | val AUC \[CI\]      | val EV \[CI\]            |
| ------- | ------------------ | ------------------- | ------------------------ |
| Airline | 0 (first user msg) | 0.77 \[0.59, 0.89\] | 0.20 \[0.05, 0.37\]      |
| Airline | 0.25               | 0.74 \[0.51, 0.90\] | 0.15 \[-0.20, 0.44\]     |
| Airline | 0.5                | 0.74 \[0.54, 0.90\] | 0.15 \[-0.24, 0.44\]     |
| Airline | 0.75               | 0.82 \[0.65, 0.94\] | 0.32 \[0.04, 0.56\]      |
| Airline | 1.0                | 0.91 \[0.79, 0.99\] | 0.48 \[0.15, 0.74\]      |
| Retail  | 0                  | 0.46 \[0.25, 0.67\] | -0.007 \[-0.054, 0.041\] |
| Retail  | 0.25               | 0.55 \[0.35, 0.73\] | -0.003 \[-0.064, 0.049\] |
| Retail  | 0.5                | 0.71 \[0.51, 0.86\] | 0.051 \[-0.014, 0.096\]  |
| Retail  | 0.75               | 0.74 \[0.56, 0.86\] | 0.083 \[0.009, 0.128\]   |
| Retail  | 1.0                | 0.85 \[0.74, 0.94\] | 0.192 \[0.058, 0.289\]   |

Pseudo token-level EV, averaging the five prefix predictions with equal weight (a
stand-in for the critic seeing every prefix): **Airline 0.239 \[-0.007, 0.440\]; Retail
0.062 \[0.007, 0.102\]**. Official-train-only training (216 / 495 trajectories;
`probe_bow.txt`) gives Airline 0.18 / 0.15 / 0.18 / 0.27 / 0.41 and Retail -0.03 / -0.01
/ 0.01 / 0.04 / 0.05, i.e. the Airline first-message signal does not need the easy pack
tasks, while Retail gets worse without them.

Interpretation. (a) Airline's first-message signal (EV ~0.2) matches the first-token
ceiling and equals what r510's critic reaches (0.14-0.18). (b) The extra Airline signal
later in the trajectory (EV 0.3-0.5 at 75-100%) is event-driven and is what a
token-level critic would have to capture. (c) Retail has no signal until about half of
the conversation and only moderate signal at the end; pseudo token-level EV 0.06 is a
rough upper reference for what any critic can get from this data and a +/-0.03
validation noise makes this undetectable on 40 trajectories. Caveats: bag-of-words sees
user-sim stop tokens (`###STOP###`, transfer tokens) that the critic also sees; the
training and validation rollouts are not independent of the critic's own training
distribution; EV here is trajectory-level, not token-pooled with real position weights.

### 4.4 Is the critic under-fit? (`val_decomp.py`, `headnorm.py`, `train_ev.py`)

For each validation step I split the token-pooled residual into the part from
trajectory-mean error and the part from within-trajectory prediction movement, and
computed the optimal affine rescale of the trajectory means (token-weighted):

| run, step, domain | EV     | EV if every token used its trajectory mean | between-traj SD of predictions | slope needed | EV after affine recalibration (in-sample, 2 params) | AUC  |
| ----------------- | ------ | ------------------------------------------ | ------------------------------ | ------------ | --------------------------------------------------- | ---- |
| r499, 94, Airline | 0.025  | 0.028                                      | 0.017                          | 11.9         | 0.175                                               | 0.88 |
| r499, 94, Retail  | -0.001 | 0.002                                      | 0.007                          | 3.2          | 0.004                                               | 0.48 |
| r510, 60, Airline | 0.168  | 0.200                                      | 0.114                          | 2.1          | 0.274                                               | 0.83 |
| r510, 60, Retail  | 0.028  | 0.039                                      | 0.021                          | 8.1          | 0.167                                               | 0.60 |
| r479, 94, Airline | 0.038  | 0.076                                      | 0.033                          | 8.8          | 0.352                                               | 0.86 |
| r479, 94, Retail  | -0.078 | -0.001                                     | 0.027                          | 0.44         | 0.001                                               | 0.66 |

(The recalibrated Airline values are in-sample fits on 20 trajectories; under a
no-signal null the expected in-sample R^2 is about 1/(n-1) = 0.05, so subtract about
that. The Retail recalibrated r510 0.167 is not trustworthy: AUC 0.60 on 9 failures.)

Evidence for under-fit rather than no-signal on Airline:

- Prediction amplitude: r499 predictions span 0.72-0.96 at the token level in the
  step-50 training batch (`train-metrics/step-000050.json`, `new_value/min,max`),
  validation between-trajectory SD 0.01-0.02; the value-head weight norm is 0.0007 ->
  0.0013 -> 0.0015 (steps 5 / 30 / 94), max component 1e-4 (`headnorm.txt`); r510's is
  0.0024 (step 65), where Airline validation EV is larger. Backbone MLP weights move
  only 0.3-0.5% relative to the pretrained weights in 94 steps (layers 15 and 31,
  `headnorm.txt`).
- Train EV on fresh rollouts: r499 0.038 (epoch 1) and 0.036 (epoch 2), r510 0.036 /
  0.120 / 0.171 / 0.193 (epochs 1-4; epochs 2+ revisit tasks with fresh rollouts, so
  part of this is task-identity memorisation). Validation Airline EV for r510 sits in
  the same range; the train-validation gap is not the bottleneck.
- Level handling differs by data mix: validation bias (token-pooled) is +0.14 to +0.35
  on Airline for r499 at steps 0-94 but -0.10 to +0.09 for r510 (`bias.py`), and r510
  also has the larger amplitude (|w| 0.0024 vs 0.0015) and higher Airline EV; this is
  consistent with the mixed set (label mean 0.78, more failures per batch) carrying more
  gradient signal per step than the pack-heavy set (label mean 0.91).
- The bias is by construction frozen in practice (at most 94 x 5e-6 = 4.7e-4 of
  movement; every saved bf16 checkpoint reads 0.9102 for r499 and 0.7812 for r510, bf16
  resolution is about 0.004), as predicted in README 16.14.3. Level is handled by
  initialisation only; any level shift between pack training data and test data (r499:
  validation bias +0.24 on Airline, README) has to be undone by the 1e-4-sized weights
  and therefore never is, so the "A" problem is only partly solved. Because token-pooled
  EV removes the global bias, it does not penalise this, but SAO advantages `R - V` do.
- Loss context (`train_ev.txt`): in r499, the window-pooled training MSE exceeds the
  batch target variance by only 0.005-0.025 (e.g. 0.105 vs 0.097 at steps 1-10, 0.048 vs
  0.040 at steps 71-80), except in the windows holding the official tasks (steps 41-50:
  0.306 vs 0.173). The critic stays close to a constant predictor throughout. Guess (not
  tested): with 91% of labels equal to 1 the Adam-normalised updates mostly move the
  mean.

Evidence against "the whole signal is already learned": a trajectory-mean slope of 12
(r499) and the BoW comparison above (token-level 0.24 vs 0.025-0.17).

### 4.5 What the critic's Airline signal is

Correlation of the critic's trajectory-mean prediction with the per-task success rate
from experiment 475 (`oracle.py`): Airline 0.70 (r479), 0.57 (r499), 0.81 (r510); Retail
0.10, 0.25, 0.04. Partial correlation of prediction with the outcome given that rate:
Airline 0.01 (r479), 0.42 (r499), -0.12 (r510); Retail 0.10, -0.06, 0.23. So in two of
three runs the Airline ranking is task difficulty, which is the first-token ceiling
(0.59), not outcome-deciding events. In r499 part of the ranking exceeds task difficulty
(partial 0.42), but with 20 trajectories that value has a wide interval. Length is not
the explanation in r479 (corr(length, outcome) -0.19) but is correlated with outcome in
r510 (Airline +0.53; failures end early), so r510's Airline gain may partly be a length
cue.

## 5. Token-level view (question 3)

Per-token predictions were not stored by any run. I reconstructed them on CPU for the
**r499 final checkpoint** (`checkpoints/.../critic/epoch1epochstep46globalstep93`,
validation step 94): `tokpred.py` loads the checkpoint with the repo's own
`Qwen35TokenCriticForCausalBackbone` in bf16, runs the 60 stored validation sequences
from `fixed-validation-rollouts.pt` (no GPU; about 4-5 min per 11k-token trajectory, 3
processes x 24 threads, `nice 10`), and keeps the values on the `loss_mask` tokens
(`tokpred_r499_p{0,1,2}.npy`); `tokan.py` analyses them (`tokan_r499.txt`).

**Validity check.** Token-pooled EV recomputed from my per-token values: Airline 0.0245
(stored 0.0246), Retail -0.0014 (stored -0.0014), overall 0.0329 (stored 0.0330);
per-trajectory min/mean/max also match the stored rows (airline:24: mean 0.8206 vs
0.8203, min and max identical). The CPU path therefore reproduces the training engine's
values; the same script works for any other checkpoint (r510, intermediate steps).

Findings (r499, step 94; 20 Airline trajectories with 8 failures, 40 Retail with 5; all
AUCs have intervals of about +/-0.15-0.2):

- **Predictions barely move and what moves is mostly structural.** Mean value by
  quintile of action tokens: Airline successes 0.858 / 0.850 / 0.851 / 0.854 / 0.859
  (flat), failures 0.844 / 0.822 / 0.834 / 0.846 / 0.853; Retail successes 0.899 ->
  0.888, failures 0.896 -> 0.878 (5 trajectories). A linear trend in position explains
  17% of within-trajectory value variance.
- **The critic's Airline separation peaks in the middle of the trajectory and decays
  toward the end.** AUC of the per-decile mean prediction against the outcome, deciles
  0-9 of action tokens: 0.64, 0.61, 0.80, **0.92**, 0.83, 0.85, 0.83, 0.76, 0.79,
  **0.71**. The bag-of-words model goes the other way: 0.77 at the first message, 0.91
  on the complete trajectory (section 4.3). The change relative to the trajectory's own
  first decile (the part that reflects events rather than task identity) has AUC at most
  0.78 (deciles 1-9: 0.42, 0.70, 0.78, 0.72, 0.75, 0.71, 0.66, 0.66, 0.64). Retail
  decile AUCs scatter between 0.19 and 0.64 around chance.
- **It does react to event types, in a plausible direction but with tiny amplitude, and
  the reaction is not tied to the outcome.** Change in mean value between consecutive
  assistant turns: after a tool response -0.008 (Airline, n=139) and -0.0085 (Retail,
  n=292); after a user message +0.008 (n=70) and +0.013 (n=168); after a tool response
  that contains an error -0.025 (Retail, n=8; Airline has n=1); on a `transfer_to_human`
  call -0.018 (n=10) and -0.023 (n=6). Observation type (tool vs user) explains 12%
  (Airline) and 40% (Retail) of the turn-to-turn change; the final outcome explains 0.0%
  in both. The mean change per turn is the same for successes and failures (Airline
  -0.0027 vs -0.0027; Retail -0.0012 vs -0.0015).
- **Largest single moves are 0.05-0.09, and the biggest are false alarms.** The largest
  |change| is airline:37 (a *successful* upgrade): -0.094 on the turn after the tool
  response confirming the upgrade and +0.083 on the turn after the user confirmed it.
  `turn_events_r499.pkl` holds the top events with their observation text.

Reading: the r499 critic carries a task-difficulty level (first-token ceiling), a
position trend and a tool/user sawtooth, each with amplitude of 0.01-0.03. It has not
yet learned outcome-deciding events (wrong tool call, user giving wrong info, policy
violation): its mean change per turn is identical in successes and failures. Because the
whole movement is two orders of magnitude below the label scale, EV cannot show learning
even where ranking exists (section 4.4). For r510 and r479 the same script is needed; I
did not run them (about 1.5 h CPU each).

## 6. Cheapest reproduction and discriminating experiments (question 4)

Already done on CPU and sufficient to reproduce "Airline small positive, Retail ~0"
without any critic training: the BoW probe of section 4.3 and the permutation null of
section 3.

GPU experiments I propose (not run). All use only stored data, so API cost is 0. Assumed
hardware: 4-8 A100-80GB (the box has 8; GPUs 4-7 are in use by r510 until it finishes).
Compute estimate: mean sequence 10.4k tokens, 15.5M tokens per 1494 rollouts; training
cost about 8 x 4e9 x tokens FLOP (fwd + bwd + checkpointing) at about 100 TFLOPS
effective per A100.

Preparation (CPU, 1-2 h to write, not GPU): `scripts/tau2/offline_critic_fit.py` that
(i) reads the training jsonl (fields `input_ids`, `loss_mask`, `reward`), (ii) builds
the same rows as `train_critic._value_target_row`, (iii) trains with
`Tau2HeadFSDPPPOCritic` (same engine, no SGLang, no rollout), (iv) evaluates on the
three `fixed-validation-rollouts.pt` (180 trajectories) with `evaluate_tau2_critic`.
Hold out the 60 validation tasks (they are test tasks, never in training).

| ID  | Experiment                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | What it discriminates                                                                                                                                                                                                                                                                                                          | Command sketch                                                                                                                                                                     | Cost                                                                  |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| E1  | Frozen-feature linear probe. One forward pass of the base Qwen3.5-4B (and optionally the r510 final critic backbone) over the r510 train rollouts (1280 when complete, 14M tokens) + 180 validation trajectories; save final-layer hidden states at every assistant-turn last token plus ~16 random action tokens per trajectory (fp16, a few hundred MB); fit logistic/ridge heads on CPU with task-grouped CV; report token-weighted EV and AUC per domain, Airline/Retail separately, with the validation bootstrap of section 3. | If probe Airline EV >= 0.2 and Retail > 0.05: features hold the signal, the critic fails because of optimisation. If probe ~ BoW or below: base features do not carry it.                                                                                                                                                      | `CUDA_VISIBLE_DEVICES=4 python scripts/tau2/extract_hidden.py --ckpt <base> --rollouts <r510 rollout/0> --val <fixed-validation-rollouts.pt> --out feats.pt` (to write; ~60 lines) | 0.3-0.7 GPU-h forward (14.6M tokens x 8e9 FLOP = 1.2e17 FLOP), no API |
| E2  | Offline replay of r510 data at the original settings: lr 5e-6 body and head, 4 epochs, batch 16.                                                                                                                                                                                                                                                                                                                                                                                                                                     | Is the poor result reproduced offline (val Airline EV 0.02-0.18, Retail ~0)? Makes every later arm comparable.                                                                                                                                                                                                                 | `scripts/tau2/run.sh data offline-critic-fit NAME critic.optimizer.lr=5e-6 critic_head_bias_init=0.78` (entry point does not exist yet; it is the script described above)          | 4 epochs x 14M tokens: about 5 GPU-h (1.3 h on 4 GPUs)                |
| E3  | Same replay, head lr 1e-4 with 10-step warmup (body lr 5e-6).                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Is the amplitude (head lr) the limit? Prediction: Airline EV rises toward the recalibrated value 0.2-0.3, Retail stays \< 0.07.                                                                                                                                                                                                | `... critic_head_lr=1e-4 critic.optimizer.warmup_steps=10`                                                                                                                         | 5 GPU-h                                                               |
| E4  | Same replay, body lr 2e-5, head lr 1e-4.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Is the backbone drift (now 0.5%) the limit?                                                                                                                                                                                                                                                                                    | `... critic.optimizer.lr=2e-5 critic_head_lr=1e-4`                                                                                                                                 | 5 GPU-h                                                               |
| E5  | Planted-signal control (synthetic known value): same replay with the labels replaced by a trajectory-level function the critic can see, e.g. `y = 1[number of tool_response messages containing "Error" >= 1]` or `y = 1[first-user-message contains "cancel"]`, both computed on CPU from the stored text.                                                                                                                                                                                                                          | If E2 settings reach EV >= 0.5 on planted labels in 1-2 epochs, the pipeline (loss, lr, steps, masks) can learn trajectory-level signal at this budget and real-label failure is about signal/noise; if it stays near 0, there is an optimisation/pipeline problem. This is the cleanest "optimisation vs no signal" splitter. | `... --relabel planted_tool_error`                                                                                                                                                 | 1-2 epochs on the 400 shortest rollouts: under 0.5 GPU-h              |
| E6  | Sigmoid + BCE head (README 16.14.3 backup option) at E3 settings.                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Gradient scale and amplitude with a logit head.                                                                                                                                                                                                                                                                                | needs ~30 lines in the engine subclass                                                                                                                                             | 5 GPU-h                                                               |

Recommended order: E1 and E5 first (about 1 GPU-h together, they decide the story), then
E2/E3 (about 10 GPU-h), then E4/E6 only if E3 shows amplitude is not the whole limit.
Total about 10-12 GPU-h for the decisive subset, about 25 GPU-h for all.

One more cheap, API-free item: to get per-token predictions for any checkpoint without
GPU, run `tokpred.py` on CPU (bf16, about 4 min per 11k-token trajectory with 24 threads
under load; 60 validation trajectories = about 1.5 h on 3 x 24 threads). It reproduces
the engine's per-trajectory min/mean/max to 3-4 digits (section 5). The engine's own
stored report cannot give per-token values, so for a regular per-step token trace add a
`values` field to `_metric_row` (one list per trajectory; 190k floats per report, 0.8
MB) -- that is a code change I did not make.

## 7. Ranked fixes (question 5)

| Rank | Fix                                                                                                                                                                                                                                  | Expected effect on validation EV                                                                                          | Cost                                                                                                                      | Why                                                                                                                                                                                                |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | Raise the value-head learning rate (E3: 1e-4 with warmup) and/or backbone lr 2e-5                                                                                                                                                    | Airline +0.1 to +0.2 (towards the recalibrated 0.17-0.27); Retail ~0 to +0.03                                             | config only; one 5 h online run, or 5 GPU-h offline first                                                                 | slope needed is 2-12, head norm 0.0015, no train-val gap. Risk: level jumps seen in smoke 498 (0.78 -> 0.63 after 1 step at 1e-4); mitigate with warmup, or freeze bias and use head lr 3e-5 first |
| 2    | Change the validation protocol: report calibrated EV (affine), AUC with CI and a permutation p-value per domain, average the last 3-4 validation steps, validate on >= 3 draws per task (180 trajectories, about 2M DeepSeek tokens) | 0 (measurement), but removes the +/-0.05 to +/-0.1 selection noise                                                        | about 2M API tokens per run; zero GPU                                                                                     | Section 3: single-run EV differences below 0.1 are not distinguishable on Airline                                                                                                                  |
| 3    | Sigmoid/BCE head or per-batch target centring                                                                                                                                                                                        | Airline +0.05 to +0.15 if amplitude-limited; unclear                                                                      | ~30 lines + 5 GPU-h                                                                                                       | the logit head makes the amplitude grow geometrically in the weights instead of linearly; guess, not tested                                                                                        |
| 4    | More Retail failures: oversample official Retail train tasks with mixed outcomes, keep the 122 informative tasks, add failure-enriched Retail tasks (README 16.14.5 shows 26 informative Retail pack tasks only)                     | Retail pseudo-token ceiling 0.06; realistic +0.02                                                                         | 1 online run (DeepSeek ~14M prompt tokens, ~\$ small at night rate)                                                       | 29 Retail failures in 94 steps is the binding constraint                                                                                                                                           |
| 5    | Lower-noise labels: Monte-Carlo average of several continuations from the same prefix (k >= 4 branches) as target                                                                                                                    | Retail could approach its within-task variance floor (ceiling 0.24 at first token is only a lower bound for later tokens) | needs an environment/simulator fork at arbitrary prefixes (not available in tau2 user simulator today); 4-8x rollout cost | removes the 52%-of-tasks-mixed label noise; large engineering cost                                                                                                                                 |
| 6    | Unfreeze attention / LoRA, longer training                                                                                                                                                                                           | unknown; probably small until 1 is fixed                                                                                  | 1 online run                                                                                                              | do after E3/E4                                                                                                                                                                                     |

Not recommended: more of the same easy pack data. 70% of r479/r499 episodes are tasks
that are 100% correct (README 16.14.5); they teach the level only, and the level is
already set by initialisation.

## 8. What extra data would settle the open points

- Per-token (or per-turn) critic values at each validation step: add a `values` list to
  `_metric_row` in `scripts/tau2/train_critic.py` (about 190k floats per report).
  Without it, only checkpoints plus CPU re-evaluation (section 5) give the token view.
- Three or more outcome draws per validation task, collected once and reused (about 2M
  DeepSeek tokens per 60 x 3 draws, per README 16.14.4 usage). The 7-draw ceiling above
  used four draws from experiment 475 plus three draws from the three runs; a dedicated
  60 x 4-8 set with stored token ids would give a proper ceiling for token-level EV, not
  only the first-token one.
- Token ids for the ceiling experiments 473-476, 481, 482: only `episodes.jsonl`
  (outcomes) is stored, so no prefix-conditional analysis is possible on them.
- Prefix-conditional outcomes: for a handful of failing trajectories, re-roll the
  conversation from several prefixes with the same user simulator to measure how much
  the success probability really changes at the decisive turn. This is the only direct
  way to separate "event-decided" from "user-noise-decided" outcomes; the tau2 user
  simulator has no prefix-fork API today (guess: needs a conversation-replay wrapper),
  so I did not estimate its cost.
- More Retail failures (currently 29 in 674 training trajectories for r479/r499, 46 in
  442 for r510).

## 9. Limits of this study

- No per-token critic predictions were stored by the runs; section 5 uses a CPU
  re-evaluation of the r499 final checkpoint only (r510 and r479 not done). With 8
  Airline failures, token-level AUC statements have intervals of +/-0.15-0.2.
- I did not run any critic training, so claims about "optimisation limit" rest on
  amplitude, head-norm, slope and train-vs-val evidence, not on an intervention; E3/E5
  are the interventions.
- BoW probe uses all three runs' rollouts (same actor, fixed-v2 for r479 vs fixed-v3 for
  r499/r510 data); train and validation tasks are disjoint (validation = official test).
- r510 was still training; its numbers are for steps \<= 65.
- The ceiling statistics use outcomes of 475 (fixed-v2 data) while r499/r510 validate on
  fixed-v3.
