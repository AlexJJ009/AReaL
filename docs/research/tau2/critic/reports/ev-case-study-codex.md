# Why tau2 critic-only training barely reaches positive EV

## Conclusion {#progress}

**The evidence supports weak learned within-domain prediction, limited and shifted
supervision, and substantial validation uncertainty—not a finding that tau2 has no
learnable value signal.** Airline has clear task-level signal, and 510 learns more of it
on its saved validation draw; Retail has weaker task-level signal and much poorer
failure coverage. Existing runs cannot separate insufficient optimization from
representation/generalization limits.

The decisive results are:

- **510 at step 60:** Airline EV **0.168, 95% trajectory/task-bootstrap interval
  \[−0.039, 0.349\]**; Retail **0.028 \[−0.037, 0.064\]**. Its gain over 499 is not
  statistically resolved. Retail's positive value disappears if one long failed task is
  removed. 510 was still running; the training audit snapshot reaches step 63, not final
  step 80.
- **Constant bias is not the EV explanation.** The implementation already subtracts mean
  residual. Mean-init helped remove random-head token scatter, but calibration alone
  cannot recover the missing ranking signal. Even test-fitted affine token calibration
  would only raise 510 to about **0.183 Airline / 0.032 Retail**.
- **The quoted 0.61/0.20 ceiling is task-only, not a bound on all-prefix EV.** Every
  task in the repeated bank has varying initial user text, so it is not established as a
  strict bound on the actual first response prefix either. Within-task reward variance
  also does not isolate user noise from actor/judge/execution noise.
- **The cheapest discriminating next experiment needs no new API data:** re-forward a
  completed checkpoint to reproduce the poor metric, extract frozen features from stored
  rollouts once, then compare ridge with longer/higher-LR head fitting on CPU using
  task-disjoint dev selection. Prepared scripts and exact commands are below; the
  primary feature pass is estimated at **1–3 GPU-hours, \$0 API**. GPU execution remains
  a proposal.

## Goal and boundaries {#goal}

Answer the five questions in the critic EV case-study issue using existing artifacts,
CPU analysis and source inspection. All scripts and derived data stay in this directory.
The repository and run directories are read-only; no GPU job, pueue task, API call or
commit is authorized. The independent Claude report has not been read.

## Measurement and uncertainty: questions 1–2 {#evidence}

### What EV actually measures

For each trajectory $i$, the report stores the action-prefix token count $n_i$, error
sum $e_i$, squared-error sum $s_i$, and target sums. Let $N=\sum_i n_i$,
$b=\sum_i e_i/N$, and $T=\sum_i n_iR_i^2/N-(\sum_i n_iR_i/N)^2$. The reported quantity
is

$$
\mathrm{EV}=1-\frac{\sum_i s_i/N-b^2}{T}.
$$

This is **token-weighted**, on the shifted assistant-action loss mask, with the terminal
reward repeated at every evaluated prefix. It is neither equal-weight trajectory EV nor
$1-\mathrm{MSE}/T$. A constant shift of every prediction changes bias and MSE but leaves
EV unchanged. Correcting bias still matters to SAO advantages; it cannot by itself
explain or repair low EV. In particular, README 16.14.4's statement that Airline EV
mainly lacks level calibration overstates what its evidence establishes.

`analyze_validation.py` recovered every Airline/Retail EV in 84 saved reports to
absolute tolerance $10^{-10}$. Each bootstrap draw samples complete trajectories with
replacement within a domain, preserves all their token statistics, and recomputes the
ratio. There is one trajectory per task, so trajectory and task cluster bootstrap
coincide here. We used 20,000 percentile draws, seed 20261014; zero-target-variance
draws are undefined and excluded, with their fraction saved. Resampling individual
tokens would be invalid because thousands of tokens share one terminal label.

### Final saved points, not selected best checkpoints

| Run / step    | Airline failures / 20 | Airline EV \[95% cluster interval\] | Retail failures / 40 | Retail EV \[95% cluster interval\] |
| ------------- | --------------------: | ----------------------------------: | -------------------: | ---------------------------------: |
| previous / 24 |                     5 |           −0.594 \[−1.427, −0.499\] |                    5 |          −0.920 \[−4.424, −0.462\] |
| 479 / 94      |                    10 |             0.038 \[−0.022, 0.084\] |                    5 |          −0.078 \[−0.307, −0.032\] |
| 480 / 94      |                    16 |            −0.066 \[−0.262, 0.011\] |                   21 |           −0.018 \[−0.050, 0.005\] |
| 499 / 94      |                     8 |              0.025 \[0.002, 0.070\] |                    5 |           −0.001 \[−0.024, 0.007\] |
| 500 / 20      |                     9 |             0.016 \[−0.005, 0.036\] |                    6 |            0.019 \[−0.011, 0.034\] |
| 510 / 60      |                     8 |             0.168 \[−0.039, 0.349\] |                    9 |            0.028 \[−0.037, 0.064\] |

Sources: saved validation reports, all per-step results, bootstrap results and source
hashes. Run 500 failed after this checkpoint; 510 is still training. Earlier
official/Telecom and Instruct runs have different evaluation protocols or actors, so
this table is descriptive, not a controlled treatment comparison.

The standard error at 510/60 is approximately **0.100 Airline and 0.028 Retail**; at
499/94 it is 0.017 and 0.009. Thus there is no universal “20-task EV error bar”: a
near-constant predictor has EV near zero on almost every resample and can have a
deceptively tight interval. Precision around zero does not establish that the domain has
no learnable signal. At 499, Airline has a small positive pointwise interval; it is not
evidence of a practically strong critic or a result corrected for inspecting many
checkpoints.

### Which differences survive sampling uncertainty?

Within a run, every checkpoint is evaluated on identical trajectories, so differences
use **paired** bootstrap draws.

| Contrast         | Domain  | EV change \[95% paired interval\] | Interpretation                                                                                     |
| ---------------- | ------- | --------------------------------: | -------------------------------------------------------------------------------------------------- |
| 479 step 90 → 94 | Airline |           +0.112 \[0.066, 0.198\] | The jump is a real change on this fixed validation draw; it is not newly sampled validation noise. |
| 479 step 90 → 94 | Retail  |           +0.272 \[0.167, 0.998\] | Clear reduction of negative EV; final EV remains negative.                                         |
| 510 step 25 → 50 | Airline |          +0.078 \[0.0005, 0.144\] | Borderline pointwise evidence; not robust to checkpoint-selection correction.                      |
| 510 step 50 → 60 | Airline |          +0.031 \[−0.070, 0.129\] | Generalized improvement is unresolved.                                                             |
| 510 step 50 → 60 | Retail  |           +0.029 \[0.013, 0.058\] | Pointwise improvement over the same trajectories; final positivity remains unresolved.             |

Across runs, the rollouts differ even when task identities agree. Independent trajectory
resampling gives 499/94 → 510/60 changes of Airline +0.144 \[−0.070, 0.320\] and Retail
+0.029 \[−0.035, 0.073\]. Resampling matched task IDs jointly gives almost the same
conclusion: Airline \[−0.064, 0.312\], Retail \[−0.036, 0.075\]. See extra analysis.
Neither establishes a stable gain from the mix treatment; initialization, training data,
order, training randomness, and validation draws are not separately randomized. 479
versus 499 also changes fixed-v2 to fixed-v3.

These intervals describe task/trajectory sampling conditional on the observed model and
empirical task population. They do **not** measure training-seed variability,
conditional rollout randomness on the exact fixed 60 tasks, or guarantee nominal
coverage with only five failures. Repeated independent rollouts on the same tasks,
scored by every compared checkpoint, are needed for the second uncertainty component.
Use a common existing rollout bank first, keep task clusters intact, and reserve
genuinely new held-out tasks for a final generalization claim. No new API collection is
needed to make the present cross-checkpoint comparison fairer.

### Token variation versus trajectory ranking

Because $R_i$ is constant within a trajectory, the residual variance decomposes exactly
into variation of trajectory-mean residuals plus within-trajectory prediction variation:

$$
\mathrm{EV}_{token}=\mathrm{EV}_{trajectory\ mean,\ token\ weighted}
-\frac{\sum_i n_i\operatorname{Var}_t(V_{it})/N}{T}.
$$

| Run / step    | Domain  | EV of trajectory means, token weighted | Within-trajectory EV penalty | Actual token EV |
| ------------- | ------- | -------------------------------------: | ---------------------------: | --------------: |
| previous / 24 | Airline |                                 −0.013 |                        0.581 |          −0.594 |
| 479 / 94      | Airline |                                  0.076 |                        0.038 |           0.038 |
| 479 / 94      | Retail  |                                 −0.001 |                        0.078 |          −0.078 |
| 499 / 94      | Airline |                                  0.028 |                        0.004 |           0.025 |
| 499 / 94      | Retail  |                                  0.002 |                        0.003 |          −0.001 |
| 510 / 60      | Airline |                                  0.200 |                        0.031 |           0.168 |
| 510 / 60      | Retail  |                                  0.039 |                        0.011 |           0.028 |

This identifies a substantial early failure mechanism: random-head token variation. At
499, that variation is largely removed, yet trajectory means barely separate outcomes.
At 510, Airline trajectory ranking is stronger; Retail remains weak. The decomposition
is descriptive: replacing a causal value function with the eventual trajectory mean uses
future information and is **not** a valid online critic fix. Genuine Bayesian value
updates can vary along a trajectory too; the penalty alone does not prove those changes
are wrong.

Even an optimally fitted affine transformation of **all token predictions on the same
test set** yields only EV 0.038/0.00009 for 479 Airline/Retail, 0.044/0.00037 for 499,
and 0.183/0.032 for 510. These are optimistic in-sample diagnostics, not deployable
calibrated results. They show that offset/scale calibration alone cannot approach the
quoted Airline task-identity ceiling. Correlation of trajectory-mean prediction with
estimated task difficulty is a different metric and does not overturn this result.

### What the stored data says about learnable signal

The 473–476 ceiling rollouts were found under the artifact root's
`user-sim-20261012/api-user/`. Run 475 is `deepseek-test-u0p1/episodes.jsonl`. Runs 473
(`deepseek-base-u0p0`) and 476 (`doubao-base-u0p0`) contain **164**, not 60, tasks; we
filtered to 475's 60 fixed-test identities before reproducing README's comparison.
data_ceiling_check.py reproduces the task-group estimates below using the repository
ceiling implementation.

| Repeated-rollout condition       | Airline task-only ceiling \[95%\] | Retail task-only ceiling \[95%\] |
| -------------------------------- | --------------------------------: | -------------------------------: |
| 473, DeepSeek actor T0 / user T0 |            0.550 \[0.272, 0.792\] |          0.105 \[−0.046, 0.245\] |
| 474, DeepSeek actor T0 / user T1 |            0.566 \[0.306, 0.798\] |          0.172 \[−0.032, 0.403\] |
| 475, DeepSeek actor T1 / user T0 |            0.614 \[0.341, 0.845\] |           0.204 \[0.025, 0.352\] |
| 476, Doubao actor T0 / user T0   |            0.475 \[0.167, 0.722\] |           0.171 \[0.004, 0.333\] |

These are estimates of $\operatorname{Var}(\mathbb E[R\mid task])/\operatorname{Var}(R)$
under uniform trajectory/task weighting, **not** the ceiling of the report's
length-weighted all-prefix value prediction. They are not even established as strict
bounds on the first actual response token: inspection of the 475 dialogues found varying
initial user messages for **all 20 Airline and all 40 Retail tasks**, with four distinct
initial messages for 19 and 39 tasks respectively. The model can already observe this
varying information. Whether it predicts reward is untested, but the assumption of an
identical initial observed state is false.

475 has Airline 7/20 tasks failing all four trials, 6 succeeding all four, and 7 mixed;
Retail has 24/40 succeeding all four, 16 mixed, and none failing all four. A CPU
diagnostic estimates each task's rate from the **other three trials**, then predicts the
held-out fourth: EV is **0.486 Airline and −0.061 Retail**. Using the same four outcomes
to both estimate and score produces misleadingly optimistic 0.711 and 0.403. Sources:
data_ceiling_fixed_test.json. The held-out-repeat check proves exploitable task-level
structure in this Airline sample. In Retail, rate-estimation noise from three labels can
consume the modest oracle advantage; it does not prove that semantic sharing across
tasks or later events is useless.

The within-task share is about 0.096/0.249 = 39% in Airline and 0.108/0.136 = 80% in
Retail under 475. Calling all of it “user noise” is not identified by these experiments:
actor sampling, user generation, judge decisions and execution nondeterminism all remain
possible contributors. Greedy actor runs still vary, but their intervals overlap the T1
run, and DeepSeek's default thinking path makes the configured user temperature 0 an
unreliable determinism claim. The four repeats also leave substantial uncertainty in
individual task probabilities.

### Training exposure and optimizer evidence

The 479 training rollouts are present as replayable JSONL, including token IDs, masks,
turns and readable segments:

`tau2-training-20261012-critic-data/critic-qwen35-formal-v1/logs/root/tau2-critic-data/critic-qwen35-formal-v1/rollout/0/*.jsonl`.

The exact episode/path inventory is data_train_labels_479.json; the corresponding
other-run inventories and data_summary.json contain the checks below. 510 training is
frozen at the separately recorded **step-63 / 1008-generated-episode** readback;
validation comparisons stay at step 60. Generated records are not automatically accepted
optimizer inputs: the executor dumps before acceptance, and 500 demonstrably has 365
generated episodes for only 22 updates. Completed 479/499 counts match their scheduled
1494 episodes, but exact accepted-record reconstruction would still be needed for a
bitwise training replay.

| Run          | Airline episodes / failures | Retail episodes / failures | All-success batches | Successful optimizer updates |
| ------------ | --------------------------: | -------------------------: | ------------------: | ---------------------------: |
| 479          |                   820 / 110 |                   674 / 29 |               29/94 |                           94 |
| 480          |                   820 / 372 |                  674 / 195 |                0/94 |                           94 |
| 499          |                    820 / 94 |                   674 / 29 |               30/94 |                           94 |
| 510 snapshot |                   572 / 147 |                   436 / 45 |                2/63 |                           63 |

In 479, **Retail pack has only 11 failures across 9 distinct tasks** out of 536
episodes; official Retail contributes 18 failures across 15 tasks. Token-weighted
success is 0.953 for Retail overall. Airline pack succeeds on 0.904 of episodes but only
0.803 of action tokens because failures are longer; official Airline succeeds on 0.383
of episodes. Thus the label imbalance and distribution mismatch are real, but the
episode-level “almost all ones” description exaggerates Airline's token-objective
imbalance.

499's train token-success means are Airline 0.799 and Retail 0.953; 510's are 0.630 and
0.866. The corresponding fixed validation token means are 499 0.612/0.839 and 510
0.713/0.784. These are the appropriate quantities for the token-MSE comparison; episode
success and token success should not be interchanged. Mix reduces Retail's imbalance and
changes Airline's weighted target distribution substantially. It also changes order and
repeats, so it is not an isolated class-balance intervention.

480 has **567 failures**, no all-success batches, and still poor within-domain EV. Its
different actor/backbone prevents a controlled comparison, but it rules out failure
scarcity as a universal explanation of every run. For 510, Airline has 75 distinct
failed tasks and Retail 34 in the snapshot; having more failures has not yet established
reliable Retail generalization.

The optimizer actually ran: all read update-success flags are 1 and counters progress
continuously. There is one optimizer update per batch of 16 episodes, not one per token,
microbatch or DP rank. The config uses plain token-mean MSE, $\gamma=\lambda=1$, no KL,
no value clipping, constant LR $5\times10^{-6}$, float32 optimizer weights, and clipping
at 1. Attention modules are frozen, **not the whole backbone**: 510's `critic.log:215`
records 1,304,961,536 frozen and 2,900,792,321 trainable parameters. Those 2.9B
parameters receive only 63–94 global updates here. Zero head initialization suppresses
backbone gradient on the first update only; subsequent updates can train the unfrozen
body.

510's last-20-batch train EV averages 0.116, versus 499's 0.029; 16 versus 12 of those
20 batches have positive EV. 510 first-20 versus last-20 batch MSE is 0.189 → 0.177.
These are **different batches evaluated before their update**, not a fixed-training-set
learning curve, and pooled EV includes domain differences. They support some learning
but cannot establish convergence, per-domain training fit, or an optimization
bottleneck. High unclipped gradient norms (510: 6.50–63.26) do not prove updates vanish
under Adam. Neither bf16 rounding nor skipped updates is supported as the cause.

**Inference:** the leading Airline explanation is limited supervised exposure combined
with task/distribution mismatch and evaluation uncertainty. **Unresolved hypotheses:**
LR too small, insufficient epochs, attention freezing, unsuitable prefix weighting, and
representation limits. Existing runs do not independently vary these factors. Retail
additionally has a demonstrably weaker task-rate signal and sparse failure coverage.
**The evidence does not justify “no learnable signal,” nor identify one dominant
optimization defect.**

### Token/event view: question 3

Ordered values and hidden states were not stored. The fixed PT contains inputs, masks,
turns and rewards; validation JSON contains only moments and extrema. Therefore no CPU
analysis of these files can honestly say *at which token* a value drops, or measure a
pre/post-event value change. We can nevertheless establish that some failed trajectories
retain high values at **every scored prefix**:

| 510/60 failed trajectory | Action-prefix tokens | Minimum / mean / maximum V |
| ------------------------ | -------------------: | -------------------------: |
| Airline task 18          |                3,554 |      0.691 / 0.866 / 0.988 |
| Airline task 45          |                3,109 |      0.496 / 0.579 / 0.918 |
| Retail task 45           |                1,440 |      0.852 / 0.928 / 0.988 |
| Retail task 101          |                8,738 |      0.719 / 0.865 / 0.984 |

inspect_events.py decoded these cached trajectories on CPU, preserving token spans and
turn boundaries; event_manifest.json binds them to the PT hash. Two concrete cases guide
a re-forward:

- **Retail 45:** after the user confirms an exchange and asks about a gift-card refund
  (tokens 8128–8202), the assistant discusses PayPal and ends by asking for confirmation
  again (action span 8203–8625). No exchange tool call occurs in the generated dialogue.
  The recorded reward is 0, yet every evaluated prefix has $V\ge0.8516$. This is a
  verified incomplete-action/high-value pairing; whether the precise reward failure was
  that omission or another evaluator condition remains unverified. Source: decoded
  trajectory, lines 340–378.
- **Retail 101:** the tool response identifies item 6477915553 as Sneakers, while the
  user and final assistant discussion call it a speaker; the last user also asks whether
  all three purifier constraints were met. These are visible consistency checks, not
  proof of the evaluator's reason for reward 0. Candidate event positions are the
  confirmation-sensitive exchange call and the last user span 18187–18250, followed by
  action span 18251–18518. Source: decoded trajectory, lines 939–1035. Whether V
  responds to these events cannot be recovered from extrema.

Airline 45 contains a transfer after the user repeatedly refuses one; Airline 18
completes several cabin downgrades yet receives 0. These are useful adverse cases
because apparent tool success or policy language is not itself ground-truth success. The
current data does not expose a verified event-level causal reward annotation.

Retail task 101 supplies **8.25% of all Retail action tokens and 38.2% of Retail failure
tokens**. Removing it changes EV from 0.0280 to −0.00753. Airline leave-one-task-out EV
ranges from 0.110 (without failed task 45) to 0.240 (without failed task 18). Thus the
longer Retail trace materially drives the positive token result; it is not evidence of
broad event recognition.

The cheapest way to answer the missing temporal question is one offline forward of the
existing checkpoint on these 60 cached trajectories. Save every masked value, align it
to the prefix **before** the next action token, and report first/last, action-progress
quartiles and pre/post-event values separately by outcome and domain. An event is
available only after its text enters the prefix. Match recomputed mean/SSE to the
original report before interpreting the curves. No rollout or judge API is needed.

## Cheapest discriminating experiment: question 4 {#decisions}

**Recommendation:** use completed **499** as the reproducible poor-result case, re-score
its fixed validation cache once, then extract frozen features from its existing train
rollouts and fit heads entirely on CPU. This avoids waiting for or touching live 510,
fixes labels across optimization comparisons, and removes rollout/API cost. 499's
near-zero Retail EV after corrected initialization is a cleaner starting point than the
old random-head failure.

Three runnable scripts are included: offline_probe.py, head_adam.py, and
verify_values.py. They do not use the online trainer or contact a simulator. The
existing `train_critic.py` is **not** an offline replay entrypoint; its cached
validation does not prevent new training rollouts, and its validator requires attention
freezing. Do not queue that online command as an “API-free offline test.”

### Prepared experiment and decision rules

1. **Reproduce the poor score:** forward the saved 499/94 checkpoint over its 60 fixed
   trajectories and save every scored value. Compare masks, targets, trajectory means,
   MSE and EV with the saved report. The proposed numerical tolerance is 0.01 absolute
   for EV/MSE/trajectory mean; this is an engineering guard, not an empirically
   established export-error bound. If it fails, stop interpretation and resolve loader,
   mask, precision or kernel differences first. Then inspect the 510 event examples with
   the separately specified step-60 checkpoint.
1. **Test accessible signal without SGD:** use the original frozen base model to extract
   32 uniformly sampled action-prefix hidden states per trajectory, with weight $n_i/32$
   (or $n_i/k_i$ for shorter samples). This approximates the token objective without
   saving all token features. Fit ridge using train-only feature standardization; choose
   regularization on a domain-stratified, task-disjoint 20% dev split. All repeats of a
   task stay on one side. Report per-domain test EV, not just pooled EV. Run a
   domain-preserving episode-label permutation control. Repeat per domain if pooled
   ridge is driven mainly by domain identity.
1. **Test head update budget and LR on identical features:** fit a zero-weight,
   mean-initialized raw-feature head with Adam, clip 1, batch 16 complete episodes, LR
   $5\times10^{-6}$ versus $5\times10^{-5}$, for 2/10/20 epochs. The script records
   fixed train/dev curves, update count and the dev-selected checkpoint, then evaluates
   fixed test once. Ridge bypasses head SGD conditioning; raw-feature Adam preserves the
   original head's feature scale. It does **not** reproduce evolving MLP/backbone
   training.
1. **Only if necessary, extract features from the trained 499 backbone too:** good ridge
   EV on base or trained features establishes available linear signal that the saved
   score failed to exploit. If both ridge and Adam generalize poorly despite fitting a
   nonconstant-label training subset, investigate coverage/shift/representation. If a
   high-dimensional head cannot fit even a small fixed mixed-label subset, inspect the
   feature/mask/target pipeline and conditioning before buying more online data.

The ridge script includes true-label and episode-permuted-label same-input memorization
controls; these require nonconstant labels. Memorizing shuffled labels demonstrates
capacity, **not** meaningful generalization. A failed frozen linear probe does not
establish absence of nonlinear or later-prefix signal. A positive held-out probe with a
negative permutation control is the useful discrimination. A trained-backbone probe can
distinguish an inadequate final score from a representation change, but cannot by itself
assign causality to frozen attention.

For a claim of improvement, use the same saved test trajectories for every candidate,
bootstrap **paired task/episode clusters**, and retain bias/MSE next to EV. The NPZ
includes task, episode and sampled prefix positions for this. Do not select alpha, LR,
epoch, domain variant or checkpoint on the official test; make those choices on dev. The
old official test has already been inspected extensively, so confirmatory model
selection ultimately needs separate held-out tasks. Proposed success criteria: numerical
reproduction first; then per-domain paired improvement interval above zero and a
positive per-domain EV interval on an independent assessment, without worse calibration.
Current data may be too small to certify a modest improvement even when its point
estimate rises.

### Exact commands to queue later — not executed in this investigation

All destinations remain under this case-study directory. The selected GPU number below
is a **future allocation placeholder**; it must refer to a free allocated GPU when
queued, not a card occupied by 510. These commands do not start SGLang or call an API.

```bash
SRC=/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math
CASE=/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-critic-ev-case-study/codex
ART=/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math
RUN=$ART/tau2-training-20261013-critic-head/critic-head-qwen35-v2
MODEL=/data_storage/yl_test/lgx/data-1/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
CKPT=$RUN/checkpoints/root/tau2-critic-head/critic-head-qwen35-v2/critic/epoch1epochstep46globalstep93
TRAIN=$RUN/logs/root/tau2-critic-head/critic-head-qwen35-v2/rollout/0
TEST=$RUN/tau2-critic-fit-evidence/fixed-validation-rollouts.pt
export PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# GPU loader smoke, then exact-mask saved-checkpoint value reproduction.
CUDA_VISIBLE_DEVICES=0 "$SRC/.venv/bin/python" "$CASE/offline_probe.py" extract --model "$CKPT" --rows "$TEST" --limit 2 --out "$CASE/499-smoke.npz"
CUDA_VISIBLE_DEVICES=0 "$SRC/.venv/bin/python" "$CASE/offline_probe.py" extract --model "$CKPT" --rows "$TEST" --out "$CASE/499-trained-test.npz"
CUDA_VISIBLE_DEVICES= "$SRC/.venv/bin/python" "$CASE/verify_values.py" --values "$CASE/499-trained-test.values.json" --report "$CASE/validation_snapshot/499/validation-step-000094.json" --out "$CASE/499-reproduction.json"

# One frozen-base forward per stored trajectory; no new rollout.
CUDA_VISIBLE_DEVICES=0 "$SRC/.venv/bin/python" "$CASE/offline_probe.py" extract --model "$MODEL" --rows "$TRAIN" --out "$CASE/499-base-train.npz"
CUDA_VISIBLE_DEVICES=0 "$SRC/.venv/bin/python" "$CASE/offline_probe.py" extract --model "$MODEL" --rows "$TEST" --out "$CASE/499-base-test.npz"

# CPU heads. Fit selects regularization on dev; final test is reporting only.
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=8 "$SRC/.venv/bin/python" "$CASE/offline_probe.py" fit --train "$CASE/499-base-train.npz" --test "$CASE/499-base-test.npz" --out "$CASE/499-ridge.json"
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=8 "$SRC/.venv/bin/python" "$CASE/offline_probe.py" fit --train "$CASE/499-base-train.npz" --test "$CASE/499-base-test.npz" --permute --out "$CASE/499-ridge-permuted.json"
for domain in airline retail; do
  CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=8 "$SRC/.venv/bin/python" "$CASE/offline_probe.py" fit --train "$CASE/499-base-train.npz" --test "$CASE/499-base-test.npz" --domain "$domain" --out "$CASE/499-ridge-$domain.json"
done
for lr in 5e-6 5e-5; do
  for epochs in 2 10 20; do
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 "$SRC/.venv/bin/python" "$CASE/head_adam.py" --train "$CASE/499-base-train.npz" --test "$CASE/499-base-test.npz" --bias 0.91 --lr "$lr" --epochs "$epochs" --out "$CASE/499-adam-$lr-$epochs.json"
  done
done

# Conditional second representation: reuse the trained-test features above.
CUDA_VISIBLE_DEVICES=0 "$SRC/.venv/bin/python" "$CASE/offline_probe.py" extract --model "$CKPT" --rows "$TRAIN" --out "$CASE/499-trained-train.npz"
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=8 "$SRC/.venv/bin/python" "$CASE/offline_probe.py" fit --train "$CASE/499-trained-train.npz" --test "$CASE/499-trained-test.npz" --out "$CASE/499-trained-ridge.json"

# Separate 510/60 event readback, pinned to an existing checkpoint rather than a future final.
MIX=$ART/tau2-training-20261013-critic-head/critic-head-qwen35-mix-r2
MIXCKPT=$MIX/checkpoints/root/tau2-critic-head/critic-head-qwen35-mix-r2/critic/epoch2epochstep19globalstep59
CUDA_VISIBLE_DEVICES=0 "$SRC/.venv/bin/python" "$CASE/offline_probe.py" extract --model "$MIXCKPT" --rows "$MIX/tau2-critic-fit-evidence/fixed-validation-rollouts.pt" --out "$CASE/510-step60-test.npz"
CUDA_VISIBLE_DEVICES= "$SRC/.venv/bin/python" "$CASE/verify_values.py" --values "$CASE/510-step60-test.values.json" --report "$CASE/validation_snapshot/510/validation-step-000060.json" --out "$CASE/510-step60-reproduction.json"
```

**Cost estimates, explicitly guesses pending the two-trajectory GPU smoke:**
60-trajectory checkpoint reproduction/event extraction: **0.05–0.25 GPU-hours each** on
one 80GB GPU; primary base-feature extraction over 1494 train + 60 validation episodes:
**1–3 GPU-hours**; optional trained-backbone extraction: another **1–3 GPU-hours**.
Every proposed command has **\$0 API cost**. CPU ridge/Adam sweeps require no further
GPU time. At 32 prefixes × 2560 float32 features, 1554 trajectories use about **0.51
GB** uncompressed feature storage; ridge converts arrays to float64 and needs additional
working memory. Sampling reduces stored features, not the full-sequence forward cost.

These budgets are informed by saved 510 last-20 timing: critic-values forward 3.84 s per
16 episodes on four training GPUs, critic training 10.53 s, versus rollout 120.18 s. The
forward timing corresponds to roughly 0.42 aggregate GPU-hours for 1554 episodes before
sequential single-GPU inefficiency, model loading, longer-tail sequences and writing
outputs. The proposed 1–3 GPU-hour allowance is a planning estimate, not a measured
benchmark of the new script.

A later **full-backbone offline** experiment could replay 80 updates over 1280 episodes
in an estimated 20–40 minutes on four GPUs (1.3–2.7 GPU-hours), or 10 passes in roughly
12–24 GPU-hours, with \$0 API cost. This is **not yet queueable**: it needs an
accepted-record replay trainer preserving the original loss, masks, freezing, optimizer
and distributed reduction. No such CLI was found, and none is presented as if it exists.
The prepared cached-feature Adam sweep is the immediate low-cost optimization test;
full-backbone and unfreezing ablations are conditional follow-up work.

## Candidate fixes ranked: question 5 {#next_steps}

All expected effects below are **hypotheses**, except the observed benefit of removing
random-head scatter. No fix is established to make Retail EV reliably positive.

| Priority | Change                                                                                             | Expected effect and reason                                                                                                                                                                                                                                     | Incremental cost / decision                                                                                                                                                                                                                                           |
| -------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1        | Reuse recorded data; solve the head accurately, then increase head updates only if dev supports it | Highest information per cost. Ridge versus raw-feature Adam separates readable signal from head optimization/conditioning. It directly tests whether 80–94 updates were sufficient for this representation.                                                    | Prepared base-feature pass 1–3 GPU-hours, then CPU sweeps; \$0 API. Do this first.                                                                                                                                                                                    |
| 2        | Improve hard-task/failure coverage and mix official/pack tasks throughout training                 | Plausibly the largest practical gain for Airline, and a prerequisite for learning diverse Retail failures. 479's 11 Retail pack failures across 9 tasks are weak coverage. 510 moves in this direction, but its apparent gain is not yet statistically secure. | Start with existing 479/510 rollouts and task-disjoint replay, \$0 API. Outcome-balanced resampling changes the value prior: use sampling weights or recalibrate on the deployment distribution. New rollout spending is deferred until replay shows what is missing. |
| 3        | Change prefix/trajectory weighting after viewing event curves                                      | May help if long repetitive or early uncertain prefixes dominate useful event supervision. Compare original token mean with equal-trajectory weighting and turn-boundary sampling on the same data; report each under both objectives.                         | No additional GPU if cached prefix features suffice; fuller features or backbone retraining add the measured replay cost. \$0 API. A gain on a reweighted metric alone is not a gain on original token EV.                                                            |
| 4        | More full-backbone offline passes / LR sweep, keeping attention frozen initially                   | Plausible when fixed-training-set loss remains high and cached-head evidence suggests accessible signal. Fixed-data training curves are currently missing.                                                                                                     | Requires the replay harness; estimated 1.3–2.7 GPU-hours per 1280-episode pass, \$0 API. Do not pay for repeated simulators just to vary the optimizer.                                                                                                               |
| 5        | Unfreeze attention or adapt selected attention modules                                             | Could improve event information routing if learned features, not just the head, are limiting. Existing 2.9B trainable parameters and changing MLP inputs mean freezing is not proof of incapacity.                                                             | Implementation plus controlled offline ablation; GPU/memory cost above or comparable to replay must be measured. No defensible exact budget without its implementation. Defer until the cheaper probes fail.                                                          |
| 6        | Sigmoid/BCE or another bounded probabilistic head                                                  | Could help calibration/stability for binary returns. It does not create missing outcome information; the mean-init runs already removed most random-head instability. Plain MSE is a valid probability regression loss.                                        | First test on cached features for CPU cost, \$0 API. Lower priority for within-domain EV than coverage and optimization.                                                                                                                                              |

A **measurement change** should precede all comparisons: use the same rollout bank for
both models, report per-domain task-cluster intervals and bias/MSE, and separate
first-prefix from later-prefix behavior. It improves confidence in the conclusion, not
the critic itself. Re-score the existing 475 repeated bank if its token/mask
representation can be reconstructed and verified; do not silently substitute text
re-tokenization for original rollout tokens. More repeats on fixed tasks reduce
conditional rollout uncertainty, while more independent tasks address task-sampling
uncertainty; neither is interchangeable with more tokens from the same episode. No API
rerun is necessary for the prepared experiment.

## Evidence, limits and delivery status {#scope}

The five issue questions are answered at the level permitted by stored artifacts: exact
metric/noise analysis; observed versus unidentified signal sources; bounds and decoded
examples for token/event behavior; runnable offline probe and head-budget experiments;
and a ranked fix plan. **Unresolved:** exact temporal value responses, real-model probe
results, causal attribution of within-task noise, attention/LR/full-backbone ablations,
and 510's final 80-step outcome. They require the specified additional forwards or
experiments. Nothing here claims those GPU results were obtained.

The investigation used the repository `.venv`, `PYTHONPATH` pointing to the requested
repo, and `CUDA_VISIBLE_DEVICES=` for every executed Python analysis. No GPU work, pueue
action, DeepSeek call, commit, run-directory edit or repo edit was performed. The Claude
case-study directory/report was not read. Existing unrelated untracked repo files were
left untouched. Scripts, source-report snapshots, decoded excerpts and derived
statistics are all in this output directory.

The new extractor was checked by source inspection, not a real GPU model forward. Its
output records sampled-prefix identities, input hashes and model metadata; model weights
are inventoried by file size rather than fully content-hashed. It verifies that biased
critic checkpoints contain the score tensors. Kernel/FLA/flash-attention compatibility,
memory usage, numerical reproduction and wall time remain untested. The extractor's
base-model score is random and must never be interpreted as a trained value curve. The
ridge/Adam scripts received CPU synthetic checks, including destroyed-feature controls;
these validate the analysis code, not tau2 learnability.

Reproduce the completed CPU analysis from this directory:

```bash
SRC=/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math
export PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$SRC/.venv/bin/python" analyze_validation.py
OPENBLAS_NUM_THREADS=1 "$SRC/.venv/bin/python" analyze_extra.py
"$SRC/.venv/bin/python" data_audit.py
"$SRC/.venv/bin/python" data_ceiling_check.py
"$SRC/.venv/bin/python" data_initial_user_check.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$SRC/.venv/bin/python" inspect_events.py
OPENBLAS_NUM_THREADS=1 "$SRC/.venv/bin/python" offline_probe.py smoke
OMP_NUM_THREADS=1 "$SRC/.venv/bin/python" head_adam.py --smoke
```

`analyze_validation.py` and the training audit read a live run, so rerunning later can
add new steps and change the snapshot. The report's numbers are bound to the saved
JSON/CSV and validation snapshots, not a promise that 510 remains at step 60/63. The
analysis deliberately corrupted SSE once to verify that the reconstruction check detects
incorrect statistics. The initial analysis included the previous run's constant-reward
Telecom group, correctly encountered undefined EV, and was restricted to the requested
Airline/Retail domains. The tokenizer's default local-cache lookup failed without making
a network request; the verified explicit local snapshot path succeeded.

### Source anchors

Paths below are relative to repository
`/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math`, inspected at
HEAD `890e5ee45d47d4dd52ce43814a8819270fdad91e`.
[code_audit.md](ev-case-study-codex-code-audit.md) and
[data_audit.md](ev-case-study-codex-data-audit.md) preserve the bounded investigations
and additional locators. The original task and README remain the
requirements/background; calculations use raw files rather than copying their
interpretations.

| Claim                                                                          | Source locator                                                                                                                                   |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Historical/current EV removes bias; macro MSE differs from trajectory-mean MSE | `scripts/qualification/critic_pretrain.py:138–171`; same formula checked at historical commit `04be68ab`                                         |
| Shifted action-prefix mask and constant terminal target                        | `scripts/tau2/train_critic.py:545–568`; metric row `:590–625`                                                                                    |
| Ordered values discarded; validation performs no optimizer update              | `scripts/tau2/train_critic.py:676–731`; PT cache save `:1027`                                                                                    |
| MSE objective and update granularity                                           | `areal/trainer/ppo/critic.py:96–174`; `areal/trainer/rl_trainer.py:1010`; `areal/engine/fsdp_engine.py:905–931,1054`                             |
| Attention freeze scope                                                         | `areal/engine/fsdp_utils/critic_freeze.py:22–81`; `areal/engine/fsdp_engine.py:619`                                                              |
| Mean-initialized head and optional head LR                                     | `examples/tau2/critic_head.py:47–92`; `scripts/tau2/train_critic_head.py:25–69`                                                                  |
| Config values                                                                  | `examples/tau2/config_critic_production.yaml:152–182`; actual per-run `tau2-critic-fit-evidence/resolved-config.json` and train metric counters  |
| Replay dump occurs before final acceptance                                     | `areal/infra/workflow_executor.py:1112,1537`                                                                                                     |
| Ceiling estimator and cluster bootstrap                                        | `docs/research/2026-10-09-tau2-high-baseline-audit/scripts/critic-data/ev_ceiling.py:20–36`; raw 473–476 paths in `data_ceiling_fixed_test.json` |
| Validation values and uncertainty                                              | `validation_snapshot/<run>/validation-step-*.json`; original absolute paths and SHA256 in `validation_analysis.json`                             |
| Training exposure, optimizer counters and checkpoint metadata                  | `data_summary.json`, `data_train_labels_*.json`, `data_timing_loss_windows.json`; each run's original paths are retained                         |
| Initial user variation and selected token spans                                | `data_initial_user_variation.json`, `event_manifest.json`, `events_*.json/.txt`                                                                  |

The report uses the requested incremental standalone file and independent reading
review. The work-report runtime's Git-exclusion initialization was not used because the
repository is explicitly read-only; no formal runtime delivery receipt is claimed.
Scope-specific content, evidence checks and actual independent review are preserved
locally instead.
