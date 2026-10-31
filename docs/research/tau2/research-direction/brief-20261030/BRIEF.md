# Research brief: PPO for multi-turn agents trained against unreliable users, and what corruption-robust bandits suggest for the critic loss

Status as of 2026-10-30. Self-contained: a reader with no access to the chat history or the training
machine should be able to follow it. Numbers are copied from the repository documents cited in each
section; anything that is an inference rather than a measurement is marked as such.

## 0. One-paragraph summary

We train an LLM customer-service agent with PPO on τ²-bench (airline, retail, telecom), where the
environment includes an LLM *user simulator*. The outcome reward $R\in\{0,1\}$ depends on both the agent
and the user, and simulated users (like real ones) often cause failures the agent did not cause: they
confirm and hang up in the same message, contradict their own task, or invent facts. Our current method
gives the **critic** (not the actor) privileged access to the user's script (the instructions the
simulator received). With $\lambda=1$ this is a pure baseline change: it cannot change the expected
gradient, only its variance. A 2×2 experiment (script-aware vs standard critic × GAE $\lambda=1$ vs a
length-adaptive $\lambda$) plus a seed replicate shows that the script-aware critic fits better
(explained variance 0.52–0.53 vs 0.40) but the downstream eval gain is small and only partly
reproducible. We now want to **change the loss** rather than the critic's input, and the starting point
is two papers on contextual bandits with adversarially corrupted rewards (§6), one of which is by our
collaborator. We want design candidates that transfer the "uncertainty-weighted regression" idea from
those papers to PPO's critic and/or advantage, with an honest account of the bias each introduces.

## 1. Setting

- **Benchmark**: τ²-bench. Each task has a database, a domain policy the agent must follow, tools, and a
  *user scenario* given only to the user simulator: reason for calling, known/unknown info, task
  instructions, persona. Success is judged by DB state checks plus natural-language assertions
  (an LLM judge).
- **Models**: actor and critic both start from `Qwen/Qwen3-4B-Instruct-2507`. Training user simulator:
  self-hosted Qwen3.5-4B (T=0). Training domains: airline (417 tasks) + retail (337 tasks); telecom was
  removed from training for the runs below.
- **RL algorithm**: PPO, one rollout per task, 128 episodes per step, 120 steps (~20 epochs), actor lr
  $10^{-6}$, critic lr $10^{-5}$, no KL, no entropy bonus, one minibatch per step (so the PPO ratio clip
  never triggers), batch-whitened advantages, value clip 0.5.
- **Eval**: held-out test split (airline 20 tasks, retail 40, telecom 40), 4 trials per task, metric
  **mean@4** = average success over the 4 trials, all temperatures 0. Two evaluators: a strong user
  (DeepSeek-Flash as user and judge) and the training-like user (Qwen3.5-4B).
- **Noise floor** (measured): re-evaluating the same checkpoint at T=0 moves the total by 2–3pp and
  single domains by up to ~6pp; retail with 40 tasks has a standard error of roughly 6pp.

## 2. The research object

From the project's decision record (ADR 0011, 2026-10-13): **the object of study is unreliable feedback
from real users; the simulator is a controllable proxy.** Real users also confirm and leave, change their
mind, or withhold information. We cannot edit a real user's prompt or replay them, so "resample the user"
is not a valid baseline. Training uses unreliable users; evaluation uses reliable users and checks policy
compliance.

Phenomenon that motivates it (same agent, Qwen3.5-4B, τ² retail): only swapping the user simulator moves
success from 85.0% to 65.0%. Agent-caused failures stay at 10–15; user-caused failures range from 7 to 45.
Even with the strongest simulator, users cause 29% of failures. Over five simulators (1,200 episodes), 168 of 385
failures were caused by the user. Failure codes we use: U1 = user confirms and ends the conversation in the same
message; U3 = user violates their own task or invents facts; U4b = user leaves before the task is done.

## 3. How a user error enters the PPO gradient

Notation. Agent tokens $t=0,\dots,L-1$ (user turns and tool outputs are not time steps). $R$ sits on the
last agent token, $\gamma=1$. $V_t$ is the critic value before token $t$. GAE:

$$
\hat A_t=\sum_{k\ge0}\lambda^k\delta_{t+k},\qquad \delta_t=r_t+V_{t+1}-V_t .
$$

With $\lambda=1$, $\hat A_t=R-V_t$. The critic is regressed on $G_t=R$ (critic $\lambda=1$ always):

$$
\mathcal L_{\text{critic}}=\frac1N\sum_t\max\Big((V_t-R)^2,\ \big(V_t^{\text{old}}+\mathrm{clip}(V_t-V_t^{\text{old}},\pm0.5)-R\big)^2\Big).
$$

**Exact split by speaker.** Let $V_k$ be the reading before agent turn $k$, $U_k$ the reading after the
agent finished turn $k$ but before the user replies, and $V_{K+1}=R$. For a token $i$ in turn $t$,

$$
R-V_i=\underbrace{(U_t-V_i)+\sum_{k>t}(U_k-V_k)}_{\text{agent segment}}+\underbrace{\sum_{k\ge t}(V_{k+1}-U_k)}_{\text{user segment}} .
$$

This is an identity. If the critic is self-consistent across a user reply, $U_k=\mathbb E[V_{k+1}\mid h_k^+]$,
the user segment has conditional mean zero.

**Two harms** (this distinction drives everything below):

1. **Variance.** A user who randomly hangs up makes the user segment a large negative number that lands
   on all earlier, correct agent tokens. The expected gradient is unchanged; learning is slower.
2. **Biased target.** If unreliability is *predictable* (e.g. this user hangs up 40% of the time right
   after "shall I confirm?"), an accurate critic already lowers $U_k$ after the question, so the agent
   segment correctly says "asking for confirmation is bad *for this user*". That is right for
   $J=\mathbb E[R]$ and wrong for "agent competence and policy compliance".

**Hard conclusion: no advantage estimator that is unbiased for $J$ can fix harm 2**, because harm 2 *is*
$J$. Fixing it requires changing the objective: weight trajectories or users differently, or move some
tokens out of the loss. Variance reduction (including a better baseline) only addresses harm 1. This is
why the corruption-robust literature is relevant: its estimators are deliberately not unbiased for the
corrupted objective.

## 4. What has been tried

| Method | What it does | Outcome |
| --- | --- | --- |
| Uniform β | multiply every user segment by $\beta=0.5$ | Mid-dialogue user segments are ~0 ($+0.006\pm0.061$), so β acts only on the last reply, where U1 is only 11–12% and the rest are agent errors that surface at settlement. Retail at step 120 was 13.75pp *below* plain PPO. |
| Rule-gated β | $\beta=0$ only where a U1 rule fires | High recall, but counterfactual precision ~58%: often the agent's plan was already wrong before "yes + STOP". |
| LLM-judge gating | an LLM reads the whole trajectory and assigns blame | Recognizes U3, but precision 76% on the training distribution; blames the user for 12.8% of agent failures; 33.5k tokens per call. |
| Truncate + bootstrap | cut at the user error, bootstrap from $U_k$ | Depends on a detector; $U_k\approx V_t$, so it behaves like masking. |
| Inverse-variance advantage weighting (paused draft, 2026-10-11) | weight each user's advantages by $1/\hat\sigma_u^2$, with $\hat\sigma_u^2$ read from the critic's squared residual at the agent's last action | Paused after a math review found two design errors (below). |
| **Script-aware critic (current)** | insert the user scenario before the dialogue in the critic's input only; actor input, actor loss and GAE unchanged | Legal baseline (the script is fixed before the episode and does not depend on agent actions). Results in §5. |

The paused weighting draft matters because any "downweight unreliable samples" proposal will meet the
same two objections, quoted from the internal review:

- **S1: the measurement point was chosen after the fact.** "The last agent action" depends on whether the
  episode terminated, so the residual there is selection-conditioned. Counterexample: the agent is
  correct up to the confirmation, the user ends early with probability $\rho$; the residual at the
  respective last positions gives $\mathbb E[e^2]=\rho(1-\rho)^2$, which is 0.009 at $\rho=0.9$. A user who
  almost always ends early gets a *large* inverse-residual weight. Fix: measure at a boundary
  identifiable online, without future termination information.
- **S2: squared residual identifies prediction risk, not unreliability.** At a fixed boundary,
  $\mathbb E[(R-V)^2\mid H,u]=p(1-p)+(p-V)^2$ with $p=\Pr(R=1\mid H,u)$. The first term is uncertainty
  given current information, which can come from the user *or* from hidden task conditions, unobserved
  DB state, or agent actions not yet taken; the second is critic error. Without repeated prefixes,
  counterfactuals or structural assumptions, one interaction cannot separate "user noise the agent
  cannot influence" from "a failure the agent could have avoided by asking differently". Also,
  $\rho$ large does not imply variance large: with $\sigma_R^2=\rho(1-\rho)$, a $\rho=0.99$ user has
  variance 0.0099, smaller than a normal user's 0.09, so naive inverse variance *upweights* the worst user.

## 5. Current results (Qwen-user training, telecom removed, step 120)

mean@4 in %, temperatures 0, infra failures excluded (0 in all rows). EV is the critic's explained
variance $1-\mathrm{Var}(V-R)/\mathrm{Var}(R)$ before the update, averaged over steps 101–120. "SAO λ" is
a length-adaptive GAE $\lambda=1-1/(1.5L)$ per trajectory, which makes the actor bootstrap from the critic.

| Config | Critic | Actor λ | DeepSeek user: airline / retail / telecom | Qwen user: airline / retail | EV |
| --- | --- | --- | --- | --- | --- |
| step 0 | — | — | 13.8 / 45.0 / 49.4 | 16.3 / 36.3 | — |
| run 3′ | standard | 1 | 18.8 / 54.4 / 55.0 | 21.2 / 48.8 | 0.40 |
| run 2′ seed 42 | script-aware | 1 | 17.5 / 65.6 / 67.5 | 23.1 / 43.1 | 0.52 |
| run 2′ seed 43 | script-aware | 1 | 18.8 / 60.6 / 53.1 | 25.0 / 41.3 | 0.53 |
| run 5 | standard | SAO | 13.8 / 56.3 / 54.4 | 12.5 / 44.4 | 0.48 |
| run 4 | script-aware | SAO | 22.5 / 55.6 / 51.9 | 23.8 / 41.9 | 0.52 |

Reading (inference, not established fact):

- The script-aware critic fits better on both seeds (EV 0.52/0.53 vs 0.40). That is the one robust finding.
- Seed 42 showed retail +11.3pp and telecom +12.5pp over run 3′ (paired bootstrap CIs excluded 0). Seed 43
  reproduces about half of the retail gain (60.6; two-seed mean 63.1, +8.7pp) and none of the telecom gain
  (53.1 vs 55.0). With $\lambda=1$ this is expected: a better baseline only reduces variance.
- With SAO λ the actor *uses* the critic's values, so a better critic could in principle move credit to
  the token where the agent erred. The pre-registered 2×2 interaction
  (run 4 − run 2′) − (run 5 − run 3′) was negative (retail −11.9, telecom −15.0 with seed 42; −6.9 / −0.6
  with seed 43 substituted). The hoped-for mechanism did not show up.
- Under the training-like Qwen user, the script-aware runs are slightly *below* run 3′ on retail on both
  seeds, within noise.
- Limitations: one seed for runs 3′/4/5; run 5 and seed 43 used a different (but same-model-family) judge
  during training; many comparisons, no multiple-comparison correction; per-episode results for these
  evals were lost with the training machine, so new paired CIs cannot be computed. Attached CSVs contain
  all checkpoints (steps 30/60/90/120) and per-step training reward and EV.

Diagnostic evidence about what the critic sees (from case studies, DeepSeek-user trajectories): when the
agent writes the wrong DB change, the remaining "surprise" at settlement is −0.26 for the script-aware
critic vs −0.45 for the standard one, i.e. the script-aware critic drops earlier. An earlier probe found
the standard critic's $U$ predicts "was the agent's proposal correct" with AUC only 0.573.

## 6. The two papers we want to borrow from

1. **He, Zhou, Zhang, Gu. "Nearly Optimal Algorithms for Linear Contextual Bandits with Adversarial
   Corruptions"** (arXiv 2205.06811, 2022). Rewards are perturbed by an adversary with total budget
   $C=\sum_t|c_t|$. The algorithm (CW-OFUL) is optimistic and fits the reward model by **weighted ridge
   regression where each sample's weight is capped by its confidence width**: roughly
   $w_t=\min\{1,\alpha/\|x_t\|_{\Sigma_{t-1}^{-1}}\}$. High-uncertainty samples get small weight, which bounds
   how much any single corrupted label can move the estimate. Regret is near-optimal for known and unknown
   $C$ and reduces to the uncorrupted rate at $C=0$.
2. **Qi, Ban, Banerjee, He. "Robust Neural Contextual Bandit against Adversarial Corruptions"** (NeurIPS
   2024; Yikun Ban is our collaborator). R-NeuralUCB extends this to neural reward models: per-sample
   weights from the gradient-feature norm $\|g(x;\theta)/\sqrt m\|_{\bar\Sigma^{-1}}$ (an NTK-style
   uncertainty), and a **context-aware weighted L2 loss specific to each candidate arm**: when scoring a
   high-uncertainty candidate, training samples are weighted more conservatively. Regret analysis without
   the usual arm-separateness assumption.

**Mapping we propose (to be checked, not assumed):**

| Bandit paper | Our PPO setting |
| --- | --- |
| context $x_t$ / arm | dialogue prefix $h_t$ (+ script $z$) at an agent decision point |
| corrupted reward $r_t=\tilde r_t+c_t$ | outcome $R$ flipped by a user error (U1/U3/U4b); "clean" reward = what a reliable user would give |
| corruption budget $C$ | number of episodes whose outcome a user error changed (measured: 168 of 385 failures in 1,200 episodes across five simulators were user-caused; 7–45 per simulator in the retail comparison) |
| weighted ridge / weighted GD on the reward model | weighted critic regression $\sum_t w_t (V_t-R)^2$, and/or weighted actor loss $\sum_t w_t\hat A_t\nabla\log\pi$ |
| uncertainty $\|x\|_{\Sigma^{-1}}$, $\|g\|_{\bar\Sigma^{-1}}$ | candidates: critic-ensemble disagreement, last-layer feature covariance of the value head, size of the user segment $|V_{k+1}-U_k|$, script-conditioned vs script-free critic disagreement |

Differences we already see: bandits have one-step rewards and an adversary with a budget; we have long
episodes, a single delayed binary reward, and a "corruption" that is a stochastic user policy, partly
predictable from the script. In the bandit papers the weighted estimator is the *reward model used for
action selection*; in PPO the critic is only a baseline when $\lambda=1$, so robustifying the critic
changes nothing in expectation unless the actor bootstraps from it ($\lambda<1$) or the weights also
enter the actor loss.

## 7. What we want help with

1. **Candidate designs** (3–5) for changing the critic loss and/or the advantage, inspired by the
   uncertainty-weighted regression above. For each: the exact formula in our notation; which harm (§3)
   it targets; what bias it introduces relative to $J$ and what objective it optimizes instead; how it
   avoids objections S1 and S2 in §4; compute cost for a 4B critic.
2. **The uncertainty signal.** What is a defensible LLM-critic analogue of
   $\|x\|_{\Sigma^{-1}}$? Note the confound in S2: high uncertainty may mean "hard task", not "unreliable
   user". Is there a way to use the user script (which we have at training time) to separate them?
3. **Where the weight should act**: critic regression only, actor loss only, or both, given that with
   $\lambda=1$ only the actor-side weight changes the expected gradient.
4. **Related work** beyond the two bandit papers: corruption-robust RL with function approximation,
   robust/weighted value regression, learning from noisy rewards in RLHF/RLVR, multi-agent credit
   assignment where another agent's mistakes contaminate the return. For each, what it would contribute
   and whether it collides with our framing.
5. **One falsifying experiment per design**, within our budget: a run is ~14.5 hours on 8 GPUs; eval
   noise is ~6pp per domain, so the design must predict an effect larger than that or a cheaper
   diagnostic (e.g. offline critic forward passes on 8.7k fixed trajectories) must be able to refute it.

## 8. Source documents (in the repository `AlexJJ009/AReaL`, branch `codex/sao-math`)

- `docs/research/tau2/README.md`: project-level conclusions.
- `docs/research/tau2/research-direction/problem-and-plan.md`: problem, PPO derivation, current method.
- `docs/research/tau2/research-direction/discussion-20261026-qwenuser-runs.md`: full advantage and loss
  formulas with a worked numeric example of $\lambda=1$ vs SAO λ.
- `docs/research/tau2/research-direction/appendix-method-draft-16.11.md` and
  `appendix-math-review-codex.md`: the paused weighting draft and its review.
- `docs/research/tau2/experiments/2026-10-24-qwenuser-no-telecom.md`: run cards and results of the 2×2
  and the seed replicate.
- `docs/research/tau2/research-direction/related-work.md`: related work as of 2026-10-11.
- `docs/research/tau2/decisions/`: ADRs (0011 is the research object).

The internal documents are written in Chinese; formulas and tables are language-independent.
