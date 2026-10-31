I am a PhD student working on RL post-training of LLM agents. The attached `BRIEF.md` describes my
project: PPO on τ²-bench where an unreliable user simulator corrupts the outcome reward, and a
script-aware critic that improved critic fit but not reliably the agent. The two attached PDFs are
contextual-bandit papers on adversarial reward corruption; the second is by my collaborator, and the idea
is to borrow their uncertainty-weighted regression for our critic loss or advantage.

Please read `BRIEF.md` fully first (especially §3, the two harms and the "unbiased estimators cannot fix
harm 2" argument, and §4, the two review objections S1/S2 that killed our earlier weighting draft). The CSVs
hold all eval and per-step training numbers; the other attachments are the internal derivations (in
Chinese) if you need detail.

Then answer, in this order:

1. Check the mapping table in BRIEF §6. Where does the bandit → PPO analogy break, and what would have to
   be true for the papers' guarantees to say anything about our setting?
2. Propose 3–5 concrete designs. For each give: the formula in the brief's notation; which harm it targets;
   the objective it actually optimizes (its bias relative to $J=\mathbb E[R]$); why it survives S1 and S2,
   or honestly where it does not; the cost for a 4B critic.
3. For each design, the cheapest experiment that could falsify it, given a ~6pp eval noise floor and an
   available set of 8.7k fixed trajectories for offline critic forward passes.
4. Related work I should read beyond the two papers (corruption-robust RL with function approximation,
   weighted/robust value regression, noisy rewards in RLHF/RLVR, credit assignment with other agents).
   Give real citations with links, say what each contributes, and flag any that already does what we
   would propose.

Mark clearly what is established in the literature versus your own conjecture. Do not re-propose
inverse-variance weighting from the critic's squared residual unless you explain how it avoids S1 and S2.
