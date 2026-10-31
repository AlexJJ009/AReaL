I am a PhD student working on RL post-training of LLM agents. The attached `BRIEF.md` describes my
project: PPO on τ²-bench, where an unreliable user simulator also decides whether an episode succeeds,
and a critic that sees the user's script. The experiments are done, and the results are weaker and
less clear-cut than the story we started with. I want help deciding what to do next.

Two things, in whatever order and depth you think the evidence supports:

1. **Is our research question the right one?** We framed it as "user errors corrupt the agent's credit,
   so separate the user's effect from the agent's". Given what the results show (BRIEF §5: the critic
   fits better but the agent gains little and inconsistently, and the length-adaptive λ variant did not
   help), what do you think the real problem is? It may be narrower, broader, or different from what we
   wrote. Point to the specific numbers or failed attempts that make you think so.
2. **Method ideas.** Given that clarified problem, what would you try? I mean ideas that change the
   learning signal (the reward, the critic's loss or target, the advantage, which tokens or episodes are
   trained on), not more tuning. My collaborator suggested two contextual-bandit papers on adversarial
   reward corruption (attached PDFs; BRIEF §6 summarizes them and sketches a mapping). Use them if they
   help, say so if they do not fit. Other directions are equally welcome.

For each idea, tell me what result in our existing data, or what cheap experiment, would show it is
wrong. BRIEF §4 lists what has already failed and why (including the review objections S1/S2 to our
earlier weighting draft), so build on those lessons rather than around them. Separate what is
established in the literature from your own conjecture, and give real references with links.

The CSVs have every checkpoint and per-step training stats if you want to check a claim against the
data. The other attachments are our internal derivations, written in Chinese.
