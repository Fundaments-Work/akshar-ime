# MBR Consensus Selection — Experiment Log (2026-09-22)

Status: MEASURED (probe only, never shipped), **FALSIFIED**. Probe removed;
no engine code was written. No test-split touch — valid killed it.

## Hypothesis

Per `docs/research/rerank-beyond-linear.md` §1 (Kumar & Byrne 2004): pick
the candidate minimizing expected loss under the model's own posterior
instead of the MAP 1-best. Inference-only, no training distribution to
mismatch (the top-2 discriminator's killer), O(k²) string comparisons.
Should attack top-2 ties + matra mass via consensus.

## Measured (valid native n=798, model top-1 655 = 82.08%)

Softmax posterior over reranked scores, T ∈ {0.25, 0.5, 1, 2} × K ∈
{5, 8, 10} × {char-edit, matra-insensitive} loss = 24 configs:

- Best: **+2/798 (+0.25pp)** at (T=0.25, matra_ins, any K) — noise
  (SE ≈ 12 hits). Flips +2/−0: consensus with a sharp posterior just
  re-selects the 1-best.
- Everything else negative, sharply worse as T rises (T=2 edit: −128).
  Flatter posteriors herd toward popular wrong forms — the exact failure
  mode the research note flagged.
- matra_ins strictly dominates edit loss everywhere (consensus on
  consonant skeletons is saner than on raw strings), but its ceiling is
  still +2.

## Verdict

FALSIFIED. The n-best list is a converged neighborhood, not a diverse
field: every candidate already survived beam + rerank, so consensus adds
no information beyond the ranking. MBR helps diverse lists (MT systems,
ASR lattices); ours agree with each other too much for voting to work.
Per the report's own prediction, this diagnoses the posterior as
miscalibrated for consensus — but recalibrating the posterior is just the
reranker-training problem (#2) by another name. Next: pairwise ranking
loss (same list, better objective) or morphology (new information).
