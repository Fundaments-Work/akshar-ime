# Top-2 Discriminator — Experiment Log (2026-09-22)

Status: IMPLEMENTED, MEASURED 3 WAYS, **FALSIFIED**. Code removed
(`src/core/top2.rs`, `src/bin/build/build_top2.rs`, engine step 1c,
`--top2` harness flag, `examples/top2_forensics.rs`); this log preserves
the numbers. No test-split touch was ever needed — valid killed it three
times.

## Hypothesis

Half of native misses are a binary top-1-vs-2 choice (forensics, valid
native n=798: gold@1 655, gold@2 53 = +6.64pp ceiling, gold@3-5 28,
missing 62). A linear L2-logistic model on nine rank2−rank1 difference
features (full-model margin + frequency/morphology/shape differences)
should resolve the choice theceiling the blend gets wrong. Forensics
supported it: rank-2 gaps run half the winners' moat, and frequency
decides wrong in 40% of winnable cases.

## Attempt 1 — full features (margin + strings), all train pairs

89,804 keep / 21,607 swap examples from a stride-24 decode of all 3.59M
train pairs; train agreement 82.3% at tau=0. Valid: monotone destruction
— accuracy ROSE with tau at every stratum (fewer swaps = better),
converging to baseline from below. Best operating point: never swap.

## Attempt 2 — string-only (`--no-margin`)

Diagnosis: margin features are computed on EM-memorized train pairs, so
their distribution doesn't transfer to unseen spellings. Zeroed idx 1, 8
and retrained. Valid: same monotone shape, τ=2 reaches 441/519 vs 443
baseline. Diagnosis insufficient.

## Attempt 3 — domain-matched (`--min-freq 10` + `--no-margin`)

Diagnosis: the stride sample is entity-tail-dominated (24% swap rate)
while valid native is ~8% — wrong subpopulation. Trained on frequent-gold
pairs only (10,560 keep / 1,316 swap, train agreement 90.7%). Valid:
τ≥1 reproduces baseline EXACTLY (443/519 — learned to abstain everywhere),
τ=0 loses by 1. Never above baseline at any operating point.

## Verdict

FALSIFIED. The discriminator is made of the same ingredients as the
ranker (model scores, frequencies, shapes) — it cannot see what the
ranker cannot. When the full pipeline puts gold 2nd, nine differences of
the same signals carry no transferable correction: three training regimes,
every tau swept, best result is a tie via abstention. The top-2 choice
needs NEW information per decision (e.g. sentence context at the moment
of choice, or morphological analysis), not a reweighting of old signals —
the same moral as the matra and conditional-gamma falsifications.

## Repro

Builder mirrored the engine's decoder-band order exactly (decode_union @
50 + rerank_with_norm with model norm/sparse/dense, minus empty context).
τ sweeps ran on VALID via `evaluate_aksharantar --top2` (flag removed
with the code). Train/valid/test hygiene held throughout: test split was
never touched for this experiment.
