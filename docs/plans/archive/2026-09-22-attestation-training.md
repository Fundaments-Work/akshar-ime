# Attestation-Weighted Training — Experiment Log (2026-09-22)

Status: IMPLEMENTED, TRAINED (`train-mid-att`, ~10 min wall... actually
596s), MEASURED, **KILLED as a ship candidate**. Code kept (flag-gated,
default-off, byte-identical); model artifact kept local + gitignored
(`data/akshar_attested.model`, 11.37 MB — never upload, never default).

## Hypothesis

Per `docs/research/sota-signals.md` T4 (Roark/Dakshina §4.2): weight EM/LM
pairs by attestation so dominant romanization conventions dominate
training. Implementation: `train --attestation` pre-pass counts the corpus
(1.48M types, same holdout split — 14,465 lines, matching the audit),
weight `w = 1 + floor(ln(1+f))` per pair (max 11, mean ~2.3), fed through
the pre-existing `add_pair_weighted` path (integer weights keep KN
counts-of-counts valid). Reranker sampling stayed uniform (isolate EM/LM).
`make train-mid-att` reproduces (500k reranker pairs, 5 epochs, 12 EM iters).

## Measured (test, attested vs baseline 80.98/45.32/29.01)

| Split | baseline | attested | Δ |
|---|---|---|---|
| AK-Freq top-1 | 80.98% (1707) | 80.60% (1699) | −8 |
| AK-NEI top-1 | 45.32% (533) | 45.92% (540) | +7 |
| AK-NEF top-1 | 29.01% (237) | 29.74% (243) | +6 |
| pooled | 2477/4101 | 2482/4101 | +5 (+0.12pp) |

Smoke anecdote: `dhanyabad` top-1 flipped to धन्यबाद over धन्यवाद —
attestation boosting a frequent-but-wrong convention is visible, not just
theoretical.

## Verdict

KILLED per the pre-registered criterion (entity gain without native loss):
entities +13 combined, native −8 — the familiar strata-robbery pattern in
miniature, and every cell within noise (pooled +5, SE ≈ 31). Log-dampened
corpus-frequency EM/LM weighting does not move the needle. What this rules
out: convention dominance via unigram-frequency weights. What it leaves
open: entity-upsampled EM (weight by entity-ness, not frequency — the
opposite direction) and multi-ref-aware reranker loss.
