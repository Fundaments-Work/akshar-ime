# Factored Matra Model — Experiment Log (2026-09-22)

Status: IMPLEMENTED, MEASURED, **FALSIFIED as a word-level prior rerank**.
Code removed (was `src/core/matra.rs`, `examples/matra_skeleton.rs`);
this log preserves the math and the numbers so the idea is not retried blind.

## Hypothesis

P(D|R) = P(C|R) · P(M|C,R) by the chain rule (exact, no approximation).
C = halant-preserving consonant skeleton (conjuncts stay consonantal),
M = per-akshara vowel-sign assignment, modelled per position:

    P(M|C,R) = prod_i P(m_i | c_i, pos_i, rho(R,i,n))

with Dirichlet-smoothed MLE (α = 0.5, same floor as the KN continuation):

    P(m|c,pos,rho) = (count + α·q(m|c)) / (total + α)

plus a matra bigram P(m_i|m_{i-1}). Closed form, no EM (skeleton/matra split
is deterministic given the Devanagari side), ~18k contexts from 3,588,793
pairs. Motivation: 51.9% of native misses are matra-class; the current
reranker's matra features (dense 10–20) count vowel signs globally without
conditioning on consonant or roman, so a conditioned factor looked like the
missing interaction.

## Measured (shipped engine, AK-Freq n=2108, top-1 80.98% reproduced)

- 182/401 misses are pure vowel-sign errors (halant-preserving skeleton
  equal); gold string in top-10 for 144 → rerank ceiling +6.8pp.
- Head-to-head (table cost gold vs engine top-1): gold wins **78/182 (42.9%)**.
  As an override it loses points; it would flip 992/1707 correct top-1s.
- Iteration 1 (roman window = first vowel letter): 79/182.
- Iteration 2 (+ doubling flag `oo` vs `u`): 77/182 — test inputs themselves
  undermark length (`rukho`, not `rookho`), so the cue never fires.
- Iteration 3 (+ matra bigram): 78/182, risk up to 58% — both candidates are
  fluent strings; the table re-learns the frequency prior the engine already
  has. Residual errors are cases where that prior itself prefers the wrong
  reading (MANUAL W4: needs context/morphology, not a second prior).

Per-akshara confusion concentrates exactly where roman is ambiguous:
र+ु→र+ू (14), त+ि→त+ी, inherent-a↔ा, anusvara placement (िँ↔िं).

## Verdict

FALSIFIED as a word-level prior. The ambiguity is in the input, not the
model — no per-position roman feature can resolve `u`→उ/ू when the user
typed `u`. Do not retry as a prior/override. Reusable parts, if the top-2
discriminator (devanagari-script-plan step 2) wants them: skeleton()/split()
decomposition (3 lines), proportional roman windows, Dirichlet backoff
pattern. Cost of keeping: module + probe + container table for −0pp —
correctly deleted per speed/storage/accuracy discipline.

## Repro

The probe streamed `data/aksharantar/train_devanagari.jsonl` (3.59M pairs),
built the table, and scored all 2108 AK-Freq cases through `ImeEngine` at
beam 64. Test-file assertions that held: skeleton keeps halants, observed
pattern costs less than unseen, unseen contexts back off to valid
probabilities. All removed with the code; rerun from this spec if doubted.
