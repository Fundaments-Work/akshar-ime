# Experiment report: which ideas carry the accuracy, and by how much

Date: 2026-09-08. Tree: `5cc341a`. Model: `data/akshar.model`.
Machine source: `docs/generated/report_matrix.json` (all tables below are
transcribed from it — no hand-edited numbers).

## Question

The literature review (`papers_to_read/`) attributes transliteration accuracy
to five ideas: direct orthographic mapping (Li–Zhang–Su 2004), discriminative
reranking with joint context features (Goto et al. 2003), generate-then-rescore
(Finch & Sumita 2010), higher-order smoothing (Kneser–Ney tradition, KenLM
2011), and exact search over the hypothesis lattice (Dijkstra 1959, Mohri 2002,
Eppstein 1998). This report tests each idea against our system by switching it
off (or down) and measuring what breaks.

## Method

- Harness: `evaluate_aksharantar` (word accuracy, strict single gold) and
  `eval_ime` (first-suggestion rate, top-5 visibility, keystroke savings,
  latency, 95% CI) — both in `src/bin/evaluate/`.
- Data: `data/aksharantar/test_devanagari.jsonl`, 4,101 held-out cases, three
  strata: AK-Freq (n=2,108 frequent words), AK-NEI (n=1,176 in-vocabulary
  named entities), AK-NEF (n=817 out-of-vocabulary named entities).
- Ablations use only runtime env switches (`AKSHAR_*`, see `src/core/mod.rs`
  lines 28–32) — no retraining, so every run decodes the identical 4,101 cases
  with the identical model. Runs are deterministic: the baseline configuration
  was executed three times (two word-eval, one via the variants/discount runs)
  and returned bit-identical percentages every time.
- Significance rule of thumb: on AK-Freq, 0.3pp ≈ 6 cases — treat anything
  smaller as noise. Strata NEI/NEF are smaller (±~3pp noise on NEF). Only
  `eval_ime` prints confidence intervals; word-eval deltas below the rule are
  reported as "~0".

## H1 — Discriminative rerank beats raw generative scores (Goto et al. 2003)

`AKSHAR_NO_RERANK=1` ranks by raw decoder score, skipping the rerank stage.

| Stratum | Full (top-1 / top-5) | No rerank | Δ top-1 |
|---|---|---|---|
| AK-Freq | 81.83 / 92.22 | 75.38 / 90.65 | **+6.45pp** |
| AK-NEI | 47.79 / 69.81 | 38.78 / 66.92 | **+9.01pp** |
| AK-NEF | 31.21 / 53.00 | 24.85 / 48.84 | **+6.36pp** |
| IME view | 62.0 / 78.0 | 54.8 / 75.5 | **+7.2pp** (CIs [60.4,63.5] vs [53.2,56.4] — non-overlapping) |

Verdict: **confirmed**. Note on honesty: the archive cites "+12pp" for the
reranker, measured against an older, weaker generator. Against the current
generator the rerank contribution is +6.5pp — still the largest single
component after the generator itself, and largest on named entities (+9pp).

## H2 — Generate-then-rescore beats either stage alone (Finch & Sumita 2010)

| Configuration | AK-Freq top-1 | AK-Freq top-5 |
|---|---|---|
| Full (generate + rescore) | **81.83** | **92.22** |
| Generator only (`NO_RERANK`) | 75.38 | 90.65 |
| Lexicon only (`TRIE_ONLY`) | 66.18 | 72.91 |
| Generator without trie-union pass (`NO_TRIE_UNION`) | 81.17 | 91.08 |

Verdict: **confirmed**. The combination beats both components, mirroring
Finch & Sumita's joint-multigram + SMT result. The trie-union pass itself is
worth +0.66pp top-1 — small but above noise.

## H3 — The learned mapping carries weight beyond heuristics (Li et al. 2004)

| Configuration | AK-Freq top-1 | Δ vs full |
|---|---|---|
| Full (γ blend) | 81.83 | — |
| Heuristics only (`GAMMA=0.0`) | 81.02 | −0.81 |
| Dense learned weights only (`GAMMA=1.0`) | 76.47 | −5.36 |
| Sparse table dropped (`NO_SPARSE`) | 81.93 | +0.10 (~0) |

Verdict: **nuanced — the uncomfortable finding**. The hand-built heuristics
(including the word-frequency prior) carry *more* accuracy than the learned
dense weights alone, and the 2²⁰ sparse table contributes nothing measurable
(+0.10pp top-1, −0.05pp top-5 — noise). What this does *not* say: the EM
chunk alignments are useless — they build the candidate generator, which H1/H2
show is responsible for 75 of our 82 points, and that cannot be ablated without
retraining. What it *does* say: inside the rerank blend, learning adds ~0.8pp
over heuristics. The "learning" story in our docs was stronger than this
measurement supports; the frequency prior is doing the heavy lifting.

## H4 — Higher-order smoothing matters (Kneser–Ney / KenLM lineage)

| Configuration | AK-Freq top-1 | AK-NEI top-1 | AK-NEF top-1 |
|---|---|---|---|
| Full (trigram KN) | 81.83 | 47.79 | 31.21 |
| Bigram only (`NO_TRIGRAM`) | 77.94 (−3.89) | 45.07 (−2.72) | 27.29 (−3.92) |
| Fixed-discount fallback (`KN_FIXED_DISCOUNT=1`) | 81.83 (0.00) | 47.79 (0.00) | 31.21 (0.00) |

Verdict: **confirmed for order, null for the fallback**. Trigram context is
worth ~4pp on every stratum. The fixed-discount switch changes nothing —
either the fallback never triggers on this data or it is behaviorally
identical here; kept as a safety path, not an accuracy feature.

## H5 — Search quality: beam width and k-best depth (Dijkstra / Mohri / Eppstein)

| Beam | AK-Freq top-1 / top-5 | Δ top-1 vs 64 |
|---|---|---|
| 8 | 80.55 / 89.47 | −1.28 |
| 16 | 81.31 / 90.75 | −0.52 |
| 64 (default) | 81.83 / 92.22 | — |
| 128 | 81.69 / 92.41 | −0.14 / +0.19 top-5 |

| Rerank depth | AK-Freq top-1 / top-5 |
|---|---|
| 8 | 81.69 / 92.27 |
| 24 (default) | 81.83 / 92.22 |
| 64 | 81.74 / 92.27 |

Verdict: **confirmed — beam 64 is the knee**. Search error is real at beam 8
(−1.3pp), nearly gone at 16 (−0.5pp), gone at 64. Rerank depth saturates by 8:
exact k-best machinery à la Eppstein would buy nothing measurable on top-1/5
with the current generator — the errors are modeling errors, not search
errors. (NEF top-5 keeps climbing slightly with beam: 49.20 → 50.43 → 53.00 →
52.51 — rare words benefit most from wider search.)

## H6 — Variant normalization (Karimi's "transliteration variants" challenge)

`AKSHAR_NO_VARIANTS=1`: 81.83 / 92.22 on every stratum — **exactly 0.00**,
bit-identical to baseline.

Verdict: **confirmed zero (negative result)**. The normalizer variants never
change the outcome on 4,101 cases. Either the test set contains no variant
cases or the emission model already covers them (the manual's own 0.00pp
measurement agrees). Candidate for removal, not for praise.

## H7 — The IME reality check

| View | 1st-suggestion | Top-5 visible | Keystrokes saved | Latency |
|---|---|---|---|---|
| Full engine | 62.0% | 78.0% | 2.7% | 0.794 ms |
| No rerank | 54.8% | 75.5% | 2.3% | 0.733 ms |

Verdict: **measured**. Word accuracy (81.8%) flatters; the prefix-IME task
scores 62.0% first-suggestion. The rerank stage costs ~0.06 ms/query for
+7.2pp — the best accuracy-per-microsecond trade in the system. Keystroke
savings (2.7%) confirm prefix completion is a minor convenience, not a
product story.

## What the numbers add up to (AK-Freq top-1 decomposition)

Generator 75.38 → +trie-union 0.66 → +trigram-over-bigram 3.89 (overlapping,
not additive) → +rerank 6.45 → **81.83**. Heuristics 81.02 vs dense 76.47:
inside the final blend, the corpus frequency prior dominates learned weights.
Sparse table, variants, discount fallback: 0.00 combined.

## Relation to prior (archive) experiments — cited, not re-run

- Corpus cleaning + dedup: +0.33pp native top-1 (`2026-09-03-accuracy-experiments`).
- Word-frequency prior: +5.64pp, largest rerank contributor (same).
- Word-bigram table: removed, 19.5 MB for +0.16pp (same).
- 5× reranker data: no gain; cause identified as feature ceiling, not data
  (`2026-09-06` measure commit). H3 above independently corroborates: the
  ceiling is real — heuristics saturate, dense weights add little.

## Threats to validity

1. Single test split, clean words only (the Aksharantar paper's own
   limitation). Noisy/code-mixed IME input is unmeasured.
2. Single gold per case: legitimate variants count as errors; the 82.16%
   multi-reference figure is ad hoc (see data/README) — strict accuracy here
   is a lower bound on user-perceived quality.
3. No phoneme-pivot baseline exists, so Li et al.'s anti-phoneme claim is
   supported by architecture, not by a controlled experiment.
4. NEF stratum (n=817) has ±~3pp noise — small deltas there (e.g. beam-128
   +0.25pp) are suggestive, not conclusive.
5. Latency from one machine (release profile); relative deltas are what
   transfer, not absolutes.

## Reproduction

```bash
BIN=target/release/evaluate_aksharantar
DS=data/aksharantar/test_devanagari.jsonl
$BIN --dataset $DS --topk 5 --show-misses 0
AKSHAR_NO_RERANK=1 $BIN --dataset $DS --topk 5 --show-misses 0
AKSHAR_TRIE_ONLY=1 $BIN --dataset $DS --topk 5 --show-misses 0
AKSHAR_NO_TRIE_UNION=1 $BIN --dataset $DS --topk 5 --show-misses 0
AKSHAR_NO_SPARSE=1 $BIN --dataset $DS --topk 5 --show-misses 0
AKSHAR_GAMMA=0.0 $BIN --dataset $DS --topk 5 --show-misses 0
AKSHAR_GAMMA=1.0 $BIN --dataset $DS --topk 5 --show-misses 0
AKSHAR_NO_TRIGRAM=1 $BIN --dataset $DS --topk 5 --show-misses 0
AKSHAR_NO_VARIANTS=1 $BIN --dataset $DS --topk 5 --show-misses 0
AKSHAR_KN_FIXED_DISCOUNT=1 $BIN --dataset $DS --topk 5 --show-misses 0
AKSHAR_BEAM=8 $BIN --dataset $DS --topk 5 --show-misses 0
AKSHAR_BEAM=16 $BIN --dataset $DS --topk 5 --show-misses 0
AKSHAR_BEAM=128 $BIN --dataset $DS --topk 5 --show-misses 0
AKSHAR_RERANK_DEPTH=8 $BIN --dataset $DS --topk 5 --show-misses 0
AKSHAR_RERANK_DEPTH=64 $BIN --dataset $DS --topk 5 --show-misses 0
target/release/eval_ime
```
