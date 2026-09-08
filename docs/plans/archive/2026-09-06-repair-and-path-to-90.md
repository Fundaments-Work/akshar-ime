# Repair and path to 90% — measured audit, fix plan, improvement plan

> **Archive snapshot — frozen.** Phases A and B landed 2026-09-06; C–F remain
> open (live status: `docs/MANUAL.md` §12). Sizes below are decimal MB (11.9 MB
> = 11.37 MiB; 9.34 MB = 8.91 MiB); Brotli re-verified 2026-09-08 as 6.72 /
> 4.94 MiB. All accuracy figures re-verified identically after v1.2.0.

Date: 2026-09-06. **Phases A and B landed the same day** — see §9 for what
changed and what it measured. Phases C-F remain open. Supersedes the baseline arithmetic in
`2026-09-05-path-past-90.md` (whose 81.40% figure no longer describes the
shipped default) and the discounting sections of `2026-09-05-mathematics.md`.

All numbers below were **measured on this tree** against `data/akshar.model`
(11.9 MB, built 2026-09-06 15:58) on the 4,101-case Aksharantar test
split. Nothing was retrained.

---

## 0. Measured status

### AK-Freq (native) top-1 / top-5

| Config | top-1 | top-5 |
| :--- | ---: | ---: |
| **Default, as committed** | **50.95%** | **81.07%** |
| `AKSHAR_FUZZY_CORPUS_SIZE=0` | 81.74% | 92.27% |
| `AKSHAR_CORPUS_FUZZY_BASE=1000` | 81.74% | 92.27% |
| beam 256 (fuzzy off) | 81.83% | 92.46% |

Named entities move the same way: NEI 47.62 → 25.68, NEF 31.21 → 20.81 when the
corpus-fuzzy path is enabled at its default band.

### γ (reranker blend) sweep, fuzzy off

| γ | meaning | AK-Freq top-1 | top-5 |
| ---: | :--- | ---: | ---: |
| 0.0 | pure heuristic, trained model unused | 81.02% | 92.08% |
| **0.3** | **shipped** | **81.74%** | **92.27%** |
| 0.6 | | 81.31% | 92.41% |
| 0.85 | | 78.75% | 92.03% |
| 1.0 | pure trained discriminative model | 76.47% | 91.46% |

**The trained reranker, used alone, is 4.55pp worse than the 3-parameter
heuristic it is supposed to improve on.** Its entire net contribution at the
tuned blend is +0.72pp. γ=0.3 is not a cautious discount — it is the dilution
point at which a broken model stops doing damage.

### Other measured facts

- Latency **3.39–3.61 ms**/query (`evaluate`, k=5–10, beam 64). Not sub-ms.
- Cold start **~4.3 s**, of which ~1.3 s is building the corpus SymSpell.
- Beam 64 → 256 buys **+0.09pp** top-1. Search is not the bottleneck.
- Collision bound **Acc\* = 99.15%** — only 0.85% of cases are unwinnable by any
  string-only system with a unigram prior. 90% is not blocked by the data.
- 92 unit tests pass. None of them detect a 31-point accuracy regression.

---

## 1. Defect register

Severity: **S1** = costs accuracy now; **S2** = corrupts the overnight retrain;
**S3** = latency/size; **S4** = correctness debt, no current impact.

| # | Sev | Defect | Location |
| :-- | :-- | :--- | :--- |
| D1 | S1 | Corpus-fuzzy band `850k` sits above the decoder band `FRESH_SCALE 800k`, so every fuzzy hit displaces the reranker's top-1. Costs **−30.8pp** top-1. At band `1000` top-5 is identical to the path being off, so it adds **zero recall**. | `engine.rs:623`, `669–680` |
| D2 | S1 | `SymSpell::lookup` returns unverified delete-set intersections — a necessary condition only. At `max_edit_distance=1` it returns true-distance-2 matches, with no distance weighting or penalty. The user-fuzzy path above it *does* verify (`min_roman_distance`); this one does not. | `fuzzy/symspell.rs:45` |
| D3 | S2 | MKN discounts clamped to `d1≤0.9, d2≤0.9, d3≤0.95`. Chen-Goodman requires `0 ≤ D_i ≤ i`; real corpora give `d2≈1.0–1.4`, `d3≈1.5–2.5`, so both pin at the ceiling and modified KN degenerates to absolute discounting at `d≈0.9` — worse than the `0.75` it replaced. `n[3]==0` also escapes the guard, yielding `d3=3→0.95`. | `em_trainer.rs:359–361` |
| D4 | S2 | `W_DENSE`, `MEAN_DENSE`, `STD_DENSE` are frozen constants. `train.rs` **imports and never retrains them**; the generator named in the file header (`train_reranker.rs`) does not exist in the repo. Retraining EM shifts the `emit`/`lm` distributions these stats normalize against, silently miscalibrating all 29 dense features. | `reranker_weights.rs`, `train.rs:13,369,459` |
| D5 | S2 | Chunked LR schedule collapses: `lr *= 0.8` fires `epochs−1` times *per batch*, then `lr *= 0.995`. At `epochs=3` that is `0.6368`/batch → **lr = 4.4e-9 by batch 36**. `make train-full` (3.59M pairs, 36 batches) trains effectively on the first ~500k pairs and no-ops the remaining ~3M. Training on more data currently makes the reranker *worse*. | `train.rs:414,429` |
| D6 | S2 | No regularization, no held-out early stopping, no dev-loss monitoring on a 2^20-slot (1M-parameter) sparse table. Shipped table has 18,041 non-zero slots; quantization scale is set by a single outlier (`min −127, max +42`), compressing every other weight's resolution. | `train.rs:500–510,528` |
| D7 | S1/S2 | No end-of-word symbol in the LM. `word_start` exists, `word_end` does not. Trigram mass is deficient by exactly the word-final fraction (`disc` divides by `c_ab`, which counts `(a,b)` occurrences with no successor), and the model cannot express "this is a plausible way for a word to end" — directly relevant since ~52% of top-1 misses are matra-only. | `em_trainer.rs:build_kn_lm` |
| D8 | S4 | Trigram interpolates against the **highest-order** bigram (`model.bigram_weight(b,c)`), not the continuation estimate `N1+(•,b,c)` KN requires. The bigram→unigram step does this correctly; the trigram→bigram step does not. | `em_trainer.rs:build_kn_lm` |
| D9 | S2 | EM discards words whose forward mass `z < 1e-12`. f64 subnormals start near 1e-308, so this threshold is ~296 orders of magnitude too conservative: long or flat-emission words are silently dropped, biasing EM toward short easy words. | `em_trainer.rs:581,693` |
| D10 | S4 | `weight as u64` truncates fractional weights to 0; when it does (or under `set_lm_ingestion(false)`), the `if *e == 0` guards re-increment `continuation` and `distinct_bigrams` on *every* occurrence instead of the first, corrupting KN continuation counts. Latent only because `train.rs` passes integer weights ≥ 1. | `em_trainer.rs:173,182,191` |
| D11 | S3 | `TranslitModel::akshara_id` is a linear scan over the akshara vocabulary. `build_corpus_symspell` calls it ~500k times at startup — most of the 4.3 s cold start. | `translit_model.rs:64`, `engine.rs:82` |
| D12 | S3 | Beam dedup key `(pos, prev2, prev, phash)` is fully redundant — the path hash determines the other three, so no states ever merge and a HashMap is built per step for nothing. | `decoder.rs:251–254` |
| D13 | S3 | `rerank_with_table` runs `akshara::segment` plus ~15 `str::matches` scans and `HashMap<String,_>` lookups per candidate × 50 candidates per keystroke. This is the 3.4 ms. | `reranker.rs:316–330` |
| D14 | S4 | ~1,350 lines of dead code: `crf.rs` (702) is never constructed — nothing sets `DecoderConfig::crf` to `Some`; `pair_model.rs` + `finalize_pair` (~650) has zero callers and is not in `UnifiedModel`. The MKN work in the last commit was documented as applied to these. | `crf.rs`, `pair_model.rs`, `em_trainer.rs:808` |
| D15 | S1 | No accuracy regression test. A 31-point drop shipped through 92 green tests. | `tests/` |
| D16 | S3 | `analyze_errors` and `probe_model` hardcode `data/translit_model.bin` / `data/word_freq_text.bin`, which the unified-container pipeline no longer produces. Both panic. This is the W0 measurement gate the previous plan says to run first. | `analyze_errors.rs:112,123`, `probe_model.rs:13` |
| D17 | S4 | Docs overstate on six counts — see §6. | `README.md`, `docs/` |

---

## 2. Phase A — stop the bleeding (no training, ~half a day)

Recovers the 31 points already lost. Nothing here needs a retrain.

**A1. Fix the fuzzy score band and verify distances.** (D1, D2)
Move corpus-fuzzy strictly below the decoder band and gate it on a real edit
distance. Concretely: score it as a *fallback* tier (base ≈ 5k, alongside
`LEXICON_ONLY_SCORE`) rather than a *dominant* one, and run bounded
Damerau-Levenshtein against the query with a per-distance penalty before
admitting a candidate. Reuse the existing `bounded_levenshtein` /
`min_roman_distance` helpers the user-fuzzy path already uses — the machinery
exists, path 4b just skips it.
*Expected: 50.95 → ~81.7 top-1. Verify the fuzzy path then earns >0 recall; if
it still contributes nothing at the corrected band, delete it.*

**A2. Add the accuracy regression test.** (D15)
A 200–400 case sample from the test split asserting AK-Freq top-1 ≥ 80% and
top-5 ≥ 90%, run in CI. Cheap, and it is the control that makes every later
experiment trustworthy. Do this **before** A1 so it fails first, then goes green.

**A3. Repair the measurement tooling.** (D16)
Point `analyze_errors` and `probe_model` at `UnifiedModel` instead of the
deleted loose `.bin` files, and make the dataset path a flag. Keep
`examples/diag_reranker.rs` (added during this audit) as the reranker-health
probe: it reports sparse-table occupancy, quantization saturation, and dev-loss
per epoch. Without these three, Phase C is unmeasurable.

**A4. Cold-start and per-keystroke latency.** (D11, D12, D13)
Build an `akshara → id` `FxHashMap` once on model load; drop the no-op beam
dedup; cache per-candidate reranker features (`segment` result, matra counts)
so they are computed once per candidate string rather than once per keystroke.
*Expected: 4.3 s → well under 1 s cold start; 3.4 ms → ~1–1.5 ms. Sub-ms likely
needs Phase D as well.*

**Gate:** AK-Freq top-1 ≥ 81.5% at default settings, regression test green.

---

## 3. Phase B — make training correct (before the overnight run, ~1 day)

Every item here changes what the overnight job produces. Landing them after the
run means running it twice.

**B1. Unclamp the MKN discounts.** (D3)
`d1.clamp(0.0, 1.0)`, `d2.clamp(0.0, 2.0)`, `d3.clamp(0.0, 3.0)`; extend the
degenerate guard to `n[1]==0 || n[2]==0 || n[3]==0` and fall back to the fixed
`0.75` in that case. Assert `0 ≤ λ < 1` for every context as a training-time
invariant.

**B2. Fix the LR schedule.** (D5)
Decay per *global epoch*, not per batch — or use a flat `lr` with AdaGrad's own
adaptation, which is what AdaGrad is for. Sanity check: the last batch's
effective lr should be within ~1 order of magnitude of the first's, not nine.
*This is the single change that makes `train-full`'s extra 3M pairs count.*

**B3. Recompute the dense weights and normalization stats.** (D4)
Either restore a `train_reranker` stage that fits `W_DENSE` on the current
model, or — simpler and strictly better — **fold the 29 dense features into the
same softmax objective as the sparse table** so both are learned jointly against
the current EM/LM output. In either case `MEAN_DENSE`/`STD_DENSE` must be
recomputed from the new decode distributions, not inherited. Regenerate
`reranker_weights.rs` as a build artifact of the run.

**B4. Regularize and early-stop.** (D6)
L2 on the sparse table (or a magnitude floor that prunes slots below a
threshold at pack time), a held-out split (`valid_devanagari.jsonl` is already
there and unused by the reranker), dev loss printed per epoch, and stop on dev
regression. Set the quantization scale from a high percentile (e.g. 99.9th) with
explicit clipping instead of from the single max weight.

**B5. Add the end-of-word symbol.** (D7)
`word_end[a] = −log P(</w> | a)` built symmetrically with `word_start`, added
once at path completion in the decoder. Fixes the trigram mass deficiency and
gives the LM a way to reject implausible word endings.

**B6. Fix the EM underflow guard.** (D9)
Scale each forward/backward column and accumulate in log space (or track a
per-column scale factor). Log how many pairs the old threshold was discarding —
that number is the size of the training set you have been leaving on the floor.

**B7. Fix the latent count-corruption guards.** (D10)
Make `continuation`/`distinct_bigrams` increment on genuine first insertion
(`Entry::Vacant`), independent of the weight value. Keep weights as `f64`
throughout or document the integer contract.

**B8. Resolve the dead code.** (D14)
Either wire `crf.rs` into `UnifiedModel` and the decoder config with a trained
model behind it, or delete it and `pair_model.rs`. 1,350 lines of unreachable
code is why the last commit's math landed in modules that never ship. If the
lattice CRF is still wanted (it is S3 in the old roadmap), that is Phase C work
— but it should not sit half-built in the tree meanwhile.

**Gate:** a `make train-quick` run reproduces Phase A accuracy within noise, and
dev loss decreases monotonically with a sane final learning rate.

---

## 4. Phase C — the overnight run and the climb to 90 (multi-cycle)

Headroom, using the previous plan's top-50 oracle of 94.40% (re-verify with the
repaired `analyze_errors` first — this is the one number in the old plan I could
not check, because the binary panics):

```
ranking gap    94.40 − 81.74 = 12.66pp   ← where the work is
generation gap 100   − 94.40 =  5.60pp
```

**C1. Retrain on the full 3.59M with B1–B7 landed.** This is the overnight job.
With the LR fix (B2) and joint dense+sparse training (B3), this is the first run
in which the extra ~3M pairs actually reach the objective. Re-sweep γ afterwards
— **if the reranker is fixed, γ should move well above 0.3.** γ staying at 0.3
is the signal that B3/B4/B6 did not take, and is the cheapest diagnostic you
have.

**C2. Factored matra model.** The previous plan's centrepiece, and the analysis
behind it is sound: ~52% of top-1 misses are matra-only, so a model that
resolved vowel-length/nasal assignment perfectly would score ~91.1%. This
remains the highest-value single lever, and B5 (`word_end`) is a cheap down
payment on it — most matra errors are word-final.

**C3. Fix the trigram's lower-order estimate.** (D8) Build the continuation
counts `N1+(•,b,c)` and interpolate against those. Textbook KN; currently the
trigram backs off to a raw-count bigram, which over-weights frequent bigrams in
exactly the contexts where backoff matters most.

**C4. Reconsider the candidate-union score space.** The u64 additive bands
(`FRESH_SCALE`, `LEXICON_ONLY_SCORE`, `user_trie_base`, `corpus_fuzzy_base`) are
hand-tuned magic numbers in which D1 was possible in the first place, and they
discard the reranker's calibration at union time by squashing it through
`800_000/(1+cost)`. Replacing the union with a single log-linear score — every
source contributing a *feature*, not a band — removes an entire class of defect
and is a prerequisite for tuning the sources jointly rather than by hand.

**Order of expected value:** C1 (unblocks everything) → C2 (largest single
lever) → C4 (removes defect class, enables joint tuning) → C3 (textbook
correctness, modest points).

---

## 5. Phase D — sub-millisecond

Phase A4 should land ~1–1.5 ms. Closing to sub-ms, in order of value:

1. **Rerank fewer candidates.** Depth 50 is fixed. A cascade — score all 50 with
   the cheap heuristic, run full dense+sparse features on the top ~10 — cuts D13
   by ~5× at negligible accuracy cost. This is the big one.
2. **Intern the vocabulary.** Replace `HashMap<String, u32>` frequency lookups
   with an id-keyed table resolved once at decode time; the reranker currently
   hashes Devanagari strings repeatedly per keystroke.
3. **Incremental decoding.** An IME re-decodes a growing prefix on every
   keystroke and throws away the lattice each time. Caching the beam per prefix
   and extending it by one character is the structural fix, and it is worth more
   than all the micro-optimization above combined.
4. Re-measure only after 1–3; do not chase constants before the cascade lands.

**Size** is not currently a problem: 11.9 MB desktop / 9.34 MB browser against a
4.92 MB Brotli budget that is already met. B4's magnitude pruning of the sparse
table (18,041 of 1,048,576 slots are non-zero) should shrink it further. No
dedicated size phase is warranted.

---

## 6. Phase F — documentation truth pass

Correct, with the measured numbers:

- `README.md:15` — "top-1 **81.45%**" → the shipped default measures 50.95%; the
  corrected default should measure ~81.7%. State which config the number is for.
- `README.md:13` vs `README.md:26` — 11.1 MB and 30.59 MB for the same artifact.
  Actual: `akshar.model` 11.9 MB, `akshar_wasm.model` 9.34 MB (documented 8.91).
- `MODEL_TRAINING_AND_OPTIMIZATION.md` — "sub-ms … intact" → 3.4–3.6 ms.
  *(corrected 2026-09-06)*
- `MODEL_TRAINING_AND_OPTIMIZATION.md` §7.1–7.2 — MKN attributed to `AkLm`/
  `PairModel`, which never ship; phonetic edit costs described but not
  implemented. *(corrected 2026-09-06)*
- `ARCHITECTURE.md:133` and `MODULES.md:226` — document corpus bigrams and
  `data/word_bigrams.bin`, both removed in `UnifiedModel` v4.
- `plans/2026-09-05-mathematics.md:169` — still states absolute discount δ=0.75.
- Nothing documents that `crf.rs` and `pair_model.rs` are unreachable.

**Standing rule:** no accuracy or latency claim in any doc without the config it
was measured under and the date. Every number in this file has both.

---

## 7. Sequencing

```
A1 A2 A3 A4   half a day, no training      →  recovers ~31pp, ~2-3x latency
B1..B8        one day, no training          →  makes the overnight run count
C1            overnight                     →  first run where 3.59M pairs land
C2 C4 C3      multi-cycle                   →  the climb to 90
D1..D3        parallel with C               →  sub-ms
F             continuous                    →  after each measured change
```

**Do not start C1 until Phase B is landed.** D3, D4 and D5 each independently
corrupt the result; running overnight on the current trainer produces a model
whose LM is over-discounted, whose dense features are normalized against stale
statistics, and whose sparse table has seen ~14% of the corpus.

## 8. What would falsify this plan

- If, after B1–B7 and C1, the γ sweep still peaks at ≈0.3, the reranker's problem
  is not supervision or calibration and the diagnosis in §0 is wrong.
- If the repaired `analyze_errors` reports a top-50 oracle materially below
  94.40%, the ranking/generation split above is wrong and C2 should yield to
  generation work.
- If A1 leaves the corpus-fuzzy path at zero measured recall, it should be
  deleted rather than repaired, and D2 becomes moot.


---

## 9. Landed 2026-09-06

### Phase A — no retraining required

| Defect | Change | Measured effect |
| :-- | :--- | :--- |
| D1, D2 | **Corpus-SymSpell candidate source removed entirely.** Given a fair test (edit distances verified against the indexed roman form, distance penalty applied), it still cost 9.6pp top-1 at a band where it could compete and was byte-identical to being switched off at any safe band — 1945/2108 either way. It never contributed recall at any setting. The vowel-length alternations it duplicated (`ee→ii`, `oo→uu`) are already handled upstream by `normalizer.rs`. | AK-Freq top-1 **50.95 → 81.74%**, top-5 **81.07 → 92.27%**; NEI 25.68 → 47.5%, NEF 20.81 → 31.21% |
| D15 | `tests/accuracy_regression.rs` — 400-case AK-Freq sample, floors at top-1 ≥ 76% / top-5 ≥ 88%. Skips cleanly when `data/` is absent. | sample reads 81.50 / 91.50, tracking the full split |
| D16 | `analyze_errors` and `probe_model` read `UnifiedModel` instead of the deleted loose `.bin` files; `--dataset`/`--model` are now flags. | the W0 measurement gate runs again |
| D11 | `TranslitModel::akshara_index` — O(1) akshara-string lookup, built on load. | cold start **4.35 s → 2.42 s** |
| D12 | No-op beam dedup removed (the key was a function of the path hash, so it could only merge on collisions). | accuracy unchanged, one hash map per step saved |
| D13 | Rerank cascade: full dense+sparse extraction for the top `AKSHAR_RERANK_DEPTH` (default 24) by heuristic rank; the tail keeps heuristic order below the scored block. | depth 24 measures **81.83 / 92.22**, no worse than scoring all 50 |

### Phase B — changes the overnight run's output

| Defect | Change |
| :-- | :--- |
| D3 | Discount clamps corrected to the Chen-Goodman bounds `0 ≤ D_i ≤ i` (were `0.9/0.9/0.95`); `n[3] == 0` added to the degenerate-fallback guard. |
| D5 | Learning-rate schedule rebuilt: the per-batch `0.8^(epochs-1) × 0.995` compounding is gone, replaced by a single anneal to `LR_FINAL_FRACTION` (0.1) of the initial rate across the whole run. The last batch of a 36-batch run now trains at **0.005 instead of 4.4e-9**. |
| D4 | `UnifiedModel` **v5** carries `dense_mean`/`dense_std`, computed during training with Welford's algorithm over every candidate scoring. `reranker::DenseNorm` prefers the model's statistics and falls back to the compiled-in constants for v1-v4 containers. The frozen `MEAN_DENSE`/`STD_DENSE` can no longer silently decalibrate a retrained model. v4 stays byte-readable. |
| D6 | Sparse-table quantization scale now comes from the 99.9th percentile of non-zero weights with explicit clipping, not from the single largest weight (the shipped table had one weight at −127 while the rest of the distribution sat in a few levels). Saturation count is reported. |
| D9 | **Scaled forward-backward.** Each forward column is rescaled and the same factors are divided out of the backward pass so they telescope, leaving a single `1/c_j` correction per column. The `z < 1e-12` floor — 296 orders of magnitude above the f64 subnormal limit — is now `z <= 0.0`. Three tests cover it: equivalence to an unscaled reference implementation, posterior mass summing to the observation weight, and a 12-akshara word at `z ≈ 2.4e-16` that the old floor discarded. |
| D10 | Continuation/`distinct_bigrams` counts key off `Entry::Vacant` (genuine first insertion) rather than `*e == 0`; weights round rather than truncate. |

### Still open

- **D7** (no end-of-word symbol), **D8** (trigram interpolates against the
  highest-order bigram rather than continuation counts) — both need a model
  format change and belong with the next retrain.
- **D14** — `crf.rs` (702 lines) and `pair_model.rs` (~650) remain unreachable.
  Left in place deliberately: the lattice CRF is S3 in the older roadmap and
  deleting a half-built planned feature is the author's call, not a cleanup.
- **D17** — docs corrected in this pass; see §6.

### Corrected: the oracle, now that `analyze_errors` runs

Measured at beam 256 on the full test split, replacing the previous plan's
unverifiable 94.40%:

| | AK-Freq | ALL |
| :--- | ---: | ---: |
| engine top-1 (strict / multi-ref) | 81.74% / 82.07% | 61.89% / 62.23% |
| decoder oracle @2 | **89.8%** | 71.4% |
| decoder oracle @5 | 92.4% | 78.1% |
| decoder oracle @50 | **94.3%** | 85.1% |
| matra-only share of misses | **51.9%** | 37.2% |
| top-1 if the matra class were solved | **91.22%** | 76.08% |
| CER (engine top-1) | 3.90% | 11.66% |

Two things this sharpens:

1. **Oracle@2 is 89.8%.** The ranking headroom is not spread across 50
   candidates — 8.1 of the 12.6 available points are a binary decision between
   the top two. A discriminator trained specifically on that pair is a smaller,
   better-posed problem than a 50-way ranker, and it is most of the way to 90.
2. **The matra number holds.** 51.9% of AK-Freq misses are matra-only, and
   solving that class alone reaches 91.22% — confirming C2 as the right
   centrepiece, and D7 (`word_end`) as its cheapest down payment, since matra
   errors concentrate word-finally.


---

## 10. 100k smoke train, 2026-09-06

`train --reranker-pairs 100000 --epochs 3 --iterations 12` on the post-Phase-B
tree, evaluated against the pre-existing `data/akshar.model`.

| | baseline `akshar.model` | smoke `akshar_smoke.model` |
| :--- | ---: | ---: |
| AK-Freq top-1 / top-5 | 81.83% / 92.22% | 81.55% / 92.31% |
| AK-NEI top-1 / top-5 | 47.79% / 69.81% | 48.04% / 69.47% |
| AK-NEF top-1 / top-5 | 31.21% / 53.00% | 30.60% / 53.00% |
| All top-1 / top-5 | 62.0% / 78.0% | 61.79% / 77.93% |

**Result: no measurable accuracy change.** At n=2,108 one standard error on
AK-Freq is ≈0.84pp, so −0.28pp is noise. State this plainly rather than
attributing it to any of the Phase B changes.

### What the smoke run did establish

1. **The pipeline runs end-to-end** with every Phase B change, and the v5
   container round-trips through save/load.
2. **The reranker trains.** Loss 1.6529 → 1.5881 → 1.5691 across three epochs,
   train top-1 54.08% → 55.28% → 55.65%.
3. **D4 was real, and is now contained.** The run's own dense statistics come
   out at `emit` mean 5.562 / std 3.300 and `lm` mean 24.638 / std 7.030,
   against compiled-in constants of 4.865 / 3.275 and 21.831 / 5.738. The `lm`
   feature moved 0.49σ in mean and 22% in spread from a *100k* retrain. Before
   v5 those weights would have been applied to mis-standardised inputs with
   nothing to detect it.
4. **D3 was real.** The true discounts measure `d1=0.6049 d2=1.0362 d3=1.4443`
   (bigram) and `d1=0.6696 d2=1.0949 d3=1.4330` (trigram). Under the old clamps
   d2 was cut to 0.9 (−13%) and d3 to 0.95 (−34%), pinning both at the ceiling
   exactly as predicted. The discounts now span 0.60–1.44.
5. Quantization saturation dropped to 19 slots of 1,048,576, with the clip at
   the 99.9th percentile (1.72990) rather than at a lone outlier.

### What the smoke run could NOT establish

**It does not exercise D5, the learning-rate fix — the single change with the
most at stake for the overnight run.** `use_chunked` requires
`num_train_pairs > 200_000`; a 100k run takes the non-chunked path, where the
old schedule was already sane. The collapse only compounds across batches.

To validate D5, run **`--reranker-pairs 500000`** (5 batches, ~40 min) and check
that the per-batch loss keeps falling into the last batch instead of flattening
after the second. That is the gate for `train-full`, not this smoke test.

### Standing question: does modified KN earn its place?

The discounts are now correct, and correcting them changed nothing measurable.
The comparison arm — `AKSHAR_KN_FIXED_DISCOUNT=1`, a single δ=0.75 — decides
whether the three-discount machinery is worth keeping at all. If it measures
the same, MKN is complexity without benefit and should be deleted in favour of
the one-line absolute discount.


---

## 11. Component ablation with paired significance, 2026-09-06

Every earlier ablation in this repo compared two independent accuracy numbers.
That understates the evidence: both systems see the same cases, so the question
is not whether the marginals differ but whether the *discordant* cases lean one
way. These use **McNemar's test** on paired per-case outcomes
(`examples/ablate_paired.rs`), two-sided, exact binomial on the discordant pairs.

`w->r` = cases the full system got wrong and the ablated one got right;
`r->w` = the reverse. Reproduce with the `AKSHAR_*` switches in `core::ablation`.

### Removing the 2^20 sparse reranker table (~1M parameters, 1.00 MB)

| split | n | full | ablated | delta | w->r | r->w | p |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| AK-Freq | 2108 | 81.83% | 81.93% | +0.09 | 15 | 13 | 0.851 |
| AK-NEF | 817 | 31.21% | 29.38% | −1.84 | 3 | 18 | **0.0015** |
| AK-NEI | 1176 | 47.79% | 46.85% | −0.94 | 13 | 24 | 0.099 |
| ALL | 4101 | 61.98% | 61.40% | −0.59 | 31 | 55 | **0.013** |

**Verdict: keep, narrowly.** It does nothing at all on native words (p=0.85 —
the +0.09 is noise), and its entire measurable value is named entities, where
it is significant on AK-NEF and on the pooled set. A paper reporting only
AK-Freq cannot justify 1M parameters for it; one that reports named entities
can.

### Removing the 29-feature dense reranker (gamma = 0, heuristic only)

| split | n | full | ablated | delta | w->r | r->w | p |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| AK-Freq | 2108 | 81.83% | 81.02% | −0.81 | 15 | 32 | **0.019** |
| AK-NEF | 817 | 31.21% | 30.23% | −0.98 | 15 | 23 | 0.256 |
| AK-NEI | 1176 | 47.79% | 46.17% | −1.62 | 18 | 37 | **0.015** |
| ALL | 4101 | 61.98% | 60.91% | −1.07 | 48 | 92 | **0.0003** |

**Verdict: keep.** Significant on the headline split and highly significant
pooled. Small in absolute terms, but real.

### The two are not independent

"Remove both" reproduces "remove dense" **exactly** — identical accuracies and
identical discordant counts on every split. At `gamma = 0` the blend returns
`-heur` and the sparse contribution is computed and then discarded, so the
sparse table can only ever act *through* the dense score. Any future ablation
must vary them in that order; reporting them as two independent +Xpp
contributions would be wrong.

### Standing conclusion

Both survive their own removal test, so neither is dead weight. What the
numbers do say is narrower than the docs used to claim: on native words the
whole discriminative stack is worth **+0.81pp**, and the 1M-parameter sparse
table contributes **none** of it.
