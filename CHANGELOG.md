# Changelog

## v1.2.0 — 2026-09-08

Engineering-health release: hand-rolled plumbing replaced with maintained
crates, plus an intuitive evaluation harness. **No accuracy change** — every
figure from v1.1.0 reproduced identically (`make eval`: 81.83/47.79/31.21;
`make eval-full`: 61.98%, MRR 0.6906; `make ablate`: all six rows identical).

### Replaced with libraries

- `evaluate` argument parsing → `clap 4` derive (same flags/defaults).
- Bootstrap PRNG (hand-rolled SplitMix64) → `rand_chacha::ChaCha8Rng`
  (seed 42, same protocol; no modulo bias).
- Devanagari classification input → NFC-normalized via
  `unicode-normalization` before the engine's own akshara segmenter.
- Test roman-key index/dedup → `fst::Set`; grapheme CER →
  `unicode-segmentation`; hot buffers → `smallvec`.
- `Trie::find_word_id_by_devanagari` O(n) linear scan → in-memory index
  (on-disk format unchanged, rebuilt on load).
- `criterion` benches added (`benches/decode.rs`); wasm lib check still clean.

### Added

- `make eval-ime`: plain-language report (correct-first-time, visible-in-top-5,
  keystrokes-saved) plus machine JSON at `docs/generated/eval.json`, the single
  source every number in the docs regenerates from.
- Two findings disclosed in the manual: 16 duplicate roman keys in the test set,
  and prefix completion saving only ~2.7% keystrokes.

### Docs

- `data/README.md`: fixed stale container size (30.59 → 11.37 MB) and result
  (82.02/92.17 → 81.83/92.22%); Brotli 4.92 → 4.94 MB.
- `README.md` / `docs/MANUAL.md`: re-verification provenance (2026-09-08),
  honest IndicXlit reranked figures (86.6% native with LM rerank), Nepali-scope
  and IBus-tag disclosures.

## v1.1.0 — 2026-09-06

A correctness, performance, and cross-platform release. Includes native IBus engine, WebAssembly browser build, trained unified models, and source documentation. Every figure below was measured on the 4,101-case Aksharantar test split; see `docs/MANUAL.md` for method.

### Highlights

- **Accuracy Recovery**: Recovered 30.79pp of native top-1 (50.95% → 81.83%).
- **5.2× Faster Decoding**: 3.5 ms → 0.67 ms per query with byte-identical output.
- **WebAssembly Build**: Fully verified browser runtime via `wasm32-unknown-unknown` and JS wrapper (`js/akshar-ime.js`).
- **42-Page Source Manual**: Mathematical, architectural, and evaluation manual (`docs/MANUAL.md` / `AksharIME-Manual.pdf`).

### WebAssembly & Cross-Platform

- Verified WebAssembly compilation target (`wasm32-unknown-unknown`).
- `make check` now runs both `check-native` and `check-wasm` (compiling and linting under `-D warnings`), guaranteeing browser compatibility across releases.
- Clean JS API (`createEngine(model, lexicon, weights)`) for web applications with offline transliteration.

### Fixed — accuracy

- **Recovered 30.79pp of native top-1** (50.95% → 81.83%). The corpus-wide
  SymSpell source scored at 850,000, above the decoder band (`FRESH_SCALE`
  800,000), so any fuzzy hit displaced the reranker's top-1 — and it verified no
  edit distances. Given a fair test (distances verified, per-edit penalty) it
  still cost 9.6pp at a competing band and was byte-identical to being disabled
  at any safe band. Removed.
  Named entities recovered likewise: AK-NEI 25.68 → 47.79%, AK-NEF 20.81 → 31.21%.

### Fixed — training mathematics

- **Modified Kneser-Ney discounts** were clamped to ≤ 0.9/0.9/0.95. Chen-Goodman
  requires `0 ≤ D_i ≤ i`; this corpus yields `D₂ = 1.036`, `D₃ = 1.444`, so both
  were pinned at the ceiling and MKN had degenerated into absolute discounting.
- **Scaled forward-backward in EM.** The old `z < 1e-12` floor sat 296 orders of
  magnitude above the f64 subnormal limit and silently discarded long or
  flat-emission words from training. Now Rabiner-scaled with a per-column
  `1/c_j` posterior correction, covered by three tests including equivalence to
  an unscaled reference.
- **Learning-rate collapse.** The schedule compounded `0.8^(epochs-1) × 0.995`
  *per batch* — 4.4e-9 by batch 36, so `train-full` trained on roughly the first
  500k pairs and no-opped the rest. Now anneals to 10% of the initial rate
  across the whole run. Validated on a 5-batch run.
- **Stale normalisation statistics.** `MEAN_DENSE`/`STD_DENSE` were compiled-in
  constants no training stage refreshed; a retrain moves `lm` by 0.49σ. Container
  **v5** now carries `dense_mean`/`dense_std`. v1–v4 still load.
- Continuation counts key off first insertion, not a non-zero weight.
- Sparse-table quantisation scale from the 99.9th percentile, not a lone outlier.

### Performance

- **3.5 ms → 0.67 ms per query (5.2×), output byte-identical.** `bigram_weight`
  and `trigram_weight` scanned successor rows linearly, ~50,000 scans per query.
  Rows are stored ascending, so this is a binary search.
- O(n) beam pruning via `select_nth_unstable_by`; arena cells materialised only
  for pruning survivors.
- **Cold start 4.35 s → 1.55 s** — `akshara_id` was a linear scan called ~500k
  times at start-up.

### Removed

~1,862 lines, no measurable accuracy change:

- `crf.rs` (702) — never constructed; `DecoderConfig::crf` was always `None`.
- `pair_model.rs` + `finalize_pair` + `pair_transitions` (816) — zero callers,
  never packed into the container.
- `lexicon.rs` + `build_lexicon` (344) — dead by construction; the shipped path
  never loaded one. 0.00pp on every split.

`ImeEngine::from_bytes` and the WASM `createEngine` keep their `lexicon`
parameter as an ignored no-op, so existing JS callers still work.

### Added

- **`docs/MANUAL.md` / `AksharIME-Manual.pdf`** — a 42-page source manual:
  mathematics, architecture, training, evaluation method, ablations with
  McNemar significance, performance, deployment, defect register, roadmap.
- `tests/accuracy_regression.rs` — a 30.8pp regression once shipped through 92
  green unit tests. Now guarded.
- `tests/fuzzy_behavior.rs` — pins fuzzy behaviour and records defect D18.
- Held-out dev monitoring during reranker training, with best-by-dev packing.
- Runtime ablation switches (`core::ablation`) so the ablation table is
  reproducible from the shipped binary.
- `make manual`, `make check`, `make release-check`, `make ablate`,
  `make profile`, `make train-mid`.

### Measured, and honest about it

- The discriminative reranker used alone is **worse** than the 3-parameter
  heuristic (76.47% vs 81.02%). Net contribution +0.81pp.
- 5× more reranker training data changed nothing. Cause: `W_DENSE` has never
  been refit by this pipeline (manual §12.3).
- Modified Kneser-Ney is not measurably better than a single δ = 0.75.
- The 2²⁰ sparse table does nothing on native words (p = 0.851); its value is
  named entities only.
- Defect **D18**: the user fuzzy path is structurally unreachable.

### Known limitations

One language, one test set. Named entities well behind the neural baseline.
Browser profile not re-measured since this release.
