# Changelog

## Unreleased

### Shared lexicon: compaction, not cross-lingual transfer (measured, 2026-09-22)

The shared automaton (previous entry) could plausibly help a low-resource
language two ways: ranking could leak another language's frequency signal
in, or the dictionary-constrained decode pass could surface a word as
generatable because *some* language attested it, not necessarily the query's
own. The first is impossible by construction (`lexicon.lang_index` resolves
strictly per-language). The second is a real code path
(`LexiconDict::is_word` checks only FST finality, language-blind) — so it
was tested rather than assumed: built an isolated lexicon from only Bodo +
Dogri's text (`examples/swap_lexicon.rs` swaps it into a copy of the shipped
model, EM/LM/reranker weights unchanged), and diffed `eval_langs` against
the shared 8-language original. Bodo's test hit counts were **identical**,
935/2244 top-1 and 1270/2244 in-list@8 either way; Dogri differed by 1 and 4
cases out of 2,000, inside one standard error. `docs/MANUAL.md` §8.7 has
the full ablation and the likely reason (the lexicon's `count >= 2` floor is
low enough that real Bodo vocabulary mostly clears it unassisted). The
"Contribution and prior art" claim (§2.1) is corrected to state plainly what
this does and does not show, rather than leaving the transfer question
open for a reader to assume the favourable answer.

### Multi-language Devanagari support: shared lexicon, language-conditioned ranking (2026-09-22)

The engine served one language's word-frequency priors (Nepali) under a
"language-agnostic" label — the phonetics were script-general, but ranking
was capped by whichever language the shipped vocabulary happened to be built
from. Now genuinely multi-language: Hindi, Marathi, Nepali, Sanskrit,
Konkani, Maithili, Bodo, Dogri, one phonetic engine, one shared ranking
automaton, per-language conditioning. MANUAL.md §2, §8.6, §17.1 have the full
writeup; summary here.

#### Added — the shared lexicon (`src/core/lexicon.rs`, `build_lexicon`)
- One minimal FST across all eight languages' words, keyed one byte per
  Devanagari codepoint, values a de-duplicated palette of per-language
  log-quantised frequency levels. Replaces the single-language
  `HashMap<String, u32>` vocabulary and `WordTrie` at once — same automaton
  is both the ranking frequency source and the decoder's dictionary
  constraint (`Dictionary` trait, `decoder.rs`; `WordCounts` trait,
  `reranker.rs`; both old implementations still work behind the same traits).
- Built from an IndicCorp v2 **sample** per language (~300 MB/language,
  pinned + SHA-256 verified, `scripts/data-manifest.tsv`,
  `make data-fetch`), not the 113 GB full corpus — scoped down mid-session
  after reconsidering the minimalism/accuracy tradeoff.
- Container v8 (`unified.rs`): `lexicon: Option<LexiconData>`, supersedes
  `vocab_freq` when present. v1–v7 files still load.
- Language conditioning via feature augmentation [Daumé III 2007]: shared +
  per-language-tagged sparse features, an 8×29 `dense_lang_weights` offset
  matrix, language dropout in training, per-language `gamma_lang`
  (`calibrate_blend`). `ImeEngine::set_language`/`languages()`.

#### Fixed — three reasons the reranker never measured as beating the heuristic
- **Reranker trained on one language.** Aksharantar's train file is sorted
  by language; `--reranker-pairs 100000` trained on Hindi only, regardless
  of how many languages the EM/LM saw. `prepare_pairs` now shuffles
  (fixed seed) before any pair count is taken.
- **EM/LM memorized the reranker's own eval pairs.** Nothing partitioned
  train from valid/test at the word level. `prepare_pairs` now drops any
  train pair whose native word also appears in valid or test.
- **Dense-feature normalization drifted from training to inference**
  (15–25% scale mismatch after a retrain moved the `emit`/`lm`
  distributions). `train.rs` now recomputes `dense_mean`/`dense_std` via
  Welford's algorithm every run instead of trusting stale v5 constants.

#### Fixed — three more, found auditing the decoder/EM aligner for the above
- **Beam search mixed candidate lengths**, biasing toward short spellings.
  Rewritten position-synchronous (`stacks: Vec<Vec<Hyp>>` by Roman byte
  position). +4.6pp macro in-list@8, no latency cost; beam width halved
  (64→32) afterward with no further loss.
- **`</w>` end-of-word LM signal trained but never charged at decode time**,
  and pruned out of packed containers as apparently unused.
  `TranslitModel::end_weight` now applies it. +1.2pp macro top-1
  (Sanskrit +3.6pp).
- **Sanskrit avagraha (U+093D) and inverted chandrabindu (U+0900) segmented
  as phonetically-emitting aksharas**, corrupting EM alignment for 51,000+
  Sanskrit pairs. Reclassified as combining marks.

#### Added — evaluation
- `eval_langs` (`make eval-langs`): per-language, per-stratum top-1/top-3/
  in-list@k/MRR, `--lang-aware`, `--ceiling n` (generation vs. ranking
  loss), lenient scoring (nukta/chandrabindu/anusvara equivalence classes).
- `eval_session` (new): cold, uniformly-weighted accuracy is the right
  number for cross-paper comparison but understates real use two ways —
  real typing is frequency-weighted, not uniform, and the engine remembers
  corrections (`user_confirms`) that `eval_langs` never exercises.
  Frequency-weighted sampling + a live learning loop, pooled over 8
  languages × 20,000 draws on `data/akshar.model`: cold 73.37%, 2nd
  encounter 99.62%, blended session 97.28% (top-1). Reported as a ceiling
  under a charitable correction-always-lands assumption, not a replacement
  for the cold number.
- `data/pairs/{train,valid,test}.jsonl` (`prepare_pairs`, `make
  data-prepare`): 7,308,261 / 30,565 / 48,277 pairs across 8 languages,
  replacing the single-language `data/aksharantar/*.jsonl` split for all
  multi-language numbers. Per-language counts are far from uniform (Bodo
  34,480 / Dogri 1,276 train pairs vs. 1.2–2.3M for the other six) — both
  languages' `in@50` generation ceiling is measurably lower, a data
  problem `--reranker-pairs` cannot fix.

#### Changed — model size
- Desktop container capped at 24.44 MB (300k-word/language lexicon +
  relative-entropy trigram pruning, `prune_lm`, threshold swept
  empirically rather than reusing the old single-language default) to fit
  a 25 MB budget. Cost of the cap, measured: pooled native top-1 60.69% vs.
  62.61% uncapped/unpruned (−1.9pp for −13 MB).
  Browser: `make web-model` output grew to 23.30 MB / 13.97 MB Brotli
  (~3× the single-language 4.94 MB Brotli baseline) because the shared
  lexicon ships in both containers unchanged; a browser-specific smaller
  lexicon cap is the lever if that needs to come down further and has not
  been built.

#### Docs
- MANUAL.md: multi-language architecture (§2, §8.6), corrected "not yet
  multi-language" scope claims (§17.4), new defect-register entries
  (§17.1, D19–D24), evaluation methodology for the new pipeline and
  `eval_session` (§12), rebuilt Source map (§16) and container-version
  table (§10.2) against the current tree, References additions ([Daumé III
  2007]; FST-transliteration prior art). Also corrected, while verifying
  citations against their actual publications: a reference's title
  ([Kirov et al. 2024] was cited as "...for Input Method Editors"; the
  paper is "Context-aware Transliteration of Romanized South Asian
  Languages") and roughly fifteen `§N` cross-references left pointing at
  the wrong section by an earlier reorganization this changelog does not
  otherwise record — neither audit is claimed exhaustive.

### Sentence context via corpus bigrams
New information, not new parameters: the engine now blends P(cur | prev)
from a pruned corpus bigram table into the pre-squash reranker margin
(engine step 1b). `set_context_word` was an empty stub — `evaluate_sentences`
measured isolated words while claiming context — and is now the context half
of `user_confirms` (sets prev, clears the prev-dependent suggestion cache).

- Table: `build_corpus_bigrams` (`make ctx-model`), count >= 20, current word
  in-vocab; 436k pairs as a gitignored sidecar (`data/corpus_bigrams.bin`).
  Ships outside the container for now (v7 interning to u32 vocab ids is the
  follow-up: ~24 MB string-keyed -> ~5 MB packed).
- Blend, not override: `CorpusCtx::bonus` is the bigram log-ratio vs the top
  candidate at weight `CTX_W = 0.5` (`AKSHAR_CTX_W`, off at 0), applied to the
  top 24 pre-squash, with an abstention rule (no vote unless prev is observed
  with cur or top — otherwise it re-ranks by unigram ratio and regresses).
  `AKSHAR_NO_CORPUS_CTX=1` ablates; no table or empty prev is byte-identical.
- Measured (`make eval-sentences`, 1000 held-out sentences, synthetic
  best-chunk roman): word@1 87.58% -> **88.80% predicted** / 88.86% oracle
  (+1.22pp realistic), sentence-exact 23.70% -> 26.80%. Predicted within
  0.06pp of oracle: error propagation is negligible.
- Cost: +0.12 ms/word on the sentence harness (0.42 -> 0.54 ms); isolated
  AK-Freq unchanged (no prev, no table in that path).
- Harness honesty: `evaluate_sentences --ctx-mode off|oracle|predicted`
  (default off = the old isolated number).

### Joint knob tuning (infrastructure; defaults unchanged)
- New `tune_weights` bin: coordinate descent over `AKSHAR_GAMMA`,
  `AKSHAR_BEAM`, `AKSHAR_RERANK_DEPTH`, `AKSHAR_NO_TRIE_UNION` on valid
  (exact top-1 counts via subprocess env; OnceLock flags can't vary
  in-process), plus a `CTX_W` line search on disjoint sentences.
- Outcome: nothing shipped. Pooled-valid tuning (+3.27pp) cratered test
  native −4.18pp (wrong objective: it robbed the native family for entity
  strata). True top-1 tuning converged to near-defaults (+4 hits valid
  AK-Freq). Baked beam=128/depth=16 experimentally, ship-gate on test traded
  native −5 for entities +9 at ~2x latency — reverted. Lesson recorded in the
  constants' doc comments: valid's strata mix does not transfer; the remaining
  path is conditional-γ per stratum, not better scalars.
- `evaluate_aksharantar` now prints `(top1 h/n, top5 h/n)`; the old bare
  `(h/n)` was top-5 counts and once sent the tuner chasing the wrong metric.
- Conditional frequency-rank blend tried and removed same-day: every
  `AKSHAR_GAMMA_HI` > LO harmed native while pooled rose (constraint held);
  converged to uniform. Reweighting cannot create signal — entities need
  entity features, not weight.

### Morphology union table (infrastructure; nil accuracy on valid)
- `MORPH_SUFFIXES` 34 → 164: Snowball Hindi port (BSD, provenance comments)
  + longest-first ordering (fixes old first-match mis-strips) + consonant
  gate for leading-implicit-a forms (`akshara::is_consonant`); unit-pinned
  (table length, ordering, gate fall-through).
- Measured nil on valid (−5 pooled, noise); ceiling probe caps the track at
  13 rescuable valid-native misses, mostly via the old 34 — Nepali/Marathi
  transcription cancelled on cost/benefit. Table stays as zero-cost gated
  data; future retrains train sparse template 5 on the new firings.

### Attestation-weighted training (flag; killed as ship candidate)

- `train --attestation` (`make train-mid-att`): log-dampened corpus-frequency
  EM/LM pair weights through the pre-existing `add_pair_weighted` path
  (integer weights keep KN valid); reranker sampling uniform; output to a
  non-default artifact only. `train-mid` run: entities +13 combined, native
  −8, pooled +5 (noise) — killed per criterion. Log:
  `plans/archive/2026-09-22-attestation-training.md`.

### Playground: separate self-contained repo (no CDN)

- `apps/playground/` moved OUT to the private repo
  `Fundaments-Work/akshar-playground` (nested here only under gitignored
  `deploy/`): clean editor shell (word/document, copy/download/clear,
  learned-reset; no integration guide, no emojis), engine + wrapper + model
  bundled under `public/` — no CDN, no external requests, no build step.
  Deploys to Cloudflare Workers (`wrangler deploy`) or Pages; verified
  serving locally (all assets 200).
- Main repo keeps the engine (`packages/`) only: `wasm-serve` /
  `playground-build` / `playground-deploy` targets pruned from the
  Makefile; MANUAL deployment chapter + AGENTS.md map updated.

### Correctness fixes + pipeline repair (2026-09-10)

Correctness-first audit: 14 code fixes across the engine, FFI, IBus layer,
trainer, and CI, each logged in `docs/plans/correctness-audit-2026-09-10.md`.
**No accuracy lost** — `make train-mid` retrain reproduces the v1.2.0 model
within bootstrap noise (`make eval`: 80.98/45.32/29.01 vs 81.02/45.75/29.50
on the previous artifact with identical code; pooled CIs overlap).

#### Fixed — engine
- `Trie::get_top_k_suggestions` max-heap inversion returned the wrong k
  (e.g. {3,1} instead of {3,2}); now a `Reverse` min-heap (+ regression test).
- `ModelDecoder::build_edges` byte-slicing panicked on non-ASCII input;
  non-ASCII queries now yield `[]` (+ regression test).
- Reranker ignores non-finite dense weights/features, rejects negative/NaN
  sparse scales, and falls back on NaN `AKSHAR_GAMMA` (also filtered at source).
- Learned-state import now clears the suggestion cache; new
  `ImeEngine::reset_learned_state()` replaces the WASM reset path that
  silently dropped the unified vocab/word-trie/sparse table.

#### Fixed — FFI / IBus
- C API: `static mut` engine → `OnceLock<Mutex<Option<…>>>`; NULL inputs
  yield `"[]"`/no-op instead of UB; no `unwrap()` across `extern "C"`;
  missing/non-UTF8 config dir falls back to an in-memory engine;
  learning persists best-effort on every confirm (a kill no longer loses
  the whole session).
- IBus engine: NULL guards on all suggestion paths, `finalize` frees the
  preedit string and lookup table, UTF-8-safe BackSpace, `g_utf8_strlen`
  cursor position, bounds-checked candidate selection, checked
  `ibus_bus_request_name`.

#### Fixed — training pipeline
- `DenseStats` accumulated z-scores and the pack step overwrote the result
  with stale constants — retrained models were silently decalibrated.
  Now accumulates raw features and packs the fresh statistics
  (emit 5.462±3.283, lm 25.080±6.886 on `train-mid`).
- Full runs consumed every pair and left the dev set empty; the tail is now
  always reserved and best-by-dev packing engaged on `train-mid`
  (batch-2 weights over the overfit final).
- Phase-2 vocabulary now honors the shared `holdout` split (14,465 lines).

#### Docs
- Headline numbers re-measured on the retrained model (README, MANUAL §1,
  `data/README.md`, regenerated `docs/generated/eval.json`); 2026-09-08
  ablation/McNemar tables kept as the dated significance record.
- `AKSHAR_FUZZY_BASE` default corrected to 600,000 (was 50,000 pre-D18).

#### CI
- Now enforces the documented gate: `cargo fmt --check`, release-profile
  clippy/tests with `-D warnings`, and the wasm target check.

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
