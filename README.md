# Akshar Devanagari IME

Roman-script input method for the Devanagari **script** — genuinely across
languages, not one language under a script-general label. One phonetic
engine and one shared ranking lexicon serve Hindi, Marathi, Nepali,
Sanskrit, Konkani, Maithili, Bodo and Dogri; a language tag conditions
ranking explicitly. You type `namaste`, it offers `नमस्ते`.

![CI](https://github.com/sapienskid/akshar-ime/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue)

**No neural network at runtime.** An EM-trained source-channel model over
orthographic syllables, a modified Kneser-Ney syllable language model, and a
linear discriminative reranker. Inference is beam search plus dot products —
no tensor library, no GPU.

**24.44 MB desktop for all 8 languages, 13.97 MB Brotli in the browser.
1.16 ms per query.**

## What contributes what

Measured by ablation on the single-language predecessor of this model,
reproducible from a shipped binary (`make ablate`); not yet re-run on the
8-language lexicon (`docs/MANUAL.md` §13):

| Ranking stage | `AK-Freq` top-1 | Δ |
| :--- | ---: | ---: |
| raw decoder order (`emit + lm`) | 75.38% | — |
| + frequency heuristic | 81.02% | **+5.64** |
| + 29 dense features | 81.93% | +0.91 |
| + 2²⁰ sparse table (shipped) | 81.83% | −0.10 |

Removing the trigram language model costs **−6.03pp**; it is the single
largest component. Two components measured since, directly on the
multi-language model: the lexicon/language-conditioned ranking itself
(macro native top-1 ~50.3% phonetics-only → ~64.4% with it, validation
split, `AKSHAR_NO_RERANK=1`) and end-of-word LM scoring (+1.2pp macro
top-1). See the manual for per-stratum numbers with McNemar p-values.

## Measured performance

Held-out multi-language test split (34,011 native-word + 14,266
named-entity cases across all 8 languages, `data/pairs/test.jsonl`, built
by `make data-prepare` from AI4Bharat Aksharantar), measured on
`data/akshar.model`, language-aware (`make eval-langs`):

| Lang | native top-1 | in-list@8 | entity top-1 | IndicXlit AK-Freq (plain / +LM) |
| :--- | ---: | ---: | ---: | ---: |
| Hindi | 53.42% | 81.45% | 43.59% | 58.6 / 67.9 |
| Marathi | 66.78% | 83.79% | 37.15% | 74.7 / 85.5 |
| Nepali | 78.32% | 92.55% | 34.97% | 80.2 / 86.6 |
| Sanskrit | 75.30% | 92.07% | 18.48% | 81.6 / 90.1 |
| Konkani | 56.24% | 79.15% | 30.94% | 65.4 / 76.3 |
| Maithili | 69.12% | 90.98% | 38.78% | 78.7 / 87.6 |
| Bodo | 41.67% | 56.60% | 18.40% | 74.8 / 78.4 |
| Dogri | 32.80% | 54.20% | — | — |
| **pooled** (n=34,011 / 14,266) | **60.69%** | 81.27% | **31.59%** | |

This is the cold, uniformly-weighted number — the one comparable to
IndicXlit's own published methodology, and this system trails its
word-LM-reranked column in most languages (only Nepali is close,
unreranked). It is **not** what returning-user accuracy looks like: sampling
by real word frequency instead of uniformly, with the engine's actual
`user_confirms` learning path exercised, pooled top-1 rises to **97.28%**
(`eval_session`, `docs/MANUAL.md` §12.10 — reported as a ceiling under a
charitable assumption, not a substitute for the table above).

| | Desktop | Browser |
| :--- | ---: | ---: |
| Container | 24.44 MB | 23.30 MB |
| Brotli | n/a (native binary) | 13.97 MB |
| Query latency | 1.16 ms ($k=8$, beam 32) | not re-measured on this artifact |
| Cold start | not re-measured on this artifact | — |

## Documentation

**[`docs/MANUAL.md`](docs/MANUAL.md)** — the complete reference, 56 pages.
Build a PDF with `make manual`.

| Part | Contents |
| :--- | :--- |
| Theory | source-channel formulation, akshara segmentation, EM with scaled forward–backward, modified Kneser-Ney, decoding, discriminative reranking |
| Implementation | module map, the shared multi-language lexicon, container format, performance engineering |
| Practice | training pipeline, **evaluation methodology** (including cold vs. session accuracy), ablations with McNemar significance, deployment |
| Status | contribution and prior art, known defects, experimental record, roadmap |

Every metric is defined formally, every technique is credited to its authors,
and every term is in the glossary. Threats to validity are stated explicitly.

The live defect register and roadmap are `docs/MANUAL.md` §17–18;
`docs/plans/archive/` preserves the experiment log, literature review and
research agenda from development — what was tried and rejected, not just what
shipped.

## Quick start

```sh
make release            # build
sudo make install       # install to IBus
make restart-ibus       # NOT as root
```

```rust
use akshar_ime::ImeEngine;
let mut engine = ImeEngine::new();
let suggestions = engine.get_suggestions("namaste", 5);
engine.user_confirms("namaste", "नमस्ते");   // adaptive learning
```

Browser: build the engine for the web (`make wasm`), served by the standalone playground repo.

## Reproducing the numbers

```sh
make data-fetch SET=aksharantar        # pinned, checksummed source data
make data-fetch SET=indiccorp-v2-sample
make data-prepare                      # builds data/pairs/{train,valid,test}.jsonl
make lexicon                           # builds data/lexicon.bin
make train-full                        # ~4h; train-mid (~40min) for a faster check
make model                             # calibrate blend + fit the size budget -> data/akshar.model
make test                              # tests, including the accuracy regression guard
make eval-langs                        # per-language IME metrics (this README's table)
cargo run --release --bin eval_session -- --lang-aware   # cold vs. session accuracy
```

`data/` is gitignored — none of the above exists after a plain clone, and
`make release` still builds without it (`build.rs` degrades gracefully).
Each step above is independently re-runnable; `train`/`train-full` auto-detect
the files `data-prepare`/`lexicon` produce over the single-language
predecessor. Verified end-to-end at `--smoke` scale
(`cargo run --release --bin train -- --smoke`, ~6s, exercises every stage of
the same code) rather than by timing the full 4h run; CI builds and tests the
code but does not run this pipeline. `--reranker-pairs` (set by the
`train-*` targets) sizes the reranker's training set only — the EM model,
language model and lexicon always use everything available.

Legacy single-language harness, kept for historical comparability
(`docs/MANUAL.md` §12.6): `make eval-full`, `make eval-errors`, `make ablate`.

## Honest limitations

- On the metric comparable across systems (cold, uniform top-1), this does
  **not** beat IndicXlit's word-LM-reranked numbers in most of the eight
  languages — only Nepali is close, unreranked (table above). The defensible
  claim is architectural (no neural runtime, ~24 MB, sub-2ms CPU decode),
  not an accuracy-SOTA one.
- Bodo and Dogri are limited by how little clean training text exists for
  them (34,480 and 1,276 pairs after cleaning) — a generation-ceiling
  problem, not just a ranking one; more reranker supervision alone will not
  close their gap.
- Named entities are well behind native-word accuracy in every language
  (31.59% pooled vs. 60.69% native).
- The eight-language browser profile has not been measured for query
  latency; its size (13.97 MB Brotli, ~3× the single-language predecessor,
  because the shared lexicon ships unchanged) has been.
- The ablation table above is the single-language predecessor's; it has not
  been re-run on the current 8-language lexicon model.
- `--reranker-pairs 0` (the ~7.3M-pair full run) has not been retried since
  the three reranker training defects below were fixed. It was tested and
  falsified under the broken pipeline — whether more supervision helps once
  training data is actually multi-language and leak-free is open, not
  re-confirmed negative.
- Three defects explain why the reranker had never measured as beating the
  3-parameter frequency heuristic despite being individually correct code:
  the reranker trained on Hindi only (Aksharantar's train file is
  language-sorted and was never shuffled before subsetting), the EM/LM
  had usually already memorized the words the reranker was later scored on
  (no word-level train/eval split existed), and dense-feature normalization
  silently drifted out of sync with retrains. All three are fixed in
  `prepare_pairs`/`train.rs`; see `docs/MANUAL.md` §17.1, D19–D21.

Manual chapter 17 lists every known defect.

## License

MIT.
