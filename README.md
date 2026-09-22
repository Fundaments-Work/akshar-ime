# Akshar Devanagari IME

Roman-script input method for the Devanagari **script** — not one language.
You type `namaste`, it offers `नमस्ते`, whether the word is Hindi, Nepali,
Marathi, Sanskrit, or any other language written in Devanagari lipi.

![CI](https://github.com/sapienskid/akshar-ime/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue)

**No neural network at runtime.** An EM-trained source-channel model over
orthographic syllables, a modified Kneser-Ney syllable language model, and a
linear discriminative reranker. Inference is beam search plus dot products —
no tensor library, no GPU.

**11.37 MB desktop, 4.94 MB Brotli in the browser. 0.72 ms per query.**

## What contributes what

Measured by ablation, reproducible from the shipped binary (`make ablate`).
Component table measured 2026-09-08; re-measured 2026-09-10 on the retrained
model (full 80.98%, trigram −6.03pp, sparse −0.48pp, dense −0.24pp — §9):

| Ranking stage | `AK-Freq` top-1 | Δ |
| :--- | ---: | ---: |
| raw decoder order (`emit + lm`) | 75.38% | — |
| + frequency heuristic | 81.02% | **+5.64** |
| + 29 dense features | 81.93% | +0.91 |
| + 2²⁰ sparse table (shipped) | 81.83% | −0.10 |

Removing the trigram language model costs **−6.03pp** on the current model
(−3.89pp on the 2026-09-08 model); it is the single largest
component. See the manual for per-stratum numbers with McNemar p-values.

## Measured performance

Held-out AI4Bharat Aksharantar test split (4,101 cases), measured
2026-09-10 on the retrained `data/akshar.model` (`make train-mid`
with the corrected pipeline: raw dense statistics, reserved dev set,
holdout-filtered vocabulary) at default settings
(`make eval`, `make eval-full`, `make ablate`).
The 2026-09-08 model measured 81.83/47.79/31.21 on the same harness;
the retrain reproduces it within bootstrap noise (pooled CIs overlap).

| Split | n | top-1 | top-5 |
| :--- | ---: | ---: | ---: |
| `AK-Freq` (native words) | 2,108 | **80.98%** | 91.84% |
| `AK-NEI` (named entities) | 1,176 | 45.32% | 70.15% |
| `AK-NEF` (named entities) | 817 | 29.01% | 51.53% |
| All cases | 4,101 | 60.40% | 77.59% |

Pooled top-1 bootstrap 95% CI [58.94%, 61.89%]; MRR 0.6798; CER on `AK-Freq`
top-1 is 3.90% (measured 2026-09-08, not re-measured on the retrain). For reference, IndicXlit (an ~11M-parameter transformer) reports
80.25% top-1 on the native split and 52.67% on named entities *without* LM
reranking; with its word-unigram rerank it reaches 86.6% / ~62% on Nepali
([Madhani et al. 2023], Table 6). Neither figure is re-measured here, and no
controlled head-to-head is claimed — native accuracy here is comparable to the
unreranked baseline, named-entity accuracy is well behind either way.

| | Desktop | Browser |
| :--- | ---: | ---: |
| Container | 11.37 MB | 8.91 MB |
| Brotli | 6.72 MB | 4.94 MB |
| Query latency | 0.72 ms (re-measured 2026-09-10, $k=10$, full set) | not re-measured since 2026-09-06 |
| Cold start | ~1.5 s | — |

## Documentation

**[`docs/MANUAL.md`](docs/MANUAL.md)** — the complete reference, 52 pages.
Build a PDF with `make manual`.

| Part | Contents |
| :--- | :--- |
| Theory | source-channel formulation, akshara segmentation, EM with scaled forward–backward, modified Kneser-Ney, decoding, discriminative reranking |
| Implementation | module map, container format, performance engineering |
| Practice | training pipeline, **evaluation methodology**, ablations with McNemar significance, deployment |
| Status | known defects, experimental record, roadmap |

Every metric is defined formally, every technique is credited to its authors,
and every term is in the glossary. Threats to validity are stated explicitly.

The live defect register and roadmap are `docs/MANUAL.md` §11–13;
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

Browser: `make wasm-serve`, then see the manual's deployment chapter.

## Reproducing the numbers

```sh
make test          # tests, including the accuracy regression guard
make eval          # accuracy by split
make eval-full     # bootstrap CIs, MRR, per-query latency
make eval-ime      # plain-language report + machine JSON (docs/generated/eval.json)
make eval-errors   # oracle curves, error taxonomy, CER, collision bound
make ablate        # component contributions
```

Training (`make train-mid`, `make train-full`) is documented in the manual.
Note that `--reranker-pairs` sizes the reranker's training set only — the EM
model and language model always use all 3.59M pairs.

## Honest limitations

- Script-general engine, still biased priors: the akshara model, decoder, and
  scoring are language-agnostic (nothing downstream reads a language tag —
  `data/README.md`), but the frequency vocabulary is news-domain-heavy, the
  34-suffix strip list is Nepali-specific, and the test set is one
  Devanagari benchmark. Making the *priors* as script-general as the *engine*
  is the accuracy program (`docs/MANUAL.md` §§11–13,
  `docs/plans/devanagari-script-plan.md`).
- One benchmark, one test set (Aksharantar, 4,101 cases).
- Named entities are well behind the neural baseline.
- Reranking is worth +6.45pp overall on the 2026-09-08 model (+0.24pp dense+sparse
  on the retrained model at full 80.98% vs heuristic-only 80.74%), but **most of
  that has always been a 3-parameter frequency heuristic** — the 10⁶-parameter
  learned stage added +0.91pp then, and its
  sparse half adds nothing on native words (p = 0.851, measured 2026-09-08).
- 5x more reranker training data was tested and changed nothing; the cause is
  that `W_DENSE` has never been refit by this pipeline (manual §12.3).

Manual chapter 11 lists every known defect.

## License

MIT.
