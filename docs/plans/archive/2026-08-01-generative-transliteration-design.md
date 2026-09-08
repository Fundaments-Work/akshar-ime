# Generative Transliteration Core — Design

> **Archive snapshot — frozen.** Original design as of 2026-08-01. The shipped
> system differs (defect fixes 2026-09-06, dependency migration v1.2.0); the
> current design is in `docs/MANUAL.md`.

Date: 2026-08-01
Status: Implemented

## Overview

The engine is a pure-Rust generative transliteration model trained by
expectation-maximisation over the Aksharantar Nepali corpus (2.4M roman →
Devanagari pairs). Devanagari is a regular abugida; the roman side carries the
ambiguity, so the model is built as a source-channel model:

```
P(D | R) ∝ P(R | D) · P(D)
```

* `P(R | D)` — a transliteration table `P(roman_chunk | akshara)` trained by EM.
* `P(D)` — a Kneser-Ney akshara n-gram language model.

Decoding is beam search over an akshara lattice. There is no neural network:
all parameters are counts converted to probabilities (EM, maximum likelihood,
absolute-discount smoothing) and the runtime is a shortest-path search over
negative-log weights.

## The transliteration model

Following Li, Zhang & Su (ACL-04, *A Joint Source-Channel Model for Machine
Transliteration*):

```
P(R | D) = sum over segmentations of R into chunks s_1..s_n aligned to the
           aksharas a_1..a_n of D, of  prod_j P(s_j | a_j)
```

### EM alignment
Each Devanagari word is segmented into aksharas (`akshara.rs`). The roman word
is a sequence of characters; a *segmentation* assigns each akshara a contiguous
chunk of 0..=MAX_CHUNK (5) roman characters. Forward-backward EM over the
alignment space maximises the corpus likelihood. Emissions are seeded from a
deterministic codepoint aligner and **schwa-augmented**: every bare-consonant
akshara also seeds its `chunk + "a"` (and schwa-dropped) forms, so EM can
discover the medial-schwa convention (`cha → च`).

The E-step is embarrassingly parallel over pairs and runs on `std::thread`
scoped threads.

### Kneser-Ney akshara LM
- `word_start`: `P(a | word start)` from corpus word-initial frequencies.
- `bigram`: `P_KN(a_j | a_{j-1})` with absolute-discount backoff.
- `trigram`: `P_KN(a_k | a_{j-1}, a_j)` — disambiguates word patterns such as
  the trigram एउटा resolving `eutako`.

### Decoder
Beam search over the akshara lattice. Each edge consumes 1..=MAX_CHUNK roman
chars and emits one akshara, weighted by emission + LM (tunable `lm_weight`,
default 1.0). Results are the k lowest-total-weight complete paths. Paths use a
persistent cons-cell arena (O(1) extension) and a u64 path-hash for cheap beam
dedup that preserves top-k diversity.

### Discriminative reranking
A log-linear reranker (`reranker.rs`) rescored the decoder's k-best list with a
feature vector (emission, LM, length, lexicon exact-match) whose weights are
trained by MERT (`train_reranker`). In practice the trained weights reduce to
the generative balance, so the reranker defaults to the pure generative score —
it remains the structured entry point for future evidence features.

## Architecture

```
ImeEngine (src/core/engine.rs)          — one decoder, one evidence score
  ├─ ModelDecoder (decoder.rs)          — persistent-path beam search
  ├─ Reranker (reranker.rs)             — log-linear k-best reranking
  ├─ TranslitModel (translit_model.rs)  — emissions + KN bigram/trigram LM
  ├─ RomanLexicon (lexicon.rs)          — corpus roman→devanagari (exact match)
  ├─ Trie / SymSpell / ContextModel     — user learning + typo tolerance
  └─ LearningEngine / persistence       — user-confirmation feedback
```

Offline tools:
- `train_model` — builds `data/translit_model.bin` (EM + LM), threaded.
- `build_lexicon` — builds `data/roman_lexicon.bin` from the corpus.
- `train_reranker` — MERT weight tuning over the dev split.
- `evaluate_model` / `evaluate_aksharantar` / `evaluate_nepali_transliteration`
  — benchmarks.

## Results (Aksharantar Nepali test, 4,101 words)

| Metric | ImeEngine | IndicXlit (neural, reference) |
|---|---|---|
| Native top-1 | 73.5% | 80.3% |
| Native top-5 | 89.7% | — |
| Named-entity top-1 | 28–39% | 52.7% |
| Keystroke latency | ~3 ms (7 ms max) | — |

Full held-out IME eval (`data/eval/aksharantar_test.tsv`, 4,101 real cases from
the Aksharantar test split): ~54% top-1 / ~74% top-5 overall. The decoder alone
reaches 75.3% top-1 on native words.

## Rebuilding

```bash
cargo run --release --bin train_model     # ~2-3 min threaded, 12 EM iterations
cargo run --release --bin build_lexicon   # ~5 s
cargo run --release --bin train_reranker  # MERT weights from the dev split
cargo run --release --bin evaluate_model -- --beam 64      # decoder-only
cargo run --release --bin evaluate_aksharantar             # IME engine
```

## Known limitations

- **Vowel length is not encoded in the roman corpus** (`la` = both ल and ला).
  The model disambiguates with the LM; a real Nepali word-frequency corpus would
  close most of the remaining gap.
- Named entities and some >5-char conjuncts (e.g. `न्थ्यौं`) remain hard.
- The Aksharantar lexicon is news-mined compounds; prefix-completion floods
  with compounds, so the IME uses exact-match lexicon evidence only.
