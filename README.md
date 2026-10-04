# Akshar Nepali IME

Fast, lightweight, and intelligent phonetic Input Method Engine (IME) for **Nepali**.

Akshar converts Roman-script typing into Devanagari. One unified phonetic transducer, an orthographic syllable language model, and a frequency-ranked lexicon of 700,000 Nepali words. You type `namaste`, it offers `नमस्ते`.

![CI](https://github.com/Fundaments-Work/akshar-ime/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## Scope

**Nepali only.** The engine ships one language's priors. The phonetic core is
still script-general, but the vocabulary, frequency tables and benchmarks are
Nepali's. Earlier versions were multi-language across eight Devanagari
languages; that edition was retired when its per-language corpora were dropped
and it can no longer be rebuilt. `docs/MANUAL.md` §2.3 keeps its numbers,
marked historical.

---

## Key Features

* **No Neural Network at Runtime:** Built on an Expectation-Maximization (EM) source-channel model over orthographic syllables, a modified Kneser-Ney syllable language model, and a linear discriminative reranker. Inference uses position-synchronous beam search and dot products — no GPU, no PyTorch/ONNX runtime.
* **Compact & Fast:** **9.6 MB** unified desktop model (`data/akshar.model`) covering 700,000 Nepali words. Query latency is **0.84 ms** on a standard CPU.
* **Adaptive Learning:** The engine learns from user confirmations (`user_confirms`), caching personal vocabulary and frequent spellings.
* **Linux Desktop & WebAssembly:** Native C IBus integration for Linux desktops (`devanagari-smart`) alongside a WebAssembly target for browser integration.

---

## Measured Accuracy

Held-out Aksharantar Nepali test split (`nep_test.json`, 4,101 AK-Freq cases).
**These figures are pooled over common words and named entities** — the cleaned
splits no longer carry the `source` field, so the two strata cannot be
separated (`data/README.md`).

| Metric | Result | Description |
| :--- | ---: | :--- |
| **Top-1** | **60.35%** | First suggestion correct, cold, no memory between queries |
| **Top-5** | **76.15%** | Correct candidate in the top 5 |
| **In-List@8** | **78.32%** | Correct candidate in the top 8 |
| **Reachable@50** | **84.00%** | Ceiling for the current generator — see below |
| **Lenient Top-1 / In-List@8** | **61.59% / 79.08%** | Counting standard orthographic equivalence classes |

`reachable@50 = 84.00%` is the hard ceiling on top-1 without changing candidate
generation: **16.00%** of queries never surface the right word at all, while
**23.65pp is pure ranking headroom**. `docs/MANUAL.md` §2.3 has the full
comparison against the retired eight-language build and the methodology.

---

## Quick Start

### Build & Install (Linux / IBus)

```sh
# 1. Build release artifacts (Rust library + C IBus engine)
make release

# 2. Install to system IBus directories
sudo make install

# 3. Reload your IBus daemon (WITHOUT sudo)
make restart-ibus
```

Then open **Settings > Keyboard > Input Sources** and add
**Nepali — नेपाली (Akshar)**.

> **The engine needs `data/akshar.model` to be useful.** Without it,
> `ImeEngine` falls back to a small built-in default model. Release downloads
> ship the trained model as a separate asset; install it as
> `/usr/share/akshar-ime/akshar.model`, or point `AKSHAR_DATA_DIR` at a
> directory containing it.

### Rust Library Usage

```rust
use akshar_ime::ImeEngine;

let mut engine = ImeEngine::new();

// Transliterate Roman input to Devanagari
let suggestions = engine.get_suggestions("namaste", 5);

// Adaptive learning on user selection
engine.user_confirms("namaste", "नमस्ते");
```

---

## Developer Guide

For prerequisites, environment setup, and coding conventions, see
**[`CONTRIBUTING.md`](CONTRIBUTING.md)**.

### Common Developer Commands

```sh
make release      # Build Rust library and C engine in release mode
make debug        # Fast debug build
make test         # Run test suite and regression guards
make check        # Required gate: cargo fmt + clippy -D warnings + tests + wasm check
make wasm         # Build WebAssembly package
make eval         # Run the Nepali benchmark evaluation
```

### Reproducing the Model from Scratch

The pipeline needs two inputs, both described in
[`data/README.md`](data/README.md): `data/nepali_corpus.txt` (1.80 GB) and
`data/aksharantar/nep_{train,valid,test}.json`. `data/` is gitignored, so place
those two before starting.

```sh
make data-prepare ASSUME_SOURCE=AK-Freq          # -> data/pairs/*.jsonl
make lexicon LEXICON_ARGS="--max-words 700000"   # -> data/lexicon.bin (4.5 MB)
make train                                      # -> data/akshar.model
make model                                      # calibrate + prune (9.6 MB)
make eval SPLIT=test                             # score the benchmark
```

There is no `data-fetch` step: the corpus and the Aksharantar splits are
vendored locally, and the scripts that used to fetch the other seven languages
have been removed.

---

## Documentation

* **[`docs/MANUAL.md`](docs/MANUAL.md)** — mathematical formulations, phonetic algorithms, module architecture, evaluation methodology, and the defect register (§17).
* **[`docs/experiments/`](docs/experiments/)** — dated experiment logs: question, method, numbers, and the decision taken.
* **[`data/README.md`](data/README.md)** — corpus and dataset provenance, sizes, checksums, and what is gitignored.
* **[`CONTRIBUTING.md`](CONTRIBUTING.md)** — developer setup, code standards, and PR workflows.
* **[`CHANGELOG.md`](CHANGELOG.md)** — version history and defect resolutions.

---

## License

MIT.