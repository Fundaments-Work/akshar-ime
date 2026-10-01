# Akshar Devanagari IME

Fast, lightweight, and intelligent phonetic Input Method Engine (IME) for the **Devanagari script**.

Akshar provides universal Roman-to-Devanagari transliteration. One unified phonetic transducer, an orthographic syllable language model, and a shared ranking lexicon cover the entire Devanagari writing system. You type `namaste`, it offers `नमस्ते`.

![CI](https://github.com/Fundaments-Work/akshar-ime/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## Key Features

* **No Neural Network at Runtime:** Built on an Expectation-Maximization (EM) source-channel model over orthographic syllables, a modified Kneser-Ney syllable language model, and a linear discriminative reranker. Inference uses position-synchronous beam search and dot products — no GPU, no PyTorch/ONNX runtime.
* **Compact & Fast:** **24.4 MB** unified desktop model (`data/akshar.model`) covering over 1.2 million Devanagari words. Query latency is **~1.1 ms** on standard CPU.
* **Adaptive Learning:** The engine actively learns from user confirmations (`user_confirms`), dynamically caching and adapting to personal vocabulary and frequent spellings.
* **Linux Desktop & WebAssembly:** Native C IBus integration for Linux desktops (`devanagari-smart`) alongside a WebAssembly target for browser integration.

---

## Measured Accuracy

Evaluated across the comprehensive Devanagari benchmark (48,277 test pairs from AI4Bharat Aksharantar; `data/pairs/test.jsonl`):

| Metric | Result | Description |
| :--- | :---: | :--- |
| **In-List@8** | **81.27%** | Correct candidate appears in top-8 suggestions |
| **Lenient In-List@8** | **82.41%** | In-list including standard orthographic equivalence classes |
| **Top-3 Accuracy** | **75.54%** | Correct candidate appears in top-3 suggestions |
| **Cold 1-Shot Top-1** | **60.69%** | First-try top candidate across the pooled multi-domain test set |
| **Core Vocabulary Top-1** | **67.0% – 78.3%** | First-try top candidate on primary Devanagari vocabulary |
| **Interactive Session Top-1** | **97.28%** | Real typing session accuracy with adaptive user learning active |

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

Then open **Settings > Keyboard > Input Sources** and add **Devanagari (Akshar)**.

### Rust Library Usage

```rust
use akshar_ime::ImeEngine;

let mut engine = ImeEngine::new();

// Transliterate Roman input to Devanagari
let suggestions = engine.get_suggestions("namaste", 5);
assert_eq!(suggestions[0].0, "नमस्ते");

// Adaptive learning on user selection
engine.user_confirms("namaste", "नमस्ते");
```

---

## Developer Guide

For prerequisites, environment setup, and coding conventions, see **[`CONTRIBUTING.md`](CONTRIBUTING.md)**.

### Common Developer Commands

```sh
make release      # Build Rust library and C engine in release mode
make debug        # Fast debug build
make test         # Run test suite and regression guards
make check        # Required gate: cargo fmt + clippy -D warnings + tests + wasm check
make wasm         # Build WebAssembly package
make eval         # Run Devanagari benchmark evaluation
```

### Reproducing Models from Scratch

The complete dataset and model pipeline is reproducible using the Makefile:

```sh
make data-fetch    # Download pinned, checksummed source datasets
make data-prepare  # Build normalized pair datasets (data/pairs/{train,valid,test}.jsonl)
make lexicon       # Build universal Devanagari FST lexicon (data/lexicon.bin)
make train         # Train emissions, LM, and reranker
make model         # Calibrate blend and prune to 25 MB desktop budget
```

---

## Documentation

* **[`docs/MANUAL.md`](docs/MANUAL.md)**: Mathematical formulations, phonetic algorithms, module architecture, and complete evaluation methodology.
* **[`CONTRIBUTING.md`](CONTRIBUTING.md)**: Developer setup, code standards, and PR workflows.
* **[`CHANGELOG.md`](CHANGELOG.md)**: Version history and defect resolutions.

---

## License

MIT.
