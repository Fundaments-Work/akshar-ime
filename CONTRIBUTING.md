# Contributing to Akshar Devanagari IME

Thank you for contributing to Akshar Devanagari IME! This document provides instructions for developers joining the project to ensure a smooth, structured development workflow.

---

## 1. System Requirements & Setup

Akshar is primarily built in **Rust** (2021 edition) with a lightweight **C wrapper** for the Linux IBus input daemon, and an optional **WebAssembly** target.

### Prerequisites (Ubuntu / Debian / Fedora / Arch)

* **Rust (stable):** `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
* **WebAssembly Target:** `rustup target add wasm32-unknown-unknown`
* **C Build Tools & Libraries:**
  * Ubuntu/Debian: `sudo apt install build-essential pkg-config libibus-1.0-dev libjansson-dev`
  * Fedora: `sudo dnf install gcc pkgconf-pkg-config ibus-devel jansson-devel`
  * Arch: `sudo pacman -S base-devel pkgconf ibus jansson`

---

## 2. Developer Workflow

We use a clean, centralized `Makefile` for developer tasks. Avoid running ad-hoc cargo commands that bypass the validation gates.

### Daily Commands

```sh
make release        # Build release Rust library and C IBus engine
make debug          # Fast debug build
make test           # Run the test suite and regression guards
make check          # Full gate: cargo fmt + clippy -D warnings + tests + wasm check
```

> [!IMPORTANT]
> **Always run `make check` before submitting a pull request.** Both native and WASM32 targets must pass with zero warnings.

---

## 3. Repository Structure

```
akshar-ime/
├── Cargo.toml            # Rust crate configuration & dependencies
├── Makefile              # Minimal build and automation recipes
├── devanagari-smart.xml  # IBus engine descriptor
├── src/
│   ├── lib.rs            # Public Rust library entry
│   ├── c_api.rs          # C FFI layer for IBus
│   ├── ibus_engine.c     # Native Linux IBus engine bridge
│   ├── wasm.rs           # WebAssembly bindings (feature = "wasm")
│   ├── learning.rs       # Interactive session learning & adaptation
│   ├── persistence.rs    # User dictionary and configuration persistence
│   ├── core/             # Phonetic decoder, FST lexicon, LM & reranker
│   ├── fuzzy/            # Canonicalization & typo tolerance
│   └── bin/              # CLI tools for training, preparation & evaluation
├── packages/             # WASM package integration and build scripts
├── scripts/              # Data download manifests & helper scripts
├── tests/                # Behavioral and regression tests
├── docs/                 # Reference manual (MANUAL.md) and technical docs
└── data/                 # Model and datasets (gitignored; preserved locally)
```

---

## 4. Local Installation & Testing with IBus

To test your local build directly in your desktop environment:

```sh
sudo make install    # Copies binaries and component files to system dirs
make restart-ibus    # Restarts your user IBus daemon (DO NOT run with sudo)
```

Then select **Devanagari (Akshar)** in your desktop's Keyboard / Input Source settings.

---

## 5. Data & Evaluation Pipeline

The engine evaluates across the complete Devanagari script benchmark:

```sh
make eval                  # Evaluate against data/pairs/test.jsonl
make eval SPLIT=valid      # Evaluate on validation split for tuning
make wasm                  # Build WASM package
```

If you need to reproduce or retrain model artifacts from scratch:
```sh
make data-fetch            # Download datasets
make data-prepare          # Normalize word pairs
make lexicon               # Build universal Devanagari FST lexicon
make train                 # Train emissions, LM, and reranker
make model                 # Calibrate blend and prune to 25 MB budget
```

---

## 6. Coding Standards & Git Guidelines

1. **Formatting:** Enforced via `cargo fmt`. Run `cargo fmt` before staging changes.
2. **Lints:** Enforced via `cargo clippy --all-targets -- -D warnings`. Code must be warning-free.
3. **Branching:**
   * Branch off `main` for all features and fixes: `git checkout -b feature/your-feature`.
   * Submit changes via Pull Request to `main`.
4. **Devanagari First:**
   * Akshar is designed as a universal phonetic engine for the **Devanagari script**.
   * Avoid language-specific or regional bias in general documentation and user-facing CLI tools.
