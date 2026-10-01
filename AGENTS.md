# AGENTS.md — Akshar Devanagari IME

Rust crate `akshar_ime` (lib entry `src/lib.rs`, public API `ImeEngine::get_suggestions` / `user_confirms`) + C IBus engine (`src/ibus_engine.c`) + browser WASM build (`packages/engine-wasm`, `packages/engine-js`). Module map: `src/core/` (decoder, engine, reranker, LM, unified container, shared lexicon), `src/fuzzy/`, `src/learning.rs`, `src/persistence.rs`, `src/c_api.rs` (native), `src/wasm.rs` (`wasm` feature).

## Commands (use the Makefile, not raw cargo)

- `make release` — builds Rust lib + C IBus engine. `make test` = `cargo test --release` (includes accuracy regression guard). Single test: `cargo test --release <name>`.
- `make check` — **required gate**: `cargo fmt --check` + `cargo clippy --release --all-targets -- -D warnings` + `cargo test --release` + `check-wasm`. Native-green does not imply wasm-green, so `make check-wasm` is not optional. Needs `rustup target add wasm32-unknown-unknown`.
- `make eval [SPLIT=valid|test]` — runs Devanagari benchmark evaluation (`eval_langs` over `data/pairs/$(SPLIT).jsonl`). `cargo run --release --bin eval_session -- --lang-aware` measures cold vs. adaptive learning session accuracy.
- `make train [PAIRS=500000] [EPOCHS=5]` — trains the engine. Writes `data/akshar.model` directly. `make model` (= `calibrate` then `promote`) calibrates blend weights and fits the 25 MB size budget.
- `make data-fetch SET=aksharantar|indiccorp-v2-sample` → `make data-prepare` → `make lexicon` → `make train` → `make model` is the full from-scratch sequence; each step is idempotent and independently re-runnable. Verified end-to-end at `--smoke` scale (`cargo run --release --bin train -- --smoke`, ~6s).
- `make wasm` — builds the WebAssembly package via `packages/engine-wasm/build.sh`.
- IBus deploy: `sudo make install` then `make restart-ibus` **without sudo**. CI needs `libibus-1.0-dev libjansson-dev` and runs `make release`.

## Data and models — gitignored, absent on fresh clones

- `data/` is fully gitignored (only `data/README.md` tracked). Corpora and model files are not committed; models ship via `make release-upload TAG=vX.Y.Z` (attaches to a GitHub release).
- `build.rs` emits an empty `reranker_weights_sparse.bin` placeholder when `data/` is missing so the crate still compiles; the unified container (`data/akshar.model`) carries the real weights. `tests/accuracy_regression.rs` skips (does not fail) without model + dataset.
- Never delete or clobber existing datasets in `data/`. Deletions under `data/` are final.

## Gotchas

- Ablation/debug env flags live in `src/core/mod.rs`: `AKSHAR_NO_TRIGRAM`, `AKSHAR_NO_TRIE_UNION`, `AKSHAR_NO_SPARSE`, `AKSHAR_NO_RERANK`, `AKSHAR_NO_VARIANTS`, `AKSHAR_GAMMA`, plus `AKSHAR_BEAM`, `AKSHAR_CACHE_SIZE`, `AKSHAR_DATA_DIR`, `AKSHAR_RERANK_DEPTH` (see `engine.rs`, `reranker.rs`). Score-band order matters: user-trie base (900k) > decoder, fuzzy base (50k) below it.
- Pinned wasm deps: `wasm-bindgen =0.2.100`, `web-sys =0.3.77`; `packages/engine-wasm/build.sh` installs exactly `wasm-bindgen-cli 0.2.100 --locked`. Do not bump one without the other.
- `docs/MANUAL.md` is the source of truth (theory, module map, eval method, defect register §17). Always treat it as pure markdown. Trust `Makefile`/code over prose when they conflict.
