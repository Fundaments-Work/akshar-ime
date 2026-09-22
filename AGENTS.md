# AGENTS.md — Akshar Devanagari IME

Rust crate `akshar_ime` (lib entry `src/lib.rs`, public API `ImeEngine::get_suggestions` / `user_confirms`) + C IBus engine (`src/ibus_engine.c`) + browser engine (`packages/engine-wasm`, `packages/engine-js`; the deployable playground lives in the separate private repo `Fundaments-Work/akshar-playground`, nested here only under gitignored `deploy/`). Module map: `src/core/` (decoder, engine, reranker, LM, unified container), `src/fuzzy/`, `src/learning.rs`, `src/persistence.rs`, `src/c_api.rs` (native only), `src/wasm.rs` (`wasm` feature only).

## Commands (use the Makefile, not raw cargo)

- `make release` — builds Rust lib + C engine. `make test` = `cargo test --release` (includes accuracy regression guard). Single test: `cargo test --release <name>`.
- `make check` — **required gate**: `cargo fmt --check` + `cargo clippy --release --all-targets -- -D warnings` + tests + wasm target check. Native-green does not imply wasm-green (a broken wasm build shipped in v1.1.0), so `make check-wasm` is not optional. Needs `rustup target add wasm32-unknown-unknown`.
- `make eval | eval-full | eval-ime | eval-errors | ablate` — accuracy by split, bootstrap CIs + latency, JSON report to `docs/generated/eval.json`, error taxonomy, component ablations. Profile: `cargo run --release --example profile_decode`.
- `make train-quick | train-mid | train-full` — differ **only** in `--reranker-pairs` (100k / 500k / all 3.59M). EM emissions + KN LM always ingest all pairs; chunked mode engages only above 200k, so validate trainer changes with `train-mid`, never `train-quick`. Full run takes ~4h.
- `make web-model` — builds browser container `data/akshar_wasm.model` via trigram pruning (`TRIGRAM_THRESHOLD`, default `3e-2`). `make wasm` builds the browser engine; the playground ships from the separate `Fundaments-Work/akshar-playground` repo (see `docs/plans/playground-extension.md`).
- IBus deploy: `sudo make install` then `make restart-ibus` **without sudo**. CI needs `libibus-1.0-dev libjansson-dev` and runs `make release`.

## Data and models — gitignored, absent on fresh clones

- `data/` is fully gitignored (only `data/README.md` tracked). No corpora or `*.model` files exist after clone; never commit binaries or anything under `src/`. Models ship via `make release-upload TAG=vX.Y.Z` (attaches to a GitHub release).
- `build.rs` emits an empty `reranker_weights_sparse.bin` placeholder when `data/` is missing so the crate still compiles; the unified container (`data/akshar.model`, desktop) / (`data/akshar_wasm.model`, browser) carries the real weights. `tests/accuracy_regression.rs` **skips** (does not fail) without model + dataset. Do not invent `data/` fixtures.
- Vocabulary merges are explicit (a news-only vocab once silently overwrote the real one, −1.2pp). There is no backup dir; deletions under `data/` are final.

## Gotchas

- Ablation/debug env flags live in `src/core/mod.rs`: `AKSHAR_NO_TRIGRAM`, `AKSHAR_NO_TRIE_UNION`, `AKSHAR_NO_SPARSE`, `AKSHAR_NO_RERANK`, `AKSHAR_NO_VARIANTS`, `AKSHAR_GAMMA`, plus `AKSHAR_BEAM`, `AKSHAR_CACHE_SIZE`, `AKSHAR_DATA_DIR`, `AKSHAR_RERANK_DEPTH` (see `engine.rs`, `reranker.rs`). Score-band order matters: user-trie base (900k) > decoder, fuzzy base (50k) below it — a misordered candidate source once caused a 30-point top-1 regression through a green unit suite; check `engine.rs` bands if top-1 moves.
- Pinned wasm deps: `wasm-bindgen =0.2.100`, `web-sys =0.3.77`; `wasm/build.sh` installs exactly `wasm-bindgen-cli 0.2.100 --locked`. Do not bump one without the other.
- `docs/MANUAL.md` is the source of truth (theory, module map, eval method, defect register §§11–13); `docs/plans/archive/` is the tried-and-rejected experiment log. PDF needs pandoc + xelatex with a Devanagari-covering font (`make manual`). Trust `Makefile`/code over prose when they conflict.
