# AGENTS.md — Akshar Nepali IME

Rust crate `akshar_ime` (lib entry `src/lib.rs`, public API `ImeEngine::get_suggestions` / `user_confirms`) + C IBus engine (`src/ibus_engine.c`) + browser WASM build (`packages/engine-wasm`, `packages/engine-js`). Module map: `src/core/` (decoder, engine, reranker, LM, unified container, shared lexicon), `src/fuzzy/`, `src/learning.rs`, `src/persistence.rs`, `src/c_api.rs` (native), `src/wasm.rs` (`wasm` feature).

**Scope: Nepali only.** The engine ships one language's priors. The phonetic core is still script-general and `LangCond` still conditions on a language tag, but that tag can only select Nepali, and the pipeline builds a single-language lexicon. Do not reintroduce per-language data paths or multi-language claims — the eight-language edition is retired and its numbers are historical (MANUAL §2.3).

## Commands (use the Makefile, not raw cargo)

- `make release` — builds Rust lib + C IBus engine. `make test` = `cargo test --release` (includes accuracy regression guard). Single test: `cargo test --release <name>`.
- `make check` — **required gate**: `cargo fmt --check` + `cargo clippy --release --all-targets -- -D warnings` + `cargo test --release` + `check-wasm`. Native-green does not imply wasm-green, so `make check-wasm` is not optional. Needs `rustup target add wasm32-unknown-unknown`.
- `make eval [SPLIT=valid|test] [K=8]` — runs the Nepali benchmark (`eval_langs` over `data/pairs/$(SPLIT).jsonl`). `make eval-session` measures cold vs. adaptive-learning session accuracy.
- `make train [PAIRS=500000] [EPOCHS=5]` — trains the engine. Writes `data/akshar.model` directly. `make model` (= `calibrate` then `promote`) calibrates blend weights and fits the size budget.
- `make data-prepare [ASSUME_SOURCE=AK-Freq]` → `make lexicon [LEXICON_ARGS="--max-words 700000"]` → `make train` → `make model` is the full from-scratch sequence; `ASSUME_SOURCE` is a **placeholder label only** — the cleaned splits dropped `source`, so the native/entity split is unrecoverable and every figure on `test.jsonl` is pooled across both. Each step is idempotent and independently re-runnable. Verified end-to-end at `--smoke` scale (`cargo run --release --bin train -- --smoke`, ~6s). **There is no `data-fetch` step** — the corpus and Aksharantar splits are vendored under `data/` (see `data/README.md`).
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
- **Log experiments in `docs/experiments/YYYY-MM-DD-<slug>.md`.** Record the question, the method, the numbers, and the decision taken. Cite them from MANUAL rather than restating figures there, so a paper can trace a claim back to the measurement that produced it. If an experiment contradicts something already written in MANUAL, correct MANUAL and say which claim changed — do not quietly overwrite a number.

## Current work: candidate-generation redesign

Scoped in `docs/experiments/2026-10-04-canonical-key-retrieval-ceiling.md`. Read it before touching retrieval or scoring.

- Baseline is established and re-measured on the Nepali-only pipeline: **60.35% top-1, 76.15% top-5, 78.32% in-list@8, 84.00% reachable@50**, 0.836 ms, 9.61 MB container — all figures **pooled over native words and named entities**, on the same 4,101 rows where the old 8-language artifact gave 60.40% / 77.59% / 24.44 MB / 0.794 ms. The old "78.32% native" was the 2,108-case native subset and is not comparable.
- **reachable@50 = 84.00% is the ceiling for the current generator.** So the error splits into 16.00% generation (gold never surfaced) and **23.65pp of ranking headroom** (top-1 → reachable). Ranking is the bigger prize and costs no artifact size — attack it first, with the factored matra term, not with a new index. Lexicon width is settled: 700k beats 300k on every metric (+1.34pp top-1 for +2.45 MB).
- **21.68% of queries are not in the top 8; 16.00% are not surfaced at all even at 50 candidates.** Rescoring cannot fix the latter; only candidate generation can. Judge retrieval work on in-list@8, not top-1.
- **D25 (open):** the normalizer expands up to 6 query variants but `engine.rs` decodes only `variants[0]`, so the expansion is inert for cold start. `saathee`→`साथिए` while `saathi`→`साथी`. Most likely resolved for free by folding vowel-length equivalence into the tier table's deviation term.
- The collapsed-key inverted index measures a **49.18%** retrieval ceiling (82.5% gate; shipped decoder is 60.35% top-1), so it is added **alongside** the lattice decode, never instead of it.
- φ is derived empirically, **keeping the inherent schwa** — the literal spec rule costs 16.2pp of exact key match.
- Lexicon is 700k words (`data/lexicon.bin`, 4.49 MB); artifact budget raised to 12–15 MB.
- The baseline is trained and passing; none of the redesign is implemented yet. Next: verify D2 actually cut the 36.24% gold-absence figure against the 700k lexicon, then build the derived-φ tool and A/B **in-list@8** with the key index added as a third candidate source.
