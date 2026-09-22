# Correctness Audit — Experiment Log (2026-09-10)

Scope: Correctness first, fast checks only (no `train-full`, no PDF rebuild).
Baseline: v1.2.0, `make eval` 81.83/47.79/31.21 per README.

## Hypotheses under test
- H1 `Trie::get_top_k_suggestions` max-heap inversion drops correct top-k (`src/core/trie.rs:131-163`).
- H2 `ModelDecoder::build_edges` byte-slices panic on non-ASCII roman (`src/core/decoder.rs:568-582`).
- H3 C API `static mut` + false NULL-tolerance = UB/crash (`src/c_api.rs:10,67-95`).
- H4 IBus NULL `suggestions_json` deref segfaults; finalize leaks (`src/ibus_engine.c:91-93,58-66`).
- H5 `train.rs` discards fresh dense stats; full run empties dev set; Phase-2 ignores holdout.
- H6 Cache survives state import; WASM reset degrades unified engine; gamma NaN poisons scores.

## Exp 1 — Trie top-k (H1)
- Repro: insert freqs 1,2,3 prefix "a", k=2 → expect {3,2}, buggy code yields {3,1}.
- Fix: `BinaryHeap<Reverse<(u64,WordId)>>`, peek=min, threshold=min.
- Verify: new unit test `top_k_returns_largest` + `cargo test --release trie`.

## Exp 2 — Decoder non-ASCII (H2)
- Repro: `decode("café")`, `decode("नमस्ते")` → old code panics on `&roman[pos..pos+l]`.
- Fix: `roman.get(pos..pos+l)` + early `[]` when `!roman.is_ascii()` after lowercase.
- Verify: `decode_never_panics_on_non_ascii` test.

## Exp 3 — C API hardening (H3)
- Fix: `OnceLock<Mutex<ImeEngine>>`, NULL guards returning `[]`/no-op, `CString::new` fallback (no unwrap across FFI), `get_dictionary_path()->Option`, persist-on-confirm (best-effort `save_dictionary`).
- Verify: `cargo test --release`, `cargo clippy --release --all-targets -- -D warnings`.

## Exp 4 — IBus C hardening (H4)
- Fix: NULL guards on all 3 `akshar_ime_get_suggestions` sites; finalize frees `preedit_string`+`table` (unref); UTF-8-safe BackSpace + `g_utf8_strlen` cursor; bounds-check cursor pos; check `ibus_bus_request_name` result.
- Verify: `make release` compiles C engine (needs ibus/jansson); else `gcc -fsyntax-only`.

## Exp 5 — Train pipeline (H5)
- Fix: push raw dense features to `DenseStats` then pack computed mean/std; reserve dev tail even on full run; filter Phase-2 vocab lines via `is_holdout(line, DEFAULT_HOLDOUT_DENOM)` (document denom=0 opt-out).
- Verify: `cargo test --release`, `cargo clippy`; NOT running train-mid/full per fast-check scope.

## Exp 6 — Cache/WASM/reranker/docs/CI (H6)
- Fix: clear cache in `apply_to_engine`; WASM reset preserves unified vocab/word-trie/sparse (rebuild from stored unified fields, not bare `from_model`); gamma/sparse-scale finite-checks; sync `AKSHAR_FUZZY_BASE` doc 50000→600000; CI adds fmt + `--release` clippy/test + wasm check.
- Verify: `cargo fmt --check`, `cargo test --release`.

## Results
- Exp 1 PASS: `top_k_returns_largest_not_smallest` ({3,2}), `top_k_k_zero_returns_empty`. Old max-heap would have returned {3,1}.
- Exp 2 PASS: `decode_never_panics_on_non_ascii` (café/नमस्ते/Zürich/emoji/mixed). Guard `!is_ascii → []` + `get()` + boundary skip.
- Exp 3 PASS: `OnceLock<Mutex<Option<ImeEngine>>>`, NULL→`[]`/no-op, no unwrap across FFI, `Option<PathBuf>` config, persist-on-confirm best-effort.
- Exp 4 PASS: NULL guards (3 sites), finalize frees table+preedit, UTF-8 BackSpace, `g_utf8_strlen` cursor, cursor bounds-check, `request_name` checked. `make release` links clean.
- Exp 5 PASS (compile-gated, no retrain per fast-check scope): raw-feature `DenseStats` + pack computed mean/std; dev tail always reserved (`Pre-decoding ... (dev reserved: N)`); Phase-2 skips `is_holdout(line,200)`. Next validation: `train-mid` + `eval`/`ablate` before any model release.
- Exp 6 PASS: `apply_to_engine` clears cache + `clear_suggestion_cache()`; WASM reset in place (`reset_learned_state`); NaN-guard gamma (source + blend), finite dense/NaN-safe sparse-scale; `AKSHAR_FUZZY_BASE` doc 50000→600000; CI = fmt + `--release` clippy/test + wasm check.
- Gate 2026-09-10: `cargo fmt --check` clean; `cargo clippy --release --all-targets -- -D warnings` clean; `cargo test --release` 86 lib + 1 accuracy + 6 fuzzy green; `cargo check/clippy --features wasm --target wasm32-unknown-unknown` clean; `make release` clean.

## Train-mid validation (2026-09-10, run after the fixes above)

Baseline model backed up to `/tmp/opencode/akshar.model.baseline` before training.

### Finding 0 — README numbers are stale vs the shipped model (pre-existing, not our regression)
`make eval` with PRISTINE (stashed) code on `data/akshar.model` (Sep 9): AK-Freq 81.02/91.75, AK-NEI 45.75/70.24, AK-NEF 29.50/51.90 — byte-identical to our patched code. README claims 81.83/47.79/31.21 (measured 09-08). So the Sep-9 model already differed from the measured one; our changes are accuracy-neutral. README/MANUAL need a re-measure note.

### Train-mid run (500k reranker pairs, 5 epochs, 12 EM iters; 571s total)
- Phase 1 EM: 128.76s, aksharas 16552, chunks 101007; MKN discounts d2=1.0369/d3=1.4438 (confirms unclamped Chen-Goodman regime).
- Phase 2 vocab: **skipped 14,465 held-out lines (1-in-200)** — holdout fix verified live; vocab 468,511 words.
- Phase 3 reranker (chunked, 5 batches): `Pre-decoding candidates for 500000 training pairs (dev reserved: 4000)` — dev-reservation fix verified; dev-before 54.64%; batch dev top-1: 67.98 → 67.87 → 64.52 → 65.48 → 66.36. **Best-by-dev packing engaged** (batch-2 weights, dev loss 1.0423, over final 1.0790) — the overfit guard that was dead on full runs now works.
- Phase 4 pack: sparse 20,570 non-zero/1M (1.96%); **dense stats raw: emit mean 5.462 std 3.283, lm mean 25.080 std 6.886** (previously would have packed ~0/~1 z-scores over stale constants) — dense-stats fix verified. Artifact 11.37 MB. Smoke test: namaste→नमस्ते, pustak→पुस्तक, sarkar→सरकार, pani→पानी top-1.

### Eval of retrained model vs baseline (same harness, same split, n=4101)
| Split | Baseline | Retrained | Δ |
| :--- | ---: | ---: | ---: |
| AK-Freq top-1/top-5 | 81.02 / 91.75 | 80.98 / 91.84 | −0.04 / +0.09 |
| AK-NEI top-1/top-5 | 45.75 / 70.24 | 45.32 / 70.15 | −0.43 / −0.09 |
| AK-NEF top-1/top-5 | 29.50 / 51.90 | 29.01 / 51.53 | −0.49 / −0.37 |
| Pooled top-1 (95% CI) | 60.64 [59.08, 62.13] | 60.40 [58.94, 61.89] | n.s. (CIs overlap) |
| MRR | 0.6809 | 0.6798 | n.s. |
| Latency | 0.716 ms | 0.724 ms | flat |

### Ablation of retrained model (AK-Freq top-1)
full 80.98; −NO_TRIGRAM 74.95 (−6.03); −NO_TRIE_UNION 80.50 (−0.48); −NO_SPARSE 80.50 (−0.48); −NO_VARIANTS 80.98 (−0.00); gamma=0 80.74 (−0.24). No component collapsed; trigram still dominant (larger than the −3.89pp on the old model — expected since fresh dense stats shift blend balance); sparse-native now −0.48 vs −0.10 before.

### Conclusion
Pipeline fixes validated end-to-end with no accuracy loss (all deltas within bootstrap noise). `data/akshar.model` is now the retrained artifact (backup of previous at `/tmp/opencode/akshar.model.baseline`, retrained copy at `/tmp/opencode/akshar.model.trainmid`). Follow-ups (not done): `train-full` still untested overnight; README §measured-performance + MANUAL re-verification provenance still cite the 09-08 numbers and should be updated to the table above.
