# Devanagari-Script Plan (live)

Research program (2026-09-22): `docs/plans/research-agenda.md` — ranked bets
with pre-registered falsification criteria, sourced from
`docs/research/sota-signals.md`, `docs/research/indic-morphology.md`,
`docs/research/rerank-beyond-linear.md`. The steps below are the pre-research
record, kept for provenance.

Goal: one **script-first** phonetic engine for all languages written in
Devanagari lipi (Hindi, Nepali, Marathi, Sanskrit, …). Language-agnostic by
construction; judged at the script level.

## Non-goals

- No per-language models, forks, or language tags in the pipeline.
- No new delivery surface (playground, extension) before accuracy moves.
- No "exponential" claims: progress is measured in percentage points with
  McNemar p-values per `docs/MANUAL.md` §8.4.

## Current state (2026-09-10)

- Core is already script-general: akshara segmentation, EM emissions, KN LM,
  decoder, scoring — nothing downstream reads a language tag
  (`data/README.md`, MANUAL §11).
- Priors are not: frequency vocabulary is news-domain-heavy, the 34-suffix
  strip list is Nepali-specific, eval is one benchmark (Aksharantar, 4,101).
- Headroom is ranking, not generation: oracle@50 94.3% vs top-1 81.8% on
  AK-Freq; 51.9% of misses are matra-only; oracle@2 already 89.8%.

## Accuracy program (ordered by expected points/effort)

1. **Factored matra model** — STATUS 2026-09-22: tried, measured, FALSIFIED
   as a word-level prior rerank; code removed, log kept at
   `docs/plans/archive/2026-09-22-factored-matra-model.md` (head-to-head
   78/182, would flip 992/1707 correct — a second prior cannot beat the
   prior; residual needs context/morphology). Redirects to step 2.
2. **Sentence context — LANDED 2026-09-22 (+1.22pp realistic).**
   `set_context_word` was an empty stub; now sets prev + clears cache.
   `CorpusCtx` (count>=20 sidecar, 436k pairs, `make ctx-model`) blends the
   bigram log-ratio pre-squash at `CTX_W=0.5` with an abstention rule and
   `AKSHAR_NO_CORPUS_CTX` ablation. Sentence word@1 87.58% -> 88.80%
   predicted (88.86 oracle); AK-Freq byte-identical. Follow-ups: v7
   interning (24 MB -> ~5 MB, kills the +0.12 ms/word string hashing too),
   wasm-profile cutoff. Rejected along the way: unconditioned bonus
   (regressed -0.5pp via unigram double-count), top-24 cascade cap
   (-0.40pp top-5).
3. **Top-2 discriminator — FALSIFIED 2026-09-22.** Three training regimes
   (full features, string-only, domain-matched), every τ swept: no
   operating point above baseline; best is a tie via abstention. Log:
   `plans/archive/2026-09-22-top2-discriminator.md`. The choice needs new
   information per decision, not reweighted old signals.
4. **Conditional blend — FALSIFIED 2026-09-22.** Per-candidate γ from
   frequency rank (low γ where the prior is trustworthy, high where OOV):
   every hi>lo harmed native (hi=0.8: 425/519) while pooled rose — the
   constraint rejected all of them; converged to uniform (0.3, 0.3).
   Mechanism: reweighting two signals cannot create a third. The learned
   model is entity-blind (γ=1.0 is 5pp worse than the heuristic); entities
   need entity INFORMATION (context bigrams helped; sparse NEF features
   help), not more weight on blind features. Code removed; tuner kept with
   the constrained objective for future knob searches.
3. **Script-wide frequency priors** — replace the single-domain vocab with a
   balanced multi-source Devanagari prior; keep per-language strata, headline
   the pooled script number.
4. **Language-agnostic morphology** — generalize the 34-suffix strip into a
   learned or shared-script suffix model.
5. **Log-linear candidate fusion** — one tunable score, removing the `u64`
   band defect class (30.8pp regression, D18).
6. **Stratified eval** — per-language splits + pooled headline; Dakshina as a
   second benchmark; never tune on the test split.

Each step ships with `make eval` / `make ablate` numbers and a MANUAL §9/§11
update, or it does not ship.

## Delivery surfaces (after accuracy)

- **Playground** (`apps/playground`): Cloudflare Workers + R2-hosted
  `akshar_wasm.model`, editor users can type/write in. Reuses `js/` wrapper
  + `wasm/pkg` as-is; no engine fork.
- **Chrome extension** (`apps/extension`): Manifest V3 content script around
  the same wrapper. No model fork; learns on-device only.
- **Repo shape** (when surfaces land): `apps/{playground,extension}`,
  `packages/{engine-js,wasm}`, `docs/` stays the manual + this plan +
  `plans/archive/`. No `src/` binaries, no committed `data/`.

## Repro gates

`make check` (fmt + clippy `-D warnings` + tests + wasm target check),
`make eval` / `make ablate` for every accuracy claim.
