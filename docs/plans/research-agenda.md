# Research Agenda (live, 2026-09-22)

Source reports: `docs/research/sota-signals.md`, `docs/research/indic-morphology.md`,
`docs/research/rerank-beyond-linear.md` (three background research agents,
primary sources cited in each). This file ranks the bets and pre-registers
what kills each one. Rule: no bet runs without a falsification criterion;
no bet ships without a test confirmation.

Standing evidence constraining every bet below:

- Native 80.98% / NEI 45.32% / NEF 29.01% (test); oracle@2 89.8% native.
- Four falsifications on record: matra prior, pooled scalar tuning,
  conditional-γ, top-2 discriminator (see `plans/archive/`). Shared moral:
  reweighting old signals fails; only new information has ever moved a number
  (context bigrams +1.22pp on sentences).
- Valid/test distribution gap: valid's strata mix (Dakshina/Samanantar/
  IndicCorp majority) rewards what test punishes. Tune per-stratum or on
  native-family valid; distrust pooled.

## Ranked bets

### 1. MBR consensus selection — FALSIFIED 2026-09-22 (best +2/798, noise)

24 configs (T × K × {edit, matra-insensitive}) on valid native: flat
posteriors herd to popular wrong forms; sharp posteriors re-select the
1-best. Converged n-best lists carry no votable information. Log:
`plans/archive/2026-09-22-mbr-selection.md`. The posterior-miscalibration
diagnosis reduces to reranker training (bet #2).

### 2. Pairwise ranking loss for the sparse/dense trainer (train-mid gate)

Source: `rerank-beyond-linear.md` §2 (Hopkins & May 2011; Shen et al. 2004).
Same linear model, same features — objective matched to the top-1 decision:
gold-vs-rival hinge instead of listwise CE. Zero inference change.
Kill: `train-mid` + valid shows no top-1 gain over CE with identical decoding
(never `train-quick`: chunked mode engages only above 200k).

### 3. Script-general morphology union table — LANDED AS INFRASTRUCTURE, NIL ACCURACY (2026-09-22)

Snowball Hindi port done (164 entries, longest-first, consonant gates,
`akshara::is_consonant` helper, unit-pinned) plus a real ordering bugfix
(old first-match returned को for ambiguous readings). Measured: no valid
movement (−5 pooled, noise). Ceiling probe then killed the follow-up:
only 13/143 valid-native misses are OOV-with-strippable-stem, nearly all
via the OLD 34 suffixes — so transcribing 200+ Nepali rules is cancelled
on cost/benefit (≤13 cases, uncertain flip rate). Table stays as
zero-cost gated data (future retrains will train sparse template 5 on the
new firings); Sanskrit stays excluded per the survey. If Hindi eval ever
exists, re-judge the Snowball half there.

Source: `indic-morphology.md` proposal. Snowball Hindi suffixes (BSD) +
Koirala-I/Shrestha-128 Nepali (transcribe with citations) + Pimpale/
Majgaonker Marathi sets; longest-match; existing `stem_f >= 5` vocab gate
makes cross-language collisions safe by construction. Same two call sites,
no new deps, no per-language branches. Sanskrit excluded (sandhi).
Kill: no top-1/top-5 move on `eval-ime`/`eval-errors` per-language splits
with morph attribution via `AKSHAR_NO_RERANK`/`AKSHAR_NO_SPARSE` ablations.
VERDICT 2026-09-22: Snowball half landed (ordering bugfix included), nil
valid movement; ceiling probe capped the whole track at 13 rescuable
valid-native misses (mostly old-34 suffixes) — Nepali transcription
cancelled, see header.

### 4. NE routing — MEASURED 2026-09-22, routing designs killed, generation-side indicated

Burial analysis (test NE): freq-burial 14%, demotion +140/−138 wash,
wider-beam contraindicated (depth-200 loses −24 net — distractors beat the
ranker), 46% gold-absent-from-50. Attenuation/demotion/router all parked.
Remaining: attestation-weighted or entity-upsampled training (`train-mid`,
~40 min machine time). Log: `plans/archive/2026-09-22-ne-routing.md`.

### 5. Attestation-weighted training pairs (data-side, cheap)

Source: `sota-signals.md` T4 (Roark §4.2). Weight EM/reranker pairs by
variant attestation instead of uniform — teaches which romanization
conventions dominate, attacks the oracle@2 tie class. Multi-ref-aware
reranker loss (any attested variant counts).
Kill: no valid movement after a `train-mid`; costs one training run to test.

### 6. Goto-style source-context + segmentation features (after #2)

Source: `rerank-beyond-linear.md` §6 (Goto 2003, local PDF verified).
Roman-bigram × akshara conjunctions (we have none — all lexicalized
templates are target-side/endpoint-anchored) + chunking-validity features
(emission-rank/entropy consumed). Linear-compatible, hashed as usual.
Runs after the objective is right (#2): better features under a wrong
objective repeat the gamma story. Kill: template ablation shows nothing.

### Explicitly not queued

- Forest/exact-k-best/ROVER: generation loss is 5.7%, not the binding
  constraint; revisit only if NE oracle coverage collapses.
- k-best MIRA: keep as challenger to #2 if pairwise stalls, not parallel.
- MERT dense+gamma: subsumed by the `tune_weights` coordinate descent
  already run (converged to near-defaults).
- LLM/API/neural transfer: ceiling reference only (Azam et al. 2025).
- Sanskrit suffix stripping: unsound under sandhi (see morphology report).

## Measured: error census (2026-09-22, test AK-Freq, 401 misses)

| Class | misses | gold in top-10 | reading |
|---|---|---|---|
| Substantive (consonants differ; incl. anusvara-vs-conjunct, loans) | 192 | 49% | half needs generation/data help — the hard core |
| Other-matra | 55 | 69% | rerank-side |
| i-length | 42 | 86% | rerank-side |
| schwa | 37 | 84% | rerank-side |
| halant/conjunct | 36 | 53% | split: half rerank, half generation |
| u-length | 27 | 85% | rerank-side |
| nasal order | 12 | 67% | rerank-side |

Vowel-length (u+i = 69) at ~85% in-beam; length + schwa + nasal = 118
misses at ~82% in-beam — the addressable ranking mass, favoring bets #1
(MBR consensus), #2 (pairwise loss), #6 (Roman-context features).
Substantive/halant misses with gold absent favor data/channel bets (#5)
and morphology (#3). Probe removed after measurement; the coarse
matra/halant/substantive split lives on in `analyze_errors`.
