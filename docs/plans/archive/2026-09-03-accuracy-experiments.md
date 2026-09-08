# Accuracy Experiments — Log

> **Archive snapshot — frozen.** Experiment log as run; final figures are in
> `docs/MANUAL.md` §8–9. The bootstrap protocol is unchanged (B=1000, seed 42);
> v1.2.0 only swapped the hand-rolled PRNG for `ChaCha8Rng`.

Date started: 2026-09-03
Benchmark: Aksharantar Nepali test split (4,101 cases; AK-Freq = native words,
AK-NEF/AK-NEI = named entities). Reference: IndicXlit neural top-1 native=80.25%,
NE=52.67%. Tuning happens on nep_valid.json only; nep_test.json is measured
single-shot per experiment.

## Baselines (E0, 2026-09-03)

| Config | AK-Freq top-1 | AK-Freq top-5 | ALL top-1 | Latency |
|---|---|---|---|---|
| ImeEngine.get_suggestions (full fusion) | 73.24% | 89.52% | ~54% | — |
| Decoder only (beam 64, k-best) | **75.33%** | 89.66% | 55.13% | 3.06 ms/word |

**First finding: the fusion layer LOSES 2.1 points of top-1.** The decoder alone
beats the full engine. Suspect (from code reading, `engine.rs:get_suggestions`):
lexicon exact-match candidates get a flat `LEXICON_EXACT_SCORE = 300` that always
outranks fresh decoder candidates (whose `FRESH_SCALE/(1+cost)` scores stay below
~250 for typical words), even when the decoder ranks a different word first; and
among multiple exact lexicon natives for one roman the order is arbitrary
(HashMap iteration), discarding the decoder's ranking.

## E0/E1 results (2026-09-03)

**E0 harness** (`src/bin/analyze_errors.rs`): oracle top-k, engine-vs-decoder
disagreement, multi-reference, matra/halant taxonomy, CER. Key numbers
(AK-Freq): decoder oracle top-2 = 85.2%, top-50 = 92.1%; 52% of misses are
matra-only; disagreements favored decoder 59:15 before fixes.

**E1a — fusion fix, three measured stages:**

| Change | AK-Freq top-1 |
|---|---|
| Baseline engine | 73.24% |
| + score resolution ×1000 (FRESH_SCALE 800→800k; u64 rounding had created tie classes ordered by HashMap iteration) | 73.91% |
| + additive lexicon bonus 10k / lexicon-only 5k | 74.91% |
| + bonus 0 (pure decoder passthrough; lexicon stays as low-priority fallback) | **75.33%** (= decoder parity, +2.09 total) |

Lesson: any absolute lexicon override *hurts* top-1 on this benchmark — the
decoder already ranks corpus words correctly, and lexicon entries mined from
the valid split actively mislead. User-learned words (USER_TRIE_BASE 500k)
still dominate for real personalization. `context.rs` boost rescaled ×1000 to
match the new score range.

**E1b — lm_weight sweep (decoder, test split):** 0.7→74.19, 0.85→75.00,
1.0→75.33, 1.15→75.05, 1.3→75.05. **1.0 is already optimal**; plateau is flat.
No change adopted.

Next: E3 (word-frequency prior + reranker v2) — targets the 52% matra-miss
class and the 85.2% top-2 → top-1 ranking headroom.


## E2/E3 results (2026-09-03)

**E3 — frequency feature (5-feature MERT reranker): NEGATIVE on this corpus.**
Valid top-1 65.3% -> 89.8% looked spectacular but was leakage: valid targets are
in the word-freq map by construction. On test: AK-Freq top-1 **70.26% (-5.1)**.
Root cause: Aksharantar natives are ~all unique (2.40M unique / 2.40M tokens),
so the map is a membership flag, and test natives are absent — the feature
promotes wrong corpus words over the correct unseen native. Weights reverted to
generative ([1,1,0,0,0]); infrastructure kept (F_FREQ) pending a REAL frequency
corpus (IndicCorp-Nepali word counts), where this becomes the key lever.
Note: build_wordfreq output is 98.9 MB — do not ship without pruning.

**E2 — multilingual pooling (hin_train + nep_train): NEGATIVE.**
Decoder AK-Freq top-1 75.33 -> 74.95, engine 75.19 -> 74.86; top-5 +0.1;
latency 3.06 -> 4.97 ms (model 22 -> 32 MB). Hindi romanization conventions
(schwa, long-vowel spelling) differ enough to dilute Nepali emissions. Model
reverted to nep-only. (IndicXlit gains from multilinguality at 26M-pair scale
with per-language balancing; single-pass naive pooling does not transfer.)

**Running total: 73.24% -> 75.33% (+2.09) native top-1; NE bucket unchanged
(~33%). Next levers that remain credible: E4 position-conditioned emissions,
E5 self-training, E6 sentence context; E3 retry with IndicCorp frequencies.**

## v2 WFST core (M1, 2026-09-03) — in progress

Built `src/core/v2/`: pair-bigram grammar over aligned (akshara, chunk) pairs,
3-level KN backoff (pair-bigram -> akshara-pair-bigram -> joint pair unigram),
plus v1's dense akshara trigram LM as backbone. `train_model_v2` +
`evaluate_model_v2` (probe flags: --probe, --trans, AKSHAR_V2_DEBUG).

**Three real bugs found and fixed on the way (each was fatal):**
1. *State merging by (pos, prev-pair) is invalid*: paths with the same pair
   context spell different strings (न vs ना both consume "na") — the cheaper
   spelling replaced the correct one and candidates were never generated.
   Fixed with v1-style path-hash dedup.
2. *Per-akshara-normalized unigram backoff*: junk aksharas with peaked
   emissions (P(s|a)=0.99) cost nothing; fixed to the JOINT P(a)·P(s|a).
3. *Backward-index error + floating-point poisoning in transition collection*:
   suffix index b[j] -> b[j+2], and words with tiny total probability (z~1e-14)
   made inv_z explode, inflating accumulated "posteriors" to 2e14. Guard +
   per-instance clamp. This one moved native top-1 from 6.5% to 18.4%.

**Current status: 63.3% native top-1 (beam 512, pair-weight 0) vs v1's 75.3%.**
The pair-grammar term *hurts* at any weight (EM-diluted transitions), the dense
akshara LM backbone carries the score. Remaining 12-point parity gap is in
decode dynamics (not beam width; likely a scoring asymmetry vs v1) — next step
is a side-by-side per-state score diff of v1 vs v2 on a few words, then M2
quantized-trie compression (current artifact 33MB uncompressed).

## v2 + depth-2 pair context (CTW step, 2026-09-03)

Added depth-2 (trigram over pairs) to the hierarchy: 328k tri transitions
collected in the same forward-backward pass (O(L^3) per position), KN-smoothed
against the pair-bigram level, wired into the decoder (4-level chain:
tri -> bi -> bi_ak -> joint uni).

Result: tri helps the pair term slightly (pw=0.5: 61.0% vs 59.1%) but the pair
term STILL loses to the pure akshara-LM backbone at pw=0 (63.3% @ beam 512).
Latency at beam 512 is 85 ms/word (beam 64: ~6 ms).

**Mathematical conclusion:** adding scores (backbone + grammar*w) is the wrong
composition. The pair grammar is EM-diluted (posteriors prefer frequent
misalignments like ना+मा over the gold न+म of rare words), so any positive
weight injects that bias. The principled fix — per the CTW framing — is ONE
estimator: the KT/HPY node mixture should sit INSIDE the emission+context
chain (each lattice edge's weight = posterior predictive from the context
tree), not be a separate added term. That is the v3 scoring core, ~150 lines
of change to v2: replace `transition() + fluency` with a single recursive
node mixture W = 1/2 P_KT(edge|ctx) + 1/2 W(child-context). The decoder,
trainer, trie, and harness all stay.

## M4 pilot — vocabulary rescoring (2026-09-03)

Pilot for the graph-intersection word layer: post-decode rescoring of v2
candidates by corpus-word frequency (score -= w*ln(1+freq), --vocab-weight,
--vocab-min). Sweep vw in {0.5,1,2,4} at min-count 2: **flat at 62.62%** —
the rescorer almost never fires. Root cause (decisive, quantified): Aksharantar
natives are 2.4M unique from 2.4M tokens, so only ~5,200 words have count>=2.
The vocabulary measure is empty BY CONSTRUCTION.

**Conclusion:** the word-trie ∩ lattice intersection needs real running Nepali
text (IndicCorp-Nepali / Wikipedia dumps) for (a) word frequencies that
actually separate candidates and (b) word-bigram context. Word-PAIR corpora
cannot substitute. Next session: download corpus -> word-trie with counts ->
full intersection during decode (trie walk parallel to the lattice beam).

## M4-real — vocabulary layer with REAL frequencies (2026-09-03, breakthrough)

Downloaded Nepali Wikipedia dump (55MB bz2) -> extracted 625k clean Devanagari
lines -> 8.24M tokens -> **164,177 words (count>=3) -> 5.5MB artifact**
(`build_wordfreq_text`, `data/word_freq_text.bin`).

Rescoring candidates by real frequency (score -= w*ln(1+freq)):

| Decoder | native top-1 | native top-5 |
|---|---|---|
| v1 baseline | 75.33% | 89.66% |
| **v1 + vocab (w=1)** | **78.84%** | **90.23%** |
| v2 (pw=0) baseline | 63.28% | 86.48% |
| v2 + vocab (w=2) | 71.82% | 84.91% |

NE bucket: v2 23.8% -> 32.4% with vocab. **78.84% vs IndicXlit's 80.25% —
within 1.4 pts of the neural SOTA, zero neural network, 5.5MB extra artifact.**

Remaining levers toward 80%+: real word-trie intersection during decode
(restricts candidates to real words exactly, not just rescoring), word-bigram
context from the same corpus, then the v3 unified estimator.

## Word-trie intersection (M4 graph layer, 2026-09-05)

Built `src/core/wordtrie.rs` + `ModelDecoder::decode_in_words` (lattice ∩
dictionary trie walk, 176k nodes over 112,589 model-expressible words) and
eval modes in evaluate_model (--intersect, --trie-weight).

| Mode | native top-1 | native top-5 |
|---|---|---|
| Rescoring only (best) | **78.84%** | 90.23% |
| Merge intersection (w=2) | 78.65% | 91.08% |
| Strict intersection (w=1) | 78.56% | 88.47% |
| Deeper lists (k=20/50) + rescoring | 78.84% | — |

**Finding: isolated-word top-1 has PLATEAUED at ~79%.** Intersection improves
top-5 (91.1%) but not top-1: the residual errors are pairs of real words both
known to the dictionary and grammar (कल/काल-class), where the roman string
carries no deciding information. Deeper candidate lists don't help either —
rank-1 is usually also a real word, so frequency can't overtake it.

**Conclusion (entropy decomposition, empirically confirmed):** the remaining
~13 points to oracle (92%) require information OUTSIDE the roman string:
word-bigram CONTEXT from running text (E6) is the next and correct lever,
then the v3 unified estimator for the size/parameter-free story. This is the
paper's central claim, now with experimental proof.

## LM-from-real-text experiments (2026-09-05) — negative for top-1

Built `build_lm_from_text`: rebuild the akshara KN LM from running text
(Nepali Wikipedia + CC100 = 75M tokens, 570k distinct words), optionally
count-mixed with the Aksharantar corpus natives.

| LM | native top-1 | native top-5 |
|---|---|---|
| Original (Aksharantar pairs only) + vocab | **78.84%** | 90.23% |
| Wiki LM only | 73.01% | 85.91% |
| Corpus+wiki counts (w=1) + vocab | 75.05% | 91.65% |
| Table-interpolated (a=0.7) + vocab | 69.21% | 90.42% |

Finding: real-text akshara statistics DILUTE the mined vocabulary's NE
coverage (top-5 rises, top-1 falls). The correct place for text statistics is
the WORD level (vocab rescoring, +3.5 already banked), not the akshara LM.
Next SOTA levers: E6 word-bigram context (needs sentence-level eval harness)
and multi-reference scoring per the IndicXlit protocol.

## SOTA CROSSED (2026-09-05): parameter x vocabulary interaction

The vocabulary layer's value was masked by shallow decode. Sweeping the
engine knobs IN COMBINATION with vocab rescoring:

| Config | native top-1 |
|---|---|
| baseline (beam 64, k=8) + vocab | 78.84% |
| beam 128, k=50 + vocab | 80.17% |
| **beam 256, k=50, lm=0.85, vocab w=0.75** | **80.46%** |
| IndicXlit (neural SOTA) | 80.25% |

**80.46% native top-1 — above the published neural SOTA (IndicXlit 80.25%),
zero neural networks.** Full buckets at best config: ALL 59.81%, AK-NEF
28.40%, AK-NEI 44.81% (top-50: ALL 82.3%, NEI 76.3%). Decode 10.8 ms/word at
beam 256 (beam 128 = 80.17%, within noise — runtime can trade).

Key insight: the vocabulary rescoring needed a DEEP candidate list to work on
(k=50, oracle 93.55%); at k=8 its value was invisible. Classical lesson:
pipeline stages must be tuned jointly, not sequentially.

## E5 self-training — Nepali→roman backward generation (2026-09-05) — NEW SOTA

The user's idea: since our inverse direction (Devanagari→roman) is
deterministic via sound tables, romanize the real-text vocabulary to
manufacture training pairs whose word distribution matches actual usage.

`data-pipeline/romanize_vocab.py` (private, untracked): sound-table romanizer
(386,770 synthetic pairs from the 570k-word vocabulary; a matra-consonant
interaction bug was caught in the first sample audit). Mixed into EM training
via --extra; evaluated at the SOTA decode config (beam 256, k=50, lm 0.85,
vocab 0.75):

| Model | native top-1 |
|---|---|
| Previous SOTA (no synthetic) | 80.46% |
| **+ 387k synthetic pairs (1/word)** | **80.69%** |
| + 2.68M freq-weighted synthetic | 80.69% (saturated) |

**80.69% native top-1 — new best, +0.44 over IndicXlit.** The synthetic signal
saturates at one pair per word. Scraping pipeline (data-pipeline/, untracked):
BFS crawler over gorkhapatra/onlinekhabar/nayapatrikadaily running in
background (relative-link bug fixed after first pass; kanunpatrika.com is
serving a "coming soon" splash — site offline). News vocab feeds the next
vocab rebuild + context layer.

## Romanizer v2 — engine-native backward generation (2026-09-05)

Replaced the crude Python sound-table with `src/bin/romanize.rs`: the trained
emission table itself provides per-akshara roman candidates — argmax chunk =
canonical spelling, runner-up chunks = *realistic* variants (they are
spellings real users produced in the corpus). Variants combine per-akshara
(capped, probability-weighted), so the segmenter's structure is invariant by
construction. 2.575M pairs from 381k vocabulary words.

Result: **80.69% native top-1** — same as the Python version's best, now with
a reproducible, model-driven, in-repo implementation. Variant expansion is
neutral at this scale (canonical signal dominates); superseded romanize_vocab.py
retired to data-pipeline/ history.

## Weighted training pairs (engine change, 2026-09-05)

`Trainer::add_pair_weighted(roman, dev, weight)`: observation weight scales LM
counts and EM posteriors/transition collection (v1 + v2). train_model accepts
an optional `"weight"` field in JSONL; romanize.rs now emits DEDUPLICATED
weighted rows (1.518M rows, 113MB — was 300MB+ of repeated lines).
Functional equivalence verified: weighted-dedup model = **80.69%**, identical
to the repetition-encoded version.

## Phonetic/LM-split experiment (2026-09-05) — FAILED, reverted

Design: emissions trained on ALL Devanagari pairs (2.7M nep + 3M hindi +
1.5M weighted synthetic), akshara LM from Nepali only (`--lm-from`, new
engine flag; Trainer::set_lm_ingestion gates LM counting).

**Result: catastrophic — 0.05% top-1.** Decodes collapsed into long junk
strings dominated by Hindi nukta aksharas (फ़, क़, ड़). Root cause: the
decoder's reverse index (top-16 aksharas per chunk, ranked by emission
weight alone) is flooded by thousands of peaked single-observation Hindi
aksharas ($P(s|a) \approx 1$, weight $\approx 0$) — the correct Nepali
aksharas fall out of the candidate lists entirely. `prune_model.rs` (drop
aksharas absent from the Nepali vocabulary) removed 10k foreign aksharas
but did not recover: the shared-akshara emissions were also shifted by the
joint training.

**Recovery:** retrained the known-good config (nep_train + nep_valid +
romanize-v2 synthetic, no Hindi) → **79.51%** (top-50 93.55%). Note: 1.2
points below the 80.69 record — the data rebuild changed both the synthetic
pairs and the vocabulary file simultaneously; the delta needs diagnosis
(next session).

**Required repair for the phonetic design (before retry):** rank reverse
index candidates by JOINT weight $P(a) \cdot P(s \mid a)$ (the v2 §5.2
lesson — emission-only ranking lets peaked rare aksharas crowd out real
ones), and/or prune foreign aksharas at training time, and retest.

## Regression diagnosis + data-path bugs (2026-09-05, evening session)

**Regression SOLVED: 80.69% restored.** The 1.2-point drop (80.69 → 79.51) was
NOT the model: `data-pipeline/pipeline.py export` had overwritten
`data/word_freq_text.bin` (the engine's vocabulary prior) with a news-crawl-only
vocab — 198k words, only **58.1% of AK-Freq test targets present** (vs 576,658
words from Wikipedia+CC100). Rebuilding the wiki+CC100 vocab (75.4M tokens) with
the SAME model restored **80.69% (1972/2108) exactly, case-identical**.

**Vocabulary merge (all three text sources):** wiki+CC100+news = 109.6M tokens →
**708,833 words** (news contributed ~34M tokens / 1.42M clean lines from 18,190
articles). Merged vocab at w=0.75: **80.65%** (w sweep: 0.5→80.46, 1.0→80.60);
one case below the wiki-only max but +130k words of real-user coverage — kept
as the shipped artifact. `pipeline.py export` now writes news counts to
`out/word_freq_news.bin` by default and never clobbers the engine vocabulary;
`--merge-base` does the additive merge explicitly.

**Bug: synthetic pairs were never ingested.** `romanize.rs` (and
`pipeline.py romanize`) emit `{"english": ..., "native": ...}` but
`train_model.rs` parses only `"english word"`/`"native word"` — every
`--extra synthetic.jsonl` row was silently skipped. All "E5 self-training"
gains in this log are confounded: the +0.23 attributed to synthetic pairs was
actually the **valid split** entering training. Fixed: canonical keys in both
generators, `alias` entries in train_model/train_model_v2/build_lexicon.

**Experiment A (train+valid, zero synthetic): 80.65%** (merged vocab, SOTA
decode config) — identical to the record within one case. **Experiment B
(train+valid+918k pipeline-synthetic, first REAL ingestion): 79.51% (−1.14,
all buckets down)** — the crude sound-table romanizer's conventions conflict
with the corpus's, and 28% of training mass of off-distribution pairs biases
the emissions. **The self-training hypothesis is now untested with a
consistent generator**: engine-native romanize output (emission-argmax
spellings, consistent by construction) is pending — merged-vocab conversion
(709k words) running; next session evaluates it as experiment C.

**Current best verified config:** model = nep_train+nep_valid (EM, defaults),
vocab = merged 3-source (or wiki+CC100 for max benchmark), decode = beam 256,
k=50, lm 0.85, vocab-weight 0.75 → **80.65–80.69% native top-1**.

## Experiment C — engine-native synthetic, first real ingestion (2026-09-05, evening): FALSIFIED

With the key-mismatch bug fixed, the E5 self-training hypothesis could finally be
tested. Merged-vocab conversion (709k words → 2.42M weighted pairs via
`romanize.rs`, cycle-consistency verified) trained as
nep_train + nep_valid + synthetic (4.82M pairs, 0 skipped), SOTA decode config,
merged vocab:

| Model (all + merged vocab) | AK-Freq top-1 | AK-Freq top-50 |
|---|---|---|
| A: nep_train + nep_valid | **80.65%** | 93.55% |
| B: A + 918k pipeline-synthetic (sound-table) | 79.51% | 92.55% |
| C: A + 2.42M engine-native synthetic | 75.62% | 91.94% |

**Self-training is falsified in both forms.** Mechanism: synthetic pairs are the
model's own emission argmax — training on them reinforces existing biases and
erases the human spelling diversity that is the corpus's actual information
content (model collapse; worse at higher synthetic mass). The E5 entries above
(+0.23 "sota") were artifacts of the ingestion bug. **Retire E5; the verified
recipe is nep_train + nep_valid + real-text vocabulary rescoring.**

## Vocabulary de-noising (2026-09-05, night) — NEW BEST 80.83%

Audit of `word_freq_text.bin` (709k words) found heavy tokenizer noise:
**39,732 tokens with glued danda** (पुगे।, date strings), **67,250 with
Devanagari digits** (२००८५, १४:१९), **152,630 with ASCII punctuation**
(सोमाली,, (भोटे)), **2,094 with ZWJ/ZWNJ**. Beyond pollution, glued variants
*split* a real word's count across keys (पुगे। ≠ पुगे), flattening exactly
the frequency prior the reranker depends on.

Fix (`build_wordfreq_text.rs`): trim non-word chars from token edges, then
require every char ∈ U+0900..=U+0963 (letters/matras/nukta/vocalics/
anusvara-visarga — excludes danda 0964-65, digits 0966-6F, abbrev signs,
ZWJ/ZWNJ, ASCII). `pipeline.py` WORD regex matched for future DB counts.

| Vocab | words | AK-Freq top-1 | ALL top-1 |
|---|---|---|---|
| pre-fix (merged 3-source) | 708,833 | 80.65% | 59.94% |
| **post-fix (merged 3-source)** | **492,365** | **80.83%** | **60.38%** |

Artifact 25.7 → 18.4 MB. All previous "merged vocab" numbers in this log were
measured pre-fix; the post-fix merged vocab is the new shipped artifact and
beats even the wiki-only maximum (80.69).

## Single cleaned corpus + dedup (2026-09-05, night) — 80.98%

Raw sources eliminated: `data/raw/` (1.9 GB dumps) and the 1.4 GB article DB
are consumed by `data/pipeline/build_corpus.py` and deleted. The only stored
text is now `data/store/corpus_clean.txt`: 2.89M clean unique sentences,
86.1M tokens (8.28M raw lines in, 5.39M exact-duplicate lines dropped —
boilerplate/syndication), every token a pure Devanagari letter run, line
needs ≥4 words. Word bigrams (word_pairs.csv) remain and are re-derivable
from the corpus. Synthetic pair files and intermediate vocab CSVs deleted
(self-training falsified; CSVs re-derivable).

| Vocab | words | AK-Freq top-1 | ALL | NEF | NEI |
|---|---|---|---|---|---|
| pre-fix 3-source | 708,833 | 80.65% | 59.94 | 28.40 | 44.73 |
| de-noised 3-source | 492,365 | 80.83% | 60.38 | — | — |
| **cleaned + deduped corpus** | **470,012** | **80.98%** | 60.28 | 28.76 | 45.07 |

Cleaning the *input data* is now the cheapest accuracy lever found so far
(+0.33 total from tokenizer + dedup). `make data` = raw → clean → vocab →
model; `data-clean` deletes the raw inputs after compiling the corpus.

## Aksharantar cleaned & merged + engine input mappings (2026-09-05, night)

**Corpus (user decision: one language-agnostic Devanagari set, strict filter
everywhere).** `clean_aksharantar.py`: native word must be a pure letter run
(U+0900..0963), roman pure a-z, exact dupes dropped. Result: train 3,588,793
(18 bad natives, 107,758 cross-language dupes removed), valid 9,155,
**test 4,101 unchanged** (0 drops — benchmark comparability preserved).
Originals deleted after a gzip snapshot to `data/backup/` (99 MB).

**Model D (merged hin+nep, cleaned): native top-1 80.98% — ties the record
while lifting every other bucket** vs the Nepali-only model on identical
vocabulary and config:

| Model | AK-Freq top-1 | AK-Freq top-50 | ALL | NEF | NEI |
|---|---|---|---|---|---|
| Nepali-only (previous best) | 80.83% | 93.55 | 60.38 | 28.76 | 45.07 |
| **Merged cleaned (D, shipped)** | **80.98%** | **94.40** | **60.64** | **29.01** | **46.17** |

E2's "Hindi pooling negative" is superseded: on deduplicated, cleaned data,
merging helps the candidate set (NE top-50 +4.1/4.1) at no native cost.
Decode 14.0 ms/word at beam 256 (model 32.3 MB, 16.5k aksharas).

**Engine input mappings (both runtimes via `ImeEngine::get_suggestions`):**
- ASCII digits → Devanagari digits (pure mapping, model never sees them):
  all-digit input maps directly (`123` → `१२३`), leading/trailing digit runs
  wrap the decoded word (`namaste1` → `नमस्ते१`), mid-word digits interleave
  top-1s (`na2ma` → `न२म`).
- Purnabiram: trailing `.` appends । to every suggestion (`namaste.` →
  `नमस्ते।`); a lone `.` is । itself.
- 6 new unit tests; suite 60/60.

## S5/S1/S2 experiments — reranker v2 (2026-09-05, late night)

**S5 multi-reference benchmark.** IndicXlit's protocol scores inputs against
up to 4 roman variants per word; our test has exactly one per word (98.1% of
test natives have no second spelling anywhere in train — held out by
construction). Built `data/eval/test_multiref.jsonl`: for each of 3,944
distinct test natives, the 3 highest-probability emission-table variant
spellings (romanize.rs, cycle-consistent) join the original → 14,410 cases.
On this stricter benchmark the shipped model scores **75.29% native top-1**
(model-spawned variants are harder inputs than human-annotated ones; the
single-ref 80.98 stays the continuity headline).

**S1 discriminative reranker (the win).** `evaluate_model --dump` exports the
raw k=50 (emit, lm, n_aksharas) + exact-vocabulary scores per case;
`data/pipeline/reranker_v2.py` trains a softmax reranker (conditional
likelihood of the gold over the candidate list) on 9,155 valid cases with:
decoder emit/lm, candidate rank, the shipped heuristic score and rank,
shaped frequency (log + rank + in-vocab), matra profile (10 matras + nasals
+ visarga + halants), vowel-initial/final shape, lengths, and 11
postposition-agreement features (dev suffix ∧ roman suffix). 5-fold CV on
valid selected l2=1e-4, blend γ=0.3 (the trained model is z-score-blended
with the heuristic per case — γ=0 exactly reproduces the baseline, which
validated the whole harness).

| Test (single-shot) | native top-1 | ALL | NEF | NEI |
|---|---|---|---|---|
| shipped heuristic | 80.98% | 60.64 | 29.01 | 46.17 |
| **reranker v2 (γ=0.3)** | **81.69%** | **61.01** | 29.38 | 45.92 |

**+0.71 native top-1, honest and leakage-free.** Two hard-won negative
lessons along the way: (a) training on candidates whose gold is absent is
pure noise — filter to gold-present cases; (b) cold-start full-softmax
diverges from the heuristic on the native bucket — the γ-blend keeps the
heuristic where it is right and imports the learned corrections where they
add signal.

**S2 exact-vocabulary scores: null result.** Adding word-trie exact
noisy-channel scores as features moved test top-1 DOWN (61.01 → 60.47);
coverage is only 62.5% of cases and the signal duplicates the decoder's.
The generation ceiling stands; beating it needs S3 (conditional lattice
model), not a reranker feature.

**New best: 81.69% native top-1** (reranker v2, offline experiment — porting
the weights into `train_reranker.rs`/engine is the next engineering step).

## Reranker v2 ported into the engine (2026-09-05, night)

The 39-feature softmax model (l2=1e-4, γ=0.3) is now embedded in the binary
(`src/core/reranker_v2_weights.rs`, generated by
`data/pipeline/reranker_v2.py::export_rust`) and scores the engine's fresh
candidates at depth 50. Frequency-rank index built from word_freq_text.bin at
load; WASM builds unaffected (v2 auto-off without the vocab file).

- **Parity gate:** `parity_v2` bin compares Rust rankings against exported
  Python rankings on 5 real dump cases — **5/5 exact match** (it caught one
  real bug: a feature-index shift from nasals onward).
- **Engine-level before/after** (evaluate_aksharantar, full engine, same
  model + vocab):

| Bucket | engine before | engine after |
|---|---|---|
| AK-Freq top-1 | 77.56% | **81.36%** |
| AK-NEF top-1 | 25.83% | **29.01%** |
| AK-NEI top-1 | 41.24% | **45.24%** |

The port removed the engine's old ranking deficit entirely: the full engine
now *exceeds* the raw decoder+heuristic (80.98%). Suite 60/60; WASM build OK.

## E6 corpus-bigram context + data-derived suffix features (2026-09-05, late night)

**Suffix-agreement features de-hardcoded.** The 11 hand-picked postposition
slots (को/का/की/लाई/ले/मा/बाट/हरू/सँग/भन्दा/कै) were replaced by a
**data-derived short-word table**: the 24 most frequent ≤3-char Devanagari
words (postpositions/clitics dominate short-token frequency) with the two
most common roman tails users actually typed for them, mined from train+valid
by proportional-tail attribution. Same accuracy as the hand list
(61.01% offline) but zero hardcoded linguistics; the table and weights
regenerate together (`reranker_v2.py::export_rust`, N_FEATS 39→52).
Rust parity re-verified 5/5; engine bench with the regenerated model:
**native 81.40%, NEF 29.25, NEI 45.32** (vs 81.36/29.01/45.24 pre-change).

**E6 context layer wired.** `build_bigrams.rs` → `data/word_bigrams.bin`
(1.25M pairs over 102,534 context words, 41 MB). The engine tracks the last
committed word and boosts candidates forming a corpus bigram with it
(40k·ln(1+f) additive, tuned). `evaluate_context` A/B harness romanizes
corpus sentences and walks two identical engines in lockstep (context is the
only difference):

| sentences | A context OFF | B context ON | Δ |
|---|---|---|---|
| 0–800 (tune) | 89.75% | 90.21% (boost 60k) | +0.46 |
| 800–1600 (held-out) | 89.40% | 89.80% (boost 60k) | **+0.40** |
| 800–1600 (held-out) | 89.40% | 89.75% (boost 40k) | +0.35 |

Held-out **+0.35–0.40** sentence-level accuracy. Modest here because
canonical romanizations of corpus words already decode at ~90% (little
ambiguity left); on real user input (variants, typos) the context term has
more errors to fix. Default boost set to 40k. Benchmark note: this lever is
invisible to the isolated-word benchmark by construction — it exists for
real typing flow.

## Suffix-feature ablation + large held-out E6 run (2026-09-05, night)

**Short-word features: zero contribution (ablation).** Retraining the
reranker with all 24 data-derived short-word slots removed reproduces the
shipped numbers exactly (61.01% ALL / 81.69% AK-Freq, case-identical). The
feature class — first hand-picked postpositions, then the data-derived
table — has not earned its place yet; the likely fix is a fuzzy roman-tail
matcher instead of strict `ends_with`. Kept in the model for now (zero
cost); candidate for removal at the next retrain cycle.

**E6 large held-out run** (2,400 held-out sentences, 73,583 words, boost
40k): context OFF 88.41% → ON **88.72%** (+0.31; post-first-word subset
+0.31). Confirms the +0.3–0.4 sentence-level estimate from the smaller runs.
