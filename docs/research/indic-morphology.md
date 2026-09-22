# Indic Morphology for a Script-General IME

Research question: what existing morphological analyzers, stemmers, or suffix
inventories cover Devanagari-script languages (Hindi, Nepali, Marathi,
Sanskrit), and how can a script-general engine use them without per-language
forks — specifically to replace the 34-item hand-listed Nepali suffix strip
in `src/core/reranker.rs` (`MORPH_SUFFIXES`, used by `get_morph_suffix`,
dense feature 28 `morph_effective_log_freq`, and sparse template 5
`suffix × roman_tail`)?

Date: 2026-09-22. No code or data files touched; this is a docs-only report.

## Why the current list is shaped the way it is

The 34 entries are almost all Nepali postpositions / clitics / verb endings
(को का की मा ले लाई बाट देखि सँग सित हरू हरु जी एको एका एकी दै दा एर नु ने
छन् थिन् थियो थिए ता त्व पन पना पनि नै त भने भनी). They matter for an IME
because roman input often arrives without word spaces, so case markers fuse
onto the stem (`ketabooko` → केटाबो+को is wrong, किताब+को is right) and the
reranker must credit a candidate whose stem is in-vocabulary even when the
full fused form is not. The existing `stem_f >= 5` vocab gate is the load-
bearing safety property: any suffix strip whose stem is not in the vocabulary
contributes nothing, so **expanding the suffix union is safe by construction**
— false strips decay to zero credit. This is what makes a script-general
union list viable where a precision stemmer would be required elsewhere.

## Candidate resources (ranked by integration fit)

### 1. Snowball Hindi stemmer (Ramanathan & Rao 2003, Devanagari port) — BEST FIT

- Coverage: Hindi inflectional suffixes for nouns/verbs/adjectives.
  Longest-match single-strip over ~60 base suffixes, each stored in both
  independent-vowel and dependent-matra form (~120 entries) plus virama.
  Covers plural/oblique markers (ों ियों), verb endings (ाएगा ेंगे ोगे),
  derivational bits (ाहट कर? actually ाकर etc.). Noun-heavy masculine
  paradigm focus (`a A i…` deletions for masculine nouns, §3.1 a.v).
- License: **BSD** (Snowball site + `snowballstem/snowball` repo). Clean for
  embedding, including in a commercial/govt IME, with attribution.
- Machine-readable: yes — `algorithms/hindi.sbl` is a complete executable
  spec in Devanagari (U+0900 block), plus `snowball-data` Hindi
  vocab/stemmed pairs (65k words) for regression testing a port.
- Integration fit: **excellent**. It already answers "stem of X with suffix
  Y" in microseconds — it *is* a longest-match suffix strip, exactly the
  shape of `get_morph_suffix`. Porting the `.sbl` `among` list to a Rust
  `&[&str]` sorted longest-first is a mechanical transliteration (the Snowball
  page documents the Devanagari mapping; dependent vs independent vowel
  doubling can be collapsed since Rust does plain `ends_with`).
- Caveat: Hindi-only; several Hindi endings collide with Nepali verb forms.
  The vocab gate absorbs this. Hindi postpositions (ने को से में पर का की के)
  overlap Nepali ones usefully.

### 2. UniMorph Hindi (`unimorph/hin`) + Sanskrit (`unimorph/san`) — BEST BUILD-TIME MINING SOURCE

- Coverage: Hindi 54,438 forms / 258 paradigms, **verbs only**, sourced from
  Wikipedia; Sanskrit `san` + giant `san2` parsed from the Heritage inflector.
  Nouns/adjectives for Hindi are absent. **No Marathi (`unimorph/mar` 404) and
  no Nepali (`unimorph/nep` 404)** — verified 2026-09-22. Related Indo-Aryan
  coverage exists for Gujarati, Bengali, Braj, Assamese.
- License: Hindi **CC BY-SA 3.0**; Sanskrit data **LGPL** (inherited from the
  Heritage inflector). Both require attribution; CC BY-SA share-alike applies
  to adapted datasets. A short list of functional-morpheme *facts* mined from
  the data is low-risk, but attribute the source in the table header comment
  and keep the mining script + raw counts out of the shipped binary.
- Machine-readable: yes — `lemma \t form \t features` TSV per language,
  `pip install unimorph` available.
- Integration fit: **good, build-time only**. Do not ship paradigm lookup in
  the IME. Instead, at model-build time, align each (lemma, form) pair,
  extract the differing tail as a candidate suffix, count support across
  paradigms, keep tails with support ≥ N, and emit a static Rust table. This
  yields Hindi verb endings the Snowball list undercovers, with zero runtime
  cost and zero license contamination of the hot path (only short strings +
  attribution comment ship).
- Caveat: Wikipedia-sourced Hindi verbs are noisy; require minimum paradigm
  support and the existing stem-in-vocab runtime gate.

### 3. Nepali stemmer literature — DIRECT RULE SOURCE (papers, no public code)

Three generations, all rule/affix-stripping, all consistent with the current
architecture:

- Bal & Shrestha / Balaram Prasain lineage ("A Morphological Analyzer and a
  Stemmer for Nepali", 2014; Konstanz dissertation 2011 "A Computational
  Analysis of Nepali Morphology"): the linguistic reference — Nepali noun/
  adjective/adverb inflection and derivation inventory. No maintained code.
- Paul et al. 2014 ("An Affix Removal Stemmer for Natural Language Text in
  Nepali"): lexical-lookup + rule hybrid, affix stripping; reported ~68%
  class accuracy. Paper lists suffix classes; no repo.
- Shrestha & Dhakal 2016 ("A new stemmer for Nepali"): **128 suffix rules**,
  stepwise iterative application. The natural superset of the current 34.
- Koirala & Shakya 2020 ("A Nepali Rule Based Stemmer…", arXiv:2002.09901):
  **85 Type-I + 161 Type-II suffixes**, Paice-style evaluation, 88.78%
  accuracy on 5k words, 5.27% under-stemming error. The largest published
  Nepali inventory.
- License: papers, no code → reimplementing published rule lists as data is
  standard practice; cite the papers in comments.
- Machine-readable: no. Suffixes must be transcribed from PDFs by hand
  (one-time cost, reviewable diff).
- Integration fit: **good**. These lists drop straight into the `ends_with`
  scan. Prefer Koirala Type-I (inflectional) first, then Shrestha's 128, then
  Type-II (derivational) only if eval moves — derivational strips are where
  false-positive risk lives, though the vocab gate covers it.

### 4. Marathi suffix work — NEEDED FOR THE THIRD LANGUAGE

- Majgaonker & Siddiqui ("Discovering suffixes: A Case Study for Marathi"):
  rule-based + unsupervised learned suffixes from raw Marathi text. Key
  warning for us: **Marathi has productive single-letter suffixes**
  (ा ी े ां etc.), which a naive longest-match strip over-stems — the paper
  reports high stemming error exactly there. For IME rerank purposes the
  vocab gate again saves us, but single-char entries should go last in match
  order (longest-first already does this) and single-char strips should
  arguably require a higher `stem_f` threshold.
- Pimpale et al. 2014: Marathi stemmer built by modifying Ramanathan & Rao's
  Hindi stemmer — evidence the Snowball-Hindi list adapts to Marathi with
  small deltas, supporting the union approach.
- Dabre et al. 2012 ("Morphological Analyzer for Marathi", FSM-based, COLING):
  full analyzer for a highly inflectional language with affix stacking.
  Paper + FST, no maintained runtime; too heavy for the hot path but a good
  reference for stacked-suffix handling (strip iteratively, max 2 rounds —
  cf. Shrestha's iterative application).
- License: papers, no redistributable code found. Reimplement lists with
  citation.
- Integration fit: **medium**. Transcribe Pimpale/Majgaonker suffix sets;
  handle stacking with bounded iterative stripping.

### 5. IIIT-Hyderabad LTRC analyzers (Hindi/Marathi) — POWERFUL, WRONG SHAPE

- Coverage: Hindi (~88% on arbitrary modern text per LTRC page), Marathi,
  plus Kannada/Punjabi/Telugu. Full analysis: root + gender/number/tense.
- License: **GPL** (per `ltrc.iiit.ac.in/morph`). Copyleft — embedding in the
  crate would GPL the engine. Check with the project before even build-time
  use; safest to treat as reference only.
- Machine-readable: tarballs of Perl + GDBM + Flex (`hin_morph.tgz`,
  `mar_morph.tgz`). 2000s stack, unmaintained-looking.
- Integration fit: **poor for the hot path** — a Perl/GDBM pipeline cannot
  answer in microseconds inside `rerank()`. At best, run it offline over a
  corpus to validate/extend the mined suffix table. Ranked here for
  completeness, not recommended.

### 6. Sanskrit: Heritage Engine + Vidyut — DO NOT SUFFIX-STRIP SANSKRIT

- Sanskrit Heritage Engine (G. Huet, Inria): rule-based analysis +
  segmentation + tagging, ~25k-word lexicon. License **LGPL** (freeware
  release 2012). Online/demo plus downloadable engine — but it is a full
  grammar-backed analyzer, milliseconds-to-seconds per sentence, not a
  microsecond stemmer.
- Vidyut (`ambuda-org/vidyut`): **Rust + Python, MIT-licensed**, Paninian
  word *generator* (`vidyut-prakriya`) with MIT-shared dhātupāṭha. The only
  resource here that is both Rust-native and permissively licensed — but it
  generates rather than analyzes, and its data tables are megabytes.
- Recommendation: **exclude Sanskrit from suffix stripping entirely**.
  Sandhi makes tail-strip morphologically unsound (stems mutate at
  boundaries: e.g. tat + ca → tacc), so strips would mostly fail the vocab
  gate anyway and add only hash-table noise to sparse template 5. Sanskrit
  candidates are already served by vocab + LM + transliteration scores.
  Revisit only via a Vidyut-derived sandhi-aware splitter, offline, if
  Sanskrit eval shows a morphology-shaped error cluster.

### 7. IndoWordNet / pyiwn (CFILT, IIT Bombay) — LEXICON, NOT MORPHOLOGY

- Coverage: wordnets for 18 scheduled languages incl. Hindi, Marathi,
  Nepali, Sanskrit, linked through Hindi. Research download (LDC / CFILT
  site, license terms on download; `pyiwn` API packages data with source).
- Integration fit: **orthogonal**. A wordnet cannot answer "stem of X".
  Its value is as a stem-validity signal (is this strip target a real lemma?)
  — but the engine already has that via the trained vocab frequencies, which
  are domain-matched where WordNet is not. Do not add as a dependency; at
  most use offline to audit strips the vocab gate rejects.

### 8. TDIL (Govt. of India) resources — REFERENCE ONLY

- `tdil-dc.in` lists morphological analyzers and the IIT-B Hindi analyzer
  among funded tools. Downloads exist but licensing is government/restrictive
  and redistribution in an open-source IME is unclear. Treat as a pointer to
  papers, not as shippable data.

### 9. Neural analyzers (2023–2026) — OFFLINE MINING ONLY

- Trend: BiLSTM/Transformer morphological analyzers for Dogri (2025),
  Gujarati (2023), Telugu, and multi-task Hindi/Urdu parsers; Stanza ships a
  Hindi UD pipeline with a neural lemmatizer. Accuracy is high but inference
  is milliseconds-per-word with MB–GB weights — three orders of magnitude
  over budget for `rerank()` over ≤24 candidates, and impossible in the wasm
  build. Use class: run Stanza-Hindi offline over the training corpus to
  propose (form → lemma) pairs, then distill tails into the static table
  exactly as with UniMorph (§2). Never a runtime dependency.

## License compatibility summary

| Resource | License | Ship in binary? |
|---|---|---|
| Snowball Hindi `.sbl` suffix list | BSD | Yes, with attribution |
| UniMorph Hindi paradigms | CC BY-SA 3.0 | Mined tails + attribution comment; keep raw data out |
| UniMorph Sanskrit (Heritage-derived) | LGPL | Same as above; Sanskrit excluded from strip anyway |
| Nepali/Marathi paper rule lists | Papers (no code) | Yes, reimplemented with citations |
| LTRC Hindi/Marathi analyzer | GPL | No (reference/offline validation at most) |
| Heritage Engine | LGPL | No (too heavy regardless) |
| Vidyut | MIT | Yes in principle; generator-side, not needed now |
| IndoWordNet data | Research/restricted | No |
| TDIL downloads | Govt/restrictive | No |

## Proposal: replace the 34-item list with a mined script-general table

1. **Union, don't fork.** Build one `MORPH_SUFFIXES`-shaped table, longest-
   match first, from: (a) Snowball Hindi `.sbl` Devanagari suffixes
   (mechanical port, ~120 entries after dependent/independent collapse);
   (b) Koirala Type-I (85) + Shrestha 128 Nepali rules (dedupe against the
   current 34 — expect the 34 to be a near-subset); (c) Pimpale/Majgaonker
   Marathi sets; (d) tails mined at build time from UniMorph Hindi verbs
   (support ≥ k paradigms) and optionally Stanza-lemmatized corpus pairs.
   No language ID at runtime — the engine never knows the language, and the
   vocab gate (`stem_f >= 5`) makes cross-language collisions harmless.
2. **Keep the runtime shape.** The `ends_with` linear scan over ~300–400
   static `&str`s is still single-digit microseconds; no FST, no heap, no
   new wasm dependency. Sparse template 5 (`suffix × roman_tail`) is
   unchanged — it just sees more suffix ids, which retraining absorbs.
3. **Bound the risk points.** Single-character Marathi suffixes go last
   (longest-first ordering already guarantees this) and consider a higher
   `stem_f` floor for 1-char strips; cap iterative stripping at 2 rounds for
   Marathi-style stacking; keep Sanskrit out of the table.
4. **Validate with the existing harness, not unit tests alone.** (A
   misordered candidate-source band once caused a 30-point top-1 regression
   through a green unit suite — same hazard class.) Gate on
   `make eval-ime` / `eval-errors` top-1/top-5 deltas per language split plus
   the accuracy-regression guard, with `AKSHAR_NO_RERANK` / `AKSHAR_NO_SPARSE`
   ablations to attribute any move to the morph features specifically.
5. **Provenance in-tree.** One comment block above the table per source
   (paper citation or URL + license), so a future auditor can tell BSD
   entries from CC BY-SA-mined entries from transcribed paper rules.

Expected end state: ~300-entry static table, same two call sites, no new
dependencies, no per-language branches, each language's eval split deciding
which source earned its place.
