# Nepali Data Root — Provenance, Preparation & Structure

This directory holds the two inputs the engine is built from: a monolingual
Nepali running-text corpus, and the cleaned Aksharantar Roman-to-Nepali
transliteration pair splits. Nothing else is required — `make data-prepare`,
`make lexicon`, `make train`, `make model` run entirely off these two.

All of `data/` is gitignored except this file. The corpus and model are not
committed; models ship via `make release-upload TAG=vX.Y.Z`.

---

## 1. Unified Corpus (`data/nepali_corpus.txt`)

- **File Path:** `data/nepali_corpus.txt`
- **Total Lines:** 6,426,307 lines
- **File Size:** 1.80 GB (1,836.56 MB)
- **Character Encoding:** UTF-8, Unicode NFC normalized
- **Language / Script:** Nepali / Devanagari exclusively
- **Quality Assurance:** **0 Roman/Latin characters** across all 6.42M lines
  (verified via `[a-zA-Z]` regex pass). All English URLs, Latin loanwords,
  English headings, and mixed-script fragments were stripped.

### Provenance

The corpus is a de-duplicated merge of Nepali Wikipedia (CC-BY-SA), CommonCrawl
CC100 Nepali (CC0), a Nepali news crawl from *Gorkhapatra*, *OnlineKhabar*,
*Naya Patrika* and *Kanun Patrika*, AI4Bharat IndicCorp v2 Nepali (CC0), the
Aksharantar Nepali native-word lists, and an expanded morphological word list
(`wnepali-expanded.txt`, from `nepali-wordle/data/raw/`, CC0). The news sources
supply contemporary political, administrative, legal and geographic vocabulary
that encyclopedic text lacks.

### Preparation Pipeline

Applied in this order over every source:

1. **Extraction:** Streamed each source (news articles, the 12 pinned IndicCorp
   v2 Nepali slices, the 3 Aksharantar JSON splits, the morphological word list).
2. **Script purification:** Applied `[a-zA-Z0-9_\-\./@:#]+` regex excision of
   Latin tokens, emails, numbers and URLs, preserving Devanagari letters
   (`U+0900..=U+097F`) and Nepali punctuation (`।`, `,`, `?`).
3. **Whitespace & NFC normalization:** Collapsed internal whitespace;
   NFC-normalized all combining marks, halants and matras.
4. **Validation:** Discarded empty lines, lines with no Devanagari characters,
   and any line retaining a Latin character.
5. **De-duplication:** Identical syndicated news paragraphs collapsed to one.

Consequence for vocabulary size: this corpus yields 3,774,787 distinct
Devanagari tokens; at the `build_lexicon` floor (`--min-count 2`, 1..=24 chars)
3,741,003 survive, and `--max-words` selects the most frequent of those.

---

## 2. Aksharantar Roman-to-Nepali Pairs (`data/aksharantar/`)

- **Upstream URL:** `https://huggingface.co/datasets/ai4bharat/Aksharantar/resolve/e418c1fc928d9f5393af33268472cf20c1891be8/nep.zip`
- **Revision:** `e418c1fc928d9f5393af33268472cf20c1891be8`
- **Archive Size:** 70,147,764 bytes (66.9 MB)
- **SHA-256:** `8bdcc5957d701ea36b63c00c713b43bc30902774b70736737d480e77929de92c`
- **License:** CC-BY 4.0 / CC0 (AI4Bharat, IIT Madras)

### Split Breakdown

| Split File | Pairs | Size | Purpose |
| :--- | ---: | ---: | :--- |
| `nep_train.json` | 2,397,414 | 183.5 MB | Transliteration channel / phonetic model training |
| `nep_valid.json` | 2,804 | 214 KB | Hyperparameter tuning and development set |
| `nep_test.json` | 4,101 | 286 KB | Held-out benchmark evaluation (the **AK-Freq** split) |

Every record is a minimal JSON object with the two fields the engine uses:

```json
{"english word": "pariskrit", "native word": "परिस्कृत"}
```

### Fields Removed From Upstream

`unique_identifier`, `source` and `score` were stripped, halving file size
while preserving 100% of the pairs.

- `unique_identifier` — an arbitrary row id (`nep12345`); no algorithmic role.
- `source` — the provenance stratum: `AK-Freq` (human-curated frequent words),
  `AK-Uni` (human-validated crowdsourced), `IndicCorp` (mined by
  back-transliteration), `Wikidata` (mined entities).
- `score` — a *negative* normalized sequence log-likelihood under AI4Bharat's
  neural mining model; meaningful only for that model, not for a
  rule/lexicon-based engine.

**Consequence, and it is load-bearing.** The whole `nep_test.json` split is
`AK-Freq`, so `prepare_pairs` is invoked with `--assume-source AK-Freq` to
restore the `source` field that `eval_langs` uses to assign a stratum. The
tool reports how many rows it filled in rather than assuming silently:

```
source field absent on 2404319 rows (train 2397414, valid+test 6905); assumed "AK-Freq".
```

Only the Nepali split is vendored. Upstream publishes eight Devanagari
languages, but the engine ships Nepali-only priors, so no other split is
fetched or trained on.

---

## 3. Derived Artifacts (all gitignored, rebuilt by `make`)

| Path | Built by | Notes |
| :--- | :--- | :--- |
| `data/pairs/{train,valid,test}.jsonl` | `make data-prepare` | 2,397,403 / 2,804 / 4,101 rows |
| `data/lexicon.bin` | `make lexicon` | FST automaton, one byte per Devanagari char |
| `data/akshar.model` | `make train && make model` | The shipped container |

`prepare_pairs` normalises every split so training and scoring share one
convention: roman is trimmed and ASCII-lowercased, native is trimmed and
NFC-normalised (Aksharantar stores eyelash-ra as `र` + nukta in Marathi and
Konkani rows, which NFC folds to the single code point `ऱ`). Train rows are
additionally cleaned and de-duplicated — roman must be `a-z` only, native must
be Devanagari letters and signs only (`U+0900..=U+0963`), exact duplicate pairs
are kept once, and **any pair whose native word appears in valid or test is
dropped**, so no evaluation word is ever trained on. Train is shuffled with a
fixed seed, because a "first N rows" sample of a language-sorted upstream file
is a sample of one language.

The lexicon automaton's keys are one byte per Devanagari character and are
exactly decodable back to the surface form, so it doubles as the word store —
no separate surface-form array is kept.

---

## 4. Directory Layout

```
data/
├── README.md               # this file
├── nepali_corpus.txt       # 1.80 GB clean Nepali Devanagari corpus (6,426,307 lines)
└── aksharantar/            # Roman-to-Nepali transliteration pairs
    ├── nep_train.json      # 2,397,414 pairs (183.5 MB)
    ├── nep_valid.json      # 2,804 pairs (214 KB)
    └── nep_test.json       # 4,101 AK-Freq pairs (286 KB)
```
