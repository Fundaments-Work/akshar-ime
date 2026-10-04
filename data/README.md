# Nepali Data Root — Provenance, Preparation & Structure

This directory contains the unified monolingual Nepali text corpus and the cleaned Aksharantar Roman-to-Nepali transliteration pair splits.
All external/temporary download scripts and multi-language datasets have been removed.

---

## 1. Unified Corpus Summary (`data/nepali_corpus.txt`)

- **File Path:** `data/nepali_corpus.txt`
- **Total Lines:** 6,426,307 lines
- **File Size:** 1.80 GB (1,836.56 MB)
- **Character Encoding:** UTF-8, Unicode NFC normalized
- **Language / Script:** Nepali / Devanagari exclusively
- **Quality Assurance:** **0 Roman/Latin characters** across all 6.42M lines (verified via `[a-zA-Z]` regex pass). All English URLs, Latin loanwords, English headings, and mixed script fragments were stripped.

---

## 2. Provenance of the Merged Sources

`data/nepali_corpus.txt` is synthesized from four primary Nepali data sources:

### A. Base Clean Corpus (`corpus_clean.txt`)
- **Contribution:** 2,887,984 lines (86.1M tokens) of clean, natural running text.
- **Original Sources:**
  1. **Nepali Wikipedia (`dumps.wikimedia.org/newiki`):** Full article dump, encyclopedic sentences, articles, and biographies (CC-BY-SA).
  2. **CommonCrawl CC100 Nepali (`data.statmt.org/cc-100`):** High-volume Nepali web text filtered to Devanagari (CC0).
  3. **Nepali News Crawl:** 18,190+ full articles collected from major Nepali news publications:
     - *Gorkhapatra* (`gorkhapatraonline.com`)
     - *OnlineKhabar* (`onlinekhabar.com`)
     - *Naya Patrika* (`nayapatrikadaily.com`)
     - *Kanun Patrika* (`nkp.gov.np`)
     Provides contemporary vocabulary, political, administrative, legal, and geographic entities absent from encyclopedic sources.
- **Original Cleaning Rules:**
  - Punctuation separation (danda/purnabiram `।`, commas, question marks).
  - Removal of Latin tokens, URLs, emails, and ASCII symbols.
  - De-duplication of identical syndicated news paragraphs.

### B. IndicCorp v2 Nepali (`ne.00.txt` – `ne.11.txt`)
- **Contribution:** 538,235 lines (~300 MB) of natural Nepali sentences.
- **Source:** AI4Bharat IndicCorp v2 (CC0 license).
- **Processing:**
  - Extracted from the 12 pinned slices of IndicCorp v2 Nepali.
  - Filtered to strip all English loan fragments and foreign Latin words.
  - Normalized with Unicode NFC.

### C. Aksharantar Nepali Native Words (`nep_train.json`, `nep_valid.json`, `nep_test.json`)
- **Contribution:** 2,398,955 unique Devanagari words.
- **Source:** AI4Bharat Aksharantar dataset (CC0 / CC-BY 4.0).
- **Processing:**
  - Extracted strictly the native Devanagari word field (`"native word"`), discarding all Roman transliteration keys (`"english word"`).
  - De-duplicated to preserve each unique native word once.
  - Filtered to ensure zero non-Devanagari characters.

### D. Expanded Morphological Word List (`wnepali-expanded.txt`)
- **Contribution:** 601,133 inflected words.
- **Source:** `nepali-wordle/data/raw/wnepali-expanded.txt` (morphological expansions, verbal conjugations, honorific and case-marked forms).
- **Processing:**
  - Unicode NFC normalized, 100% pure Devanagari without non-Devanagari characters.

---

## 3. Preparation Pipeline

The consolidation was performed according to the following strict pipeline:
1. **Extraction:** Read streams from `corpus_clean.txt`, the 12 IndicCorp slices, the 3 Aksharantar JSON splits, and `wnepali-expanded.txt`.
2. **Script Purification:** Applied regular expressions `[a-zA-Z0-9_\-\./@:#]+` to excise all Latin tokens, emails, numbers, and URLs while preserving Devanagari letters (`U+0900..=U+097F`) and Nepali punctuation (`।`, `,`, `?`).
3. **Whitespace & NFC Normalization:** Collapsed internal whitespace and normalized all combining marks, halants, and matras to Unicode NFC.
4. **Validation:** Empty lines, lines without Devanagari characters, and lines containing any residual Latin characters were discarded.

---

## 4. Aksharantar Roman-to-Nepali Transliteration Pairs (`data/aksharantar/`)

The Roman-to-Nepali transliteration dataset comes directly from AI4Bharat's Aksharantar benchmark.

### A. Provenance & Upstream Checksum
- **Upstream URL:** `https://huggingface.co/datasets/ai4bharat/Aksharantar/resolve/e418c1fc928d9f5393af33268472cf20c1891be8/nep.zip`
- **Revision:** `e418c1fc928d9f5393af33268472cf20c1891be8`
- **Archive Size:** 70,147,764 bytes (66.9 MB)
- **SHA-256 Checksum:** `8bdcc5957d701ea36b63c00c713b43bc30902774b70736737d480e77929de92c`
- **License:** CC-BY 4.0 / CC0 (AI4Bharat, IIT Madras)

### B. Split Breakdown (Cleaned)
Located in `data/aksharantar/`:
| Split File | Pairs Count | Cleaned Size | Purpose |
| :--- | :--- | :--- | :--- |
| `nep_train.json` | 2,397,414 | 183.5 MB | Transliteration channel / phonetic model training |
| `nep_valid.json` | 2,804 | 214 KB | Hyperparameter tuning and development set |
| `nep_test.json` | 4,101 | 286 KB | Unseen benchmark evaluation (`AK-Freq` split) |

### C. Cleaned Record Schema
All extraneous fields (`unique_identifier`, `source`, `score`) were removed, reducing file sizes by over 50% while preserving 100% of the transliteration pairs.
Each line is now a minimal JSON object:
```json
{"english word": "pariskrit", "native word": "परिस्कृत"}
```

### D. Explanation of Removed Metadata

1. **`unique_identifier`**: An arbitrary row identifier (e.g. `nep12345`) assigned by AI4Bharat's dataset export. Serves no algorithmic purpose.
2. **`source`**: The origin of the pair within Aksharantar:
   - `AK-Freq`: Human-curated frequent words.
   - `AK-Uni`: Human-validated crowdsourced pairs.
   - `IndicCorp`: Mined via back-transliteration from web corpora.
   - `Wikidata`: Mined entity transliterations.
3. **`score`**:
   - **What it is:** In mined pairs (from IndicCorp), `score` is a negative floating-point number (e.g. `-0.0028`, `-0.1852`, `-0.2460`). For human-verified pairs (`AK-Freq`, `AK-Uni`), it was `null`.
   - **How it was measured:** In AI4Bharat's Aksharantar mining pipeline, candidate pairs were extracted from web corpora using an automated sequence-to-sequence transliteration model. The `score` represents the **normalized sequence log-likelihood** $\log P(\text{native} \mid \text{roman})$ under the mining model. Values close to $0.0$ indicate high model confidence in the alignment.
   - **Why removed:** These mining scores are specific to AI4Bharat's neural alignment model and are not needed by our rule/lexicon-based IME engine.

---

## 5. Current Directory Layout

```
data/
├── README.md               # This documentation
├── nepali_corpus.txt       # Unified 1.80 GB clean Nepali Devanagari corpus (6,426,307 lines)
└── aksharantar/            # Cleaned Roman-to-Nepali transliteration pairs
    ├── nep_train.json      # 2,397,414 training pairs (183.5 MB)
    ├── nep_valid.json      # 2,804 validation pairs (214 KB)
    └── nep_test.json       # 4,101 benchmark test pairs (286 KB)
```
