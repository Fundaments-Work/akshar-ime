# Data root — where AksharIME's data lives, where it came from, and how it is cleaned

This directory is **fully gitignored** (only this README is tracked). No data
is committed to the repository — the repo ships code; releases ship the built
artifacts. Everything below is rebuilt locally with the pipeline in
`data/pipeline/`.

## Where the data came from (provenance)

| Source | What it is | License / access |
|---|---|---|
| **Aksharantar** (AI4Bharat, IIT Madras) | 5.4M roman→Devanagari word pairs, produced by human annotators and validated with a rule-based checker. The per-language files are **merged into one language-agnostic Devanagari set**; nothing downstream reads the language a pair came from. | CC0 / some CC-BY — download from [HuggingFace](https://huggingface.co/datasets/ai4bharat/Aksharantar) |
| **Wikipedia** | Full article dump for the target language, text extracted from the XML. | CC-BY-SA — [dumps.wikimedia.org/newiki](https://dumps.wikimedia.org/newiki/) |
| **CC100** | CommonCrawl web text, the portion the creators filtered to Devanagari. | CC0 — [data.statmt.org/cc-100](https://data.statmt.org/cc-100/) |
| **akshar-ime news crawl** | 18,190+ full articles from news sites (gorkhapatra, onlinekhabar, nayapatrika, kanunpatrika), crawled with the private pipeline in `pipeline/pipeline.py`. Current-affairs vocabulary (ministers, dates, places) that encyclopedic sources lack. | public news; only derived counts ship |
| **Your own typing** | learned on-device in `~/.config/akshar-devanagari/`, never leaves the machine | — |

## How the data is cleaned (the single rule set)

All cleaning enforces one definition of a *word*: **a maximal run of
Devanagari letters U+0900..=U+0963** (consonants, vowels, matras, nukta
forms, anusvara/visarga). Everything else is stripped or dropped:

- no danda/purnabiram glued to words (`पुगे।` → `पुगे`)
- no digits — Devanagari (`२०८१`) *and* ASCII (`2024`), either glued or alone
- no ASCII punctuation/brackets/quotes (`सोमाली,` → `सोमाली`)
- no ZWJ/ZWNJ joiners, no Latin letters
- lines/sentences need ≥4 surviving words (kills headers, dates, captions)
- **exact duplicate lines/pairs removed** (news syndication and Wikipedia
  boilerplate repeat massively — 5.4M duplicate lines and 108k duplicate
  word-pairs dropped)

Why it matters: glued variants split a real word's count across dictionary
keys, flattening exactly the frequency signal the engine's reranker depends
on. Cleaning + dedup of the text corpus measured **+0.33 points** native
top-1 (79.51-equivalent configs → see
`../docs/plans/2026-09-03-accuracy-experiments.md`).

## Layout

```
data/
  aksharantar/          language-agnostic Devanagari word-pair set (strict-cleaned)
    train_devanagari.jsonl   3,588,793 pairs — EM training
    valid_devanagari.jsonl       9,155 pairs — tuning only
    test_devanagari.jsonl        4,101 cases — benchmark (single-shot measurement)
  store/
    corpus_clean.txt   THE single stored text: 2.89M clean unique sentences,
                       86.1M tokens, compiled from Wikipedia+CC100+news;
                       frequencies and bigrams are derived from it
    word_pairs.csv     word bigrams (freq>=3) — E6 context-layer data
  eval/                benchmark convenience files (TSV = test split flattened)
  pipeline/            the pipeline (private, untracked):
    pipeline.py            news crawl / count / export
    clean_aksharantar.py   strict-clean + merge the word-pair JSONs
    build_corpus.py        compile the cleaned corpus, delete raw inputs
    fetch_corpus.py        download Aksharantar from Hugging Face
    extract_wiki.py        Wikipedia dump -> text lines
    filter_cc100.py        CC100 -> Devanagari lines
    make_eval_tsv.py       regenerate eval TSV
    .venv/                 its Python environment
  backup/              gzip snapshots taken before destructive operations, and
                       akshar_full_precision.model — the f32 master kept because
                       the shipped containers are 8-bit quantized and cannot be
                       reverted without retraining
  akshar.model         BUILT — desktop container, 30.59 MB (transliteration model +
                       KN syllable LM + 470k vocab frequencies + sparse reranker
                       weights + phrase bigrams)
  akshar_wasm.model    BUILT — browser container, 8.91 MB / 4.92 MB Brotli
                       (entropy-pruned trigram LM, no phrase bigrams)
```

## Rebuild recipes (local; not Makefile targets — data is not in the repo)

```bash
# 1. Word-pair corpus (transliteration training)
python3 data/pipeline/fetch_corpus.py data/aksharantar     # download Aksharantar
python3 data/pipeline/clean_aksharantar.py                 # clean + merge -> *_devanagari.jsonl
#   originals are deleted after cleaning; keep data/backup/*.gz snapshots

# 2. Cleaned running-text corpus (the single text file)
mkdir -p data/raw
curl -sL -o /tmp/newiki.xml.bz2 https://dumps.wikimedia.org/newiki/latest/newiki-latest-pages-articles.xml.bz2
python3 data/pipeline/extract_wiki.py /tmp/newiki.xml.bz2 data/raw/newiki.txt
curl -sL -o /tmp/cc100-ne.txt.xz https://data.statmt.org/cc-100/ne.txt.xz
python3 data/pipeline/filter_cc100.py /tmp/cc100-ne.txt.xz data/raw/cc100ne.txt
# news: python3 data/pipeline/pipeline.py crawl  (articles land in the DB)
python3 data/pipeline/build_corpus.py data/store/corpus_clean.txt \
    --db data/store/corpus_text.db data/raw/newiki.txt data/raw/cc100ne.txt

# 3. One-Shot End-to-End Model Training
cargo run --release --bin train
# or simply: make train

# Full overnight whole-corpus run (3.59M pairs):
# cargo run --release --bin train -- --reranker-pairs 0 --epochs 5 --bigram-min-freq 5

# Inspect model breakdown and sub-component sizes:
cargo run --release --bin probe_model -- --model data/akshar.model --inspect

# Pack unified model with customizable bigram filtering:
cargo run --release --bin pack_model -- --bigram-min-freq 5 --out data/akshar.model

# Build the browser profile (8.91 MB / 4.92 MB Brotli). TRIGRAM_THRESHOLD trades
# size against accuracy along the curve in docs/MODEL_TRAINING_AND_OPTIMIZATION.md:
make web-model

# Re-encode an existing container into the current format (any version in, v3 out):
cargo run --release --bin repack_model -- --model data/akshar.model \
  --out data/akshar.model --compact-aksharas

# 4. Evaluate (Aksharantar test split: 4,101 cases)
cargo run --release --bin evaluate_aksharantar -- --model data/akshar.model
```

## Rules that prevent repeat incidents

- Model artifacts are bundled into the unified `data/akshar.model` container.
- No binary files are ever stored inside `src/`.
- The news pipeline exports into `data/store/` and merges into the vocab only
  via explicit `--merge-base` (a news-only vocabulary once silently overwrote
  the real one and cost 1.2 accuracy points).
- **Backup before destroying:** gzip snapshot into `data/backup/` first.
- Verified result of the current chain: **82.02% native top-1, 92.17% top-5** (desktop profile) through the
  full engine (canonical discriminative reranker + candidate union + pruned syllable lattice).
