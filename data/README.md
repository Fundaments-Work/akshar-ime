# Data root — where AksharIME's data lives, where it came from, and how it is used

This directory is **fully gitignored** (only this README is tracked). No data
is committed to the repository — the repo ships code; releases ship the built
artifacts.

> **Frozen inputs.** The `data/pipeline/` scripts and `data/backup/` snapshots
> were removed on 2026-09-08. The files below can no longer be regenerated
> locally — they are inputs, not outputs. To rebuild them from scratch you
> re-fetch the public sources in the provenance table and re-apply the cleaning
> rule set below (the rules *are* the pipeline now).

## Where the data came from (provenance)

| Source | What it is | License / access |
|---|---|---|
| **Aksharantar** (AI4Bharat, IIT Madras) | ~5.4M raw roman→Devanagari word pairs, cleaned and merged into one language-agnostic Devanagari set (3,588,793 train pairs); nothing downstream reads the language a pair came from. | CC0 / some CC-BY — [HuggingFace](https://huggingface.co/datasets/ai4bharat/Aksharantar) |
| **Wikipedia** | Full article dump for the target language, text extracted from the XML. | CC-BY-SA — [dumps.wikimedia.org/newiki](https://dumps.wikimedia.org/newiki/) |
| **CC100** | CommonCrawl web text, the portion the creators filtered to Devanagari. | CC0 — [data.statmt.org/cc-100](https://data.statmt.org/cc-100/) |
| **akshar-ime news crawl** | 18,190+ full articles from news sites (gorkhapatra, onlinekhabar, nayapatrika, kanunpatrika). Current-affairs vocabulary (ministers, dates, places) that encyclopedic sources lack. Crawl scripts removed; only derived counts survive, frozen inside the corpus. | public news; only derived counts ship |
| **Your own typing** | learned on-device in `~/.config/akshar-devanagari/`, never leaves the machine | — |

## How the corpus is used (the full chain)

`data/store/corpus_clean.txt` (1.5 GB, 2.89M clean unique sentences, 86.1M
tokens) is the system's **word-frequency source**. Exactly this, nothing more:

1. **Train time** (`src/bin/train/train.rs`): the corpus is tokenized with the
   cleaning rule set below; tokens with frequency ≥ 3 become `vocab_freq`
   (~470k words), packed into the unified container (`src/core/unified.rs`,
   v5: translit model + sparse reranker table + `vocab_freq` + dense
   normalisation constants).
2. **Runtime** (`src/core/engine.rs`): `vocab_freq` feeds three consumers —
   the `WordTrie` (trie-constrained decode pass), `FreqRanks` (the frequency
   heuristic, +5.64pp on `AK-Freq`, the single largest rerank contributor),
   and the engine's `freq` map for candidate scoring.
3. **Sentence evaluation** (`src/bin/evaluate/evaluate_sentences.rs`): reads
   the corpus with a 1-in-200 holdout (`src/core/holdout.rs`) for
   in-context accuracy.

What the corpus does **not** feed:

- the akshara KN language model and EM emissions — those come from the
  Aksharantar word pairs, not running text;
- word-bigram context — the v4 word-bigram table was removed (19.5 MB for
  +0.16pp, not shipped), and `store/word_pairs.csv` does not exist. The
  `ContextModel` (`src/core/context.rs`) learns word bigrams on-device from
  the user's own confirmations only, and `set_context_word` is an
  unimplemented stub (`src/core/engine.rs`).

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
`../docs/plans/archive/2026-09-03-accuracy-experiments.md`).

## Layout (current — 2026-09-08 cleanup)

```
data/
  aksharantar/          language-agnostic Devanagari word-pair set (strict-cleaned)
    train_devanagari.jsonl   3,588,793 pairs — EM training
    valid_devanagari.jsonl       9,155 pairs — tuning only
    test_devanagari.jsonl        4,101 cases — benchmark (single-shot measurement)
  store/
    corpus_clean.txt   THE single stored text: 2.89M clean unique sentences,
                       86.1M tokens; word frequencies are derived from it at
                       train time (see "How the corpus is used" above)
  eval/
    test_multiref.jsonl  14,410 alternative romanizations for multi-reference
                         scoring (see MANUAL §8; not wired into any harness —
                         measured ad hoc, reported alongside strict accuracy,
                         never instead of it)
  akshar.model         BUILT — desktop container, 11.37 MB (11,917,219 bytes:
                        transliteration model +
                        KN syllable LM + 470k vocab frequencies + sparse reranker
                        weights + dense normalisation constants; no word bigrams)
  akshar_wasm.model    BUILT — browser container, 8.91 MB / 4.94 MB Brotli
                        (entropy-pruned trigram LM, no phrase bigrams)
```

Removed 2026-09-08 and intentionally not restored: `backup/` (snapshots),
`akshar_mid.model` (unreferenced checkpoint — `train` rebuilds it),
`pipeline/` (all scripts + venv), `eval/valid_nep.jsonl`,
`eval/valid_native.jsonl`, `eval/aksharantar_test.tsv` (all unreferenced
scratch; the TSV regenerates from the test split if needed).

## Recipes (local; only what still runs — data is not in the repo)

```bash
# 1. Train from the frozen inputs (corpus + pairs must already exist above)
cargo run --release --bin train
# or simply: make train

# Full overnight whole-corpus run (3.59M pairs):
# cargo run --release --bin train -- --reranker-pairs 0 --epochs 5 --bigram-min-freq 5

# Inspect model breakdown and sub-component sizes:
cargo run --release --bin probe_model -- --model data/akshar.model --inspect

# Pack unified model with customizable bigram filtering:
cargo run --release --bin pack_model -- --bigram-min-freq 5 --out data/akshar.model

# Build the browser profile (8.91 MB / 4.94 MB Brotli, verified 2026-09-08).
# TRIGRAM_THRESHOLD trades size against accuracy; see docs/MANUAL.md
# "Browser profile" for the measured sizes:
make web-model

# Re-encode an existing container into the current format (any version in, v3 out):
cargo run --release --bin repack_model -- --model data/akshar.model \
  --out data/akshar.model --compact-aksharas

# 2. Evaluate (Aksharantar test split: 4,101 cases)
cargo run --release --bin evaluate_aksharantar -- --model data/akshar.model
```

Re-collecting the raw sources (no scripts left — plain downloads):
Aksharantar from the HuggingFace link above; Wikipedia from
`dumps.wikimedia.org/newiki`; CC100-Ne from `data.statmt.org/cc-100`.
Then re-apply the cleaning rule set in this file.

## Rules that prevent repeat incidents

- Model artifacts are bundled into the unified `data/akshar.model` container.
- No binary files are ever stored inside `src/`.
- A news-only vocabulary once silently overwrote the real one and cost 1.2
  accuracy points — vocabulary merges are explicit or they don't happen.
- **There is no backup directory.** `data/` is gitignored and local; deletions
  are final. Keep snapshots outside the repo if you need them.
- Verified result of the current chain: **80.98% native top-1, 91.84% top-5** (`AK-Freq`,
  desktop profile) through the
  full engine (canonical discriminative reranker + candidate union + pruned syllable lattice).
  Re-measured 2026-09-10 with `make eval` after the pipeline correctness fixes
  (`train-mid` retrain: raw dense statistics, reserved dev set, holdout-filtered
  vocabulary); pooled top-1 60.40% [58.94%, 61.89%], within noise of the
  2026-09-08 model (81.83% / 92.22%, pooled 61.98% [60.61%, 63.36%]).
```

(End of file - total 136 lines)
