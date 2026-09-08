# Data usage research — how the field uses data, and our path to 90%

> **Archive snapshot — frozen.** Literature review as of 2026-09-05. IndicXlit
> figures are without LM rerank (reranked: 86.6% Nepali native, paper Table 6).

Date: 2026-09-05. Companion to `2026-09-05-mathematics.md`, `2026-09-05-research-agenda.md`,
`2026-09-03-accuracy-experiments.md`.

---

## 1. What we own, and what each asset is (not) used for

| Asset | Size | Used for | Unused potential |
|---|---|---|---|
| Aksharantar Nepali train | 2.4M pairs | EM emissions + akshara KN LM | nothing — well exploited |
| Aksharantar Hindi train | ~3M pairs (downloaded) | **nothing** (naive pooling failed, reverted) | retry after reverse-index joint-weight fix (documented prerequisite) |
| Wikipedia + CC100 text | 75M tokens, 570k words | word **unigram** frequencies only (`word_freq_text.bin`) | — |
| News scrape DB (`data-pipeline/nepali_text.db`) | 18,190 articles, 524k distinct words, **6.5M word-bigram rows** | counted, exported to `out/word_pairs.csv` (1.25M rows ≥ freq 3) — **consumed by nothing** | E6 context layer: the data is already sitting there |
| `out/synthetic.jsonl` | 918k weighted pairs | mixed into EM training | regen from the fuller news vocabulary |
| User dictionary | on-device | personalization | unmeasured (compounding in real use) |

**The headline audit finding: our single largest collected-but-unused asset is the
word-bigram table (6.5M rows).** The E6 context layer was designed (research agenda §1)
but has no data wiring; the data side is done.

## 2. How IndicXlit actually uses data (primary source: [paper](https://aclanthology.org/2023.findings-emnlp.4/), [arXiv:2205.03018](https://arxiv.org/pdf/2205.03018))

- 26M pairs / 21 languages; a single 11M-parameter multilingual char-level
  transformer, beam 4, **re-rank top-4 with F = 0.9·T + 0.1·P** where T = model
  log-prob and P = word-unigram LM score.
- Table 6, Nepali: AK-Freq (native/frequent words) **80.1 without re-ranking,
  86.6 with top-4 unigram re-ranking**. Unigram re-ranking lifts the native
  bucket by ~12% relative on average across languages; "re-ranking doesn't
  help for named entities" (their AK-NEF Nepali ≈ 49%, NEI ≈ 55% — both
  re-ranked; NE is where our decoder+vocab also beats them: AK-NEF 28.4 vs
  their 49.1 is *their* win, but NEI 44.8 vs 55.4 — check: our numbers are
  single-reference, theirs multi-reference; see below).
- **Evaluation is multi-reference**: annotators produced up to 4 romanized
  variants per native word; test correctness counts a match against *any*
  reference variant. Our harness (single gold from `nep_test.json`) is
  therefore **strictly harder per case** — our 79.5–80.7 and their 80.1 are
  not measured on the same task, and a multi-reference mode is required for
  like-for-like claims (the experiment log's "multi-reference scoring per the
  IndicXlit protocol" todo).
- Their error analysis: **60% of errors are vowels, 25% similar consonants** —
  the same matra-dominated distribution we measure (52% matra-only).

**Implication.** Two calibration corrections before chasing 90: (a) implement
multi-reference scoring so our number is comparable to the literature; (b) note
that a *published neural system reaches 86.6% on this exact bucket with a
unigram-LM rerank over just 4 candidates* — while our rerank over 50 candidates
(oracle 93.55%) lands at 80.69. **Our ranking stage, not our candidate
generation, is the deficit.**

## 3. Context in production IMEs (primary source: [Kirov et al. 2024, Computational Linguistics](https://aclanthology.org/2024.cl-2.2/))

- Google's transliteration IMEs improve with **preceding-word context**
  (not full sentences): overall **3.3% absolute / 18.6% relative WER
  reduction**. Methods include combining non-contextual transliteration with
  language-model rescoring over word lattices.
- This validates the E6 design (conditional prior P(v|w_{i-1}) over the
  candidate set) as the production-proven lever, and the projected +2–4
  native top-1 as realistic.

## 4. Bigger Nepali text than our 75M tokens

- **IndicCorp v2** ([HF card](https://huggingface.co/datasets/ai4bharat/IndicCorpV2),
  [paper](https://aclanthology.org/2023.acl-long.693.pdf)): ~20.9B tokens over
  24 languages; **Nepali (npi_Deva) = 16.99M rows (~2.1 GB parquet)** — roughly
  3–5× our current text, from CommonCrawl with cleaning.
- Our news crawler adds fresh, domain-current text (18k articles so far) and is
  the only source whose word distribution matches what users type today.
- IndicXlit's own ablation (Table 7, row 5): adding **IndicCorp monolingual
  data for corpus mining was their single largest data win** (+8.7 avg top-1 in
  their staged ablation) — text corpora as a data source, not just as a
  frequency table.

## 5. Prior art on the ambiguity classes that block us

- Vowel-length/matra errors are the literature's dominant class (60% of
  IndicXlit errors) and no published isolated-word method resolves them from
  the string alone — consistent with our entropy analysis (roman carries zero
  bits for कल/काल).
- Everything that works injects **prior** information: unigram LM reranking
  (IndicXlit, +6.5 on Nepali AK-Freq), preceding-word context (Kirov,
  −18.6% rel. WER), user history. Our roadmap already contains all three
  layers; execution is the gap.

## 6. Prioritized program toward 90%

Ordered by (expected points) / (effort), using assets we already hold:

1. **Regression diagnosis** (rebuild drifted 80.69 → 79.51; synthetic pairs and
   vocab changed together) — recover the record first; it is 1.2 free points.
2. **Multi-reference eval mode** (IndicXlit protocol) — makes every later claim
   comparable; likely raises our reported native top-1 by accepting valid
   variants.
3. **E6 context layer wired to `word_pairs.csv`** — the 6.5M bigrams exist;
   Kirov-calibrated +2–4 native top-1, and the sentence harness from the
   research agenda. Biggest *product* lever.
4. **Reranker v2 on real features, leakage-free** — features: corpus frequency,
   decoder margin (top1−top2), word length, LM score, boundary-akshara priors;
   tuned on valid, measured on test. Target: close the ranking gap toward the
   93.55% oracle. This is the main *benchmark* lever: 80.69 → 86+ is ranking,
   90 requires the oracle itself to rise (deeper k, intersection).
5. **Scale the text corpus** — rebuild `word_freq_text.bin` from IndicCorp v2
   Nepali + news crawl; better frequency estimates directly sharpen step 4 and
   the E6 priors.
6. **Hindi pooling retry** with the joint-weight reverse-index repair — only
   after 1–4; expect transfer, not a step change.
7. **Self-training refresh** — romanize the enlarged vocabulary (news + IndicCorp).

## 7. Sources

- IndicXlit / Aksharantar: [EMNLP 2023 Findings](https://aclanthology.org/2023.findings-emnlp.4/) · [arXiv:2205.03018](https://arxiv.org/pdf/2205.03018) · [GitHub](https://github.com/AI4Bharat/IndicXlit) · [HF dataset](https://huggingface.co/datasets/ai4bharat/Aksharantar)
- Kirov, Johny, Katanova, Gutkin, Roark (2024): [Context-aware Transliteration of Romanized South Asian Languages, Computational Linguistics 50(2)](https://aclanthology.org/2024.cl-2.2/)
- IndicCorp v2: [HF dataset card](https://huggingface.co/datasets/ai4bharat/IndicCorpV2) · [Doddapaneni et al., ACL 2023](https://aclanthology.org/2023.acl-long.693.pdf)
