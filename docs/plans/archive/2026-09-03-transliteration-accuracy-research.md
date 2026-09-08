# Transliteration Accuracy Research — Findings & Feasibility

> **Archive snapshot — frozen.** IndicXlit figures below are quoted *without* LM
> reranking; reranked Nepali is 86.6% native / ~62% named-entity (paper Table 6)
> — the honest comparison target. See `docs/MANUAL.md` threats to validity.

Date: 2026-09-03
Status: Research complete; experiments ongoing (see 2026-09-03-accuracy-experiments.md)

## The 95% question

**Isolated-word top-1 of 95% is not attainable on this benchmark — by anyone.**
Calibration from published systems:

| System | Benchmark | Top-1 |
|---|---|---|
| IndicXlit (11M-param transformer, 6+6 layers, 256-dim, char-level, trained on 26M pairs / 21 languages) | Aksharantar Nepali native (AK-Freq) | **80.25%** |
| IndicXlit | Aksharantar Nepali named entities | 52.67% |
| IndicXlit | Dakshina (12 languages, avg) | ~51.8% |
| Best LLMs (GPT-class, 2025 benchmark) | Dakshina/Aksharantar | *below* IndicXlit |
| This engine (statistical, EM+KN) | Aksharantar Nepali native | 73.2% → **75.3%** (E0/E1 fixes) |

Sources: [IndicXlit repo](https://github.com/AI4Bharat/IndicXlit),
[Aksharantar paper (EMNLP 2023 Findings, arXiv:2205.03018)](https://aclanthology.org/2023.findings-emnlp.4.pdf),
[LLM transliteration benchmark (arXiv:2505.19851)](https://arxiv.org/abs/2505.19851),
[romanized Hindi/Bengali modeling (arXiv:2511.22769)](https://arxiv.org/html/2511.22769v1).

The input is information-lossy: roman script does not encode vowel length
(kal → कल/काल/काली), schwa, or nasalization. Many test pairs are undecidable
from the roman string alone; even the neural SOTA leaves ~20% on the table.

## Where the errors actually are (E0 measurements, AK-Freq n=2108)

- Decoder oracle: top-1 75.3%, **top-2 85.2%**, top-3 88.0%, top-50 92.1%.
  → The model *generates* the gold for 92% of words; ranking is the bottleneck.
- Taxonomy of top-1 misses: **52% matra-only** (vowel sign/nasal), 48%
  substantive; CER 5.1%.
- Named entities are a different regime (decoder top-1 33.8%, CER 25%) —
  coverage-limited, not ranking-limited.

## The honest 95% targets

1. Native top-1 ≥ 80% (neural parity without a neural net).
2. Top-3 coverage ≥ 95% (oracle says 88% at k=3 today; achievable with
   better emissions/data).
3. **Sentence/context-aware accuracy ≥ 95%** — the metric that matters for an
   IME; pinyin IMEs live here. With word context, "kal" following "hijo ra"
   disambiguates to कल.

## Ranked technique shortlist (expected gain on native top-1)

| # | Technique | Est. gain | Status |
|---|---|---|---|
| E0 | Fix fusion: score-resolution + lexicon override | **+2.1** | **Done (75.33%)** |
| E1 | Exact DP decoding (drop beam pruning) | 0–0.3 | pending |
| E2 | Multilingual Devanagari pooling (hin+mar+nep) | +1–2 | pending |
| E3 | Word-frequency prior + perceptron reranker v2 | +2–4 | **key lever** |
| E4 | Position-conditioned emissions P(s|a,pos) | +0.5–1.5 | pending |
| E5 | Self-training / round-trip data amplification | +1–2 | pending |
| E6 | Word-bigram context decoding (sentence level) | →95% effective | **differentiator** |
| E7 | Tiny distilled neural reranker (fallback) | +2–3 | only if plateau |

E3 attacks exactly the 52% matra-miss class: a word-frequency prior resolves
कल vs काल by corpus frequency, which the akshara LM cannot do.
