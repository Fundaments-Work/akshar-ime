# Experiment archive

Working documents from the development of Akshar IME, kept because they record
**what was tried, measured and rejected** — which the manual's final numbers
cannot show on their own. Superseded as plans; preserved as evidence.

Read in this order:

| Document | Date | What it records |
| :--- | :--- | :--- |
| `2026-08-01-generative-transliteration-design.md` | Aug 1 | Original design of the generative core: the source-channel decision, akshara units, first results. |
| `2026-09-03-transliteration-accuracy-research.md` | Sep 3 | Error analysis and a ranked technique shortlist (E0–E7) with expected gains. The "95% question". |
| `2026-09-03-accuracy-experiments.md` | Sep 3 | **The experiment log.** E0–E3 with measured deltas, the v2 WFST core, depth-2 pair context. |
| `2026-09-05-data-flow.md` | Sep 5 | How raw text becomes the artefacts a keystroke touches. |
| `2026-09-05-data-research.md` | Sep 5 | Literature review: how IndicXlit uses data, context in production IMEs (Kirov et al. 2024), larger Nepali corpora, prior art on the blocking ambiguity classes. |
| `2026-09-05-mathematics.md` | Sep 5 | Complete mathematical treatment (equations with code references). Discounting sections superseded by the Sep 6 repair doc. |
| `2026-09-06-repair-and-path-to-90.md` | Sep 6 | Measured defect audit (A–F), landed fixes, improvement plan. Phases A/B shipped; C–F open. |
| `2026-09-05-research-agenda.md` | Sep 5 | Mathematics considered but not executed: context-tree weighting, A* anytime decoding, the entropy harness, incremental decoding. |
| `2026-09-05-roadmap-to-90.md` | Sep 5 | First plan to 90%: audit of how every byte of data is used. |
| `2026-09-05-path-past-90.md` | Sep 5 | Revision of the above, with the W0 measurement-gate results. |
| `2026-09-22-factored-matra-model.md` | Sep 22 | Factored P(C\|R)·P(M\|C,R) matra model: implemented, measured, falsified as a word-level prior (78/182 head-to-head); code removed. |
| `2026-09-22-top2-discriminator.md` | Sep 22 | Top-2 swap discriminator: 3 training regimes (full/string-only/domain-matched), no operating point above baseline; code removed. |
| `2026-09-22-mbr-selection.md` | Sep 22 | MBR consensus: 24 configs on valid native, best +2/798 (noise); converged lists carry no votable information; probe removed. |
| `2026-09-22-ne-routing.md` | Sep 22 | NE burial analysis: router/demotion/attenuation killed (deep 6%, demote +140/−138, freq-burial 14%); 46% absent = generation gap, needs training; probe removed. |
| `2026-09-22-attestation-training.md` | Sep 22 | Attestation-weighted train-mid: entities +13, native −8, pooled +5 (noise) — killed as ship candidate; flag kept, artifact local-only. |

There are currently no live planning documents — `docs/plans/` holds only this
archive. The live roadmap is `docs/MANUAL.md` §12–13, and the mathematics and
the final measurements are in `docs/MANUAL.md`.

**Caution: the accuracy figures in these documents are historical.** They were
measured before the 2026-09-06 defect fixes and do not describe the shipped
system. IndicXlit comparisons are quoted without LM rerank (reranked: 86.6%
Nepali native, paper Table 6). The manual is authoritative.
