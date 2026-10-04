# Experiment Log — Canonical-Key Retrieval Ceiling

**Date:** 2026-10-04
**Question:** Should the engine adopt the specified *inverted phonetic-key index*
(deterministic canonical Roman projection → collapse into a fuzzy key → inverted
index → log-linear scoring → rule fallback), and can it reach its accuracy target?
**Answer:** adopt it as a **candidate generator only**, unioned with the existing
lattice decoder. As the sole generator it cannot reach the target — it measures
**49.18%** retrieval against an **82.5%** gate, and below the shipped decoder's
**60.40%** top-1.

Everything below was measured before any of it was implemented, which is why the
architecture was scoped this way. Inputs were `data/nepali_corpus.txt` (1.80 GB,
6,426,307 lines) and `data/aksharantar/nep_{train,valid,test}.json`. All headline
figures are on `nep_test.json`, the **4,101-case AK-Freq Nepali split** — the
benchmark the specification's own gate names (`eval --split ak-freq`).

---

## 0. Baseline being beaten

Rebuilt and re-measured on the Nepali-only pipeline, 2026-10-04
(`data/akshar.model`, 9.61 MB, `nep_test.json`, 4,101 AK-Freq cases):

| Metric | Nepali-only | 8-language artifact (historical) |
| :--- | ---: | ---: |
| top-1 | **60.35%** | 60.40% |
| top-3 | 73.15% | — |
| in-list@8 | **78.32%** | — |
| MRR | 0.672 | 0.676 |
| top-5 visible | — | 77.59% |
| lenient top-1 | 61.59% | — |
| mean query latency | not re-measured | 0.79 ms |

Retiring the other seven languages cost **0.05pp** of Nepali top-1, so the
Nepali-only rebuild is a clean baseline rather than a regression. Any
candidate-generation change is judged against **60.35%**, and — more importantly
— against the **21.68% of queries whose gold word is not in the candidate list at
all** (100 − 78.32% in-list@8). Rescoring cannot recover those; only better
generation can.

---

## 1. Method

**Projection.** Transcribed from the specification's Phase 1 tables, built from
explicit codepoints rather than string literals (ङ→`ng` … ह→`h`, the nine matras,
anusvara/chandrabindu, halant conjuncts). The specification's two schwa rules —
"drop word-final bare consonant unless the word is one akshara" and "in
`C1 V C2 a C3 V`, delete the schwa on `C2`" — were implemented as written and then
varied, because they turn out to be the dominant variable (§3, §4).

**Collapse.** The κ₁ fold set, with digraphs folded before single letters so
`shh`→`s` cannot be mis-read as `s`+`hh`: `shh,sh→s`; `chh→ch`; `ng,ny,n~→n`;
`Th→th`, `Dh→dh`, `T→t`, `D→d`, `N→n`; `w,b→v`; `aa→a`, `ii→i`, `uu→u`,
`ee→i`, `oo→u`. Skeleton (κ₃) is the consonant-only projection of the collapsed key.

**Index.** Words projected, collapsed, and bucketed; each posting list
pre-sorted by descending corpus frequency so a top-*B* slice needs no runtime
sort. Lexicon selected with `build_lexicon`'s own rule — Devanagari-only tokens,
1–24 characters, count ≥ 2, capped at `--max-words` — so the probe and the Rust
builder choose the same vocabulary.

**Distance.** SymSpell deletion neighbourhoods of the query key (delete ≤ 1 and
≤ 2 characters), and Levenshtein distance for diagnosis.

---

## 2. E1 — The specification's projection contradicts its own worked examples

Phase 1 asserts three unit tests. **None is reachable.**

| Word | Spec expects | Best any reading achieves |
| :--- | :--- | :--- |
| नमस्ते | `namaste` | `namaste` ✅ — but only if the medial schwa is kept |
| घर | `ghar` | `ghar` ✅ — same condition |
| नगरपालिका | `nagarpalika` | `nagrapaalika` / `nagarapaalika` ❌ |
| काठमाडौँ | `kaathmaan~Dau` | `kaaThamaaDaun~` ❌ |

Three separate defects:

1. **The medial-schwa rule is stated against its own example.** नमस्ते survives
   only because the स is halant-final, leaving no vowel in the `C3 V` slot. Drop
   the medial schwa on नागर-words and नमस्ते itself becomes `nmste`.
2. **`nagarpalika` needs two adjacent schwa deletions.** The rule as written
   deletes at most one, and iterating it does not produce the example either —
   after the first deletion the second schwa no longer sits in a `C1 V C2 a C3 V`
   frame. Best reachable is `nagrapaalika`.
3. **`kaathmaan~Dau` contradicts the specification's own tables**, three ways:
   ठ is `Th` in the consonant table but `th` in the example; the word uses anusvara
   (ं→`n`) but the example emits `n~` (chandrabindu); and `n~` is placed *before*
   the following syllable instead of appended.

Also unaddressed by the specification: **vocalic ृ has no entry in the consonant
or matra table at all**, yet appears in common words (`वृद्ध`, `कृष्ण`).

**Consequence.** φ cannot be taken from the specification. It has to be derived
and then measured — which is why §3 exists.

---

## 3. E2 — The projection ablation: one line of the spec costs 16.2pp

Exact collapsed-key match on the 4,101-case split. Every row is bounded above by
lexicon coverage (§4), so these are lower bounds on a perfect matcher.

| Projection variant | Exact key match | Δ |
| :--- | ---: | ---: |
| Schwa dropped — **the spec, literally** | **15.44%** | — |
| **Schwa kept** (`नमस्ते`→`namaste`, `घर`→`ghar`) | **31.63%** | **+16.19** |
| + vocalic ृ → `ri` | 32.09% | +0.46 |
| + `a`↔`e` inherent-vowel fold | 32.80% | +0.71 |
| + medial rule, V admits `a`/`aa` | see note | — |

**Headline: the schwa rule is a 2.05× penalty.** Everything else combined is worth
about 1pp. Further folding is *not* the lever — the projection's value is in what it
*preserves*, not what it merges.

*Note on the medial rule:* it is close to a no-op for नमस्ते (conjunct blocks it) but
changes नगरपालिका from `nagarapaalika` to `nagrapaalika` once `a`/`aa` are admitted as
the V3 vowel class. Either way it never reaches the specification's own example.

**Decision taken.** Derive φ empirically from the 2.4M-pair training split, keep the
inherent schwa, add vocalic ृ, and report measured key-match per level as the
acceptance criterion instead of the specification's asserted strings.

---

## 4. E3 — Retrieval ceiling of the specified cascade

Gold-word **retrieval** rate over a 300k-word lexicon. Retrieval is a hard upper
bound on top-1, so this is the architecture's ceiling.

| Strategy | Gold retrieved | Mean candidates |
| :--- | ---: | ---: |
| Collapsed key, exact | 32.36% | 0.7 |
| + SymSpell delete-1 on key | 39.01% | 3.0 |
| + SymSpell delete-2 on key | 39.31% | 7.9 |
| + skeleton key (`nmst`) | 48.43% | 15.9 |
| **+ delete-2 + skeleton** | **49.18%** | 23.1 |

Bucket occupancy is not the problem — the collapsed key holds **1.27 words/key** and
the skeleton key **2.54**, so the top-32 cap never binds. The specified design's two
structural intuitions are both **confirmed**: only real words come back (no non-word
false positives), and there is no candidate explosion.

**The failure mode is the opposite of the one anticipated: false negatives.**
Two independent causes, quantified next.

---

## 5. E4 — Why it misses: coverage vs. key distance

| | Share of the 4,101 |
| :--- | ---: |
| Gold word **absent** from a 300k lexicon | **36.24%** |
| Present, collapsed key **exact** | 32.36% |
| Present, **1** edit away | 17.22% |
| Present, **2** edits away | 7.46% |
| Present, **3+** edits away | **6.73%** |

**Cause 1 — coverage (36.24%).** A hard ceiling applied before any matching happens.
Architecture-independent, and fixed by widening the lexicon, not by modelling.

**Cause 2 — the residual misses are systematic phonological shifts, not typos.**
The 3+ edit cases are not misspellings a user could be helped with:

| Typed | Gold | Collapsed key vs. gold key | Edits |
| :--- | :--- | :--- | ---: |
| `action` | एक्शन | `action` vs `ekSan` | 5 |
| `professor` | प्रोफेसर | `professor` vs `prophesar` | 4 |
| `jailbata` | जेलबाट | `jailvata` vs `jelavat` | 4 |
| `administration` | एडमिनिस्ट्रेशन | — | 6 |
| `switzerland` | स्वीटजरल्याण्ड | — | 5 |
| `wriddha` | वृद्ध | `vriddha` vs `vaddh` | 3 |

Note `jailbata`→`जेलबाट` (`ai` vs `e`+`a`) and `professor`→`प्रोफेसर` (`e` vs `a`):
these are the *inherent-vowel ambiguity* of Sanskrit loans, and `action`→`एक्शन`
(prefixed `अ` becoming `e`). No edit distance bridges them, because they are not
distances — they are different romanization conventions.

**Why delete-2 underdelivers.** Levenshtein ≤ 2 would allow **57.02%**, but SymSpell's
deletion neighbourhood cannot express a *substitution*, only deletions. That gap —
57.02% vs. the measured 39.31% — is the price of a deletion-only index.

---

## 6. E5 — Artifact size budget

The specification requires ≤ 5.0 MB (elsewhere 5.5 MB) for a 250k–300k word lexicon
plus four inverted indexes, a unigram table, a pruned bigram table and a transducer
table.

**Flat surface arrays do not fit.** Word-store cost alone, measured over the corpus
(mean surface 25.7 bytes — Nepali is heavily suffix-inflected):

| Prune | Words | Surfaces | + u32 offsets | u16 akshara + u32 |
| :--- | ---: | ---: | ---: | ---: |
| c≥3 | 673,917 | 16.51 MB | 19.08 MB | 13.58 MB |
| c≥8 | 274,966 | 6.36 MB | 7.41 MB | 5.29 MB |
| c≥10 | 232,452 | 5.32 MB | 6.21 MB | 4.44 MB |
| c≥20 | 142,036 | 3.15 MB | 3.69 MB | 2.64 MB |

At the specification's own 250k–300k vocabulary the store alone is **6.2–7.4 MB**,
i.e. 113–147% of the whole budget before any index exists.

**But the automaton is not flat, and this changes the conclusion.** `Lexicon` stores
one byte per Devanagari character in a minimal FST whose keys `decode_key` inverts
exactly — so it *is* the word store and no separate surface array is needed:

| Build | Words | `data/lexicon.bin` |
| :--- | ---: | ---: |
| `make lexicon` | 300,000 | **1.92 MB** |
| `make lexicon --max-words 700000` | 700,000 | **4.49 MB** |

So the ≤5 MB gate is **not** impossible; a 4.49 MB lexicon leaves a workable
remainder for indexes and tables inside a 12–15 MB artifact. The budget is raised
from 5.5 MB accordingly.

**Also found: the specification's pruning rule and vocabulary target disagree.**
`c(w) ≥ 3` yields **673,917** words, not the stated 250,000–300,000. The threshold
that lands in that range is **c ≥ 8–10**.

---

## 7. Decisions taken

| # | Decision | Rationale |
| :-- | :--- | :--- |
| D1 | **Union** the key index with the lattice decode; do not replace it | §4/§5 cap the index at 49.18% < 60.40% shipped. The decoder reaches 60.40% precisely because it emits Devanagari through the transliteration model, absorbing `action`→`एक्शन`-class shifts. |
| D2 | Widen the lexicon to **700k** words | Largest single lever on the ceiling, and architecture-independent: 36.24% → ~25% absence. 4.49 MB is affordable. |
| D3 | **Derive φ empirically**; keep the inherent schwa | §2 and §3: the spec's projection is internally inconsistent and its schwa rule halves key quality. |
| D4 | Raise the artifact budget to **12–15 MB** | §6: 4.49 MB for the lexicon alone; the ≤5 MB gate was measured against a flat array and is superseded. |
| D5 | Pair the log-linear scorer with a **factored matra term** | MANUAL §16/§17.2: 75.9% of native-word errors are in-beam ties that word-level global counts provably cannot resolve. Four word-level features alone are the representation already measured insufficient. |
| D6 | Keep the existing engine behind a flag and **A/B** | Nothing above is implemented yet; a baseline must exist before any gate is meaningful. |

### Note on D1 and the earlier failed attempt

MANUAL §16 records a **corpus-wide SymSpell** that cost **−30.79pp** of native top-1
and "never contributed recall at any score band", plus a **corpus roman→Devanagari
lexicon** at **0.00pp**. Both are the direct ancestors of the specified design, and
both were scored in the hand-tuned `u64` bands. The hypothesis now under test is that
the same generator is a net win **under unified log-linear fusion** — which is
exactly MANUAL's standing open limitation. This is a hypothesis, not a result.

---

## 8. Lexicon-width attribution (D2 verified)

D2 ("widen to 700k") was the least-verified decision in this log, so it was
measured directly: the same training recipe (`--reranker-pairs 500000
--epochs 5 --iterations 12`) at two lexicon widths, scored on the same 4,101
rows with the same harnesses.

| | 300k words | 700k words | Δ |
| :--- | ---: | ---: | ---: |
| lexicon | 1.92 MB | 4.49 MB | +2.57 MB |
| container | 7.16 MB | 9.61 MB | +2.45 MB |
| top-1 | 59.01% | **60.35%** | **+1.34pp** |
| top-3 | 72.28% | 73.15% | +0.87pp |
| top-5 | 75.66% | 76.15% | +0.49pp |
| in-list@8 | 77.88% | 78.32% | +0.44pp |
| reachable@50 | 83.86% | 84.00% | +0.14pp |
| MRR | 0.662 | 0.672 | +0.010 |

**D2 confirmed**: the wider lexicon is better on every metric, at 2.45 MB. The
+1.34pp top-1 for +2.45 MB is a good trade and the lexicon is now 47% of the
container.

It also **rules out** the lexicon as the cause of the −1.44pp top-5 regression
against the retired eight-language artifact (§0): 300k is *worse* on top-5, so
narrowing the lexicon does not recover it. That regression remains unattributed;
the surviving candidates are the trigram pruning threshold, the blend weight
`gamma` calibrated on a Nepali-only validation split, and the reranker having
trained on 500k Nepali-only pairs rather than a mixed-language pool.

### The number that sets the ceiling

`reachable@50 = 84.00%`. So for the current generator:

| Band | Share |
| :--- | ---: |
| gold in top-8 (in-list@8) | 78.32% |
| gold in top-50 but not top-8 | 5.68% |
| **gold never surfaced at all** | **16.00%** |

Two consequences, and they reprioritise the work:

1. **Ranking headroom is 23.65pp** (60.35% → 84.00%), which is larger than the
   retrieval gap and costs nothing in size to attack. That is the factored
   matra term (D5), not a new index.
2. **Top-1 cannot exceed 84.00%** without changing candidate generation. The
   specification's 82.5%/84% gate sits exactly at this ceiling, so it is only
   reachable if generation improves *and* ranking becomes near-perfect.

## 9. What is not yet measured

- **Why top-5 is 1.44pp below the retired eight-language artifact.** Lexicon
  width is now excluded (§8). Remaining candidates: trigram pruning threshold,
  the blend weight `gamma` calibrated on a Nepali-only validation split, and the
  reranker having trained on 500k Nepali-only pairs rather than a mixed pool.
- Whether the union's in-list@8 exceeds **78.32%**. Still the single most
  important open number for retrieval work.
- Whether the log-linear scorer beats the `u64` bands in practice. MANUAL §16
  records that the learned reranker alone peaks at γ≈0.2–0.3 for **+0.28pp**, so
  this is genuinely open and should not be assumed.
- Latency of the union against the ≤0.85 ms target. The current model measures
  0.836 ms/query mean; p99 has not been recorded.
- Whether `--reranker-pairs 0` (all 2.4M) helps. MANUAL's null result was
  measured on the eight-language mix, where 500k pairs gave Nepali only ~62k
  reranker examples; the pool is now Nepali-only, so the null does not transfer
  and is worth ~10 min to settle.

## 10. Next steps

1. ~~Establish the Nepali-only baseline.~~ **Done** — 60.35% top-1, 78.32%
   in-list@8 (§0).
2. ~~Re-measure on the rebuilt container.~~ **Done**, same step.
3. ~~Confirm D2 (lexicon width).~~ **Done** (§8): 700k beats 300k on every
   metric, +1.34pp top-1 for +2.45 MB.
4. **Attributed ranking first, not retrieval.** The 23.65pp gap between top-1
   (60.35%) and reachable@50 (84.00%) is the larger prize and costs no size.
   Build the akshara-factored matra term (D5) and log-linear fusion (D5), and
   measure against top-1 *and* top-5.
5. Build the derived-φ tool; gate it on measured key-match per level.
6. Build the collapsed-key index + SymSpell as a *third* source beside the two
   existing ones; A/B **in-list@8** before touching the scorer. Only worth doing
   once 1–3 are exhausted, because it adds ~1–2 MB and ~0.05–0.1 ms.
7. Settle `--reranker-pairs 0` (~10 min) whenever convenient.
