# AksharIME — Research Agenda (the mathematics ahead)

> **Archive snapshot — frozen.** Open ideas, unaffected by later changes; all
> items below are still open. See `docs/MANUAL.md` for what has since landed.

Date: 2026-09-05. Companion to `2026-09-05-mathematics.md`.
Priorities ordered by user impact; §2 is the paper's core.

---

## 1. Context layer (E6) — conditional priors over the word lattice

**Model.** After the user commits word `w_{i−1}`, the current word's
candidates are rescored with a corpus word-bigram term:

$$
\mathrm{score}'(v) = \mathrm{score}(v) - \lambda_c \cdot \big(-\log P(v \mid w_{i-1})\big),
$$

$$
P(v \mid w) = \frac{\max\big(c(w, v) - \delta,\ 0\big)}{c(w)} + \lambda(w) \cdot P(v) \qquad \text{(KN over words)}.
$$

This is a *conditional* version of the vocabulary prior (mathematics doc §6):
the prior over the ambiguity set A(R) is no longer corpus-global but
conditioned on the committed left context.

**Data.** `word_pairs` table (streaming counter over the running-text DB),
pruned to freq ≥ 3; expected ~2–5M bigrams from 75M+ tokens.

**Evaluation — DONE (2026-09-05).** `evaluate_context` A/B harness (two
identical engines over romanized corpus sentences, context the only
difference). Held-out: **+0.31–0.40%** suggestion accuracy (73,583 words).
Modest vs the Kirov-calibrated +2–4 because canonical romanizations already
decode at ~89% — real user input (variants, typos) leaves more for context
to fix. Wired into the engine with boost 40k·ln(1+f).

**Design decision to make:** greedy per-word rescoring (v1) vs a word-lattice
Viterbi that defers commitment until the sentence evidence is in. Pinyin IMEs
do the latter; it is the natural home for the composed graph T∘L∘G.

---

## 2. v3 — parameter-free estimation via context-tree weighting

**Problem being solved.** Every smoothing choice so far is a hand constant
(δ = 0.75, α = 0.05, λ weights, backoff hierarchy shape). CTW replaces the
whole apparatus with a Bayesian mixture over context depths, with a regret
guarantee.

**KT estimator.** For a node with counts (n_1..n_K) over outcomes and a flat
Dirichlet(½) prior, the predictive probability of outcome i is

$$
P_{\mathrm{KT}}(i \mid \text{node}) = \frac{n_i + \tfrac{1}{2}}{n + \tfrac{K}{2}},
$$

and the node's sequential likelihood accumulates as a product of these.

**The CTW recursion.** For a context path (deepest first), the mixture
weight of depth d is updated online:

$$
W_d = \tfrac{1}{2} \cdot P_{\mathrm{KT},d} \cdot W_{d-1} + \tfrac{1}{2} \cdot W_{d+1},
$$

standardly: each tree node stores its mixture probability
$W_{\text{node}} = \tfrac{1}{2} P_{\mathrm{KT,node}} + \tfrac{1}{2} \prod_{\text{children}} W_{\text{child}}$;
the root's $W$ is the model's sequential likelihood. Regret guarantee
(Willems–Shtarkov–Tillemae 1995):

$$
\ln P_{\mathrm{CTW}}(\text{sequence}) \ \ge\ \ln P_{\text{best fixed depth}} \ -\ |T| \ln 2,
$$

i.e. within $|T| \ln 2$ nats of the *best* depth chosen in hindsight —
a guarantee no KN-tuned, and no neural, system states.

**Application to AksharIME.** The context tree lives over the alignment
lattice: node = suffix of previously emitted (akshara, chunk) pairs, exactly
the v2 pair-trie; leaves carry KT counts over next-pairs. Each lattice edge
is scored by the root mixture instead of the current
`emit + fluency + λ·pair` sum. The observed failure of additive fusion
(mathematics doc §5.4) is precisely what the mixture corrects: context depth
is chosen *per event by the evidence*, not by a global weight.

**Engineering shape.** Nodes = trie (already built); counts are small
integers → u8/u16 quantized (merges with M2 compression); per-edge cost is
O(1) amortized (one KT update + one mixture step). Expected artifact: ~4MB
total, zero tuning constants, same or better top-1 — and the paper's central
theorem.

**Open question to resolve first:** CTW is classically defined for sequence
prediction; the lattice setting has posterior-weighted (not observed)
sequences. Two defensible routes: (a) train the tree on the single best
alignment per word (Viterbi alignment, EM-hardening), or (b) fractional
counts with the mixture weights recomputed per E-step. Route (a) is simpler
and matches how the akshara LM is already counted; start there.

---

## 3. Exact anytime decoding — A* over the lattice

**Claim to establish:** provably exact k-best decoding at IME latency.

Beam search has no quality guarantee; A* with an **admissible** heuristic
does. For node u = (roman position i, LM state), define

$$
h(u) = (m - i) \cdot w_{\min}, \qquad w_{\min} = \text{minimum lattice edge weight},
$$

which never overestimates (each remaining akshara costs ≥ min edge weight).
A* then expands nodes in f = g + h order and the first complete path is
*provably* the global optimum; k-best follows by continuing expansion.
Because edge weights are bounded below (max_weight cap) and the lattice is
a DAG of width ≤ L per position, the heuristic is cheap and tight enough
that expansions stay near the greedy path on easy inputs — worst case is
the exact algorithm's own cost.

**Deliverable:** a decoder mode with the invariant "output = exact k-best",
benchmarked against beam-256 quality at competitive latency. Also the
 principled answer to the beam-width tuning we needed for SOTA.

---

## 4. The entropy harness — measuring the information budget

Formalize the decomposition in mathematics doc §8 into a tool:

1. For each test case, compute the ambiguity set A(R) ∩ Words (from the
   lattice ∩ word-trie — already implemented).
2. Bucket each miss: matra-only / schwa-only / nasal / conjunct / coverage
   (existing taxonomy).
3. Estimate the conditional entropy H(D | R, layer) as each evidence layer
   is added: none → word frequencies → previous word (bigram) → full
   sentence. Empirical estimator: plug-in entropy over the renormalized
   candidate posteriors per layer.
4. Output: bits required per error class per layer — the *budget* each
   feature must spend, and the theoretical ceiling of each.

This is the paper's analysis section: it explains, with numbers, why
isolated-word accuracy plateaus (~80%) and where sentence-level accuracy's
headroom lives.

---

## 5. Incremental decoding — keystroke-latency engineering

An IME decodes prefixes: keystroke t+1 extends the prefix decoded at t. Cache
the beam at every position; each keystroke expands only the new final edges
(~L·K edges) instead of re-running the full lattice. Combined with:

- beam 128 (loses 0.29 pts vs 256, half the cost) as the immediate stopgap;
- trie-quantized weights (M2) for cache-friendly scoring;

target: SOTA-quality suggestions at < 2 ms per keystroke, including the
vocabulary rescoring, on low-resource devices.

---

## 6. Sequencing

| Step | Output | Unlocks |
|---|---|---|
| 1. E6 context + sentence harness | DONE: +0.31–0.40 sentence-level | real-typing value grows with corpus scale |
| 2. Incremental decoding + beam 128 | < 2 ms/keystroke at SOTA quality | real-device IME, low-resource |
| 3. Entropy harness | information budget per layer | paper analysis section |
| 4. v3 CTW estimator | parameter-free, ~4MB, regret bound | paper core theorem |
| 5. Exact A* decoding | provable exactness | paper decoding section |

Steps 1–2 ship product value; 3–5 produce the publication
("Parameter-Free Universal Transliteration: Context-Tree Weighting over
Alignment Lattices"). All build on infrastructure that already exists.
