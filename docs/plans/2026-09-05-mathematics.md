# AksharIME — Complete Mathematical Treatment

Date: 2026-09-05
Status: Reference document. Companion: `2026-09-05-research-agenda.md`.
All equations match the implemented code (file references given).

---

## 1. Problem formulation and notation

- $R = r_1 r_2 \dots r_m$ — roman input string (bytes of `[a-z]`).
- $D = a_1 a_2 \dots a_n$ — Devanagari word, an **akshara** (syllable) sequence.
- An **alignment** $\sigma$ assigns each akshara $a_j$ a contiguous chunk
  $s_j = r_{i_{j-1}+1} \dots r_{i_j}$ of length $l_j \in [1, L]$, $L = 5$
  (`MAX_CHUNK = 5`), or the empty chunk (for aksharas whose sound the roman
  spelling omits).
- Alignment space: $\Sigma(R, D) = \{\sigma : s_1 s_2 \dots s_n = R\}$.

The engine solves $\arg\max_D P(D \mid R)$ over all $D$ whose akshara
sequence can consume $R$. This is exact shortest-path inference over a
weighted DAG; the statistical content is entirely in the edge weights.

---

## 2. The source-channel model

Following Li, Zhang & Su (ACL 2004), with Bayes:

$$
P(D \mid R) = \frac{P(R \mid D)\, P(D)}{P(R)}, \qquad P(R) \text{ constant per query}.
$$

**Emission model (trained, `em_trainer.rs`, `translit_model.rs`):**

$$
P(R \mid D) = \sum_{\sigma \in \Sigma(R,D)} \prod_{j=1}^{n} P(s_j \mid a_j).
$$

Each $P(s \mid a)$ is a multinomial over roman chunks per akshara; $L = 5$
because no akshara maps to more than 5 roman letters in the corpus.

**Language model (trained):** $P(D)$ is factorized over the akshara sequence
by a trigram Kneser-Ney LM (§4), with a word-start prior $P(a_1 \mid \#)$.

**Decoder (`decoder.rs`):** build the lattice DAG — node = (roman position,
previous aksharas for LM state), edge = (chunk length, akshara, weight).
Edge weight $w(e) = -\log P(s \mid a)$; path weight
$= \sum \text{emit} + \lambda_{\mathrm{lm}} \sum \text{LM}$.
Because all weights are negative logs and add along paths, the decoder is
**Viterbi shortest-path in the tropical semiring**
$(\mathbb{R}^+ \cup \{\infty\}, \min, +)$:

$$
\mathrm{best}[i][\text{state}] = \min_{\text{incoming edges } e} \big(\mathrm{best}[i - |s_e|][\text{state}(e)] + w(e)\big).
$$

Beam search (width $B = 64 \dots 256$) approximates exact k-best with
per-step truncation; the k-best list feeds the reranker and the vocabulary
layer.

**State-merging subtlety (empirically discovered, §9.1):** beam states may
be merged only when both the *future* (position, LM context) and the
*output identity* (the string spelled so far) coincide. Two paths with
equal $(pos, \text{prev-pair})$ can spell different strings — the chunk
`na` aligns to both aksharas न (na) and ना (naa) — and merging them
silently deletes candidates. The v1/v2 decoders dedup on
$(pos, \text{prev}_2, \text{prev}, \text{path-hash})$, a hash of the full
akshara path, which preserves output identity.

---

## 3. EM training of the emission table

### 3.1 Objective

Maximise the corpus likelihood over unknown alignments:

$$
\mathcal{L} = \prod_{k} \sum_{\sigma \in \Sigma(R_k, D_k)} \prod_{j} P(s_j \mid a_j).
$$

### 3.2 Forward–backward recurrences (`em_trainer.rs::e_step_chunk`)

Define $f[j][i]$ = probability that the first $j$ aksharas consume exactly
$i$ roman characters, and $b[j][i]$ = probability that aksharas $j \dots n$
consume characters $i \dots m$ (1-based over aksharas):

$$
f[0][0] = 1, \qquad f[0][i] = 0 \ (i > 0),
$$

$$
f[j][i] = \sum_{l=1}^{\min(L, i)} f[j-1][i-l] \cdot P\big(r_{i-l+1..i} \mid a_j\big),
$$

$$
b[n][m] = 1, \qquad b[j][i] = \sum_{l=1}^{\min(L, m-i)} P\big(r_{i+1..i+l} \mid a_j\big) \cdot b[j+1][i+l],
$$

$$
Z = f[n][m] \quad \text{(total likelihood of the word)}.
$$

**Posterior** of a specific edge (akshara $j$, chunk of length $l$ ending at
position $i$):

$$
\mathrm{post}(j, l, i) = \frac{f[j-1][i-l] \cdot P\big(r_{i-l+1..i} \mid a_j\big) \cdot b[j][i]}{Z} \;\le\; 1.
$$

Each posterior is a conditional probability over alignments; the corpus
counts are its accumulated expectation.

### 3.3 M-step with Dirichlet smoothing

$$
P_{\mathrm{new}}(s \mid a) = \frac{c(a, s) + \alpha \cdot P_{\mathrm{uni}}(s)}{\sum_{s'} c(a, s') + \alpha}, \qquad \alpha = 0.05,
$$

where $P_{\mathrm{uni}}(s)$ is the global chunk frequency.

### 3.4 Numerical stability (discovered empirically — v2 debugging)

For words the model assigns near-zero total probability, $Z$ can reach
$10^{-14} \dots 10^{-300}$. Then $1/Z$ explodes and every posterior
$f \cdot p \cdot b / Z$, though bounded by 1 *in exact arithmetic*, can
exceed 1 in floating point — one degenerate word injected ${\sim}10^{14}$
fake posterior mass into the pair statistics and collapsed accuracy from
18% to 6%. Guards now enforced:

$$
\text{skip word if } Z < 10^{-12} \text{ or } Z \notin \mathbb{R}_{\mathrm{finite}}; \qquad \mathrm{post} := \min(\mathrm{post}, 1).
$$

Lesson: forward–backward over learned tables **must** be defended against
its own degenerate words; the invariant $\mathrm{post} \le 1$ holds only in
exact arithmetic.

### 3.5 Observation weights (E5)

A training pair with weight $w$ (e.g. corpus frequency of a synthetic pair)
scales its LM counts by $w$ and every posterior by $w$:

$$
c(a, s) \mathrel{+}= w \cdot \mathrm{post}(j, l, i).
$$

Weighted EM with integer weights $w$ is *equivalent* to replicating the
pair $w$ times (verified empirically: identical 80.69% either way), but
stores one row instead of $w$.

---

## 4. Kneser-Ney akshara LM (`em_trainer.rs::build_kn_lm`)

Counted **deterministically** from segmented corpus words (not EM
posteriors — this density is why the akshara LM outperformed the pair
grammar, §5.4):

$$
P_{\mathrm{KN}}(b \mid a) = \frac{\max\big(c(a,b) - \delta,\ 0\big)}{c(a)} + \lambda(a) \cdot P_{\mathrm{cont}}(b),
$$

$$
\lambda(a) = \frac{\delta \cdot |\{b : c(a,b) > 0\}|}{c(a)}, \qquad
P_{\mathrm{cont}}(b) = \frac{|\{a : c(a,b) > 0\}| + 0.5}{|\Sigma_{\mathrm{bigram}}| + 0.5N},
$$

with **modified Kneser-Ney discounts** (Chen & Goodman 1999) rather than a single
$\delta$: $D_1$, $D_2$, $D_{3+}$ are estimated from the count-of-counts $n_1..n_4$
via $Y = n_1/(n_1 + 2n_2)$, and $\lambda$ becomes $\sum_b D_{c(a,b)} / c(a)$.
Each $D_i$ is constrained to $0 \le D_i \le i$ (`em_trainer.rs::modified_discounts`);
the fixed $\delta = 0.75$ survives only as the fallback when $n_1$, $n_2$ or $n_3$
is zero. Plus the word-start prior

$$
P(a \mid \#) = \frac{c_{\#}(a) + 0.5}{W + 0.5N}, \qquad W = \text{corpus word count}.
$$

All stored as $-\log$ weights (f32). Interpretation of the continuation
count: it measures *how many distinct left-contexts* $b$ appears in — the
correct KN estimate of a bigram's probability mass when the lower order is
unobserved. This is why the KN akshara LM punishes junk akshara sequences
(never observed in any context) far more than raw-count smoothing would.

---

## 5. The v2 pair grammar — and why naive fusion fails

### 5.1 Definition

v2 replaces independent edges with a context-dependent transition over
**aligned pairs** $\pi_j = (a_j, s_j)$:

$$
P(\pi_1 \dots \pi_n) = P_{\#}(\pi_1) \prod_{j \ge 2} P(\pi_j \mid \pi_{j-1}),
$$

with a 3-level backoff hierarchy: pair-bigram → (prev akshara, cur pair) →
joint pair unigram $P(a, s) = P(a) \cdot P(s \mid a)$.

### 5.2 The joint-unigram requirement

Backoff to $P(s \mid a)$ alone is wrong: an akshara with a single observed
chunk gets $P(s \mid a) \approx 1$, so *rare aksharas with peaked emissions
cost nothing* and junk paths win. The backoff target must be the joint
$P(a) \cdot P(s \mid a)$, keeping rare aksharas expensive. (Measured:
6.5% → 18.4% top-1.)

### 5.3 Transition collection (weighted forward–backward)

Posterior of a consecutive pair transition (aksharas $j-1 \to j$, chunk
split at position $i_1$):

$$
\mathrm{post} = \frac{f[j-1][i_1 - l_1] \cdot P(s_{j-1} \mid a_{j-1}) \cdot P(s_j \mid a_j) \cdot b[j+1][i_1 + l_2]}{Z}.
$$

Note the backward index: the suffix must start **after** the current pair
(1-based $b[j+1]$; boundary case $i_1 + l_2 = m$ ⇒ suffix $= 1$). An
off-by-one here ($b[j]$) makes every transition posterior ~0 and only noise
survives the count threshold — the second fatal v2 bug.

### 5.4 The dilution phenomenon (why v2 < v1)

The pair bigram is trained on **EM posteriors**, which concentrate on the
*most probable alignment per corpus pair* — and the corpus prefers frequent
misalignments: $P(\text{na} \mid \text{ना}) \approx 0.97$ versus
$P(\text{na} \mid \text{न}) \approx 0.35$, so word-initial "nam…" masses
onto the ना+मा alignment even for words whose gold aksharas are न+म. The
pair table therefore encodes the corpus's *alignment bias*, not its word
structure. Meanwhile the akshara trigram LM is counted deterministically
from gold segmentations — dense and unbiased.

**Theorem-shaped lesson:** for a grammar over latent alignments,
transition statistics inherited from the alignment posterior are biased
toward alignment-frequency, not word-frequency. Fixing this requires either
(a) word-level constraints (the $T \circ L \circ G$ layer, §6), or (b) a
single estimator that treats context depth as latent (the CTW design,
research agenda §2) — *not* additive fusion, which injects the bias with
any positive weight (measured: every $\lambda_{\mathrm{pair}} > 0$ lowered
top-1).

---

## 6. The vocabulary layer — rescoring as MAP estimation

The engine ranks the decoder's k-best list with a corpus-frequency term:

$$
\mathrm{score}'(v) = \mathrm{score}(v) - \lambda \cdot \ln\big(1 + f(v)\big), \qquad v \in \text{k-best},
$$

where $f(v)$ is the frequency of candidate $v$ in running text. This is a
log-linear opinion mixture: the decoder supplies $\ln P(v \mid R)$ (grammar
evidence) and the corpus supplies $\ln P(v)$ (prior evidence); the sum is a
MAP estimate under a prior interpolated between uniform and the corpus
distribution.

Measured on native words ($n = 2108$):

| k-best depth | baseline | with vocabulary |
|---|---|---|
| 8 | 75.33% | 78.84% |
| 50 | — | **80.46%** (beam 256, $\lambda_{\mathrm{lm}} = 0.85$, $\lambda = 0.75$) |

Two mathematics-level findings:

1. **Depth interaction**: the prior can only re-rank candidates the decoder
   emitted. The oracle at $k=50$ is 93.55%; at $k=8$ it is ${\sim}87\%$. The
   vocabulary layer's value is *bounded by candidate depth* — pipeline
   stages must be tuned jointly (a stage-wise-optimal configuration was
   stage-wise suboptimal: 78.84% at $k=8$ vs 80.46% at $k=50$).
2. **Frequency saturation**: Aksharantar's native side has 2.4M unique
   words over 2.4M tokens — its frequency distribution is nearly uniform,
   hence useless as a prior (measured flat). Real running text (75M tokens,
   570k distinct words, Zipf-distributed) is the informative prior.
   Word-pair corpora cannot substitute for running text.

---

## 7. Self-training — backward romanization (E5)

The engine's inverse map Devanagari → roman is *deterministic given the
akshara segmentation*:

$$
\mathrm{romanize}(W) = \prod_{a \in \mathrm{segment}(W)} \arg\max_{s} P(s \mid a),
$$

the emission table's own argmax per akshara. Since the segmentation of $W$
is exact and the per-akshara argmax is the most probable chunk ever
observed for it, the synthetic pair $(\mathrm{romanize}(W), W)$ is the
canonical spelling of $W$ *under the model's own statistics*.

**Validated variants:** runner-up chunks (2nd/3rd argmax) are spellings
real users produced; combining per-akshara alternatives (capped at 8 joint
spellings) yields variant pairs whose akshara alignment is invariant by
construction (noise applies to whole akshara tokens only). Variants are
down-weighted by joint probability × ¼.

Result: +0.23 (80.46 → 80.69), saturating at one pair per word — the
canonical signal dominates; variants add robustness, not benchmark accuracy.

---

## 8. Empirical error decomposition (the information budget)

### 8.1 Oracle curves (native words, $n = 2108$)

$$
P(\text{gold} \in \text{top-}k):\quad k{=}1\!:\!75.3,\quad k{=}2\!:\!85.2,\quad k{=}3\!:\!88.0,\quad k{=}5\!:\!89.7,\quad k{=}10\!:\!91.3,\quad k{=}50\!:\!92.1.
$$

The oracle is the empirical upper bound of *any* reranker over the lattice.
The gap (top-1 75.3 → top-50 92.1) is **ranking error**; the gap above
(top-50 92.1 → 100) is **generation error** (coverage).

### 8.2 Error taxonomy

Of top-1 misses: **52% matra-only** (differ from gold by vowel-length or
nasal signs only), 48% substantive. A matra error means both candidates are
real words consistent with the roman string — *the roman string carries
zero bits to separate them*. Formally, define the ambiguity set

$$
A(R) = \{D : P(R \mid D) > \tau\};
$$

for these pairs $|A(R) \cap \text{Words}| \ge 2$ with comparable priors, so
no function of $R$ alone exceeds the prior-weighted chance of picking
correctly.

### 8.3 The information budget

Each external information source shrinks the effective ambiguity set:

| Source | Acts on | Measured contribution |
|---|---|---|
| Grammar + LM (string only) | $A(R)$ ranking | 75.33% |
| Word frequency (corpus prior) | $A(R) \cap \text{Words}$, weighted | 80.46% |
| Word bigram context (previous word) | conditional prior $P(w_i \mid w_{i-1})$ | projected +2–4 (E6, pending) |
| User history (on-device) | personal $P(w)$ | unmeasured, compounding |

This is the quantitative form of the claim: isolated-word accuracy is
bounded by the string's entropy; sentence-level accuracy is not. The
measured plateau at ${\sim}79$–$80.7\%$ *is* the empirical entropy of
romanized input under this grammar.

---

## 9. Numerical-robustness ledger (all discovered the hard way)

| # | Failure mode | Symptom | Guard |
|---|---|---|---|
| 1 | Beam state merge ignores output identity | correct candidates never generated (न vs ना share pair context) | dedup key includes path hash |
| 2 | Per-akshara backoff target | rare aksharas with peaked emissions free → junk wins | joint $P(a) \cdot P(s \mid a)$ |
| 3 | Backward index off-by-one in transition collection | transition posteriors ~0; only noise counted | suffix index $b[j{+}1]$, boundary at $m$ |
| 4 | FP poisoning from degenerate words | posterior mass $2 \times 10^{14}$; counts destroyed | skip $Z < 10^{-12}$; clamp post ≤ 1 |
| 5 | Additive smoothing on large alphabets | single observation ≈ certainty ($P \approx 0.67$ at $C{=}1$) | absolute-discount KN, never additive |
| 6 | u64 score quantization | tie classes ordered by HashMap iteration | score resolution ×1000 |
| 7 | SQLite concurrent writers | "database is locked" kills crawlers | sequential writers (or WAL + busy_timeout) |

---

## 10. Source index

| Component | Code | Doc section |
|---|---|---|
| EM trainer, forward–backward | `src/core/em_trainer.rs` | §3 |
| KN LM | `src/core/em_trainer.rs::build_kn_lm` | §4 |
| v1 decoder (lattice, beam) | `src/core/decoder.rs` | §2 |
| v2 pair grammar | `src/core/v2/mod.rs` | §5 |
| Reranker + vocabulary scoring | `src/core/reranker.rs`, `src/core/engine.rs` | §6 |
| Backward romanizer | `src/bin/romanize.rs` | §7 |
| Error analysis | `src/bin/analyze_errors.rs` | §8 |
| Measurements | `2026-09-03-accuracy-experiments.md` | throughout |
