# Beyond a linear blend: non-neural reranking and decoding on n-best lists

Research note for `akshar-ime`: a beam-search decoder producing n-best lists,
rescored by a linear model (29 dense + hashed sparse features, gamma blend with
a frequency heuristic). Question: what reranking/decoding advances exist
**beyond a linear blend** that still operate on n-best lists **without neural
networks**?

Scope: inference budget is sub-millisecond per query, no GPU, no tensors.
Ranked below by fit for this pipeline. Every entry gives the paper citation,
the core idea in a few sentences, and what would need to change in our pipeline
to try it. No code or data files were modified.

## Where we stand (one paragraph)

Our trainer already implements the Collins (2000) family objective: softmax
cross-entropy over the candidate list with AdaGrad over sparse+dense weights
(`src/bin/train/train.rs`), decoding each training pair once per batch and
keeping best-by-dev-loss weights. The MANUAL (§7) is honest about the weak
point: the learned model alone is 5.36pp worse than the 3-parameter heuristic,
and gamma = 0.3 is the point where a weak model stops doing damage. The error
analysis (§9.7) says the headroom is ranking, not generation (oracle@50 94.3%
vs top-1 81.8% on AK-Freq), 8 of 12.5pp is a top-2 choice, and 51.9% of native
misses are matra-only. So techniques that (a) change the decision rule at fixed
weights, or (b) train the linear scorer against ranking loss instead of
likelihood, attack the measured gap directly. Techniques needing new search
infrastructure rank lower.

## Ranked techniques

### 1. Minimum Bayes Risk (consensus) decoding — best fit, inference-only

**Cite:** Shankar Kumar and William Byrne, "Minimum Bayes-Risk Decoding for
Statistical Machine Translation," HLT-NAACL 2004. Follow-up for compact
representations: Roy Tromble et al., "Lattice Minimum Bayes-Risk Decoding for
Statistical Machine Translation," EMNLP 2008.

**Core idea.** MAP decoding picks the single highest-scoring candidate, but the
evaluation metric (exact match, BLEU, CER) is a loss against an unknown
reference. MBR instead picks the candidate minimizing *expected* loss under the
model's own posterior: score each candidate against every other candidate as a
pseudo-reference, weight by posterior probability, and select the consensus.
Kumar & Byrne show this on N-best lists improves translation quality under
BLEU-like losses; Tromble et al. extend it to lattices and characterize which
loss functions admit efficient computation. Intuitively it prefers the candidate
that agrees most with the rest of the high-probability mass — a robust center
rather than a sharp peak.

**Why it fits here.** Zero training change: it is a decision rule applied to the
existing n-best list with existing scores. Cost is O(k^2) string comparisons
(k ~ 24–50), each a cheap edit/CER computation — microseconds, inside budget.
It directly targets our two measured structures: the top-2 coin flip (consensus
breaks ties toward the candidate sharing matras/stems with the field) and the
51.9% matra-only misses (a consensus over character n-grams favors the majority
vowel-sign choice). Sentence-level BLEU approximations in Tromble et al. map
naturally to per-query CER/exact-match gain here.

**What would need to change.** Add a post-rerank selection step in
`reranker.rs`/`engine.rs`: after gamma-blend scoring, compute posterior
weights (softmax over final scores with a temperature tuned on dev), compute
pairwise loss (start with character-level edit distance or 1 − exact match;
optionally matra-insensitive loss to mirror the error taxonomy), pick the
min-risk candidate for position 1 while leaving ranks 2–5 in model order.
Tunable: temperature, loss function, whether MBR applies to top-10 only. Risk:
if the n-best posterior is flat or miscalibrated, consensus can herd toward a
popular wrong form; evaluate with McNemar per MANUAL §9.

### 2. Pairwise / listwise ranking losses ("tuning as ranking") — best training fit

**Cite:** Mark Hopkins and Jonathan May, "Tuning as Ranking," EMNLP 2011
(pairwise RankSVM-style tuning for MT). Background: Thorsten Joachims,
"Optimizing Search Engines Using Clickthrough Data," KDD 2002 (RankSVM);
Libin Shen, Anoop Sarkar, and Franz Och, "Discriminative Reranking for Machine
Translation," HLT-NAACL 2004 (perceptron ranking on n-best). Survey framing:
Tie-Yan Liu, "Learning to Rank for Information Retrieval" (pairwise/listwise
taxonomy).

**Core idea.** Softmax cross-entropy trains the scorer to put all mass on the
gold candidate — a harder task than needed when only the top-1 order matters.
Pairwise approaches instead train on preferences (gold should outscore each
rival, or every correctly-ranked pair), reducing ranking to binary
classification on score differences; listwise variants optimize the whole
ordering. Hopkins & May show a pairwise ranking formulation of MT tuning is
simple, scalable, and competitive with MERT while handling far more features.
Shen et al. apply perceptron-style ranking directly to MT n-best lists. The
model stays linear; only the objective changes.

**Why it fits here.** Our loss/headroom mismatch is exact: we train likelihood
but evaluate top-1 exact match, and 8pp of headroom is a pairwise top-2
decision. A pairwise hinge loss on (gold, best-rival) or all-pairs within the
list optimizes precisely the decision that loses those points. Still a dot
product at inference — zero runtime cost change. Works with the hashed sparse
table as-is.

**What would need to change.** In `src/bin/train/train.rs`, replace or augment
the softmax CE batch update with a pairwise hinge/Margin update over
score differences (feature-difference vectors, so the existing AdaGrad
machinery transfers almost unchanged). Start with gold-vs-top-rival pairs,
then all-pairs with margin scaled by loss (e.g. CER between rival and gold —
this shades into structured ramp loss, Gimpel & Smith 2012). Keep best-by-dev
selection but select on dev top-1/MRR, not dev CE loss. Validate with
`train-mid` (never `train-quick`) per AGENTS.md, since chunked mode only
engages above 200k pairs.

### 3. k-best MIRA (online large-margin with per-example adaptation)

**Cite:** Taro Watanabe et al., "Online Large-Margin Training for Statistical
Machine Translation," EMNLP-CoNLL 2007. Extensions: Colin Cherry and George
Foster, "Batch Tuning Strategies for Statistical Machine Translation,"
NAACL 2012 (batch lattice and k-best MIRA for thousands of features);
David Chiang, Yuval Marton, and Philip Resnik, "Online Large-Margin Training
of Syntactic Parsers" / Chiang et al. 2008–2009 forest/k-best MIRA line.
Base algorithm: Koby Crammer and Yoram Singer, "Ultraconservative Online
Algorithms for Multiclass Problems," JMLR 2003 (MIRA family).

**Core idea.** MIRA is an online large-margin update: on each training
instance, adjust weights minimally subject to ranking the gold (or hope/fear
derivations) above competitors by a margin proportional to the loss. Unlike the
perceptron it adapts the step size per example from the margin violation, and
unlike MERT it scales to large sparse feature sets. Watanabe et al. apply it to
k-best MT reranking with loss-scaled margins; Cherry & Foster make it batch and
stable for thousands of features. It keeps the model linear and the update
sparse.

**Why it fits here.** Our sparse table (~2^20 hashed slots, AdaGrad CE) is
exactly the regime k-best MIRA was built for, and the per-example margin
adapts naturally to lists where the gold is at rank 1 (no update) vs rank 30
(large update). Loss-scaled margins let matra-only rivals incur small margins
and substantive rivals large ones — a finer instrument than 0/1 perceptron.
Inference unchanged (dot product).

**What would need to change.** Add a MIRA update path in the trainer alongside
AdaGrad: maintain the same sparse table + dense weights, compute k-best scores
per sample, find the max-violating rival under loss-scaled margin, apply the
closed-form MIRA step with clipping (C parameter tuned on dev). Compare
directly against AdaGrad-CE and pairwise-hinge under identical decoding (same
`--reranker-pairs`, same best-by-dev protocol). Watch for the known MIRA
instability on noisy lists — Cherry & Foster's averaging/regularization is the
remedy to port if vanilla MIRA oscillates.

### 4. MERT for the small dense+gamma stage (not for sparse)

**Cite:** Franz Josef Och, "Minimum Error Rate Training in Statistical Machine
Translation," ACL 2003. Lattice extension: Wolfgang Macherey et al.,
"Lattice-Based Minimum Error Rate Training for Statistical Machine
Translation," EMNLP 2008. Stabilization: George Foster and Roland Kuhn,
"Stabilizing Minimum Error Rate Training," WMT 2009; Daniel Cer et al.,
"Regularization and Search for Minimum Error Rate Training," WMT 2008.

**Core idea.** MERT directly optimizes the evaluation metric (error count) over
a small weight vector using line searches: for each weight in turn, the
piecewise-constant error surface along that direction changes only where the
top-1 ranking flips, so the exact optimum along the line is found by sweeping
intervals between flip points. Och shows this tunes a ~8-feature log-linear MT
model to BLEU far better than likelihood training. It does not scale to large
feature sets (each iteration needs full n-best rescoring per dimension), hence
the lattice extensions and the regularization literature around its
instability.

**Why it fits here.** We have a genuinely small dense stage begging for direct
metric optimization: 29 dense weights + gamma (+ LM_W, VOCAB_W heuristic
constants) controlling top-1 exact match. MERT on this ~30-dimensional vector
against dev top-1 is textbook-applicable and cheap (rescoring cached n-best
lists, no re-decoding per step). It does *not* apply to the 2^20 sparse table —
that stays with perceptron/MIRA/pairwise methods.

**What would need to change.** Freeze sparse weights, dump dev-set n-best lists
with dense features once, run coordinate-wise MERT line searches on the dense
vector + gamma against top-1 accuracy (or 1 − MRR for a smoother signal).
Regularize (Cer et al.) and re-decode once after tuning (Macherey-style outer
loop: n-best lists go stale as weights move). Honest precondition from MANUAL
§12.3: the dense weights are frozen/stale — MERT-tuning them without first
re-fitting is tuning a decalibrated stage; consider joint dense retraining
(§2–3) before or alongside MERT.

### 5. Discriminative language modeling (perceptron/CRF n-gram reranking)

**Cite:** Brian Roark, Murat Saraçlar, and Michael Collins, "Discriminative
Language Modeling with Conditional Random Fields and the Perceptron
Algorithm," ACL 2004; expanded as "Discriminative n-gram Language Modeling,"
Computer Speech and Language 2007 (Roark, Saraçlar, Collins, and Johnson).
Related: Michael Collins et al., "Discriminative Syntactic Language Modeling
for Speech Recognition," ACL 2005.

**Core idea.** A generative n-gram LM is trained to predict text; a
*discriminative* LM is trained to rank the correct hypothesis above its
n-best rivals. Roark et al. train perceptron and CRF n-gram models whose
features are word/n-gram occurrences in the candidate, on ASR n-best lists,
and show WER reductions a generative LM cannot reach because the features
penalize confusable alternatives directly. Training is online and linear;
inference is a dot product over n-gram features present in each candidate.

**Why it fits here.** Our KN akshara LM is the largest single accuracy
contributor (+3.89pp), yet it is purely generative. A discriminative unigram/
bigram layer over aksharas (or over the 34 morphological suffixes × contexts)
scored on the n-best list is the same inference shape as our sparse table but
with a principled feature class (all n-grams, perceptron-trained) rather than
7 hand templates. Cost at inference: hash lookups per candidate n-gram,
same order as current sparse extraction.

**What would need to change.** Add an akshara-n-gram template family (unigrams
+ bigrams, possibly skip-grams over matra positions given the matra-only error
mass) to the hashed sparse table, or a second table trained with averaged
perceptron alongside AdaGrad CE. Feature count explodes (all bigrams in the
list), so cap by frequency/support and rely on the existing dev-loss early
stopping against overfitting. Ablate with `AKSHAR_NO_SPARSE`-style switches to
separate the new family's contribution from the 7 hand templates.

### 6. Joint MaxEnt transliteration with chunking + bilingual context (Goto et al.)

**Cite:** Isao Goto, Naoto Kato, Noriyoshi Uratani, and Terumasa Ehara,
"Transliteration Considering Context Information Based on the Maximum Entropy
Method," ACL 2003 (local PDF: `papers_to_read/goto-etal-2003-maxent-transliteration.pdf`).
Verified from the local copy: MaxEnt model over English→katakana conversion
units that jointly scores letter-chunking plausibility and source/target
context, +63% over a context-free baseline.

**Core idea.** Rather than separating segmentation from conversion, Goto et al.
train a single maximum-entropy (log-linear) model whose features fire on the
chosen chunking *and* on source-side and target-side context of each unit. At
decode time every segmentation/conversion hypothesis is scored by the same
model, so chunk validity and contextual fit trade off inside one distribution.
The paper's ablation shows both halves matter: chunking plausibility and
context each carry accuracy.

**Why it fits here.** Our pipeline separates these concerns across stages: EM
emissions + KN LM propose, 7 sparse templates patch (final/first akshara ×
Roman char are exactly Goto-style bilingual-context features). The paper
suggests named, high-value additions in our idiom: chunking-validity features
(length-delta bucket exists; add per-chunk emission-rank/entropy features) and
*source-context* features (Roman bigrams around each boundary × proposed
akshara), which our templates currently lack — all our lexicalized templates
are target-side or endpoint-anchored.

**What would need to change.** Inference shape unchanged (more hashed
templates). Add: (a) Roman-context × akshara features (preceding/following
Roman bigram × current akshara, hashed); (b) segmentation-quality features
(mean/max emission rank consumed, number of rare-chunk edges); (c) internal
(not just endpoint) akshara × Roman-chunk conjunctions, at least for the
matra-bearing aksharas behind half our errors. Train with the existing
softmax/AdaGrad path; the Goto paper's lesson is about *which* features earn
their slots, testable by template ablation.

### 7. Two-system hypothesis rescoring by score interpolation (Finch & Sumita)

**Cite:** Andrew Finch and Eiichiro Sumita, "Transliteration Using a
Phrase-Based Statistical Machine Translation System to Re-score the Output of
a Joint Multigram Model," NEWS 2010 (local PDF:
`papers_to_read/finch-sumita-2010-multigram-smt-rescoring.pdf`). Verified from
the local copy: joint-multigram model generates the n-best list, phrase-based
SMT models rescore it, linearly interpolated scores beat *both* components
substantially on development data.

**Core idea.** Two systems with complementary inductive biases — a joint
multigram (source-target co-segmentation history) and a phrase-based SMT model
(phrase-table + LM scores) — each score the same n-best list, and a linear
interpolation of their scores reranks better than either alone. Neither system
needs to generate the other's hypotheses; rescoring is pure feature
computation per hypothesis. The paper's claim is precisely that rescoring with
*foreign* model scores (information unavailable to the generator) is where the
gain comes from.

**Why it fits here.** We already run two heterogeneous generators (free beam +
trie-constrained pass) but merge by dedup union, not by score interpolation —
each candidate carries only its own pass's scores. The free pass's LM/emission
decomposition evaluated on trie-pass candidates (and vice versa: word-frequency
priors evaluated on free-pass candidates) is the Finch & Sumita move available
for free. More ambitiously, a third lightweight scorer (e.g. a character-level
phrase-table-style model over Roman↔akshara co-segments, or the morph-stem
prior as an independent judge) adds a genuinely foreign score per hypothesis.

**What would need to change.** First, minimal: in `decode_union`, score every
merged candidate under *both* passes' models (emit/LM under the free model are
already decomposed; add trie-pass membership + word frequency as features for
free-pass candidates rather than relying on union recall). Tune the
interpolation weight on dev (this is a second gamma, MERT-tunable per §4).
Second, optional: train a small auxiliary scorer (joint-multigram-style phrase
table over the same 3.59M pairs — fast, non-neural) whose per-hypothesis score
becomes one more dense feature. Keep the cascade: foreign scores compute only
for the top-24 heuristic block.

### 8. Forest reranking (non-local features without the n-best bottleneck)

**Cite:** Liang Huang, "Forest Reranking: Discriminative Parsing with Non-Local
Features," ACL-HLT 2008. Algorithmic basis: Liang Huang and David Chiang,
"Better k-best Parsing," IWPT 2005 (exact k-best over hypergraphs / packed
forests).

**Core idea.** N-best reranking can only choose among listed hypotheses, and
widening the list has diminishing returns; forest reranking instead scores a
*packed forest* (hypergraph) of exponentially many derivations with arbitrary
non-local features (features over whole trees, not just local rules) using
approximate search (e.g. perceptron-guided beam over the forest). Huang shows
forest reranking beating 50- and 100-best reranking baselines on parsing,
precisely because the reranker explores hypotheses the k-best list never
contained.

**Why it fits here — partially.** Our oracle@50 is 94.3%: generation
covers nearly everything, so the n-best bottleneck Huang attacks is mostly
absent at k=50 — but our reranker only *sees* the top-24 cascade block, and
the cascade is a second bottleneck. The transferable idea is rescoring a
denser structure (the beam's full 50, or the lattice) with non-local features
(whole-word shape: total matra profile, suffix well-formedness) rather than
deeper lists.

**What would need to change.** Substantial: our beam discards the lattice after
decoding (only top-k strings survive), so forest-style rescoring needs the
lattice or full beam retained past decoding, plus a rescoring search over it.
That breaks the current clean decoder→reranker string interface and adds
latency risk against a 0.63–0.82 ms budget. Ranked below §§1–7 accordingly:
consider only if §§1–3 saturate, and first test the cheap version — widen
rerank depth 24→50 with non-local features (already measured: no gain at
depth 50, MANUAL §7.5), which suggests the forest machinery would buy little
here.

### 9. Exact k-best / A* decoding (Eppstein; Huang–Chiang; A* parsing lineage)

**Cite:** David Eppstein, "Finding the k Shortest Paths," SIAM J. Computing
1998 (local PDF: `papers_to_read/eppstein-1998-k-shortest-paths.pdf`;
O(m + n log n + k) implicit representation). Instantiation for language:
Huang & Chiang 2005 (above). A* search lineage for exact decoding with
admissible heuristics over lattices/hypergraphs (e.g. Klein & Manning 2001–2003
A* parsing; Pauls & Klein 2009 hierarchical A*).

**Core idea.** Beam search is approximate: it can drop the gold path early and
no reranker can recover it (our 1 − oracle@50 = 5.7% generation loss). Exact
k-best algorithms enumerate the true k shortest paths of a graph: Eppstein's
gives an implicit representation in near-linear time plus O(1) amortized per
path, and A* variants use outside/admissible heuristics to make exact search
tractable. For our lattice — a DAG over Roman positions, strictly
forward-moving — exact k-best is simpler than the general cyclic case Eppstein
solves (DAG k-shortest-paths by dynamic programming + lazy enumeration).

**Why it fits here — narrowly.** This does not improve *ranking* (the 12.5pp
gap); it converts at most the 5.7% never-generated tail into rankable
candidates, and only if the reranker then ranks them first. Latency is the
binding constraint: exact enumeration over a dense lattice (16 aksharas ×
chunk positions) risks exceeding the per-keystroke budget that the O(beam)
approximate path meets comfortably.

**What would need to change.** Replace `decode_detailed`'s beam with exact
DAG k-best enumeration (NOT the full Eppstein digraph machinery — positions
form a topological order, so a recursize/lazy k-way-merge per node suffices),
keeping the emit/LM decomposition per path for the reranker. Measure oracle@k
lift vs latency at k=10/24/50 before touching the reranker. Honest expectation:
small; the MANUAL's roadmap already concludes the gap is ranking. Ranked here
for completeness and as the principled fix if generation loss ever dominates
(e.g. on named entities, where oracle coverage is worse: 85.1% @50 pooled).

### 10. System combination by alignment and voting (ROVER / confusion networks)

**Cite:** Jonathan Fiscus, "A Post-Processing System to Yield Reduced Word
Error Rates: Recognizer Output Voting Error Reduction (ROVER)," ASRU 1997.
Word-level consensus generalization: Lidia Mangu et al., "Finding Consensus in
Speech Recognition: Word Error Minimization and Other Applications of
Confusion Networks," Computer Speech and Language 2000. MT combination line:
Rosti et al. / Matusov et al. (FST- and confusion-network-based MT system
combination, 2007–2008); Karimi et al. 2011 survey (transliteration context,
local PDF `papers_to_read/karimi-etal-2011-survey.pdf` for the task framing).

**Core idea.** Given outputs of *multiple independent systems*, align them
(dynamic programming / confusion network) and vote per slot, weighting votes by
system confidences. ROVER and confusion-network combination routinely cut WER
because independent systems make independent errors; the consensus is better
than any member. The requirement is diversity: combining copies of one system
gains nothing.

**Why it fits here — weakly.** Our candidate sources (free beam, trie pass,
fuzzy/user trie, variant rewrites) are not independent systems — they share
the EM/LM core — so classic ROVER gains should not be expected. The
narrowly-applicable residue: our errors are 51.9% matra-only, i.e. candidates
differ by single slots, which is exactly the regime where per-slot voting over
an *aligned* n-best list (character confusion network built from our own
k-best, à la Mangu et al.) can synthesize a winner that is a slot-wise
majority — closely related to MBR §1 but operating below string level, so it
can *construct* a string no decoder path produced.

**What would need to change.** Build a character/akshara-level confusion
network from the top-k list per query (multiple alignment to the 1-best,
vote per column with posterior weights), constrained so the output stays a
valid akshara sequence (project columns through the akshara segmenter).
Compare against MBR §1 head-to-head: if slot-voting never beats string-consensus
here, drop it — diversity theory predicts exactly that outcome for
single-engine lists.

## What the local PDFs contributed

- `goto-etal-2003-maxent-transliteration.pdf` (Goto et al. 2003): joint
chunking+context MaxEnt model, §6 above; motivates Roman-context features.
- `finch-sumita-2010-multigram-smt-rescoring.pdf` (Finch & Sumita 2010):
two-system rescoring by interpolation beating both members, §7 above;
motivates scoring merged candidates under both passes.
- `eppstein-1998-k-shortest-paths.pdf` (Eppstein 1998): implicit k-shortest-paths
representation, §9 above; noted as oversized machinery for our DAG lattice.
- `karimi-etal-2011-survey.pdf` (Karimi et al. 2011): transliteration task
survey framing for §10's combination discussion (not yet mined for feature
ideas — flagged as follow-up reading alongside Knight & Graehl 1998 and
Li, Zhang & Su 2004, already in `papers_to_read/`).
- Not present locally and read via web/secondary sources: Collins 2000/2005,
Och 2003, Kumar & Byrne 2004, Tromble et al. 2008, Roark et al. 2004/2007,
Fiscus 1997, Huang 2008, Huang & Chiang 2005, Watanabe et al. 2007,
Hopkins & May 2011, Cherry & Foster 2012. Before implementing §§1–4, pull the
primary PDFs (all ACL Anthology open access) and verify the update equations;
the characterizations above are from abstracts, surveys, and citing papers,
not from re-reading each primary.

## Suggested order of experiments

1. **MBR selection (§1)** — no training, days of work, attacks the top-2 gap
directly. If it fails, the posterior is miscalibrated, which itself diagnoses
the reranker.
2. **Pairwise ranking loss (§2)** — same model, same features, better-aligned
objective; needs a `train-mid` validation gate.
3. **k-best MIRA (§3)** — head-to-head with (2); keep the winner.
4. **MERT on dense+gamma (§4)** — after (2)/(3) fix dense-weight staleness, or
as a final polish on the small stage.
5. **Discriminative akshara n-grams (§5) and Goto-style Roman-context templates
(§6)** — feature work, only after the objective is right (better features
under a wrong objective repeat the §7 gamma story).
6. **Finch-style dual scoring of merged candidates (§7)** — small, parallelizable
with (5).
7. **Forest/exact-k-best/ROVER (§§8–10)** — only if generation loss, not ranking
loss, becomes the binding constraint (watch named-entity oracle coverage).

## One-line summary

Decide better with fixed weights first (MBR consensus), then train the same
linear weights against the ranking decision actually being lost (pairwise loss
/ MIRA, MERT for the dense stage); feature and search machinery come last.
