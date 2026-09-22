# SOTA transliteration signals transferable to a non-neural engine

Date: 2026-09-22. Question: what information signals drive state-of-the-art
transliteration accuracy that a **non-neural** (beam-search + linear-reranker)
engine could also use?

Primary sources actually read (local PDFs under `papers_to_read/`, extracted
via `pdftotext`; § citations below refer to those extractions):

- **Madhani et al. 2023** = Madhani et al., "Aksharantar: Open Indic-language
  Transliteration datasets and models for the Next Billion Users", EMNLP 2023
  Findings (`madhani-etal-2023-aksharantar-indicxlit.pdf`; also
  arXiv:2205.03018). The IndicXlit system + unigram rerank.
- **Roark et al. 2020** = Roark et al., "Processing South Asian Languages
  Written in the Latin Script: the Dakshina Dataset", LREC 2020
  (`roark-etal-2020-dakshina.pdf`). Pair n-gram vs LSTM vs transformer
  baselines + noisy-channel sentence decoding.
- **Karimi et al. 2011** = Karimi, Scholer & Turpin, "Machine Transliteration
  Survey", ACM Computing Surveys 43(3) (`karimi-etal-2011-survey.pdf`).
  Pre-neural taxonomy: generative / hybrid / combined (reranking) methods.
- **Hellsten et al. 2017** = Hellsten et al., Google transliteration WFSTs for
  mobile keyboards (`hellsten2017_google_translit_wfst.pdf`). Production
  non-neural reference architecture (pair-LM ∘ lexicon ∘ word-LM, ~10 MB,
  ≤20 ms).
- **Goto et al. 2003** = Goto et al., max-ent transliteration with source+target
  context (`goto-etal-2003-maxent-transliteration.pdf`).
- **Finch & Sumita 2010** = Finch & Sumita, joint-multigram + phrase-SMT
  hypothesis rescoring (`finch-sumita-2010-multigram-smt-rescoring.pdf`).
- **Azam et al. 2025** = Azam et al., "Beyond Specialization: Benchmarking LLMs
  for Transliteration of Indian Languages", arXiv:2505.19851 (abstract only,
  fetched 2026-09-22). Newest data point; contradicts one claim in our archive
  (see §7).

Web search was unavailable in this environment, so "anything newer (2023–2026)"
is covered only by Azam et al. 2025 plus the repo's own archive notes
(`docs/plans/archive/2026-09-03-transliteration-accuracy-research.md`). The
NEWS shared tasks are covered indirectly: no NEWS overview paper was read, but
Finch & Sumita 2010 reports the NEWS metric suite (top-1 accuracy, mean
F-score, MRR, MAPref) and a NEWS-style combination result, and Karimi §5.4–5.5
covers the combination methods and corpus effects that defined the NEWS era.
Anything I state about NEWS beyond that is marked as background, not verified.

Reference point for "fit" notes: this engine per `docs/MANUAL.md` §§5–8 is
EM source-channel over aksharas + modified-KN akshara trigram + trie-union
beam decode + heuristic/linear rerank (29 dense + hashed sparse, γ=0.3 blend,
depth-24 cascade), hand-tuned score bands in `engine.rs`. Current: 80.98%
AK-Freq top-1, 45.3/29.0% NE strata; oracle@2 89.8% native; 51.9% of native
misses matra-only.

## 1. Ranked list of transferable signals

Ordered by (expected gain × ease of fit × strength of source evidence).

### T1. Word-unigram prior with a properly tuned generator weight — the single biggest documented rerank signal

- **What:** rescore beam/transformer n-best with `Fc = α·Tc + (1−α)·Pc`,
  `Tc` = generator (char-level log-prob), `Pc` = word-level unigram LM score,
  **α = 0.9 tuned on dev** (Madhani §5, eq. 1: beam 4, top-4 rescoring).
- **Effect:** "+12% average across languages" (Madhani §6.1); Nepali AK-Freq
  80.25% → **86.6%** after rerank (Table 6); Dakshina micro-avg 61.4 → 72.1
  in the same table. Unigram LM trained on monolingual corpora (IndicCorp
  vocabulary sourcing, §3.3; KenLM/KN tooling used in the project, §4.1).
- **Fit: direct — and already half-built.** The engine's 3-parameter heuristic
  `emit + 0.85·lm − 0.75·log(1+f)` *is* a word-frequency prior and is worth
  +5.64pp (MANUAL §9.1). The SOTA lesson is narrower: (i) the generator keeps
  ~90% of the weight — the prior only breaks near-ties, which matches our
  oracle@2 gap exactly; (ii) α is **tuned on dev**, not hand-set, and tuning
  on the test set is a known validity threat (MANUAL §9.6). Action: retune the
  (emit, lm, freq) weights + γ on the **valid split** by grid/MERT-style
  search instead of the frozen 0.85/0.75/0.3 constants.
- **Caveat (says Madhani explicitly, §6.1):** reranking "mostly benefits the
  native language words and high resource languages"; "limited benefits for
  named entities… since named entities might not be well represented in the
  LM". So T1 is a native-word signal; applying it blindly to NEs is
  documented to do little (§4 below handles NEs separately).

### T2. Noisy-channel sentence decoding: k-best per word × word-LM combine

- **What:** extract k-best transliterations per word from the isolated-word
  model, then combine with a word-level LM (Roark §4.3.2: unpruned
  Katz-smoothed **trigram** over native-script Wikipedia). No parallel
  sentence data needed — exactly our data situation (1.5 GB running text).
- **Effect:** "large reduction in error rate versus single word
  transliteration"; noisy-channel beats seq2seq transformers on **9 of 12**
  languages, "sometimes substantially better" (Roark §4.3.3, Table 4). E.g.
  Hindi whitespace-WER: single-word pair-6g 24.6% → noisy-channel **11.0%**.
- **Fit: highest-headroom structural add.** Our headline metric is isolated
  words, but an IME *has* left context; the archive already lists word-bigram
  context as the differentiator (E6). Roark is the controlled evidence that a
  non-neural k-best + word-LM pass beats neural sentence models on most
  languages. Note the removed v4 word-bigram table cost 19.5 MB for +0.16pp
  *isolated-word* — the correct test is sentence-WER (`make eval-sentences`),
  not top-1 on AK-Freq. Also note their failure mode: highly inflected
  Dravidian (Malayalam, Tamil) favored the transformer — closed-vocabulary
  LM constrains morphological productivity, relevant to our agglutinative
  suffix handling.
- **Implementation note:** Hellsten et al. show the production form —
  compose pair-model ∘ lexicon ∘ word-LM into one transducer (CLG) with
  LOUDS/compact encodings; our two-pass union + context re-rank is the same
  family, just uncomposed.

### T3. Reranker features SOTA actually uses: generator confidence + LM + frequency — trained discriminatively, not hand-set

- **What:** Oh & Isahara 2007 (per Karimi §5.4): SVM / max-ent rerank of 7
  systems' outputs with exactly three feature families — **confidence score,
  language model, Web frequency** → 87.4–88.2% top-1 EN-JA/EN-KO. Zelenko &
  Aone 2006 (Karimi §5.6 refs): discriminative methods for transliteration.
- **Fit: direct, and indicts our pipeline.** Our dense set already contains
  all three families (emit/lm ≈ confidence+LM, log-freq ≈ frequency) — but
  `W_DENSE` was never refit by the pipeline (MANUAL §12.3) and γ=0.3 is the
  point "at which a weak model stops doing damage". SOTA's version of this
  stage is *trained* (SVM/MaxEnt/perceptron/MIRA/Collins-2000 reranking
  objective — our trainer already uses a Collins-style objective for sparse
  only). Fix = train dense+sparse jointly with held-out monitoring, per the
  manual's own diagnosis. No new features needed to start; the signal is the
  training discipline.
- **Adjacent (Finch & Sumita 2010):** generate n-best from model A, rescore
  with model B's scores, **linearly interpolate both scores with tuned
  weights** — combined "substantially" beats either component. Our γ-blend
  is this pattern with γ hand-picked; Finch tunes via MERT (Och 2003). Same
  action as T1: tune the interpolation on dev.

### T4. Attestation/frequency-weighted training and multi-reference awareness

- **What:** Dakshina lexicon training repeats each pair by attestation count
  (Roark §4.2: "each pair repeated as many times as it was attested");
  Aksharantar keeps up to 4 romanization variants per annotator + 2 per
  validator (Madhani §4.2); Dakshina round-trip validation shows 3.5–8.5%
  CER is irreducible ambiguity (Roark §3.3/Dakshina README).
- **Fit: cheap data-side wins.** (i) Weight EM/reranker pairs by variant
  attestation instead of uniform pairs — teaches the channel *which*
  romanization conventions dominate, directly attacking the oracle@2 tie
  class. (ii) Our `test_multiref.jsonl` already exists for eval; extending
  multi-ref to *training* (accept any attested variant as correct in the
  reranker loss) matches how SOTA data treats variation. (iii) The 3.5–8.5%
  round-trip CER + our 99.15% collision bound together say: ~1% is
  unwinnable, the rest is ranking — consistent with roadmap, no new ceiling.

### T5. Wider orthographic context features in the (linear) model

- **What:** Goto et al. 2003: MaxEnt transliteration with source *and* target
  context windows (3 chars each side + chunking plausibility), −63% error vs
  context-free (their claim, JP-EN); Oh et al. (Karimi §5.3): MaxEnt/DT with
  context length 3 on each side of the mapped unit; Knight & Graehl 1998
  (Karimi §5.2): WFSA channel + k-best (Eppstein 1998) + target LM.
- **Fit: direct.** Our sparse templates already use first/final
  akshara×roman-char and matra×consonant conjunctions; SOTA's pre-neural
  best adds ±3 context conjunctions and *chunking plausibility* (segmentation
  score as a feature — our decoder's segmentation is invisible to the
  reranker). Position-conditioned emissions (archive E4, `P(s|a,pos)`) are the
  generative twin of the same signal. All linear-compatible.

### T6. Origin-aware / entity-aware routing (the NE answer is *not* a bigger unigram)

- **What:** Karimi §3.4 (language of origin): "Josef" → French vs English
  pronunciation needs different target chars; extra variants/errors when
  origin is misspecified. Madhani §6.2: foreign NEs < Indian NEs < uniform
  sample — surprising ordering the authors flag as needing investigation.
  Dakshina: code-switching with English is pervasive; evaluation needs
  pass-through of Latin-script tokens (Roark §4.3.1, Table 3).
- **Fit: route, don't rerank.** Evidence says: (i) skip/attenuate the unigram
  prior on the NE path (Madhani: it doesn't help rare names); our bands
  already let trie/fuzzy sources compete — the NE path wants channel+LM
  scores *without* the news-frequency prior that buries rare names;
  (ii) origin cues (capitalization patterns, consonant clusters atypical for
  native words, English-looking affixes) as reranker features or a
  NE-vs-native classifier gating the frequency weight — the linear,
  non-neural version of origin conditioning; (iii) IME-level: pass through
  already-Latin tokens (user typed English) instead of transliterating them,
  per Dakshina's pass-through evaluation.
- **Honest limit:** genuine foreign-name accuracy is a long tail; even
  IndicXlit sits at ~38–56% on AK-NEF/NEI (Table 6). Parity target should be
  native-first; NE goal is narrowing, not closing, the gap.

### T7. Pair n-gram / joint-multigram channel as a competitive non-neural baseline family

- **What:** Roark Table 2: pair 6-gram with Witten-Bell is within **2%
  absolute CER** of LSTM/transformer on all 12 languages, best on 2/12 —
  including the sparsest-data language (Sindhi). Hellsten: same family ships
  in production keyboards.
- **Fit: validation, not a new component.** Our EM channel over akshara:chunk
  pairs *is* this family at syllable granularity. The transferable details:
  Witten-Bell/Katz smoothing comparisons for the pair model, higher pair
  order (6 vs our MAX_CHUNK=5 window — different axis but same spirit), and
  treating pair-model order as a tuned hyperparameter. If our channel ever
  looks weak, the literature says check smoothing/order before reaching for
  capacity.

## 2. Reranking signals SOTA uses (question a) — consolidated

| Signal | Source | Effect size | NN-free? |
|---|---|---|---|
| Word-unigram prior, α≈0.9, dev-tuned | Madhani §5–6.1, eq.1 | +12% avg; nep 80.25→86.6 | yes — already partially present |
| k-best × word-trigram combine (sentence) | Roark §4.3.2–4.3.3 | single 24.6→11.0 WER (hin) | yes — biggest missing piece |
| Confidence + LM + frequency, SVM/MaxEnt-trained | Oh & Isahara 2007 via Karimi §5.4 | 87–88% top-1 EN-JA/KO | yes — features present, training missing |
| Cross-model n-best rescoring, MERT-tuned interpolation | Finch & Sumita 2010 | "substantially" over either alone | yes — same pattern as γ-blend |
| Target-side n-gram LM (bi/tri) inside the generator | Karimi §5.2 (most systems bigram; K&G unigram) | baseline everywhere | yes — have KN trigram (−6pp w/o) |
| Attestation-weighted pairs | Roark §4.2 | baked into all Dakshina baselines | yes — data change only |
| Context-window conjunction features | Goto 2003; Oh et al. via Karimi §5.3 | −63% err (claimed, JP); consistent gains | yes — linear features |

What SOTA reranking does **not** use (honest negative): no SOTA system reranks
with an akshara/subword *continuation* model of our exact kind — they use
word-level priors *on top of* character generators. Our KN akshara trigram
plays both roles at once; the literature suggests splitting them (generator LM
+ word prior) rather than growing one LM to do both.

## 3. Named entities vs native words (question b)

1. **Universal difficulty ordering.** Madhani Table 6: AK-Freq ≫ AK-Uni
   (−10pp) ≫ AK-NEI ≫ AK-NEF. Roark: Perso-Arabic scripts hardest, Dravidian
   easiest (Table 2 discussion). Our strata reproduce this exactly
   (81/45/29%). Nobody treats NEs as "just more words".
2. **Unigram priors help native words, not names** (Madhani §6.1, quoted in
   T1). Mechanism: names are rare by definition. Implication: gate the
   frequency weight by entity-l likelihood instead of using one γ for all
   inputs.
3. **Origin matters more than language** (Karimi §3.4). Foreign vs Indian
   origin is annotated *in the benchmark itself* (AK-NEF vs AK-NEI) because
   grapheme–phoneme mismatch differs by origin. Our engine has no origin
   signal; even cheap proxies (unusual clusters, length, dictionary miss
   with high channel score) are untried.
4. **NE evaluation is multi-reference or it is noise.** Karimi §5.5: system
   accuracy swings up to **30%** with corpus/transliterator construction;
   single-reference eval doesn't transfer across corpora; 4–5 diverse
   transliterators needed. Our multi-ref file + bootstrap/McNemar practice
   already meets this standard — keep it for any NE claim.
5. **Mixed-script reality.** Dakshina sentences contain Latin tokens
   (code-switching), digits, punctuation; SOTA evaluation pass-throughs them
   (Roark §4.3.1). An IME that transliterates the English word inside a Hindi
   sentence is wrong by construction — detection + pass-through outranks
   another point of NE accuracy.

## 4. Non-neural / hybrid techniques competitive with transformers (question c)

- **Pair 6-gram ≈ LSTM ≈ transformer** on isolated words (Roark Table 2,
  all within 2% CER; non-neural best on 2/12). This is the headline
  existence proof for our architecture family.
- **Noisy-channel (pair-6g + Katz trigram) beats transformers 9/12** at
  sentence level (Roark Table 4). Non-neural + context > neural without it.
- **Google keyboard WFST (Hellsten 2017):** C∘L∘G composition, EM pair
  alignment with epsilons, compact LOUDS encoding, 10 MB / 20 ms budgets —
  a shipped product on exactly our technical approach.
- **Phrase-SMT + joint-multigram rescoring** beats both parents (Finch &
  Sumita 2010); **MaxEnt context models** dominate pre-neural J-E (Goto
  2003); **SVM/MaxEnt combination** reaches high-80s top-1 (Oh & Isahara
  2007, Karimi §5.4).
- **Where non-neural loses:** open-vocabulary morphology (Malayalam/Tamil
  sentence-level, Roark §4.3.3), foreign-origin NEs (all Madhani NE tables),
  and raw data-scaling races (IndicXlit +15% on Dakshina is mostly 26M-pair
  scale + multilingual transfer, §6.1/§7 — neither is a modeling idea).

## 5. What a beam-search + linear-reranker system is structurally missing (question d)

Against the above, our gaps in rough priority order:

1. **Sentence context at decode time.** The literature's largest
   isolated→context delta (halving WER) comes from a word-LM over k-best —
   we score words in isolation. `evaluate_sentences`/ctx-model is the
   instrument; a word-LM combine is the missing stage.
2. **Trained (not hand-set) combination weights.** Every SOTA blend is tuned
   (α=0.9, MERT, SVM/MaxEnt, AdaGrad-with-monitoring). Ours: frozen W_DENSE,
   hand γ, hand bands (MANUAL §§7, 8, 12.3). The defect is process, and the
   fix is dev-set tuning + joint dense/sparse training — no architecture
   change.
3. **One score space.** SOTA combines *scores* (log-linear interpolation);
   we route *candidates through bands* of hand constants (MANUAL §8 — "the
   weakest part of the architecture", two serious defects found there).
   Finch-style interpolation and Oh-style trained reranking both assume
   comparable scores; bands violate that assumption by construction.
4. **Top-2 discriminative focus.** Oracle@2 captures ~2/3 of our ranking loss
   (89.8 vs 81.8); SOTA's α=0.9 finding says the prior exists to break ties,
   not to re-order lists. A pairwise/top-2 discriminator (archive
   `2026-09-22-top2-discriminator.md`) matches the loss to the error
   distribution better than listwise softmax.
5. **Vowel/matra factorization.** SOTA error mass is vowels: 60% vowel +
   25% similar-consonant (Madhani §6.3); ours: 51.9% matra-only. Neither
   SOTA generator factorizes vowel signs from consonant skeletons — this is
   an open gap for *everyone*, and our factored-matra direction (archive
   `2026-09-22-factored-matra-model.md`) attacks shared SOTA error mass,
   not a local quirk.
6. **Segmentation/chunking as a rerank feature.** Goto's chunking
   plausibility + our invisible segmentation: the reranker never sees *how*
   the string was segmented, only the result. A segmentation-score feature
   is linear-compatible and directly SOTA-motivated.
7. **Calibrated cross-source scores.** Oh & Isahara note confidence scores
   across systems may not be comparable — the same warning our bands ignore.
   Any multi-source future (user trie, fuzzy, context) needs score
   comparability first.

## 6. What does NOT transfer (honest negatives)

- **Transformer representations and transfer at scale.** IndicXlit's edge is
  26M pairs × 21 languages with shared parameters (language-tagged
  multilingual training, temperature sampling T=1.5). There is no non-neural
  analogue of cross-lingual parameter sharing at that scale; our small
  analogue is Devanagari pooling (archive E2), worth trying but not
  comparable in expected gain. Data scale itself transfers (more pairs help
  EM/LM/vocab), but with diminishing returns for a low-capacity channel.
- **LLM prompting / API models.** Azam et al. 2025 (abstract) report GPT-family
  models generally outperforming IndicXlit on Dakshina/Aksharantar top-1/CER,
  with fine-tuned GPT-4o best on specific languages. That is API-scale neural
  inference — no signal, feature, or weight is extractable into a 11 MB /
  0.7 ms CPU engine. Track as a ceiling reference, not a technique source.
  (Correction: our archive snapshot says LLMs score *below* IndicXlit; the
  paper's abstract claims the opposite for GPT-family models. The archive
  note is stale — do not cite it without re-reading the full paper.)
- **Web-scale frequency counts at runtime.** Oh & Isahara's Web-frequency
  feature and Madhani's unigram both rest on web-scale monolingual data. What
  transfers is the *static compiled residue* (unigram table in the
  container), not live counting. Our news-domain vocab bias (MANUAL §9.6)
  is the known distortion of this residue — fix by domain-balancing the
  corpus, not by discarding the prior.
- **Learned subword/char embeddings and attention alignments.** Our
  alignment is EM over discrete chunks; attention's soft, position-free
  reordering has no counterpart — but transliteration is near-monotone
  (Karimi §4.2), so little is lost: monotonicity is why pair-models stay
  competitive (§4).
- **Language-tag conditioning as done by IndicXlit.** Our engine is
  deliberately script-wide with no language tag in the pipeline. Adding
  language conditioning would cut against the design goal; origin-gating
  (§T6) is the most tag-like mechanism consistent with it.
- **NEWS-era system combination by voting.** NEWS winners combined *diverse
  full systems* (Karimi §5.4: independent errors + selection risk — "weak
  methods may dilute"). We have one system; voting over our own beam is
  already what the reranker does. Don't build 7 weak transliterators to vote.

## 7. Suggested order of attack (for roadmap use, not a plan)

1. Dev-tuned (emit, lm, freq, γ) on the valid split — T1+T3 process fix,
   zero architecture risk.
2. Joint dense+sparse reranker training (unfreeze W_DENSE) — T3, MANUAL
   §12.3's identified cause.
3. Word-LM k-best combine for sentence context — T2, largest structural gap.
4. NE routing: attenuate frequency prior + origin features + Latin
   pass-through — T6.
5. Context-window sparse features + segmentation-score feature — T5, gap #6.
6. Factored matra modeling — gap #5, shared SOTA error mass.

## Sources

- Madhani et al. 2023 — local `papers_to_read/madhani-etal-2023-aksharantar-indicxlit.pdf`
  (§5 eq.1 α=0.9 beam-4 top-4 rescore; §6.1 +12%, NEI/NEF hardness, rerank-needs-monolingual-data;
  §6.2 AK-Uni −10pp, foreign<Indian NEs; §6.3 60/25/15 error split; Table 4 Dakshina;
  Table 6 Aksharantar incl. nep 80.25→86.6; §7 ablations; §4.1 KenLM 4-gram sampling).
- Roark et al. 2020 — local `papers_to_read/roark-etal-2020-dakshina.pdf`
  (§4.2.1 pair-6g Witten-Bell + attestation repetition; Table 2 within-2%-CER;
  §4.3 noisy-channel Katz trigram k-best combine, Table 4 9/12 wins; §4.3.1
  whitespace vs pass-through; §3.3 round-trip 3.5–8.5% CER).
- Karimi et al. 2011 — local `papers_to_read/karimi-etal-2011-survey.pdf`
  (§3.4 origin; §4.2 noisy-channel formulation/monotone alignment; §5.2–5.4
  hybrid/combined incl. Oh & Isahara 2007 SVM/MaxEnt + Web frequency, Karimi
  2008 voting; §5.5 corpus effects ±30%, 4–5 transliterators; §6 extraction).
- Hellsten et al. 2017 — local `papers_to_read/hellsten2017_google_translit_wfst.pdf`
  (C∘L∘G, EM pair alignment with ϵ, LOUDS/compact operating points §3–4).
- Goto et al. 2003 — local `papers_to_read/goto-etal-2003-maxent-transliteration.pdf`
  (source+target context windows, Models B–G, §3 results).
- Finch & Sumita 2010 — local `papers_to_read/finch-sumita-2010-multigram-smt-rescoring.pdf`
  (n-best rescoring + linear interpolation + MERT/BLEU-proxy tuning; NEWS
  metric columns: top-1/MRR/MAPref/mean F-score).
- Azam et al. 2025, arXiv:2505.19851 (abstract; GPT-family vs IndicXlit on
  Dakshina/Aksharantar, top-1 + CER + noise robustness).
- IndicXlit repo (github.com/AI4Bharat/IndicXlit, fetched 2026-09-22):
  6+6-layer 256-dim 4-head 11M-param char transformer, fairseq, beam/nbest 4,
  per-language En-Indic/Indic-En result tables.
- Dakshina repo (github.com/google-research-datasets/dakshina, fetched
  2026-09-22): lexicon attestation format, whitespace/pass-through protocol,
  CC BY-SA 4.0.
- Repo-internal: `docs/MANUAL.md` §§5–9, 11–13;
  `docs/plans/archive/2026-09-03-transliteration-accuracy-research.md`
  (whose "LLMs below IndicXlit" line §7 above corrects).
