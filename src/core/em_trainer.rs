// File: src/core/em_trainer.rs
//
// EM trainer for the generative transliteration model.
//
// Following Li, Zhang & Su (ACL-04), we learn the transliteration table
// P(chunk | akshara) from an unaligned parallel lexicon.  Each Devanagari word
// is segmented into aksharas; the roman word is a sequence of chars.  A
// segmentation of the roman word assigns each akshara a contiguous chunk of
// 0..=MAX_CHUNK roman characters.  EM (forward-backward over the alignment
// space) maximises the corpus likelihood under
//
//   P(R | D) = sum_segmentations prod_j P(chunk_j | akshara_j).
//
// Emissions are seeded from a deterministic codepoint aligner (`alignment.rs`)
// so EM starts from a sensible point, then refined with a few EM iterations.
// A Kneser-Ney akshara n-gram LM is built from the same corpus for scoring
// fresh transliterations.

use crate::core::akshara::segment;
use crate::core::alignment::align_emissive;
use crate::core::translit_model::{
    pack_chunk, pack_chunk_bytes, unpack_chunk, TranslitModel, MAX_CHUNK,
};
use std::collections::{BTreeSet, HashMap};

/// A single training pair held in memory: roman bytes + akshara id sequence.
struct Pair {
    roman: Vec<u8>,
    aks: Vec<u32>,
    /// Observation weight (frequency): scales LM counts and EM posteriors.
    weight: f64,
}

pub struct TrainerConfig {
    /// Number of EM passes over the corpus.
    pub iterations: usize,
    /// Add-alpha smoothing on the emission M-step (keeps probabilities > 0).
    pub em_smoothing: f64,
    /// Kneser-Ney absolute discount for the akshara bigram LM.
    pub kn_discount: f64,
    /// Seed emissions from the deterministic codepoint aligner before EM.
    pub seed_from_aligner: bool,
    /// Stop ingesting after this many clean pairs (iterate fast in dev).
    pub limit: Option<usize>,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            iterations: 12,
            em_smoothing: 0.05,
            kn_discount: 0.75,
            seed_from_aligner: true,
            limit: None,
        }
    }
}

pub struct Trainer {
    akshara_map: HashMap<String, u32>,
    akshara_list: Vec<String>,
    pairs: Vec<Pair>,
    limit: Option<usize>,

    // Seed counts (raw) and live emission probabilities.
    seed_counts: Vec<HashMap<u32, f64>>,
    emission: Vec<HashMap<u32, f64>>,

    // LM counts.
    unigram_counts: Vec<u64>,
    word_initial: Vec<u64>,
    total_words: u64,
    bigram_counts: HashMap<(u32, u32), u64>,
    trigram_counts: HashMap<(u32, u32, u32), u64>,
    /// continuation[akshara] = number of distinct left-context aksharas.
    continuation: Vec<u64>,
    /// successor_count[(a,b)] = number of distinct c with (a,b,c) seen.
    trigram_successors: HashMap<(u32, u32), u64>,
    /// bigram_cont_right[(b, c)] = number of distinct a such that trigram (a,b,c) is seen.
    /// Used for the correct Kneser-Ney trigram lower-order estimate: the textbook
    /// requires backing off to P_cont(c | b) = N1+(•,b,c) / Σ_c' N1+(•,b,c'), NOT
    /// the highest-order bigram P(c | b).  An earlier version used bigram_weight(b, c)
    /// here, which is the deviation documented in §6.6 of MANUAL.md.
    bigram_cont_right: HashMap<(u32, u32), u64>,
    /// total_bigram_cont[b] = Σ_c N1+(•,b,c) — denominator for P_cont(c|b).
    total_bigram_cont: Vec<u64>,

    // Global chunk unigram (for Dirichlet-smoothed emissions).
    chunk_unigram: HashMap<u32, f64>,
    total_chunk_obs: f64,

    distinct_bigrams: u64,
    /// When false, ingested pairs train the emissions only — the LM counts
    /// stay untouched (phonetic/LM split: emissions are language-independent,
    /// the LM is language-specific).
    count_lm: bool,
    pub ingested: usize,
    pub skipped: usize,
}

impl Trainer {
    pub fn new() -> Self {
        Self {
            akshara_map: HashMap::new(),
            akshara_list: Vec::new(),
            pairs: Vec::new(),
            limit: None,
            seed_counts: Vec::new(),
            emission: Vec::new(),
            unigram_counts: Vec::new(),
            word_initial: Vec::new(),
            total_words: 0,
            bigram_counts: HashMap::new(),
            trigram_counts: HashMap::new(),
            continuation: Vec::new(),
            trigram_successors: HashMap::new(),
            bigram_cont_right: HashMap::new(),
            total_bigram_cont: Vec::new(),
            chunk_unigram: HashMap::new(),
            total_chunk_obs: 0.0,
            distinct_bigrams: 0,
            count_lm: true,
            ingested: 0,
            skipped: 0,
        }
    }

    /// Builder: cap the number of clean pairs ingested (dev/testing).
    pub fn with_limit(mut self, limit: Option<usize>) -> Self {
        self.limit = limit;
        self
    }

    fn intern_akshara(&mut self, akshara: &str) -> u32 {
        if let Some(&id) = self.akshara_map.get(akshara) {
            return id;
        }
        let id = self.akshara_list.len() as u32;
        self.akshara_list.push(akshara.to_string());
        self.akshara_map.insert(akshara.to_string(), id);
        self.seed_counts.push(HashMap::new());
        self.emission.push(HashMap::new());
        self.unigram_counts.push(0);
        self.word_initial.push(0);
        self.continuation.push(0);
        self.total_bigram_cont.push(0);
        id
    }

    /// Add one (roman, devanagari) pair.
    ///
    /// All well-formed devanagari words contribute akshara LM counts.  Pairs
    /// whose roman is clean lowercase ASCII `a-z` and of sane length also feed
    /// the EM emission model (seeded from the codepoint aligner).
    pub fn add_pair(&mut self, roman: &str, dev: &str) {
        self.add_pair_weighted(roman, dev, 1.0);
    }

    /// Add a pair with an observation weight (e.g. corpus frequency of a
    /// synthetic pair).  Weight scales the LM counts and the EM posteriors.
    /// Toggle whether ingested pairs contribute to the akshara LM counts.
    pub fn set_lm_ingestion(&mut self, on: bool) {
        self.count_lm = on;
    }

    pub fn add_pair_weighted(&mut self, roman: &str, dev: &str, weight: f64) {
        if roman.is_empty() || dev.is_empty() {
            self.skipped += 1;
            return;
        }

        let aks: Vec<u32> = segment(dev)
            .iter()
            .map(|a| self.intern_akshara(a))
            .collect();
        if aks.is_empty() {
            self.skipped += 1;
            return;
        }

        // LM counts (independent of roman quality), scaled by weight and
        // gated by the phonetic/LM split flag.
        // Counts are integer-weighted; round rather than truncate so a weight
        // below 1 cannot silently become 0.
        let lmw = if self.count_lm {
            weight.round().max(0.0) as u64
        } else {
            0
        };
        for &a in &aks {
            self.unigram_counts[a as usize] += lmw;
        }
        self.word_initial[aks[0] as usize] += lmw;
        self.total_words += lmw;

        // Append an end-of-word token </w> to the akshara sequence for LM
        // counting.  The textbook KN formulation requires this so that trigram
        // mass sums to 1 over all complete words, and so the model can express
        // that certain aksharas are implausible word endings.  Without it,
        // 51.9% of native errors are matra-only and concentrate word-finally
        // (§6.6, §12, MANUAL.md).  The </w> token is never emitted at decode
        // time: it carries no Roman chunk and is excluded from the reverse
        // index when building the lattice.
        let eow_id = self.intern_akshara(crate::core::translit_model::END_OF_WORD);
        let aks_with_eow: Vec<u32> = aks.iter().copied().chain(std::iter::once(eow_id)).collect();
        // Continuation counts are TYPE counts: they must increment exactly once,
        // on the first time an n-gram is seen.  Keying that off `*e == 0` breaks
        // whenever the added weight is 0 (count_lm disabled, or a sub-1 weight
        // rounding down): every later occurrence re-counts the same type and
        // inflates `continuation` / `distinct_bigrams` without bound, corrupting
        // the Kneser-Ney continuation distribution.  Use Entry::Vacant, which
        // means "first insertion" regardless of the weight.
        //
        // Use `aks_with_eow` so the </w> token participates in bigram and
        // trigram counts — this closes the trigram probability mass and lets
        // the LM penalise implausible word endings.
        for w in aks_with_eow.windows(2) {
            let (b, c) = (w[0], w[1]);
            match self.bigram_counts.entry((b, c)) {
                std::collections::hash_map::Entry::Vacant(v) => {
                    v.insert(lmw);
                    self.continuation[c as usize] += 1;
                    self.distinct_bigrams += 1;
                }
                std::collections::hash_map::Entry::Occupied(mut o) => {
                    *o.get_mut() += lmw;
                }
            }
        }
        for w in aks_with_eow.windows(3) {
            let (a, b, c) = (w[0], w[1], w[2]);
            match self.trigram_counts.entry((a, b, c)) {
                std::collections::hash_map::Entry::Vacant(v) => {
                    v.insert(lmw);
                    *self.trigram_successors.entry((a, b)).or_insert(0) += 1;
                    // Track N1+(•,b,c): count of distinct a for each (b,c).
                    // This is the correct lower-order estimate for the trigram→bigram
                    // KN backoff, per Chen & Goodman (1999) §4.
                    match self.bigram_cont_right.entry((b, c)) {
                        std::collections::hash_map::Entry::Vacant(vv) => {
                            vv.insert(1);
                            // First time (b,c) appears — increment the denominator for b.
                            self.total_bigram_cont[b as usize] += 1;
                        }
                        std::collections::hash_map::Entry::Occupied(mut oo) => {
                            *oo.get_mut() += 1;
                            self.total_bigram_cont[b as usize] += 1;
                        }
                    }
                }
                std::collections::hash_map::Entry::Occupied(mut o) => {
                    *o.get_mut() += lmw;
                }
            }
        }

        // EM pairs: clean lowercase a-z roman, bounded lengths.
        let roman_ok = !roman.is_empty()
            && roman.len() <= 24
            && aks.len() <= 12
            && roman.bytes().all(|b| b.is_ascii_lowercase());
        if !roman_ok {
            self.skipped += 1;
            return;
        }
        if let Some(limit) = self.limit {
            if self.ingested >= limit {
                return;
            }
        }

        // Seed the emission counts from the codepoint aligner, grouped by akshara.
        // Bare-consonant aksharas also get their inherent-schwa variant (`s` + "a")
        // and schwa-dropped variant (drop a trailing "a") so EM can discover the
        // medial-schwa convention (e.g. "cha" -> च).
        let aligned = align_emissive(roman, dev);
        let mut p = 0usize;
        for &a in &aks {
            let aks_str = self.akshara_list[a as usize].clone();
            let aks_len = aks_str.chars().count();
            let mut chunk = String::with_capacity(2);
            for _ in 0..aks_len {
                if let Some(pair) = aligned.get(p) {
                    chunk.push_str(&pair.roman);
                    p += 1;
                }
            }
            if !chunk.is_empty() && chunk.len() <= MAX_CHUNK {
                self.seed_chunk(a, &chunk);
                if is_bare_consonant(&aks_str) {
                    if chunk.len() < MAX_CHUNK {
                        let mut with_a = chunk.clone();
                        with_a.push('a');
                        self.seed_chunk(a, &with_a);
                    }
                    if chunk.ends_with('a') && chunk.len() > 1 {
                        self.seed_chunk(a, &chunk[..chunk.len() - 1]);
                    }
                }
            }
        }

        let id = self.ingested;
        if self.pairs.len() <= id {
            self.pairs.push(Pair {
                roman: roman.as_bytes().to_vec(),
                aks,
                weight,
            });
        } else {
            self.pairs[id] = Pair {
                roman: roman.as_bytes().to_vec(),
                aks,
                weight,
            };
        }
        self.ingested += 1;
    }

    fn seed_chunk(&mut self, a: u32, chunk: &str) {
        let key = pack_chunk(chunk);
        *self.seed_counts[a as usize].entry(key).or_insert(0.0) += 1.0;
        *self.chunk_unigram.entry(key).or_insert(0.0) += 1.0;
        self.total_chunk_obs += 1.0;
    }

    /// Finalise: initialise emissions, run EM, build the akshara LM, serialise-ready model.
    pub fn finalize(&mut self, config: &TrainerConfig) -> TranslitModel {
        self.init_emissions(config.seed_from_aligner);
        self.run_em(config.iterations, config.em_smoothing);
        let mut model = TranslitModel {
            version: crate::core::translit_model::MODEL_VERSION,
            ..Default::default()
        };
        model.aksharas = self.akshara_list.clone();

        // Chunk vocabulary from emission keys, sorted for determinism.
        let mut chunk_set: BTreeSet<u32> = BTreeSet::new();
        for em in &self.emission {
            for &k in em.keys() {
                chunk_set.insert(k);
            }
        }
        let mut chunk_id_map: HashMap<u32, u32> = HashMap::new();
        for (i, &k) in chunk_set.iter().enumerate() {
            chunk_id_map.insert(k, i as u32);
            model.chunks.push(unpack_chunk(k));
        }

        model.emissions = self
            .emission
            .iter()
            .map(|em| {
                let mut v: Vec<(u32, f32)> = em
                    .iter()
                    .map(|(k, p)| {
                        let w = if *p > 0.0 { -p.ln() as f32 } else { 50.0 };
                        (chunk_id_map[k], w)
                    })
                    .collect();
                v.sort_by_key(|(cid, _)| *cid);
                v
            })
            .collect();

        self.build_kn_lm(&mut model, config.kn_discount);
        model
    }

    fn init_emissions(&mut self, seed_from_aligner: bool) {
        if seed_from_aligner {
            for a in 0..self.akshara_list.len() {
                let counts = std::mem::take(&mut self.seed_counts[a]);
                if counts.is_empty() {
                    continue;
                }
                let total: f64 = counts.values().sum();
                let mut em: HashMap<u32, f64> = HashMap::with_capacity(counts.len());
                for (k, c) in counts {
                    // Floor small counts so the first E-step isn't degenerate.
                    em.insert(k, (c / total).max(1e-4));
                }
                self.emission[a] = em;
            }
            return;
        }
        // Fallback init (no aligner seed): uniform over observed chunks.
        // The first EM iteration will still learn from the data.
        for a in 0..self.akshara_list.len() {
            let counts = std::mem::take(&mut self.seed_counts[a]);
            let mut em: HashMap<u32, f64> = HashMap::new();
            for k in counts.keys() {
                em.insert(*k, 1.0);
            }
            self.emission[a] = em;
        }
    }

    /// Set `AKSHAR_KN_FIXED_DISCOUNT=1` to fall back to a single absolute
    /// discount, for measuring what modified Kneser-Ney actually buys.
    fn use_fixed_discount() -> bool {
        std::env::var("AKSHAR_KN_FIXED_DISCOUNT").is_ok_and(|v| v == "1")
    }

    fn modified_discounts(
        counts: &HashMap<(u32, u32), u64>,
        counts_tri: Option<&HashMap<(u32, u32, u32), u64>>,
    ) -> (f64, f64, f64) {
        if Self::use_fixed_discount() {
            return (0.75, 0.75, 0.75);
        }
        // Chen-Goodman modified Kneser-Ney: 3 discounts from n1..n4
        let mut n = [0u64; 5];
        if let Some(tri) = counts_tri {
            for &c in tri.values() {
                if (1..=4).contains(&c) {
                    n[c as usize] += 1;
                }
            }
        } else {
            for &c in counts.values() {
                if (1..=4).contains(&c) {
                    n[c as usize] += 1;
                }
            }
        }
        // Chen & Goodman (1999) constrain each discount to 0 <= D_i <= i.  The
        // upper bounds are 1/2/3, NOT a common ~0.9: clamping D2 and D3 to 0.9
        // pins both at the ceiling on any real corpus (typical D2 ~ 1.0-1.4,
        // D3 ~ 1.5-2.5), which collapses modified Kneser-Ney back into
        // single-discount absolute discounting at d ~ 0.9 -- worse than the
        // fixed 0.75 it replaced.
        //
        // n[3] == 0 makes D3 degenerate the same way n[1]/n[2] == 0 does, so it
        // belongs in the same guard.
        if n[1] == 0 || n[2] == 0 || n[3] == 0 {
            return (0.5, 0.75, 0.95);
        }
        let y = n[1] as f64 / (n[1] as f64 + 2.0 * n[2] as f64);
        let d1 = (1.0 - 2.0 * y * n[2] as f64 / n[1].max(1) as f64).clamp(0.0, 1.0);
        let d2 = (2.0 - 3.0 * y * n[3] as f64 / n[2].max(1) as f64).clamp(0.0, 2.0);
        let d3 = (3.0 - 4.0 * y * n[4] as f64 / n[3].max(1) as f64).clamp(0.0, 3.0);
        (d1, d2, d3)
    }

    fn discount_for(cnt: u64, d1: f64, d2: f64, d3: f64) -> f64 {
        match cnt {
            1 => d1,
            2 => d2,
            _ => d3,
        }
    }

    fn build_kn_lm(&self, model: &mut TranslitModel, _delta: f64) {
        let n = self.akshara_list.len();
        let distinct = self.distinct_bigrams as f64;
        let mut unigram_kn = vec![0.0f32; n];
        for (a, w) in unigram_kn.iter_mut().enumerate() {
            let cont = self.continuation[a] as f64;
            // Floor so word-initial-only aksharas keep a finite log-prob.
            let p = (cont + 0.5) / (distinct + 0.5 * n as f64);
            *w = -p.ln() as f32;
        }

        let mut by_left: HashMap<u32, Vec<(u32, u64)>> = HashMap::new();
        for (&(b, c), &cnt) in &self.bigram_counts {
            by_left.entry(b).or_default().push((c, cnt));
        }

        let (d1_bi, d2_bi, d3_bi) = Self::modified_discounts(&self.bigram_counts, None);
        eprintln!("  [lm] bigram discounts: d1={d1_bi:.4} d2={d2_bi:.4} d3={d3_bi:.4}");
        let mut bigrams = vec![Vec::new(); n];
        let mut backoff = vec![0.0f32; n];
        for a in 0..n {
            let Some(list) = by_left.get(&(a as u32)) else {
                // No outgoing bigrams observed: pure backoff to the unigram.
                backoff[a] = 0.0;
                continue;
            };
            let total: u64 = list.iter().map(|(_, c)| c).sum();
            let disc_sum: f64 = list
                .iter()
                .map(|(_, c)| Self::discount_for(*c, d1_bi, d2_bi, d3_bi))
                .sum();
            let lambda = disc_sum / total as f64;
            backoff[a] = (-lambda.ln()) as f32;
            let mut v = Vec::with_capacity(list.len());
            for &(c, cnt) in list {
                let d = Self::discount_for(cnt, d1_bi, d2_bi, d3_bi);
                let disc = (cnt as f64 - d).max(0.0) / total as f64;
                let p_kn_c = (-unigram_kn[c as usize] as f64).exp();
                let p = disc + lambda * p_kn_c;
                let w = if p > 0.0 { -p.ln() } else { 50.0 };
                v.push((c, w as f32));
            }
            v.sort_by_key(|(id, _)| *id);
            bigrams[a] = v;
        }
        model.bigrams = bigrams;
        model.backoff = backoff;
        model.unigram_kn = unigram_kn;

        // Word-start prior: P(a | word start) from corpus word-initial counts.
        let mut word_start = vec![0.0f32; n];
        for (a, w) in word_start.iter_mut().enumerate() {
            let c = self.word_initial[a] as f64;
            // Floor keeps log finite for aksharas that never start words.
            let p = (c + 0.5) / (self.total_words as f64 + 0.5 * n as f64);
            *w = -p.ln() as f32;
        }
        model.word_start = word_start;

        // Trigram KN LM: group counts by (a,b) context.
        //
        // Textbook Kneser-Ney requires that the trigram backs off to the
        // *continuation probability* P_cont(c | b) = N1+(•,b,c) / Σ_c' N1+(•,b,c'),
        // NOT the highest-order bigram P(c | b).  The previous code used
        // `model.bigram_weight(b, c)` here, which is the deviation recorded
        // in §6.6 of MANUAL.md.  This block uses `self.bigram_cont_right` /
        // `self.total_bigram_cont` instead.
        let mut by_ctx: HashMap<(u32, u32), Vec<(u32, u64)>> = HashMap::new();
        for (&(a, b, c), &cnt) in &self.trigram_counts {
            by_ctx.entry((a, b)).or_default().push((c, cnt));
        }
        let mut ctxs: Vec<(u32, u32)> = by_ctx.keys().cloned().collect();
        ctxs.sort();
        let mut trigram_keys = Vec::with_capacity(ctxs.len());
        let mut trigrams = Vec::with_capacity(ctxs.len());
        let mut trigram_backoff = Vec::with_capacity(ctxs.len());
        let (d1_tri, d2_tri, d3_tri) =
            Self::modified_discounts(&HashMap::new(), Some(&self.trigram_counts));
        eprintln!("  [lm] trigram discounts: d1={d1_tri:.4} d2={d2_tri:.4} d3={d3_tri:.4}");

        // Total continuation mass per middle akshara b: Σ_c N1+(•,b,c).
        // This is the denominator for P_cont(c | b) and is already maintained
        // in `self.total_bigram_cont[b]`.
        let cont_total = &self.total_bigram_cont;

        for &(a, b) in &ctxs {
            let list = &by_ctx[&(a, b)];
            let c_ab = self.bigram_counts.get(&(a, b)).copied().unwrap_or(1) as f64;
            let disc_sum: f64 = list
                .iter()
                .map(|(_, c)| Self::discount_for(*c, d1_tri, d2_tri, d3_tri))
                .sum();
            let lambda = disc_sum / c_ab;
            trigram_backoff.push((-lambda.max(1e-15).ln()) as f32);
            let denom_b = cont_total.get(b as usize).copied().unwrap_or(0) as f64;
            let mut v = Vec::with_capacity(list.len());
            for &(c, cnt) in list {
                let d = Self::discount_for(cnt, d1_tri, d2_tri, d3_tri);
                let disc = (cnt as f64 - d).max(0.0) / c_ab;
                // Correct KN lower-order: continuation probability P_cont(c | b).
                let n1_bc = self.bigram_cont_right.get(&(b, c)).copied().unwrap_or(0) as f64;
                let p_cont = if denom_b > 0.0 {
                    // Floor at 0.5 / denom so no seen (b,c) pair gets zero mass.
                    (n1_bc + 0.5) / (denom_b + 0.5)
                } else {
                    // b never appears as a middle element; fall back to KN unigram.
                    (-model.unigram_kn.get(c as usize).copied().unwrap_or(12.0) as f64).exp()
                };
                let p = disc + lambda * p_cont;
                let w = if p > 0.0 { -p.ln() } else { 50.0 };
                v.push((c, w as f32));
            }
            v.sort_by_key(|(id, _)| *id);
            trigram_keys.push((a, b));
            trigrams.push(v);
        }
        model.trigram_keys = trigram_keys;
        model.trigrams = trigrams;
        model.trigram_backoff = trigram_backoff;
        model.build_trigram_index();
    }

    fn run_em(&mut self, iterations: usize, alpha: f64) {
        if self.pairs.is_empty() {
            return;
        }
        for it in 0..iterations {
            self.em_iteration(alpha);
            if it % 3 == 2 {
                eprintln!("  [em] iteration {}/{} done", it + 1, iterations);
            }
        }
    }

    fn em_iteration(&mut self, alpha: f64) {
        let n_aksharas = self.akshara_list.len();
        let mut counts: Vec<HashMap<u32, f64>> = (0..n_aksharas).map(|_| HashMap::new()).collect();

        // The E-step is embarrassingly parallel over pairs: each thread computes
        // local posterior counts for a slice, then we merge.  The emission maps
        // and pairs are read-only, so scoped threads can share them safely.
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(16);
        let pairs = &self.pairs;
        let emission = &self.emission;
        if !pairs.is_empty() && threads > 1 {
            let chunk_size = pairs.len().div_ceil(threads);
            std::thread::scope(|s| {
                let mut handles = Vec::with_capacity(threads);
                for chunk in pairs.chunks(chunk_size) {
                    handles.push(s.spawn(move || e_step_chunk(chunk, emission, n_aksharas)));
                }
                for h in handles {
                    for (a, map) in h
                        .join()
                        .expect("e-step thread panicked")
                        .into_iter()
                        .enumerate()
                    {
                        let ca = &mut counts[a];
                        for (k, v) in map {
                            *ca.entry(k).or_insert(0.0) += v;
                        }
                    }
                }
            });
        } else {
            let local = e_step_chunk(pairs, emission, n_aksharas);
            for (a, map) in local.into_iter().enumerate() {
                counts[a] = map;
            }
        }

        // --- M-step: normalise + Dirichlet-smooth toward the global chunk unigram. ---
        let total_chunk = self.total_chunk_obs.max(1.0);
        for (a, counts_a) in counts.iter().enumerate() {
            if counts_a.is_empty() {
                continue;
            }
            let total: f64 = counts_a.values().sum();
            let denom = total + alpha;
            let mut new_em: HashMap<u32, f64> = HashMap::with_capacity(counts_a.len());
            for (k, c) in counts_a {
                let prior = self.chunk_unigram.get(k).copied().unwrap_or(0.0) / total_chunk;
                new_em.insert(*k, (c + alpha * prior) / denom);
            }
            self.emission[a] = new_em;
        }
    }
}

/// Forward-backward EM E-step over a slice of pairs.
///
/// Returns per-akshara fractional posterior counts of (packed chunk -> count).
/// This is a free function so it can run on scoped threads sharing the emission
/// tables read-only.
fn e_step_chunk(
    pairs: &[Pair],
    emission: &[HashMap<u32, f64>],
    n_aksharas: usize,
) -> Vec<HashMap<u32, f64>> {
    let mut counts: Vec<HashMap<u32, f64>> = (0..n_aksharas).map(|_| HashMap::new()).collect();
    let mut f: Vec<Vec<f64>> = Vec::new();
    let mut b: Vec<Vec<f64>> = Vec::new();
    // Per-column forward scale factors (see the scaled forward-backward below).
    let mut scale: Vec<f64> = Vec::new();

    for pair in pairs {
        let m = pair.roman.len();
        let n = pair.aks.len();
        if m == 0 || n == 0 {
            continue;
        }

        // --- Forward pass (scaled). ---
        // Each column is divided by its own max so the running product stays
        // O(1); `scale[j]` records the factor.  Without this, a 10-akshara word
        // with moderately flat emissions underflows any fixed floor and gets
        // dropped from training entirely.
        f.clear();
        f.resize(n + 1, vec![0.0f64; m + 1]);
        f[0][0] = 1.0;
        scale.clear();
        scale.resize(n + 1, 1.0f64);
        for j in 1..=n {
            let em = &emission[pair.aks[j - 1] as usize];
            let prev = f[j - 1].clone();
            let mut row = vec![0.0f64; m + 1];
            for i in 0..=m {
                let maxl = MAX_CHUNK.min(i);
                let mut acc = 0.0;
                for l in 0..=maxl {
                    let key = pack_chunk_bytes(&pair.roman[i - l..i]);
                    if let Some(&p) = em.get(&key) {
                        acc += prev[i - l] * p;
                    }
                }
                row[i] = acc;
            }
            let c = row.iter().cloned().fold(0.0f64, f64::max);
            if c > 0.0 && c.is_finite() {
                for v in row.iter_mut() {
                    *v /= c;
                }
                scale[j] = c;
            }
            f[j] = row;
        }
        let z = f[n][m];
        // Guard only against a genuinely degenerate word (the model assigns it
        // no probability at all, or the arithmetic broke).  The columns are
        // rescaled above, so `z` here is O(1) and no longer shrinks with word
        // length: the old 1e-12 floor was ~296 orders of magnitude above the
        // f64 subnormal limit and silently discarded every long or
        // flat-emission word from training.
        if z <= 0.0 || !z.is_finite() {
            continue;
        }

        // --- Backward pass (scaled with the SAME factors as the forward pass,
        //     so they telescope out of the posterior). ---
        b.clear();
        b.resize(n + 1, vec![0.0f64; m + 1]);
        b[n][m] = 1.0;
        for j in (0..n).rev() {
            let em = &emission[pair.aks[j] as usize];
            let next = b[j + 1].clone();
            let mut row = vec![0.0f64; m + 1];
            for i in 0..=m {
                let maxl = MAX_CHUNK.min(m - i);
                let mut acc = 0.0;
                for l in 0..=maxl {
                    let key = pack_chunk_bytes(&pair.roman[i..i + l]);
                    if let Some(&p) = em.get(&key) {
                        acc += p * next[i + l];
                    }
                }
                row[i] = acc;
            }
            let c = scale[j + 1];
            if c > 0.0 && c.is_finite() {
                for v in row.iter_mut() {
                    *v /= c;
                }
            }
            b[j] = row;
        }

        // --- Accumulate posteriors. ---
        let inv_z = 1.0 / z;
        for (idx, &a) in pair.aks.iter().enumerate() {
            let j = idx + 1;
            // Scaling correction: the forward/backward scale products cancel to
            // a single 1/c_j for column j.
            let inv_z = inv_z / scale[j];
            let em = &emission[a as usize];
            let counts_a = &mut counts[a as usize];
            for i in 1..=m {
                let maxl = MAX_CHUNK.min(i);
                for l in 1..=maxl {
                    let key = pack_chunk_bytes(&pair.roman[i - l..i]);
                    if let Some(&p) = em.get(&key) {
                        let post = f[j - 1][i - l] * p * b[j][i] * inv_z;
                        if post > 0.0 {
                            *counts_a.entry(key).or_insert(0.0) += post * pair.weight;
                        }
                    }
                }
            }
            if let Some(&p0) = em.get(&0u32) {
                if p0 > 0.0 {
                    for i in 0..=m {
                        let post = f[j - 1][i] * p0 * b[j][i] * inv_z;
                        if post > 0.0 {
                            *counts_a.entry(0u32).or_insert(0.0) += post * pair.weight;
                        }
                    }
                }
            }
        }
    }
    counts
}

impl Default for Trainer {
    fn default() -> Self {
        Self::new()
    }
}

impl Trainer {}

/// A bare-consonant akshara carries an inherent schwa that roman spellings may
/// write as a trailing "a" or drop.  Such aksharas end in a consonant or halanta
/// (never a matra / anusvara / visarga).
fn is_bare_consonant(akshara: &str) -> bool {
    let Some(last) = akshara.chars().last() else {
        return false;
    };
    let cp = last as u32;
    (0x0915..=0x0939).contains(&cp)
        || (0x0958..=0x095F).contains(&cp)
        || matches!(
            cp,
            0x0931 | 0x0934 | 0x0978 | 0x0979 | 0x097A | 0x097B | 0x097D | 0x094D
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_consonant_detection() {
        assert!(is_bare_consonant("क"));
        assert!(is_bare_consonant("च"));
        assert!(is_bare_consonant("क्"));
        assert!(!is_bare_consonant("का"));
        assert!(!is_bare_consonant("कि"));
        assert!(!is_bare_consonant("कं"));
        assert!(!is_bare_consonant("अ"));
    }

    #[test]
    fn trainer_interns_aksharas() {
        let mut t = Trainer::new();
        t.add_pair("ka", "क");
        t.add_pair("kha", "ख");
        // The </w> end-of-word token is interned as a synthetic akshara during LM
        // counting (see §6.6 fix in MANUAL.md).  Real aksharas still appear.
        assert!(t.akshara_list.contains(&"क".to_string()));
        assert!(t.akshara_list.contains(&"ख".to_string()));
        assert!(t.akshara_list.contains(&"</w>".to_string()));
        assert_eq!(t.ingested, 2);
    }

    #[test]
    fn trainer_counts_bigrams() {
        let mut t = Trainer::new();
        t.add_pair("ka", "क");
        t.add_pair("kama", "कम");
        // The </w> synthetic token gets its own akshara slot (unigram_count = 0
        // since we only append it to bigram/trigram windows, never count it as a
        // standalone unigram).  Look up counts by ID to be position-independent.
        let ka_id = t.akshara_map["क"] as usize;
        let ma_id = t.akshara_map["म"] as usize;
        assert_eq!(t.unigram_counts[ka_id], 2);
        assert_eq!(t.unigram_counts[ma_id], 1);
        assert_eq!(t.bigram_counts.get(&(ka_id as u32, ma_id as u32)), Some(&1));
        assert_eq!(t.continuation[ma_id], 1);
    }

    #[test]
    fn trainer_skips_dirty_roman() {
        let mut t = Trainer::new();
        t.add_pair("Ka", "क");
        assert_eq!(t.ingested, 0);
        assert_eq!(t.unigram_counts[0], 1);
    }

    #[test]
    fn finalize_produces_valid_model() {
        let mut t = Trainer::new();
        t.add_pair("ka", "क");
        t.add_pair("kama", "कम");
        t.add_pair("kamal", "कमल");
        t.add_pair("namaste", "नमस्ते");
        let model = t.finalize(&TrainerConfig::default());
        assert!(model.validate());
        assert!(model.aksharas.len() >= 4);
        let kid = model.akshara_id("क").unwrap();
        let top = model.top_emissions(kid, 3);
        assert!(!top.is_empty());
        assert!(top.iter().any(|(c, _)| c == "ka"));
    }

    #[test]
    fn em_learns_expected_emission() {
        let mut t = Trainer::new();
        // म is written "ma" 3x and "m" once; EM should learn P(ma|म) > P(m|म).
        for _ in 0..3 {
            t.add_pair("nama", "नम");
        }
        t.add_pair("namaste", "नमस्ते");
        let model = t.finalize(&TrainerConfig {
            iterations: 8,
            ..Default::default()
        });
        let nid = model.akshara_id("न").unwrap();
        let p_na = model.emission_prob(nid, "na");
        let p_n = model.emission_prob(nid, "n");
        assert!(p_na > p_n, "P(na|न) should exceed P(n|न): {p_na} vs {p_n}");
        let mid = model.akshara_id("म").unwrap();
        let p_ma = model.emission_prob(mid, "ma");
        let p_m = model.emission_prob(mid, "m");
        assert!(p_ma > p_m, "P(ma|म) should exceed P(m|म): {p_ma} vs {p_m}");
    }

    #[test]
    fn pack_helpers_agree() {
        assert_eq!(
            pack_chunk_bytes(b"ka"),
            crate::core::translit_model::pack_chunk("ka")
        );
        assert_eq!(pack_chunk_bytes(b""), 0);
    }
}

#[cfg(test)]
mod scaling_tests {
    use super::*;

    /// The scaled forward-backward must produce the same posteriors as a naive
    /// unscaled implementation.  Rescaling each forward column and dividing the
    /// same factor out of the backward pass makes the scale products telescope,
    /// leaving a single 1/c_j correction per column; if that correction is wrong
    /// (or the passes use independent scales) the posteriors silently change.
    fn unscaled_posteriors(pair: &Pair, emission: &[HashMap<u32, f64>]) -> Vec<HashMap<u32, f64>> {
        let m = pair.roman.len();
        let n = pair.aks.len();
        let mut f = vec![vec![0.0f64; m + 1]; n + 1];
        f[0][0] = 1.0;
        for j in 1..=n {
            let em = &emission[pair.aks[j - 1] as usize];
            for i in 0..=m {
                let mut acc = 0.0;
                for l in 0..=MAX_CHUNK.min(i) {
                    if let Some(&p) = em.get(&pack_chunk_bytes(&pair.roman[i - l..i])) {
                        acc += f[j - 1][i - l] * p;
                    }
                }
                f[j][i] = acc;
            }
        }
        let z = f[n][m];
        let mut b = vec![vec![0.0f64; m + 1]; n + 1];
        b[n][m] = 1.0;
        for j in (0..n).rev() {
            let em = &emission[pair.aks[j] as usize];
            for i in 0..=m {
                let mut acc = 0.0;
                for l in 0..=MAX_CHUNK.min(m - i) {
                    if let Some(&p) = em.get(&pack_chunk_bytes(&pair.roman[i..i + l])) {
                        acc += p * b[j + 1][i + l];
                    }
                }
                b[j][i] = acc;
            }
        }
        let mut out: Vec<HashMap<u32, f64>> = vec![HashMap::new(); emission.len()];
        for (idx, &a) in pair.aks.iter().enumerate() {
            let j = idx + 1;
            let em = &emission[a as usize];
            for i in 1..=m {
                for l in 1..=MAX_CHUNK.min(i) {
                    let key = pack_chunk_bytes(&pair.roman[i - l..i]);
                    if let Some(&p) = em.get(&key) {
                        let post = f[j - 1][i - l] * p * b[j][i] / z;
                        if post > 0.0 {
                            *out[a as usize].entry(key).or_insert(0.0) += post * pair.weight;
                        }
                    }
                }
            }
        }
        out
    }

    fn toy_emissions() -> Vec<HashMap<u32, f64>> {
        let mut a0 = HashMap::new();
        a0.insert(pack_chunk("ka"), 0.6);
        a0.insert(pack_chunk("k"), 0.4);
        let mut a1 = HashMap::new();
        a1.insert(pack_chunk("th"), 0.5);
        a1.insert(pack_chunk("t"), 0.3);
        a1.insert(pack_chunk("tha"), 0.2);
        let mut a2 = HashMap::new();
        a2.insert(pack_chunk("ma"), 0.7);
        a2.insert(pack_chunk("m"), 0.3);
        vec![a0, a1, a2]
    }

    #[test]
    fn scaled_forward_backward_matches_unscaled() {
        let emission = toy_emissions();
        let pair = Pair {
            roman: b"kathma".to_vec(),
            aks: vec![0, 1, 2],
            weight: 1.0,
        };
        let got = e_step_chunk(std::slice::from_ref(&pair), &emission, emission.len());
        let want = unscaled_posteriors(&pair, &emission);
        for a in 0..emission.len() {
            for (k, v) in &want[a] {
                let g = got[a].get(k).copied().unwrap_or(0.0);
                assert!(
                    (g - v).abs() < 1e-9,
                    "akshara {a} chunk {k}: scaled {g} vs unscaled {v}"
                );
            }
            assert_eq!(
                got[a].len(),
                want[a].len(),
                "chunk set differs for akshara {a}"
            );
        }
    }

    /// Posterior mass over all alignments of one akshara must sum to its weight:
    /// every position of the word is explained by exactly one chunk per akshara.
    #[test]
    fn posteriors_sum_to_weight_per_akshara() {
        let emission = toy_emissions();
        let pair = Pair {
            roman: b"kathma".to_vec(),
            aks: vec![0, 1, 2],
            weight: 1.0,
        };
        let got = e_step_chunk(std::slice::from_ref(&pair), &emission, emission.len());
        for (a, counts) in got.iter().enumerate() {
            let total: f64 = counts.values().sum();
            assert!(
                (total - 1.0).abs() < 1e-9,
                "akshara {a} posterior mass {total}, expected 1.0"
            );
        }
    }

    /// A long word whose forward mass falls far below the old 1e-12 floor must
    /// still contribute posteriors instead of being silently discarded.
    #[test]
    fn long_low_probability_word_is_not_dropped() {
        // 12 aksharas each emitting at p = 0.05 gives z ~ 2.4e-16, well under
        // the floor the unscaled implementation used.
        let mut em = HashMap::new();
        em.insert(pack_chunk("a"), 0.05);
        let emission = vec![em];
        let pair = Pair {
            roman: b"aaaaaaaaaaaa".to_vec(),
            aks: vec![0; 12],
            weight: 1.0,
        };
        let got = e_step_chunk(std::slice::from_ref(&pair), &emission, 1);
        let total: f64 = got[0].values().sum();
        assert!(
            total > 0.0,
            "long low-probability word contributed no posterior mass"
        );
        assert!(
            (total - 12.0).abs() < 1e-6,
            "expected mass 12.0 (one per akshara), got {total}"
        );
    }
}
