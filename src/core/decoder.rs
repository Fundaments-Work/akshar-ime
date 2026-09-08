// File: src/core/decoder.rs
//
// Generative decoder: ranked Devanagari candidates for a roman string under
// the transliteration model.
//
// The decoder builds a lattice over roman character positions.  Each edge is a
// chunk s of 1..=MAX_CHUNK characters that some akshara a can emit, weighted by
// -log P(s | a) (emission) plus the akshara LM (word-start prior, Kneser-Ney
// bigram/trigram).  Beam search finds the k lowest-total-weight complete paths;
// the concatenated aksharas of each path form a Devanagari candidate string.

use crate::core::translit_model::{TranslitModel, MAX_CHUNK};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashMap;

const MAX_STEPS: usize = 32;
/// Drop emissions worse than this -log weight (they are alignment noise).
const MAX_EMISSION_WEIGHT: f32 = 8.0;
/// Keep at most this many aksharas per chunk in the reverse index.
const MAX_AKSHARAS_PER_CHUNK: usize = 16;

/// Tunable decoder parameters (exposed for eval-driven tuning).
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    pub beam_width: usize,
    pub max_emission_weight: f32,
    pub max_aksharas_per_chunk: usize,
    pub lm_weight: f64,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            beam_width: 64,
            max_emission_weight: MAX_EMISSION_WEIGHT,
            max_aksharas_per_chunk: MAX_AKSHARAS_PER_CHUNK,
            lm_weight: 1.0,
        }
    }
}

/// A single lattice edge: consumes `len` roman chars, emits akshara `a` at weight `w`.
#[derive(Debug, Clone, Copy)]
struct Edge {
    len: usize,
    a: u32,
    w: f32,
}

pub struct ModelDecoder {
    pub model: TranslitModel,
    /// Reverse index: chunk -> (akshara, weight), capped + sorted by weight.
    reverse: HashMap<String, Vec<(u32, f32)>>,
    /// Search / scoring configuration.
    pub config: DecoderConfig,
}

#[derive(Debug, Clone)]
struct BeamState {
    pos: usize,
    prev: Option<u32>,  // last akshara (None = word start)
    prev2: Option<u32>, // second-to-last akshara
    score: f64,         // total = emit + lm_weight * lm (for beam ordering)
    emit: f64,          // accumulated emission -log weight
    lm: f64,            // accumulated LM -log weight (unscaled)
    phash: u64,         // cheap hash of the path (dedup key)
    path: Option<u32>,  // index of the last cons cell in the path arena
}

/// One cell of a persistent (immutable) path in the arena.
struct PathCell {
    parent: Option<u32>,
    akshara: u32,
}

/// splitmix64 finaliser — cheap avalanche mixing for path hashing.
#[inline]
fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^ x >> 31
}

#[inline]
fn path_hash(parent: u64, a: u32) -> u64 {
    mix(parent ^ (a as u64).wrapping_add(0x9e3779b97f4a7c15))
}

/// A decoded candidate with its decomposed scores, for discriminative reranking.
#[derive(Debug, Clone)]
pub struct DecodedCandidate {
    pub dev: String,
    /// Sum of emission weights (-log P(R | aksharas), best alignment).
    pub emit: f64,
    /// Sum of LM weights (word-start + bigram + trigram), unscaled.
    pub lm: f64,
    /// Number of aksharas in the candidate.
    pub akshara_count: usize,
}

impl ModelDecoder {
    pub fn new(model: TranslitModel) -> Self {
        Self::with_config(model, DecoderConfig::default())
    }

    /// Build a decoder with explicit search/scoring parameters.
    pub fn with_config(model: TranslitModel, config: DecoderConfig) -> Self {
        // Build the reverse index: chunk -> (akshara, weight), capped + sorted.
        let mut reverse: HashMap<String, Vec<(u32, f32)>> = HashMap::new();
        for (a, list) in model.emissions.iter().enumerate() {
            for &(cid, w) in list {
                if w >= config.max_emission_weight {
                    continue;
                }
                if let Some(chunk) = model.chunks.get(cid as usize) {
                    reverse
                        .entry(chunk.clone())
                        .or_default()
                        .push((a as u32, w));
                }
            }
        }
        for v in reverse.values_mut() {
            v.sort_by(|a, b| a.1.total_cmp(&b.1));
            v.truncate(config.max_aksharas_per_chunk);
        }
        Self {
            model,
            reverse,
            config,
        }
    }

    /// Configure the LM weight relative to emission weights (default 1.0).
    pub fn with_lm_weight(mut self, w: f64) -> Self {
        self.config.lm_weight = w;
        self
    }

    /// Expose the reverse-index candidates for a chunk (diagnostics / search).
    pub fn chunk_candidates(&self, chunk: &str) -> &[(u32, f32)] {
        self.reverse.get(chunk).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Decode a roman string into `k` ranked (devanagari, -log score) pairs.
    pub fn decode(&self, roman: &str, k: usize) -> Vec<(String, f64)> {
        self.decode_detailed(roman, k)
            .into_iter()
            .map(|c| (c.dev, c.emit + c.lm * self.config.lm_weight))
            .collect()
    }

    /// Decode into decomposed candidates (emission + LM separately) for reranking.
    pub fn decode_detailed(&self, roman: &str, k: usize) -> Vec<DecodedCandidate> {
        let roman = roman.to_ascii_lowercase();
        let edges_by_pos = self.build_edges(&roman);
        let m = roman.len();
        if m == 0 {
            return vec![];
        }
        let k = k.max(1);

        // Persistent path arena: extending a path is O(1) (push one cons cell),
        // so the beam never clones full paths.
        let mut arena: Vec<PathCell> = Vec::with_capacity(1024);

        let mut beam: Vec<BeamState> = vec![BeamState {
            pos: 0,
            prev: None,
            prev2: None,
            score: 0.0,
            emit: 0.0,
            lm: 0.0,
            phash: 0,
            path: None,
        }];
        // Complete paths: dev string -> (score, emit, lm, akshara_count).
        let mut seen: FxHashMap<String, (f64, f64, f64, usize)> = FxHashMap::default();

        // A generated hypothesis before it is admitted to the beam.  It carries
        // its parent's arena index and its own akshara rather than an arena
        // cell: only the survivors of pruning are materialised, which cuts
        // arena writes from O(beam x edges) to O(beam) per step.
        struct Cand {
            pos: usize,
            prev2: Option<u32>,
            score: f64,
            emit: f64,
            lm: f64,
            phash: u64,
            parent: Option<u32>,
            akshara: u32,
        }
        let mut next: Vec<Cand> = Vec::with_capacity(self.config.beam_width * 16);

        for _step in 0..MAX_STEPS {
            if beam.is_empty() {
                break;
            }
            next.clear();

            for st in &beam {
                if st.pos == m {
                    let (dev, count) = self.reconstruct(&arena, st.path);
                    seen.entry(dev)
                        .and_modify(|best| {
                            if st.score < best.0 {
                                *best = (st.score, st.emit, st.lm, count);
                            }
                        })
                        .or_insert((st.score, st.emit, st.lm, count));
                    continue;
                }
                for &e in &edges_by_pos[st.pos] {
                    let emit_w = e.w as f64;
                    let fluency = match (st.prev2, st.prev) {
                        (_, None) => self.model.start_weight(e.a),
                        (None, Some(b)) => self.model.bigram_weight(b, e.a),
                        (Some(a), Some(b)) if !crate::core::ablation::no_trigram() => {
                            self.model.trigram_weight(a, b, e.a)
                        }
                        (Some(_), Some(b)) => self.model.bigram_weight(b, e.a),
                    };
                    let emit = st.emit + emit_w;
                    let lm = st.lm + fluency;
                    next.push(Cand {
                        pos: st.pos + e.len,
                        prev2: st.prev,
                        score: emit + lm * self.config.lm_weight,
                        emit,
                        lm,
                        phash: path_hash(st.phash, e.a),
                        parent: st.path,
                        akshara: e.a,
                    });
                }
            }

            // Keeping the best `beam_width` of ~5,000 hypotheses does not need
            // them ordered, so partition in O(n) instead of sorting in
            // O(n log n).  The beam's internal order is irrelevant: complete
            // paths are collected into `seen` and sorted once at the end.
            //
            // Distinct paths are deliberately kept separate rather than merged
            // by state: this decoder extracts k-best *paths*, not the 1-best
            // path per state.
            let width = self.config.beam_width;
            if next.len() > width {
                next.select_nth_unstable_by(width, |a, b| a.score.total_cmp(&b.score));
                next.truncate(width);
            }

            beam.clear();
            for c in next.iter() {
                let idx = arena.len() as u32;
                arena.push(PathCell {
                    parent: c.parent,
                    akshara: c.akshara,
                });
                beam.push(BeamState {
                    pos: c.pos,
                    prev: Some(c.akshara),
                    prev2: c.prev2,
                    score: c.score,
                    emit: c.emit,
                    lm: c.lm,
                    phash: c.phash,
                    path: Some(idx),
                });
            }
        }

        let mut results: Vec<DecodedCandidate> = seen
            .into_iter()
            .map(|(dev, (_, emit, lm, akshara_count))| DecodedCandidate {
                dev,
                emit,
                lm,
                akshara_count,
            })
            .collect();
        results.sort_by(|a, b| {
            let at = a.emit + a.lm * self.config.lm_weight;
            let bt = b.emit + b.lm * self.config.lm_weight;
            at.total_cmp(&bt)
        });
        results.truncate(k);
        results
    }

    /// Walk a persistent path in the arena back to the root, producing the
    /// Devanagari string and its akshara count.
    /// M4: decode restricted to paths that stay inside the word trie — the
    /// lattice ∩ dictionary graph intersection.  Every edge must extend the
    /// current trie node; only completions on trie terminals (real words with
    /// a corpus frequency) are returned.  Returns (dev, score, freq).
    pub fn decode_in_words(
        &self,
        roman: &str,
        k: usize,
        trie: &crate::core::wordtrie::WordTrie,
    ) -> Vec<(String, f64, u32)> {
        let roman = roman.to_ascii_lowercase();
        let edges_by_pos = self.build_edges(&roman);
        let m = roman.len();
        if m == 0 {
            return vec![];
        }
        let k = k.max(1);

        let mut arena: Vec<PathCell> = Vec::with_capacity(256);
        #[derive(Clone)]
        struct St {
            pos: usize,
            prev: Option<u32>,
            prev2: Option<u32>,
            score: f64,
            path: Option<u32>,
            wnode: usize,
        }
        let mut beam = vec![St {
            pos: 0,
            prev: None,
            prev2: None,
            score: 0.0,
            path: None,
            wnode: 0,
        }];
        let mut seen: FxHashMap<String, (f64, u32)> = FxHashMap::default();

        for _step in 0..MAX_STEPS {
            if beam.is_empty() {
                break;
            }
            let mut next: Vec<St> = Vec::with_capacity(beam.len() * 8);
            for st in &beam {
                if st.pos == m {
                    if let Some(f) = trie.freq(st.wnode) {
                        let (dev, _) = self.reconstruct(&arena, st.path);
                        seen.entry(dev)
                            .and_modify(|best| {
                                if st.score < best.0 {
                                    *best = (st.score, f);
                                }
                            })
                            .or_insert((st.score, f));
                    }
                    continue;
                }
                for &e in &edges_by_pos[st.pos] {
                    if let Some(wn) = trie.child(st.wnode, e.a) {
                        let fluency = match (st.prev2, st.prev) {
                            (_, None) => self.model.start_weight(e.a),
                            (None, Some(b)) => self.model.bigram_weight(b, e.a),
                            (Some(a), Some(b)) => self.model.trigram_weight(a, b, e.a),
                        };
                        let score = st.score + e.w as f64 + fluency * self.config.lm_weight;
                        let cell = PathCell {
                            parent: st.path,
                            akshara: e.a,
                        };
                        let idx = arena.len() as u32;
                        arena.push(cell);
                        next.push(St {
                            pos: st.pos + e.len,
                            prev: Some(e.a),
                            prev2: st.prev,
                            score,
                            path: Some(idx),
                            wnode: wn,
                        });
                    }
                }
            }
            // Trie node identity subsumes the path hash: a node is reached by
            // exactly one akshara sequence.  Dedup by (pos, prev2, prev, node).
            next.sort_by(|a, b| {
                a.pos
                    .cmp(&b.pos)
                    .then_with(|| a.prev.cmp(&b.prev))
                    .then_with(|| a.prev2.cmp(&b.prev2))
                    .then_with(|| a.wnode.cmp(&b.wnode))
                    .then_with(|| a.score.total_cmp(&b.score))
            });
            next.dedup_by(|a, b| {
                a.pos == b.pos && a.prev == b.prev && a.prev2 == b.prev2 && a.wnode == b.wnode
            });
            next.sort_by(|a, b| a.score.total_cmp(&b.score));
            next.truncate(self.config.beam_width.max(64) * 2);
            beam = next;
        }

        let mut out: Vec<(String, f64, u32)> =
            seen.into_iter().map(|(d, (s, f))| (d, s, f)).collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(k);
        out
    }

    /// W3: candidate generation constrained to valid dictionary words with detailed scores.
    pub fn decode_in_words_detailed(
        &self,
        roman: &str,
        k: usize,
        trie: &crate::core::wordtrie::WordTrie,
    ) -> Vec<DecodedCandidate> {
        let roman = roman.to_ascii_lowercase();
        let edges_by_pos = self.build_edges(&roman);
        let m = roman.len();
        if m == 0 {
            return vec![];
        }
        let k = k.max(1);

        let mut arena: Vec<PathCell> = Vec::with_capacity(256);
        #[derive(Clone)]
        struct St {
            pos: usize,
            prev: Option<u32>,
            prev2: Option<u32>,
            score: f64,
            emit: f64,
            lm: f64,
            path: Option<u32>,
            wnode: usize,
        }
        let mut beam = vec![St {
            pos: 0,
            prev: None,
            prev2: None,
            score: 0.0,
            emit: 0.0,
            lm: 0.0,
            path: None,
            wnode: 0,
        }];
        let mut seen: FxHashMap<String, (f64, f64, f64, usize)> = FxHashMap::default();

        for _step in 0..MAX_STEPS {
            if beam.is_empty() {
                break;
            }
            let mut next: Vec<St> = Vec::with_capacity(beam.len() * 8);
            for st in &beam {
                if st.pos == m {
                    if trie.freq(st.wnode).is_some() {
                        let (dev, aks_cnt) = self.reconstruct(&arena, st.path);
                        seen.entry(dev)
                            .and_modify(|best| {
                                if st.score < best.0 {
                                    *best = (st.score, st.emit, st.lm, aks_cnt);
                                }
                            })
                            .or_insert((st.score, st.emit, st.lm, aks_cnt));
                    }
                    continue;
                }
                for &e in &edges_by_pos[st.pos] {
                    if let Some(wn) = trie.child(st.wnode, e.a) {
                        let fluency = match (st.prev2, st.prev) {
                            (_, None) => self.model.start_weight(e.a),
                            (None, Some(b)) => self.model.bigram_weight(b, e.a),
                            (Some(a), Some(b)) => self.model.trigram_weight(a, b, e.a),
                        };
                        let emit = st.emit + e.w as f64;
                        let lm = st.lm + fluency;
                        let score = emit + lm * self.config.lm_weight;
                        let cell = PathCell {
                            parent: st.path,
                            akshara: e.a,
                        };
                        let idx = arena.len() as u32;
                        arena.push(cell);
                        next.push(St {
                            pos: st.pos + e.len,
                            prev: Some(e.a),
                            prev2: st.prev,
                            score,
                            emit,
                            lm,
                            path: Some(idx),
                            wnode: wn,
                        });
                    }
                }
            }
            next.sort_by(|a, b| {
                a.pos
                    .cmp(&b.pos)
                    .then_with(|| a.prev.cmp(&b.prev))
                    .then_with(|| a.prev2.cmp(&b.prev2))
                    .then_with(|| a.wnode.cmp(&b.wnode))
                    .then_with(|| a.score.total_cmp(&b.score))
            });
            next.dedup_by(|a, b| {
                a.pos == b.pos && a.prev == b.prev && a.prev2 == b.prev2 && a.wnode == b.wnode
            });
            next.sort_by(|a, b| a.score.total_cmp(&b.score));
            next.truncate(self.config.beam_width.max(64) * 2);
            beam = next;
        }

        let mut out: Vec<DecodedCandidate> = seen
            .into_iter()
            .map(|(dev, (_, emit, lm, akshara_count))| DecodedCandidate {
                dev,
                emit,
                lm,
                akshara_count,
            })
            .collect();
        out.sort_by(|a, b| {
            let at = a.emit + a.lm * self.config.lm_weight;
            let bt = b.emit + b.lm * self.config.lm_weight;
            at.total_cmp(&bt)
        });
        out.truncate(k);
        out
    }

    /// W3: Candidate Union across heterogeneous generators.
    /// Merges beam search candidates with word-trie dictionary candidates.
    pub fn decode_union(
        &self,
        roman: &str,
        k: usize,
        trie: Option<&crate::core::wordtrie::WordTrie>,
    ) -> Vec<DecodedCandidate> {
        let trie_only = crate::core::ablation::trie_only() && trie.is_some();
        let mut cands = if trie_only {
            Vec::new()
        } else {
            self.decode_detailed(roman, k)
        };
        if let Some(t) = trie.filter(|_| trie_only || !crate::core::ablation::no_trie_union()) {
            let in_words = self.decode_in_words_detailed(roman, k, t);
            let mut seen: FxHashSet<String> = cands.iter().map(|c| c.dev.clone()).collect();
            for c in in_words {
                if seen.insert(c.dev.clone()) {
                    cands.push(c);
                }
            }
        }
        cands
    }

    fn reconstruct(&self, arena: &[PathCell], path: Option<u32>) -> (String, usize) {
        let mut aks = Vec::with_capacity(12);
        let mut cur = path;
        while let Some(i) = cur {
            let cell = &arena[i as usize];
            aks.push(cell.akshara);
            cur = cell.parent;
        }
        aks.reverse();
        let count = aks.len();
        let mut s = String::with_capacity(count * 3);
        for a in aks {
            if let Some(ak) = self.model.aksharas.get(a as usize) {
                s.push_str(ak);
            }
        }
        (s, count)
    }

    fn build_edges(&self, roman: &str) -> Vec<Vec<Edge>> {
        let m = roman.len();
        let mut edges = vec![Vec::new(); m + 1];
        for pos in 0..m {
            for l in 1..=MAX_CHUNK.min(m - pos) {
                let chunk = &roman[pos..pos + l];
                if let Some(list) = self.reverse.get(chunk) {
                    for &(a, w) in list {
                        edges[pos].push(Edge { len: l, a, w });
                    }
                }
            }
        }
        edges
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::em_trainer::{Trainer, TrainerConfig};

    fn trained_model(pairs: &[(&str, &str)]) -> TranslitModel {
        let mut t = Trainer::new();
        for (r, d) in pairs {
            t.add_pair(r, d);
        }
        t.finalize(&TrainerConfig {
            iterations: 6,
            ..Default::default()
        })
    }

    #[test]
    fn decodes_single_akshara() {
        let model = trained_model(&[("ka", "क"), ("ki", "कि"), ("ku", "कु")]);
        let dec = ModelDecoder::new(model);
        let res = dec.decode("ka", 5);
        assert!(!res.is_empty());
        assert_eq!(res[0].0, "क");
    }

    #[test]
    fn decodes_word_with_matra() {
        let model = trained_model(&[("ka", "क"), ("ki", "कि"), ("kama", "कम"), ("nama", "नम")]);
        let dec = ModelDecoder::new(model);
        let res = dec.decode("kama", 5);
        assert!(!res.is_empty());
        assert_eq!(res[0].0, "कम");
    }

    #[test]
    fn decodes_namaste() {
        let model = trained_model(&[("namaste", "नमस्ते"), ("nama", "नम"), ("nadi", "नदी")]);
        let dec = ModelDecoder::new(model);
        let res = dec.decode("namaste", 8);
        assert!(!res.is_empty());
        assert!(res.iter().any(|(d, _)| d == "नमस्ते"));
        assert_eq!(res[0].0, "नमस्ते");
    }

    #[test]
    fn decodes_handles_empty() {
        let model = trained_model(&[("ka", "क")]);
        let dec = ModelDecoder::new(model);
        assert!(dec.decode("", 5).is_empty());
    }
}
