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
    /// Each akshara's lexicon key bytes (`lexicon::encode_key`), by id.
    akshara_keys: Vec<Option<Vec<u8>>>,
    /// Search / scoring configuration.
    pub config: DecoderConfig,
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

/// A position in a dictionary while a word is spelled out, opaque to the
/// decoder (a trie node, or an automaton address plus accumulated output).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DictState(pub u64, pub u64);

/// A dictionary the lattice can be intersected with: words spelled akshara
/// by akshara from `root`.
pub trait Dictionary {
    fn root(&self) -> DictState;
    /// Extend a prefix by one akshara; `None` if no word continues that way.
    fn step(&self, state: DictState, akshara: u32) -> Option<DictState>;
    /// Whether the prefix spelled so far is a whole word.
    fn is_word(&self, state: DictState) -> bool;
}

/// The per-language lexicon automaton as a decoding dictionary.  `keys[a]`
/// is akshara `a`'s key bytes ([`ModelDecoder::akshara_keys`]).
pub struct LexiconDict<'a> {
    pub lexicon: &'a crate::core::lexicon::Lexicon,
    pub keys: &'a [Option<Vec<u8>>],
}

impl Dictionary for LexiconDict<'_> {
    fn root(&self) -> DictState {
        let (addr, out) = self.lexicon.root().to_raw();
        DictState(addr, out)
    }
    fn step(&self, state: DictState, akshara: u32) -> Option<DictState> {
        let key = self.keys.get(akshara as usize)?.as_deref()?;
        let next = self.lexicon.step(
            crate::core::lexicon::LexState::from_raw(state.0, state.1),
            key,
        )?;
        let (addr, out) = next.to_raw();
        Some(DictState(addr, out))
    }
    fn is_word(&self, state: DictState) -> bool {
        self.lexicon
            .word_levels(crate::core::lexicon::LexState::from_raw(state.0, state.1))
            .is_some()
    }
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
        let akshara_keys = model
            .aksharas
            .iter()
            .map(|a| crate::core::lexicon::encode_key(a))
            .collect();
        Self {
            model,
            reverse,
            akshara_keys,
            config,
        }
    }

    /// Lexicon key bytes of every akshara, by id (for [`LexiconDict`]).
    pub fn akshara_keys(&self) -> &[Option<Vec<u8>>] {
        &self.akshara_keys
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
    ///
    /// Position-synchronous beam search over the lattice: `stacks[p]` holds
    /// the partial paths that have consumed exactly `p` roman bytes, so a
    /// beam only ever compares paths that have explained the same input.
    /// (A single beam over paths of mixed length favours the ones that have
    /// read the least -- their cost is lower only because they are shorter --
    /// and pruned the gold spelling of long-chunk words: widening that beam
    /// from 64 to 256 raised the in-top-50 rate by 5-10 points for Hindi,
    /// Konkani and Bodo.)
    ///
    /// Paths in one stack that spell the same akshara sequence differ only in
    /// how the roman was segmented; their futures are identical, so only the
    /// cheapest is kept (Viterbi recombination).  Distinct spellings are never
    /// merged: the decoder returns the k best distinct strings.
    pub fn decode_detailed(&self, roman: &str, k: usize) -> Vec<DecodedCandidate> {
        let roman = roman.to_ascii_lowercase();
        // Exp 2: the lattice indexes bytes; non-ASCII input can never match an
        // ASCII chunk and previously panicked on non-char-boundary slicing.
        if !roman.is_ascii() {
            return vec![];
        }
        let edges_by_pos = self.build_edges(&roman);
        let m = roman.len();
        if m == 0 {
            return vec![];
        }
        let k = k.max(1);
        let width = self.config.beam_width.max(1);

        /// A partial path: its arena cell is written only if it survives
        /// pruning in its own stack.
        struct Hyp {
            prev2: Option<u32>,
            prev: Option<u32>,
            score: f64,
            emit: f64,
            lm: f64,
            phash: u64,
            parent: Option<u32>,
        }
        // Persistent path arena: extending a path is O(1) (one cons cell).
        let mut arena: Vec<PathCell> = Vec::with_capacity(1024);
        let mut stacks: Vec<Vec<Hyp>> = (0..=m).map(|_| Vec::new()).collect();
        stacks[0].push(Hyp {
            prev2: None,
            prev: None,
            score: 0.0,
            emit: 0.0,
            lm: 0.0,
            phash: 0,
            parent: None,
        });

        // Keep the `width` cheapest distinct akshara sequences of a stack.  A
        // linear-time selection first cuts the stack (thousands of entries)
        // to a few times the width, so only that remainder is sorted for
        // recombination; alternative segmentations of one string rarely fill
        // more than that margin.
        let prune = |stack: &mut Vec<Hyp>| {
            let margin = width * 4;
            if stack.len() > margin {
                stack.select_nth_unstable_by(margin, |a, b| a.score.total_cmp(&b.score));
                stack.truncate(margin);
            }
            stack.sort_unstable_by(|a, b| a.phash.cmp(&b.phash).then(a.score.total_cmp(&b.score)));
            stack.dedup_by_key(|h| h.phash);
            if stack.len() > width {
                stack.select_nth_unstable_by(width, |a, b| a.score.total_cmp(&b.score));
                stack.truncate(width);
            }
        };

        for pos in 0..m {
            let mut stack = std::mem::take(&mut stacks[pos]);
            if stack.is_empty() {
                continue;
            }
            prune(&mut stack);
            for h in stack {
                // `prev` is the akshara this hypothesis ends in; materialise
                // it so its extensions can point at it.
                let path = match h.prev {
                    Some(a) => {
                        arena.push(PathCell {
                            parent: h.parent,
                            akshara: a,
                        });
                        Some(arena.len() as u32 - 1)
                    }
                    None => None,
                };
                for &e in &edges_by_pos[pos] {
                    let fluency = match (h.prev2, h.prev) {
                        (_, None) => self.model.start_weight(e.a),
                        (None, Some(b)) => self.model.bigram_weight(b, e.a),
                        (Some(a), Some(b)) if !crate::core::ablation::no_trigram() => {
                            self.model.trigram_weight(a, b, e.a)
                        }
                        (Some(_), Some(b)) => self.model.bigram_weight(b, e.a),
                    };
                    let emit = h.emit + e.w as f64;
                    let lm = h.lm + fluency;
                    stacks[pos + e.len].push(Hyp {
                        prev2: h.prev,
                        prev: Some(e.a),
                        score: emit + lm * self.config.lm_weight,
                        emit,
                        lm,
                        phash: path_hash(h.phash, e.a),
                        parent: path,
                    });
                }
            }
        }

        // Complete paths: keep the best alignment of every distinct string
        // (the same bounded pruning, with room for k results), then charge
        // the end-of-word transition.
        let mut done = std::mem::take(&mut stacks[m]);
        let keep = width.max(k) * 4;
        if done.len() > keep {
            done.select_nth_unstable_by(keep, |a, b| a.score.total_cmp(&b.score));
            done.truncate(keep);
        }
        done.sort_unstable_by(|a, b| a.phash.cmp(&b.phash).then(a.score.total_cmp(&b.score)));
        done.dedup_by_key(|h| h.phash);
        let mut seen: FxHashMap<String, (f64, f64, f64, usize)> = FxHashMap::default();
        for h in done {
            let Some(last) = h.prev else { continue };
            arena.push(PathCell {
                parent: h.parent,
                akshara: last,
            });
            let (dev, count) = self.reconstruct(&arena, Some(arena.len() as u32 - 1));
            let eow = self.model.end_weight(h.prev2, h.prev);
            let (score, lm) = (h.score + eow * self.config.lm_weight, h.lm + eow);
            seen.entry(dev)
                .and_modify(|best| {
                    if score < best.0 {
                        *best = (score, h.emit, lm, count);
                    }
                })
                .or_insert((score, h.emit, lm, count));
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
            at.total_cmp(&bt).then_with(|| a.dev.cmp(&b.dev))
        });
        results.truncate(k);
        results
    }

    /// Walk a persistent path in the arena back to the root, producing the
    /// Devanagari string and its akshara count.
    /// Decode restricted to real words: the lattice intersected with the word
    /// trie.  Every edge must extend the current trie node, and only paths
    /// that end on a word (a trie terminal) are returned.
    ///
    /// Same position-synchronous search as [`Self::decode_detailed`], with the
    /// same recombination by akshara sequence (a minimal automaton shares
    /// states between different prefixes, so its state is not an identity):
    /// the cheapest segmentation of each word prefix survives.
    pub fn decode_in_words_detailed(
        &self,
        roman: &str,
        k: usize,
        dict: &dyn Dictionary,
    ) -> Vec<DecodedCandidate> {
        let roman = roman.to_ascii_lowercase();
        if !roman.is_ascii() {
            return vec![];
        }
        let edges_by_pos = self.build_edges(&roman);
        let m = roman.len();
        if m == 0 {
            return vec![];
        }
        let k = k.max(1);
        // Most edges die on the trie, so a wider beam is nearly free here.
        let width = self.config.beam_width * 2;

        struct Hyp {
            prev2: Option<u32>,
            prev: Option<u32>,
            score: f64,
            emit: f64,
            lm: f64,
            phash: u64,
            parent: Option<u32>,
            ws: DictState,
        }
        let mut arena: Vec<PathCell> = Vec::with_capacity(256);
        let mut stacks: Vec<Vec<Hyp>> = (0..=m).map(|_| Vec::new()).collect();
        stacks[0].push(Hyp {
            prev2: None,
            prev: None,
            score: 0.0,
            emit: 0.0,
            lm: 0.0,
            phash: 0,
            parent: None,
            ws: dict.root(),
        });
        let prune = |stack: &mut Vec<Hyp>| {
            stack.sort_unstable_by(|a, b| a.phash.cmp(&b.phash).then(a.score.total_cmp(&b.score)));
            stack.dedup_by_key(|h| h.phash);
            if stack.len() > width {
                stack.select_nth_unstable_by(width, |a, b| a.score.total_cmp(&b.score));
                stack.truncate(width);
            }
        };

        for pos in 0..m {
            let mut stack = std::mem::take(&mut stacks[pos]);
            if stack.is_empty() {
                continue;
            }
            prune(&mut stack);
            for h in stack {
                let path = match h.prev {
                    Some(a) => {
                        arena.push(PathCell {
                            parent: h.parent,
                            akshara: a,
                        });
                        Some(arena.len() as u32 - 1)
                    }
                    None => None,
                };
                for &e in &edges_by_pos[pos] {
                    let Some(ws) = dict.step(h.ws, e.a) else {
                        continue;
                    };
                    let fluency = match (h.prev2, h.prev) {
                        (_, None) => self.model.start_weight(e.a),
                        (None, Some(b)) => self.model.bigram_weight(b, e.a),
                        (Some(a), Some(b)) if !crate::core::ablation::no_trigram() => {
                            self.model.trigram_weight(a, b, e.a)
                        }
                        (Some(_), Some(b)) => self.model.bigram_weight(b, e.a),
                    };
                    let emit = h.emit + e.w as f64;
                    let lm = h.lm + fluency;
                    stacks[pos + e.len].push(Hyp {
                        prev2: h.prev,
                        prev: Some(e.a),
                        score: emit + lm * self.config.lm_weight,
                        emit,
                        lm,
                        phash: path_hash(h.phash, e.a),
                        parent: path,
                        ws,
                    });
                }
            }
        }

        let mut done = std::mem::take(&mut stacks[m]);
        done.sort_unstable_by(|a, b| a.phash.cmp(&b.phash).then(a.score.total_cmp(&b.score)));
        done.dedup_by_key(|h| h.phash);
        let mut out: Vec<DecodedCandidate> = Vec::new();
        for h in done {
            let Some(last) = h.prev.filter(|_| dict.is_word(h.ws)) else {
                continue;
            };
            arena.push(PathCell {
                parent: h.parent,
                akshara: last,
            });
            let (dev, akshara_count) = self.reconstruct(&arena, Some(arena.len() as u32 - 1));
            let eow = self.model.end_weight(h.prev2, h.prev);
            out.push(DecodedCandidate {
                dev,
                emit: h.emit,
                lm: h.lm + eow,
                akshara_count,
            });
        }
        out.sort_by(|a, b| {
            let at = a.emit + a.lm * self.config.lm_weight;
            let bt = b.emit + b.lm * self.config.lm_weight;
            at.total_cmp(&bt).then_with(|| a.dev.cmp(&b.dev))
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
        trie: Option<&dyn Dictionary>,
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
        for (pos, slot) in edges.iter_mut().enumerate().take(m) {
            // Byte-safe: skip positions that are not char boundaries (only
            // reachable if a non-ASCII input slipped past the caller guard).
            if !roman.is_char_boundary(pos) {
                continue;
            }
            for l in 1..=MAX_CHUNK.min(m - pos) {
                let chunk = match roman.get(pos..pos + l) {
                    Some(c) => c,
                    None => break,
                };
                if let Some(list) = self.reverse.get(chunk) {
                    for &(a, w) in list {
                        slot.push(Edge { len: l, a, w });
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
    fn completing_a_word_pays_the_end_of_word_transition() {
        // कि only ever starts words here, की only ever ends them.  The LM
        // learns P(</w> | .. की) >> P(</w> | .. कि); a decoder that never
        // charges the end-of-word transition cannot use that, and a packer
        // that prunes </w> erases it (both shipped before this test).
        let model = trained_model(&[
            ("kitab", "किताब"),
            ("kinara", "किनारा"),
            ("kisan", "किसान"),
            ("naki", "नकी"),
            ("baki", "बकी"),
            ("saki", "सकी"),
            ("ki", "कि"),
            ("ki", "की"),
        ]);
        let short = model.akshara_id("कि").unwrap();
        let long = model.akshara_id("की").unwrap();
        let na = model.akshara_id("न").unwrap();
        assert!(
            model.end_weight(Some(na), Some(long)) < model.end_weight(Some(na), Some(short)),
            "ending in की must be cheaper than ending in कि"
        );
        let dec = ModelDecoder::new(model);
        let res = dec.decode("naki", 5);
        assert_eq!(res[0].0, "नकी", "word-final 'ki' should end in की: {res:?}");
    }

    #[test]
    fn decodes_handles_empty() {
        let model = trained_model(&[("ka", "क")]);
        let dec = ModelDecoder::new(model);
        assert!(dec.decode("", 5).is_empty());
    }

    #[test]
    fn decode_never_panics_on_non_ascii() {
        // Exp 2 (H2): pasted Devanagari / accented / emoji input must yield
        // [] instead of panicking on byte slicing.
        let model = trained_model(&[("ka", "क")]);
        let dec = ModelDecoder::new(model);
        for q in ["café", "नमस्ते", "Zürich", "🙏", "naमस्ते"] {
            assert!(dec.decode(q, 5).is_empty(), "query {q:?}");
            assert!(dec.decode_detailed(q, 5).is_empty(), "query {q:?}");
        }
    }
}
