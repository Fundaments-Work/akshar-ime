// File: src/core/translit_model.rs
//
// The generative transliteration model, trained by EM (Li, Zhang & Su, ACL-04,
// "A Joint Source-Channel Model for Machine Transliteration"):
//
//   P(R | D)  =  sum over segmentations of R into chunks s_1..s_n aligned to
//               the aksharas a_1..a_n of D, of  prod_j P(s_j | a_j)
//
// plus a Kneser-Ney smoothed akshara n-gram language model (word-start prior,
// bigram and trigram) used for scoring fresh (out-of-dictionary)
// transliterations.
//
// Everything is stored as additive negative-log weights (tropical semiring),
// so decoding is shortest-path / Viterbi over the akshara lattice.

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

/// Look up `id` in an adjacency row.
///
/// Rows are stored ascending by id -- the trainer sorts them and the container
/// delta-encodes them, which would be incorrect for any other order -- so this
/// is a binary search.  It used to be a linear scan, and with ~5,000 hypotheses
/// expanded per beam step it was the decoder's dominant cost.
#[inline]
fn find_weight(row: &[(u32, f32)], id: u32) -> Option<f32> {
    row.binary_search_by_key(&id, |(k, _)| *k)
        .ok()
        .map(|i| row[i].1)
}

/// Number of leading chars of the file format we can bump when the layout
/// changes.  Kept small and human readable.
pub const MODEL_VERSION: u32 = 2;

/// Synthetic end-of-word akshara.  The trainer appends it to every word for
/// LM counting, so P(</w> | a, b) is the probability that a word ends after
/// `a b`; it has no emissions, so the decoder never generates it.
pub const END_OF_WORD: &str = "</w>";

/// The compact, serialisable transliteration model.
///
/// A [u32] id in `aksharas` / `chunks` is an index into the corresponding
/// vocabulary vector.  Weight fields are -log probability (lower = better).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TranslitModel {
    pub version: u32,
    /// Devanagari akshara vocabulary (index = akshara id).
    pub aksharas: Vec<String>,
    /// Roman chunk vocabulary, lowercase a-z, length 1..=MAX_CHUNK.
    pub chunks: Vec<String>,
    /// Per-akshara emission list: (chunk_id, -log P(chunk | akshara)).
    pub emissions: Vec<Vec<(u32, f32)>>,
    /// Per-akshara akshara bigram list: (next_akshara_id, -log P_KN(next | this)).
    pub bigrams: Vec<Vec<(u32, f32)>>,
    /// Per-akshara bigram backoff weight (additive, added when `next` is unseen).
    pub backoff: Vec<f32>,
    /// -log P_KN(akshara) continuation unigram (for backing off to unigram).
    pub unigram_kn: Vec<f32>,
    /// -log P(akshara | word start) from corpus word-initial frequencies.
    pub word_start: Vec<f32>,
    /// Trigram LM: per (a,b) context, the successors and their -log P_KN(c|a,b).
    /// `trigram_keys[i]` is the (a,b) context for `trigrams[i]`.
    pub trigram_keys: Vec<(u32, u32)>,
    pub trigrams: Vec<Vec<(u32, f32)>>,
    /// -log lambda(a,b) for backing off to the bigram P_KN(c|b).
    pub trigram_backoff: Vec<f32>,
    /// Runtime index from (a,b) -> position in trigram_keys.  Not serialised.
    #[serde(skip)]
    pub trigram_index: FxHashMap<(u32, u32), usize>,
    /// Runtime index from akshara string -> id.  Not serialised; built on load.
    #[serde(skip)]
    pub akshara_index: FxHashMap<String, u32>,
    /// Runtime id of [`END_OF_WORD`], if the model was trained with it.
    #[serde(skip)]
    pub eow_id: Option<u32>,
}

/// Maximum roman characters a single akshara can absorb.
pub const MAX_CHUNK: usize = 5;

/// Bit position of the packed chunk-length field (must sit above the character
/// bits: 5 bits per char for MAX_CHUNK chars, then 4 length bits).
const LEN_SHIFT: u32 = 26;

impl TranslitModel {
    /// Return (akshara_id, chunk_id) maps and helper lookup methods.
    pub fn akshara_id(&self, akshara: &str) -> Option<u32> {
        // Indexed lookup.  This used to be a linear scan over the whole akshara
        // vocabulary; engine start-up calls it hundreds of thousands of times
        // while building the word trie, which made it the dominant term in a
        // multi-second cold start.
        if !self.akshara_index.is_empty() {
            return self.akshara_index.get(akshara).copied();
        }
        self.aksharas
            .iter()
            .position(|a| a == akshara)
            .map(|i| i as u32)
    }

    /// Emission weight of akshara emitting `chunk` (-log P), +inf if unseen.
    pub fn emission_weight(&self, akshara_id: u32, chunk: &str) -> f64 {
        let Some(chunk_id) = self
            .chunks
            .iter()
            .position(|c| c == chunk)
            .map(|i| i as u32)
        else {
            return f64::INFINITY;
        };
        self.emissions
            .get(akshara_id as usize)
            .and_then(|list| find_weight(list, chunk_id))
            .map(|w| w as f64)
            .unwrap_or(f64::INFINITY)
    }

    /// Emission probability P(chunk | akshara) directly.
    pub fn emission_prob(&self, akshara_id: u32, chunk: &str) -> f64 {
        let w = self.emission_weight(akshara_id, chunk);
        if w == f64::INFINITY {
            0.0
        } else {
            (-w).exp()
        }
    }

    /// Top chunks emitted by an akshara (for diagnostics / decoder candidate gen).
    pub fn top_emissions(&self, akshara_id: u32, k: usize) -> Vec<(String, f64)> {
        self.emissions
            .get(akshara_id as usize)
            .map(|list| {
                let mut v: Vec<(String, f64)> = list
                    .iter()
                    .filter_map(|(cid, w)| {
                        self.chunks
                            .get(*cid as usize)
                            .map(|c| (c.clone(), *w as f64))
                    })
                    .collect();
                v.sort_by(|a, b| a.1.total_cmp(&b.1));
                v.truncate(k);
                v
            })
            .unwrap_or_default()
    }

    /// Bigram weight between consecutive aksharas (-log P_KN), with backoff.
    pub fn bigram_weight(&self, a: u32, b: u32) -> f64 {
        if let Some(list) = self.bigrams.get(a as usize) {
            if let Some(w) = find_weight(list, b) {
                return w as f64;
            }
        }
        let backoff = self.backoff.get(a as usize).copied().unwrap_or(0.0) as f64;
        let uni = self.unigram_kn.get(b as usize).copied().unwrap_or(0.0) as f64;
        backoff + uni
    }

    /// -log P_KN(b | a) but clipped so it stays finite even for totally unseen ids.
    pub fn safe_bigram_weight(&self, a: u32, b: u32) -> f64 {
        if a as usize >= self.bigrams.len() || b as usize >= self.aksharas.len() {
            return 15.0;
        }
        self.bigram_weight(a, b).min(25.0)
    }

    /// Trigram weight -log P_KN(c | a, b) with backoff to the bigram.
    pub fn trigram_weight(&self, a: u32, b: u32, c: u32) -> f64 {
        if let Some(i) = self.trigram_index.get(&(a, b)) {
            if let Some(w) = find_weight(&self.trigrams[*i], c) {
                return w as f64;
            }
            let backoff = self.trigram_backoff.get(*i).copied().unwrap_or(0.0) as f64;
            return backoff + self.bigram_weight(b, c);
        }
        // (a,b) context unseen in training: pure bigram.
        self.bigram_weight(b, c)
    }

    /// -log P_KN(c | a, b), clipped to stay finite for pathological ids.
    pub fn safe_trigram_weight(&self, a: u32, b: u32, c: u32) -> f64 {
        if a as usize >= self.aksharas.len()
            || b as usize >= self.aksharas.len()
            || c as usize >= self.aksharas.len()
        {
            return 15.0;
        }
        self.trigram_weight(a, b, c).min(30.0)
    }

    /// Word-start prior weight for an akshara (-log P(a | word start)).
    pub fn start_weight(&self, a: u32) -> f64 {
        self.word_start.get(a as usize).copied().unwrap_or(12.0) as f64
    }

    /// End-of-word weight -log P(</w> | prev2, prev): the cost of the word
    /// ending after its last two aksharas.  Completing a path must pay it,
    /// exactly as every other akshara pays its LM transition; without it the
    /// LM cannot tell a plausible ending from an implausible one, which is
    /// where vowel-sign errors concentrate.  Zero for models trained without
    /// the token, and under the `AKSHAR_NO_EOW` ablation.
    pub fn end_weight(&self, prev2: Option<u32>, prev: Option<u32>) -> f64 {
        let (Some(eow), Some(b)) = (self.eow_id, prev) else {
            return 0.0;
        };
        if crate::core::ablation::no_eow() {
            return 0.0;
        }
        match prev2 {
            Some(a) if !crate::core::ablation::no_trigram() => self.trigram_weight(a, b, eow),
            _ => self.bigram_weight(b, eow),
        }
    }

    /// Serialise to `path` with bincode.
    pub fn save(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::BufWriter;
        let file = std::fs::File::create(path)?;
        let mut w = BufWriter::new(file);
        bincode::serialize_into(&mut w, self)?;
        Ok(())
    }

    /// Load a bincode-serialised model from `path`.
    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        use std::io::BufReader;
        let file = std::fs::File::open(path)?;
        let r = BufReader::new(file);
        let mut m: Self = bincode::deserialize_from(r)?;
        m.build_trigram_index();
        Ok(m)
    }

    /// Load from raw bytes (WASM-friendly, no filesystem).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let mut m: Self = bincode::deserialize(bytes)?;
        m.build_trigram_index();
        Ok(m)
    }

    /// Serialise to bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        Ok(bincode::serialize(self)?)
    }

    /// Rebuild the runtime trigram index (called after load / finalise).
    pub fn build_trigram_index(&mut self) {
        self.trigram_index = self
            .trigram_keys
            .iter()
            .enumerate()
            .map(|(i, &k)| (k, i))
            .collect();
        self.build_akshara_index();
    }

    /// Rebuild the runtime akshara-string -> id index.
    pub fn build_akshara_index(&mut self) {
        self.akshara_index = self
            .aksharas
            .iter()
            .enumerate()
            .map(|(i, a)| (a.clone(), i as u32))
            .collect();
        // Only a model that kept its end-of-word transitions can score word
        // endings.  Packers before the fix pruned every transition into </w>
        // while keeping the akshara, and backing off from there yields a
        // per-context constant, not P(</w> | ...): such models must decode
        // exactly as they were trained to.
        self.eow_id = self.akshara_index.get(END_OF_WORD).copied().filter(|&eow| {
            self.bigrams
                .iter()
                .any(|row| find_weight(row, eow).is_some())
        });
    }

    /// Basic sanity: model is non-empty and internally consistent.
    pub fn validate(&self) -> bool {
        if self.aksharas.is_empty() {
            return false;
        }
        let n = self.aksharas.len();
        if self.emissions.len() != n
            || self.bigrams.len() != n
            || self.backoff.len() != n
            || self.unigram_kn.len() != n
            || self.word_start.len() != n
        {
            return false;
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Internal chunk packing: a chunk of up to 4 lowercase ASCII letters is packed
// into a u32 (5 bits per letter + 4 length bits).  Used by the trainer for
// allocation-free emission lookups.
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn pack_chunk(chunk: &str) -> u32 {
    pack_chunk_bytes(chunk.as_bytes())
}

/// Pack up to MAX_CHUNK lowercase ASCII bytes into a u32 (5 bits each + length).
#[inline]
pub fn pack_chunk_bytes(bytes: &[u8]) -> u32 {
    let mut v = 0u32;
    for (i, &b) in bytes.iter().enumerate().take(MAX_CHUNK) {
        let code = if b.is_ascii_lowercase() {
            (b - b'a') as u32
        } else {
            26
        };
        v |= code << (i * 5);
    }
    v | ((bytes.len().min(MAX_CHUNK) as u32) << LEN_SHIFT)
}

pub(crate) fn unpack_chunk(packed: u32) -> String {
    let len = ((packed >> LEN_SHIFT) & 0xF) as usize;
    let mut s = String::with_capacity(len);
    for i in 0..len {
        let code = ((packed >> (i * 5)) & 0x1F) as u8;
        if code < 26 {
            s.push((b'a' + code) as char);
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_roundtrip() {
        for s in ["a", "ka", "krya", ""] {
            assert_eq!(unpack_chunk(pack_chunk(s)), s);
        }
    }

    #[test]
    fn pack_handles_non_ascii_with_escape() {
        // Uppercase chars pack as 26 (escape); roundtrip is not identity.
        let packed = pack_chunk("sTe");
        assert_ne!(unpack_chunk(packed), "sTe");
    }

    #[test]
    fn empty_model_is_invalid() {
        assert!(!TranslitModel::default().validate());
    }

    #[test]
    fn emission_weight_inf_for_unknown() {
        let m = TranslitModel {
            version: 1,
            aksharas: vec!["क".to_string()],
            chunks: vec!["ka".to_string()],
            emissions: vec![vec![(0u32, 0.0f32)]],
            bigrams: vec![vec![]],
            backoff: vec![0.0f32],
            unigram_kn: vec![0.0f32],
            word_start: vec![0.0f32],
            trigram_keys: vec![],
            trigrams: vec![],
            trigram_backoff: vec![],
            trigram_index: FxHashMap::default(),
            akshara_index: FxHashMap::default(),
            eow_id: None,
        };
        assert!(m.validate());
        assert_eq!(m.emission_weight(0, "ka"), 0.0);
        assert_eq!(m.emission_weight(0, "k"), f64::INFINITY);
    }

    #[test]
    fn bigram_backoff_uses_unigram() {
        let m = TranslitModel {
            version: 1,
            aksharas: vec!["क".to_string(), "र".to_string()],
            chunks: vec![],
            emissions: vec![vec![], vec![]],
            bigrams: vec![vec![(1u32, 2.0f32)], vec![]],
            backoff: vec![3.0f32, 0.0f32],
            unigram_kn: vec![4.0f32, 5.0f32],
            word_start: vec![6.0f32, 7.0f32],
            trigram_keys: vec![],
            trigrams: vec![],
            trigram_backoff: vec![],
            trigram_index: FxHashMap::default(),
            akshara_index: FxHashMap::default(),
            eow_id: None,
        };
        // seen: direct weight
        assert_eq!(m.bigram_weight(0, 1), 2.0);
        // unseen b=0 from a=0: backoff + unigram
        assert_eq!(m.bigram_weight(0, 0), 3.0 + 4.0);
        // a=1 has empty bigram list, backoff 0
        assert_eq!(m.bigram_weight(1, 1), 0.0 + 5.0);
    }
}
