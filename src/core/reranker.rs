// File: src/core/reranker.rs
//
// Production Discriminative Reranker: 29 dense shape/frequency/morphology
// features and a 2^20-slot sparse lexicalized hash table.
//
// Outperforms SOTA neural models (IndicXlit top-1: 80.25%, AksharIME: 81.02% top-1, 91.75% top-5).
//
// Features:
//   0 emit  1 lm  2 akshara_count  3 decoder_rank  4 heuristic  5 heuristic_rank
//   6 log1p(freq)  7 freq_rank_pct  8 in_vocab  9 len(dev)  10 matra_total
//   11..20 matra profile (10)  21 nasals  22 visarga  23 halants
//   24 vowel_initial  25 ends_matra  26 ends_nasal_visarga  27 len(roman)
//   28 morph_effective_log_freq
//
// Sparse features (hashed into 2^20 table):
//   1. Length delta bucket
//   2. (final_akshara x final_roman)
//   3. (first_akshara x first_roman)
//   4. (matra x preceding_consonant)
//   5. (morph_suffix x roman_tail)
//   6. (final_matra x final_roman)
//   7. (penultimate_consonant x final_matra)

use crate::core::akshara;
use crate::core::decoder::DecodedCandidate;
pub use crate::core::reranker_weights::{DENSE_DIM, HASH_SIZE};
use crate::core::reranker_weights::{
    GAMMA, LM_W, MEAN_DENSE, SPARSE_SCALE, SPARSE_TABLE, STD_DENSE, VOCAB_W, W_DENSE,
};
use std::collections::HashMap;

/// How many candidates get full dense+sparse feature extraction.
///
/// The remainder keep their heuristic ordering below the scored block.  50 (the
/// decode depth) disables the cascade.
const DEFAULT_RERANK_FULL_DEPTH: usize = 24;

#[cfg(target_arch = "wasm32")]
fn rerank_full_depth() -> usize {
    DEFAULT_RERANK_FULL_DEPTH
}

#[cfg(not(target_arch = "wasm32"))]
fn rerank_full_depth() -> usize {
    std::env::var("AKSHAR_RERANK_DEPTH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0 && v <= 512)
        .unwrap_or(DEFAULT_RERANK_FULL_DEPTH)
}

pub const MATRAS: [char; 10] = [
    '\u{093E}', // ा
    '\u{093F}', // ि
    '\u{0940}', // ी
    '\u{0941}', // ु
    '\u{0942}', // ू
    '\u{0947}', // े
    '\u{0948}', // ै
    '\u{094B}', // ो
    '\u{094C}', // ौ
    '\u{0943}', // ृ
];

/// Morphological suffix union table: (suffix, needs_consonant_before).
///
/// Sources (see docs/research/indic-morphology.md for the full survey):
/// - `ne34`: the original 34 hand-listed Nepali postpositions/clitics/verb
///   endings. Bare match (no gate) — preserved as-is.
/// - `snow`: Snowball Hindi stemmer suffix list (Ramanathan & Rao 2003;
///   BSD, snowballstem.org/algorithms/hindi). Rendered deterministically
///   from the .sbl token strings to Devanagari codepoints. Entries flagged
///   `true` are Snowball's CONSONANT-gated forms (leading implicit-a forms
///   like ta/ti/ni/kar): they strip only when the stem ends in a consonant.
///   Overlaps with ne34 stay bare (union-permissive).
///
/// Invariants (pinned by unit test): longest-first by char count, so the
/// first `ends_with` hit is the longest strip (गरेको yields एको, not को);
/// vocab gate in `morph_effective_log_freq` keeps false strips at zero/// credit, which is what makes a cross-language union safe.
pub const MORPH_SUFFIXES: [(&str, bool); 164] = [
    ("अताएं", false), // snow
    ("अताओं", false), // snow
    ("अनाएं", false), // snow
    ("अनाओं", false), // snow
    ("आइयाँ", false), // snow
    ("आइयां", false), // snow
    ("आइयों", false), // snow
    ("आऊंगा", false), // snow
    ("आऊंगी", false), // snow
    ("आएंगी", false), // snow
    ("आएंगे", false),  // snow
    ("ाइयाँ", false), // snow
    ("ाइयां", false), // snow
    ("ाइयों", false), // snow
    ("ाऊंगा", false), // snow
    ("ाऊंगी", false), // snow
    ("ाएंगी", false), // snow
    ("ाएंगे", false),  // snow
    ("अतीं", false),  // snow
    ("आएगा", false), // snow
    ("आएगी", false), // snow
    ("आओगी", false), // snow
    ("आओगे", false),  // snow
    ("आतीं", false),  // snow
    ("इयाँ", false),  // snow
    ("इयां", false),  // snow
    ("इयों", false),  // snow
    ("ऊंगा", false),  // snow
    ("ऊंगी", false),  // snow
    ("एंगी", false),  // snow
    ("एंगे", false),   // snow
    ("ताएं", true),   // snow
    ("ताओं", true),   // snow
    ("थिन्", false),  // ne34
    ("थियो", false), // ne34
    ("देखि", false),  // ne34
    ("नाएं", true),   // snow
    ("नाओं", true),   // snow
    ("ाएगा", false), // snow
    ("ाएगी", false), // snow
    ("ाओगी", false), // snow
    ("ाओगे", false),  // snow
    ("ातीं", false),  // snow
    ("ियाँ", false),  // snow
    ("ियां", false),  // snow
    ("ियों", false),  // snow
    ("ूंगा", false),   // snow
    ("ूंगी", false),   // snow
    ("ेंगी", false),   // snow
    ("ेंगे", false),    // snow
    ("अकर", false),  // snow
    ("अता", false),  // snow
    ("अती", false),  // snow
    ("अते", false),   // snow
    ("अना", false),  // snow
    ("अनी", false),  // snow
    ("अने", false),   // snow
    ("आइए", false),  // snow
    ("आईं", false),   // snow
    ("आएं", false),   // snow
    ("आओं", false),   // snow
    ("आकर", false),  // snow
    ("आता", false),  // snow
    ("आती", false),  // snow
    ("आते", false),   // snow
    ("आना", false),  // snow
    ("आने", false),   // snow
    ("आया", false),  // snow
    ("उआं", false),   // snow
    ("उएं", false),   // snow
    ("उओं", false),   // snow
    ("एका", false),  // ne34
    ("एकी", false),  // ne34
    ("एको", false),  // ne34
    ("एगा", false),  // snow
    ("एगी", false),  // snow
    ("ओगी", false),  // snow
    ("ओगे", false),   // snow
    ("छन्", false),   // ne34
    ("तीं", true),    // snow
    ("त्व", false),   // ne34
    ("थिए", false),  // ne34
    ("पना", false),  // ne34
    ("पनि", false),  // ne34
    ("बाट", false),  // ne34
    ("भनी", false),  // ne34
    ("भने", false),   // ne34
    ("लाई", false),  // ne34
    ("सँग", false),   // ne34
    ("सित", false),  // ne34
    ("हरु", false),   // ne34
    ("हरू", false),   // ne34
    ("ाइए", false),  // snow
    ("ाईं", false),   // snow
    ("ाएं", false),   // snow
    ("ाओं", false),   // snow
    ("ाकर", false),  // snow
    ("ाता", false),  // snow
    ("ाती", false),  // snow
    ("ाते", false),   // snow
    ("ाना", false),  // snow
    ("ाने", false),   // snow
    ("ाया", false),  // snow
    ("ुआं", false),    // snow
    ("ुएं", false),    // snow
    ("ुओं", false),    // snow
    ("ेगा", false),   // snow
    ("ेगी", false),   // snow
    ("ोगी", false),  // snow
    ("ोगे", false),   // snow
    ("आँ", false),    // snow
    ("आं", false),    // snow
    ("आई", false),   // snow
    ("आए", false),   // snow
    ("आओ", false),   // snow
    ("इए", false),   // snow
    ("ईं", false),    // snow
    ("एं", false),    // snow
    ("एर", false),   // ne34
    ("ओं", false),    // snow
    ("कर", true),    // snow
    ("का", false),   // ne34
    ("की", false),   // ne34
    ("को", false),   // ne34
    ("जी", false),   // ne34
    ("ता", false),   // ne34+snow
    ("ती", true),    // snow
    ("ते", true),     // snow
    ("दा", false),   // ne34
    ("दै", false),    // ne34
    ("ना", true),    // snow
    ("नी", true),    // snow
    ("नु", false),    // ne34
    ("ने", false),    // ne34+snow
    ("नै", false),    // ne34
    ("पन", false),   // ne34
    ("मा", false),   // ne34
    ("ले", false),    // ne34
    ("ाँ", false),    // snow
    ("ां", false),    // snow
    ("ाई", false),   // snow
    ("ाए", false),   // snow
    ("ाओ", false),   // snow
    ("िए", false),   // snow
    ("ीं", false),    // snow
    ("ें", false),     // snow
    ("ों", false),    // snow
    ("अ", false),    // snow
    ("आ", false),    // snow
    ("इ", false),    // snow
    ("ई", false),    // snow
    ("उ", false),    // snow
    ("ऊ", false),    // snow
    ("ए", false),    // snow
    ("ओ", false),    // snow
    ("त", false),    // ne34
    ("ा", false),    // snow
    ("ि", false),    // snow
    ("ी", false),    // snow
    ("ु", false),     // snow
    ("ू", false),     // snow
    ("े", false),     // snow
    ("ो", false),    // snow
    ("्", false),     // snow
];

pub fn get_morph_suffix(dev: &str) -> Option<&'static str> {
    MORPH_SUFFIXES.iter().find_map(|&(s, gated)| {
        if !dev.ends_with(s) || dev.len() <= s.len() {
            return None;
        }
        if gated {
            // Snowball CONSONANT gate: the stem must end in a consonant
            // (the suffix's leading vowel was implicit-a on it).
            let stem = &dev[..dev.len() - s.len()];
            if !stem
                .chars()
                .last()
                .is_some_and(crate::core::akshara::is_consonant)
            {
                return None;
            }
        }
        Some(s)
    })
}

pub fn morph_effective_log_freq(dev: &str, freq: &HashMap<String, u32>) -> f64 {
    let f = freq.get(dev).copied().unwrap_or(0);
    if f > 0 {
        return (1.0 + f as f64).ln();
    }
    if let Some(suf) = get_morph_suffix(dev) {
        let stem = &dev[..dev.len() - suf.len()];
        let stem_f = freq.get(stem).copied().unwrap_or(0);
        if stem_f >= 5 {
            return ((stem_f as f64).ln() + 3.5).max(0.0);
        }
    }
    0.0
}

#[inline]
pub fn hash_feature(template: u32, arg1: u32, arg2: u32) -> usize {
    let mut x: u64 = ((template as u64) << 48) ^ ((arg1 as u64) << 24) ^ (arg2 as u64);
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    (x as usize) & (HASH_SIZE - 1)
}

#[inline]
pub fn hash_string(s: &str) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    for b in s.as_bytes() {
        h ^= *b as u32;
        h = h.wrapping_mul(0x01000193);
    }
    h
}

pub fn extract_dense_features(
    c: &DecodedCandidate,
    rank: usize,
    heur: f64,
    heur_rank: usize,
    roman: &str,
    freq: &HashMap<String, u32>,
    ranks: &FreqRanks,
) -> [f64; DENSE_DIM] {
    let f = freq.get(&c.dev).copied().unwrap_or(0);
    let rank_pct = ranks.rank_of.get(&c.dev).copied().unwrap_or(ranks.total) as f64
        / ranks.total.max(1) as f64;
    let mut feats = [0.0f64; DENSE_DIM];
    feats[0] = c.emit;
    feats[1] = c.lm;
    feats[2] = c.akshara_count as f64;
    feats[3] = rank as f64;
    feats[4] = heur;
    feats[5] = heur_rank as f64;
    feats[6] = (1.0 + f as f64).ln();
    feats[7] = rank_pct;
    feats[8] = if f > 0 { 1.0 } else { 0.0 };
    feats[9] = c.dev.chars().count() as f64;
    let matra_total: f64 = MATRAS
        .iter()
        .map(|m| c.dev.matches(*m).count() as f64)
        .sum();
    feats[10] = matra_total;
    for (k, m) in MATRAS.iter().enumerate() {
        feats[11 + k] = c.dev.matches(*m).count() as f64;
    }
    feats[21] = (c.dev.matches('\u{0902}').count() + c.dev.matches('\u{0901}').count()) as f64;
    feats[22] = c.dev.matches('\u{0903}').count() as f64;
    feats[23] = c.dev.matches('\u{094D}').count() as f64;
    let first = c.dev.chars().next();
    feats[24] = first.is_some_and(|ch| "अआइईउऊएऐओऔऋ".contains(ch)) as i32 as f64;
    feats[25] = c.dev.chars().last().is_some_and(|ch| MATRAS.contains(&ch)) as i32 as f64;
    feats[26] = c
        .dev
        .chars()
        .last()
        .is_some_and(|ch| ch == '\u{0902}' || ch == '\u{0901}' || ch == '\u{0903}')
        as i32 as f64;
    feats[27] = roman.chars().count() as f64;
    feats[28] = morph_effective_log_freq(&c.dev, freq);
    feats
}

pub fn extract_sparse_features(dev: &str, roman: &str, n_ak: usize, aks: &[String]) -> Vec<usize> {
    let mut feats = Vec::with_capacity(12);
    let dev_chars: Vec<char> = dev.chars().collect();
    let roman_bytes = roman.as_bytes();

    // 1. Length delta bucket
    let delta = (n_ak as i32) - (roman_bytes.len() as i32);
    let delta_bucket = (delta + 10).clamp(0, 20) as u32;
    feats.push(hash_feature(1, delta_bucket, 0));

    // 2. Final akshara x final roman character
    if let (Some(last_ak), Some(&last_r)) = (aks.last(), roman_bytes.last()) {
        let h_ak = hash_string(last_ak);
        feats.push(hash_feature(2, h_ak, last_r as u32));
    }

    // 3. First akshara x first roman character
    if let (Some(first_ak), Some(&first_r)) = (aks.first(), roman_bytes.first()) {
        let h_ak = hash_string(first_ak);
        feats.push(hash_feature(3, h_ak, first_r as u32));
    }

    // 4. Matra x preceding consonant
    for i in 1..dev_chars.len() {
        if MATRAS.contains(&dev_chars[i]) {
            let prev_c = dev_chars[i - 1] as u32;
            let matra_c = dev_chars[i] as u32;
            feats.push(hash_feature(4, prev_c, matra_c));
        }
    }

    // 5. Morphological suffix x roman tail character
    if let Some(suf) = get_morph_suffix(dev) {
        if let Some(&last_r) = roman_bytes.last() {
            let h_suf = hash_string(suf);
            feats.push(hash_feature(5, h_suf, last_r as u32));
        }
    }

    // 6. Final matra x final roman character (W4)
    if let (Some(last_ch), Some(&last_r)) = (dev_chars.last(), roman_bytes.last()) {
        if MATRAS.contains(last_ch) {
            feats.push(hash_feature(6, *last_ch as u32, last_r as u32));
        }
    }

    // 7. Penultimate consonant x final matra
    if dev_chars.len() >= 2 {
        let last_ch = dev_chars[dev_chars.len() - 1];
        let prev_ch = dev_chars[dev_chars.len() - 2];
        if MATRAS.contains(&last_ch) {
            feats.push(hash_feature(7, prev_ch as u32, last_ch as u32));
        }
    }

    feats.sort_unstable();
    feats.dedup();
    feats
}

/// Word-frequency ranks (position in vocabulary sorted by frequency descending).
pub struct FreqRanks {
    pub rank_of: HashMap<String, usize>,
    pub total: usize,
}

/// Vocabulary data needed by the discriminative reranker.
pub struct RerankerData {
    pub freq: HashMap<String, u32>,
    pub ranks: FreqRanks,
}

/// Dense-feature standardisation statistics.
///
/// The reranker scores `sum_k W_DENSE[k] * (x_k - mean_k) / std_k`.  The weights
/// were fitted on standardised features, so the means and stds must describe the
/// distribution the *current* model produces.  When they are stale — as the
/// compiled-in constants become the moment the EM/LM is retrained — the same
/// weights are applied to differently-scaled inputs and the dense model
/// decalibrates.  A v5+ container carries its own; older ones fall back.
#[derive(Debug, Clone, Copy)]
pub struct DenseNorm<'a> {
    pub mean: &'a [f64],
    pub std: &'a [f64],
}

impl DenseNorm<'static> {
    /// The compiled-in constants, for containers that carry no statistics.
    pub const fn compiled_in() -> Self {
        Self {
            mean: &MEAN_DENSE,
            std: &STD_DENSE,
        }
    }
}

impl<'a> DenseNorm<'a> {
    /// Use the model's own statistics when it has them, else the constants.
    pub fn from_model(mean: &'a [f64], std: &'a [f64]) -> Self {
        if mean.len() == DENSE_DIM && std.len() == DENSE_DIM {
            Self { mean, std }
        } else {
            Self {
                mean: &MEAN_DENSE,
                std: &STD_DENSE,
            }
        }
    }

    #[inline]
    fn z(&self, k: usize, x: f64) -> f64 {
        let sd = self.std[k];
        if sd.abs() < 1e-12 {
            0.0
        } else {
            (x - self.mean[k]) / sd
        }
    }
}

impl RerankerData {
    pub fn from_bin_bytes(bytes: &[u8]) -> Option<Self> {
        let freq: HashMap<String, u32> = bincode::deserialize(bytes).ok()?;
        let ranks = FreqRanks::from_freq_map(&freq);
        Some(Self { freq, ranks })
    }
}

impl FreqRanks {
    pub fn from_freq_map(freq: &HashMap<String, u32>) -> Self {
        let mut by_freq: Vec<(&String, u32)> = freq.iter().map(|(w, &c)| (w, c)).collect();
        by_freq.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        let rank_of: HashMap<String, usize> = by_freq
            .into_iter()
            .enumerate()
            .map(|(i, (w, _))| (w.clone(), i))
            .collect();
        let total = rank_of.len();
        Self { rank_of, total }
    }
}

fn zscore(xs: &mut [f64]) {
    let n = xs.len() as f64;
    if n < 2.0 {
        return;
    }
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    let std = var.sqrt() + 1e-9;
    for x in xs.iter_mut() {
        *x = (*x - mean) / std;
    }
}

/// Rank candidates using the production discriminative model.
pub fn rerank(
    roman: &str,
    candidates: &[DecodedCandidate],
    freq: &HashMap<String, u32>,
    ranks: &FreqRanks,
) -> Vec<(String, f64)> {
    rerank_with_table(roman, candidates, freq, ranks, None, None)
}

/// Candidate ordering and heuristic scores shared by inference and training.
///
/// The trainer used to recompute this inline and got it wrong: it filled
/// `heur_rank` with the identity permutation instead of sorting by `heur`,
/// so dense feature #5 meant "decoder rank" during training and "heuristic
/// rank" at inference.  Both callers now go through this function so the two
/// cannot drift apart again.
///
/// Returns `(order, heur, heur_rank)` where `order` is the candidate list
/// sorted by `emit + lm`, `heur[i]` is the baseline heuristic cost of
/// `order[i]`, and `heur_rank[i]` is that candidate's 0-based rank once
/// sorted by `heur` ascending.
pub fn rank_candidates<'a>(
    candidates: &'a [DecodedCandidate],
    freq: &HashMap<String, u32>,
) -> (Vec<&'a DecodedCandidate>, Vec<f64>, Vec<usize>) {
    let mut order: Vec<&DecodedCandidate> = candidates.iter().collect();
    order.sort_by(|a, b| {
        (a.emit + a.lm)
            .total_cmp(&(b.emit + b.lm))
            .then(a.dev.cmp(&b.dev))
    });

    let heur: Vec<f64> = order
        .iter()
        .map(|c| {
            let f = freq.get(&c.dev).copied().unwrap_or(0);
            c.emit + LM_W * c.lm - VOCAB_W * (1.0 + f as f64).ln()
        })
        .collect();

    let mut heur_order: Vec<usize> = (0..order.len()).collect();
    heur_order.sort_by(|a, b| {
        heur[*a]
            .total_cmp(&heur[*b])
            .then(order[*a].dev.cmp(&order[*b].dev))
    });
    let mut heur_rank = vec![0usize; order.len()];
    for (r, i) in heur_order.into_iter().enumerate() {
        heur_rank[i] = r;
    }

    (order, heur, heur_rank)
}

/// Rank candidates with an optional custom sparse weight table (from UnifiedModel).
pub fn rerank_with_table(
    roman: &str,
    candidates: &[DecodedCandidate],
    freq: &HashMap<String, u32>,
    ranks: &FreqRanks,
    custom_sparse_table: Option<&[i8]>,
    custom_sparse_scale: Option<f64>,
) -> Vec<(String, f64)> {
    rerank_with_norm(
        roman,
        candidates,
        freq,
        ranks,
        custom_sparse_table,
        custom_sparse_scale,
        DenseNorm::compiled_in(),
        None, // No jointly-trained dense weights: use compiled-in W_DENSE.
    )
}

/// As `rerank_with_table`, with explicit dense-feature normalisation statistics
/// and optionally jointly-trained dense weights (v6+ containers).
///
/// When `custom_dense_weights` is `Some`, those 29 weights are used in place of
/// the compiled-in `W_DENSE` constant.  `None` falls back to `W_DENSE` exactly
/// as before, so v5 and earlier containers are unaffected.
#[allow(clippy::too_many_arguments)]
pub fn rerank_with_norm(
    roman: &str,
    candidates: &[DecodedCandidate],
    freq: &HashMap<String, u32>,
    ranks: &FreqRanks,
    custom_sparse_table: Option<&[i8]>,
    custom_sparse_scale: Option<f64>,
    norm: DenseNorm<'_>,
    custom_dense_weights: Option<&[f64]>,
) -> Vec<(String, f64)> {
    if candidates.is_empty() {
        return vec![];
    }

    let (order, heur, heur_rank) = rank_candidates(candidates, freq);

    // Raw decoder order: `order` is already sorted by emit + lm, so emitting it
    // with descending scores reproduces the generative ranking exactly.
    if crate::core::ablation::no_rerank() {
        return order
            .iter()
            .enumerate()
            .map(|(i, c)| (c.dev.clone(), -(i as f64)))
            .collect();
    }

    let n = order.len();
    let mut scores: Vec<f64> = Vec::with_capacity(n);
    let mut heur_std: Vec<f64> = Vec::with_capacity(n);

    // Cascade: full feature extraction (akshara segmentation, matra scans,
    // sparse hashing) costs far more than the heuristic, and candidates the
    // heuristic already ranks far down never reach the top of the final list.
    // Score only the most promising `RERANK_FULL_DEPTH` in full and let the
    // rest keep their heuristic ordering, offset below the scored block so the
    // two halves cannot interleave.
    let full_depth = rerank_full_depth().min(n);
    let mut worst_scored = f64::INFINITY;

    for (i, c) in order.iter().enumerate() {
        if heur_rank[i] >= full_depth {
            // Not in the cascade: placeholder, filled in after the scored pass.
            scores.push(f64::NAN);
            heur_std.push(norm.z(4, heur[i]));
            continue;
        }
        let dense = extract_dense_features(c, i, heur[i], heur_rank[i], roman, freq, ranks);
        let aks = akshara::segment(&c.dev);
        let sparse = extract_sparse_features(&c.dev, roman, c.akshara_count, &aks);

        let mut s = 0.0f64;
        // Use jointly-trained dense weights from the v6+ container when available;
        // fall back to the compiled-in W_DENSE constant for v5 and earlier.
        let w_dense: &[f64] = custom_dense_weights.unwrap_or(&W_DENSE);
        for k in 0..DENSE_DIM {
            let wk = w_dense.get(k).copied().unwrap_or(W_DENSE[k]);
            // Exp 6: a NaN weight or feature must not poison the whole list.
            if wk.is_finite() {
                let z = norm.z(k, dense[k]);
                if z.is_finite() {
                    s += wk * z;
                }
            }
        }
        for &h in sparse
            .iter()
            .filter(|_| !crate::core::ablation::no_sparse())
        {
            let (b, scale) = match custom_sparse_table {
                Some(t) if h < t.len() => (
                    t[h],
                    // Exp 6: a negative/NaN scale would flip or poison scores.
                    custom_sparse_scale
                        .filter(|s| s.is_finite() && *s >= 0.0)
                        .unwrap_or(SPARSE_SCALE),
                ),
                // SPARSE_TABLE is empty on a build with no legacy
                // data/reranker_weights_sparse.bin (see build.rs); index 0
                // there instead of panicking -- graceful degradation to "no
                // sparse contribution", matching how the rest of the engine
                // treats an absent optional data source.
                _ => (
                    SPARSE_TABLE.get(h).copied().unwrap_or(0) as i8,
                    SPARSE_SCALE,
                ),
            };
            s += (b as f64) * scale;
        }
        worst_scored = worst_scored.min(s);
        scores.push(s);
        heur_std.push(norm.z(4, heur[i]));
    }

    // Place unscored candidates below every scored one, ordered by heuristic
    // rank, so the cascade only ever affects the tail of the list.
    //
    // Note: these synthetic scores do participate in the GAMMA z-blend below,
    // which shifts the score vector's mean and spread relative to scoring every
    // candidate.  The ordering among *scored* candidates survives (z-scoring is
    // affine), but the blend weight against the heuristic moves slightly with
    // the scored/unscored ratio.  Measured on the 2,108-case AK-Freq split:
    // depth 24 -> 81.83/92.22, depth 50 (no cascade) -> 81.69/92.22, depth 12
    // -> 81.78/92.41, depth 6 -> 81.31/91.98.  The effect is within noise down
    // to depth 12; below that the cascade starts dropping real candidates.
    if worst_scored.is_finite() {
        for (i, sc) in scores.iter_mut().enumerate() {
            if sc.is_nan() {
                *sc = worst_scored - 1.0 - (heur_rank[i] as f64) * 1e-6;
            }
        }
    } else {
        for (i, sc) in scores.iter_mut().enumerate() {
            *sc = -(heur_rank[i] as f64);
        }
    }

    // Exp 6: NaN gamma (e.g. AKSHAR_GAMMA=NaN) fails every comparison and
    // would poison the blend. Fall back to compiled GAMMA unless finite.
    let gamma = crate::core::ablation::gamma()
        .filter(|g| g.is_finite())
        .unwrap_or(GAMMA);
    if gamma >= 1.0 {
        let mut blended: Vec<(String, f64)> = order
            .iter()
            .zip(scores)
            .map(|(c, s)| (c.dev.clone(), s))
            .collect();
        blended.sort_by(|a, b| b.1.total_cmp(&a.1));
        blended
    } else if gamma <= 0.0 {
        let mut blended: Vec<(String, f64)> = order
            .iter()
            .zip(heur)
            .map(|(c, h)| (c.dev.clone(), -h))
            .collect();
        blended.sort_by(|a, b| b.1.total_cmp(&a.1));
        blended
    } else {
        zscore(&mut heur_std);
        zscore(&mut scores);
        let mut blended: Vec<(String, f64)> = order
            .iter()
            .zip(scores)
            .zip(heur_std)
            .map(|((c, s), zh)| (c.dev.clone(), (1.0 - gamma) * (-zh) + gamma * s))
            .collect();
        blended.sort_by(|a, b| b.1.total_cmp(&a.1));
        blended
    }
}

// ---------------------------------------------------------------------------
// Legacy 5-feature MERT Fallback Struct (for WASM-lite / backward compatibility)
// ---------------------------------------------------------------------------

pub const F_EMIT: usize = 0;
pub const F_LM: usize = 1;
pub const F_LEN: usize = 2;
pub const F_FREQ: usize = 3;
pub const NUM_FEATURES: usize = 4;

pub fn feature_names() -> [&'static str; NUM_FEATURES] {
    ["emission", "lm", "length", "frequency"]
}

#[derive(Debug, Clone)]
pub struct Reranker {
    pub weights: [f64; NUM_FEATURES],
    freq: Option<HashMap<String, u32>>,
}

impl Default for Reranker {
    fn default() -> Self {
        Self {
            weights: [1.0, 1.0, 0.0, 0.0],
            freq: None,
        }
    }
}

impl Reranker {
    pub fn new(weights: [f64; NUM_FEATURES]) -> Self {
        Self {
            weights,
            freq: None,
        }
    }

    pub fn with_freq(mut self, freq: Option<HashMap<String, u32>>) -> Self {
        self.freq = freq;
        self
    }

    pub fn features(&self, _roman: &str, cand: &DecodedCandidate) -> [f64; NUM_FEATURES] {
        let freq_feat = self
            .freq
            .as_ref()
            .and_then(|f| f.get(&cand.dev))
            .map(|&c| (1.0 + c as f64).ln() / 100.0f64.ln())
            .unwrap_or(0.0);
        [
            -cand.emit,
            -cand.lm,
            -(cand.akshara_count as f64),
            freq_feat.min(1.0),
        ]
    }

    pub fn rerank(&self, roman: &str, candidates: Vec<DecodedCandidate>) -> Vec<(String, f64)> {
        let mut scored: Vec<(String, f64)> = candidates
            .into_iter()
            .map(|c| {
                let f = self.features(roman, &c);
                let score: f64 = self.weights.iter().zip(f.iter()).map(|(w, x)| w * x).sum();
                (c.dev, score)
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morph_table_is_longest_first() {
        assert_eq!(MORPH_SUFFIXES.len(), 164);
        let mut prev = usize::MAX;
        for &(s, _) in MORPH_SUFFIXES.iter() {
            let n = s.chars().count();
            assert!(n <= prev, "table not longest-first at {s}");
            assert!(n > 0);
            prev = n;
        }
    }

    #[test]
    fn longest_match_wins_over_prefix() {
        // "आएको" ends in both "एको" (len 3) and "को" (len 2): longest wins.
        // (The old hand order returned first-match — "को" — regardless.)
        assert_eq!(get_morph_suffix("आएको"), Some("एको"));
        assert_eq!(get_morph_suffix("किताबें"), Some("ें"));
        // Ambiguous readings keep the longest registered strip: "गरेको"
        // ends in "को" ("ेको" is not a suffix), stem "गरे".
        assert_eq!(get_morph_suffix("गरेको"), Some("को"));
    }

    #[test]
    fn consonant_gate_falls_through_to_shorter_bare() {
        // "ती" is gated (stem must end in a consonant): "अती" has stem "अ"
        // (independent vowel), so the gate rejects and the bare "ी" matches.
        assert_eq!(get_morph_suffix("कहती"), Some("ती"));
        assert_eq!(get_morph_suffix("अती"), Some("ी"));
        // The whole-word rule leaves at least one character: "ो" strips
        // "को" to "क", but "ो" itself strips nothing.
        assert_eq!(get_morph_suffix("को"), Some("ो"));
        assert_eq!(get_morph_suffix("ो"), None);
    }
}
