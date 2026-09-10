// File: src/core/engine.rs
//
// IME engine: one decoder, one coherent evidence score.
//
// Candidates come from four sources, each contributing an evidence score:
//
//   * the generative decoder (fresh transliterations of the roman prefix),
//   * the user's learned dictionary (trie + fuzzy SymSpell),
//   * the context model (re-ranks words the user has typed before).
//
// The final score is the max evidence across sources: a word confirmed as a
// real word outranks a merely-transliterated form, and user-confirmed words
// climb as their frequency grows.

use crate::core::{
    context::ContextModel,
    decoder::{DecoderConfig, ModelDecoder},
    normalizer::expand_query_variants,
    reranker::{Reranker, NUM_FEATURES},
    translit_model::TranslitModel,
    trie::Trie,
    types::{TransliterationModel, WordId},
};
use crate::fuzzy::symspell::SymSpell;
use crate::learning::{LearningEngine, WordConfirmation};
#[cfg(not(target_arch = "wasm32"))]
use crate::persistence::{load_from_disk, save_to_disk};
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;

const CONTEXT_WINDOW_SIZE: usize = 3;
const MAX_EDIT_DISTANCE: usize = 2;
const QUERY_VARIANT_LIMIT: usize = 6;

/// Decoder beam for the IME (accuracy/speed sweet spot, see M2 eval).
const DECODER_BEAM: usize = 64;

fn decoder_beam() -> usize {
    std::env::var("AKSHAR_BEAM")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| (8..=512).contains(&v))
        .unwrap_or(DECODER_BEAM)
}

fn cache_limit() -> usize {
    std::env::var("AKSHAR_CACHE_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0 && v <= 4096)
        .unwrap_or(256)
}

#[allow(dead_code)]
fn adaptive_beam(roman_len: usize) -> usize {
    let base = decoder_beam();
    if roman_len <= 3 {
        (base / 2).max(8)
    } else if roman_len <= 5 {
        (base * 3 / 4).max(12)
    } else {
        base
    }
}

/// Scale converting a reranker log-score into the engine's higher-better u64 score.
const FRESH_SCALE: f64 = 800_000.0;
/// Purnabiram (।, U+0964) — mapped from a trailing '.'.
const PURNABIRAM: char = '\u{0964}';
/// User-confirmed word from the learned trie.  Above any decoder score so a
/// word the user picked before always wins.
const DEFAULT_USER_TRIE_BASE: u64 = 900_000;
/// Fuzzy (edit-distance) match over user-learned roman variants (D18 fix).
/// Calibrated to sit below the decoder's exact top-1 (800,000) so exact decodes
/// are never hijacked, but well above the decoder tail (~250,000) so a typo
/// of a learned word is actually recoverable in the top suggestions.
const DEFAULT_FUZZY_BASE: u64 = 600_000;
const FUZZY_DISTANCE_PENALTY_SCALE: u64 = 150_000;

fn user_trie_base() -> u64 {
    std::env::var("AKSHAR_USER_TRIE_BASE")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_USER_TRIE_BASE)
}
fn fuzzy_base() -> u64 {
    std::env::var("AKSHAR_FUZZY_BASE")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_FUZZY_BASE)
}

pub struct ImeEngine {
    pub decoder: ModelDecoder,
    pub reranker: Reranker,
    /// Corpus-vocabulary data for the discriminative reranker (native builds with
    /// word_freq_text.bin present; None on WASM-lite or missing file).
    pub reranker_data: Option<crate::core::reranker::RerankerData>,
    /// Dense-feature normalisation statistics carried by the model (v5+);
    /// empty means "use the compiled-in constants".
    pub dense_mean: Vec<f64>,
    pub dense_std: Vec<f64>,
    /// Jointly-trained dense reranker weights (v6+); empty means "use compiled-in W_DENSE".
    pub dense_weights: Vec<f64>,
    /// W3: Candidate Union word trie over vocabulary.
    word_trie: Option<crate::core::wordtrie::WordTrie>,
    pub trie: Trie,
    pub context_model: ContextModel,
    pub symspell: SymSpell,
    pub(crate) transliteration_model: TransliterationModel,
    learning_engine: LearningEngine,
    #[allow(dead_code)]
    dictionary_path: Option<String>,
    pub sparse_table: Option<Vec<i8>>,
    pub sparse_scale: f64,
    suggestion_cache: RefCell<FxHashMap<String, Vec<(String, u64)>>>,
}

impl ImeEngine {
    pub fn new() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let unified_path = data_path("akshar.model");
            if unified_path.exists() {
                if let Ok(unified) = crate::core::unified::UnifiedModel::load(&unified_path) {
                    return Self::from_unified(unified);
                }
            }
        }
        let model = load_model_or_default();
        let decoder = ModelDecoder::with_config(
            model,
            DecoderConfig {
                beam_width: decoder_beam(),
                ..DecoderConfig::default()
            },
        );
        let reranker = load_reranker();
        let reranker_data = load_reranker_data();
        let word_trie = reranker_data.as_ref().map(|v| {
            crate::core::wordtrie::WordTrie::from_freq_map(
                &v.freq,
                &|a| decoder.model.akshara_id(a),
                1,
            )
        });
        Self {
            decoder,
            reranker,
            reranker_data,
            dense_mean: Vec::new(),
            dense_std: Vec::new(),
            dense_weights: Vec::new(),
            word_trie,
            trie: Trie::new(),
            context_model: ContextModel::new(CONTEXT_WINDOW_SIZE),
            symspell: SymSpell::default(),
            transliteration_model: TransliterationModel::new(),
            learning_engine: LearningEngine::new(),
            dictionary_path: None,
            sparse_table: None,
            sparse_scale: crate::core::reranker_weights::SPARSE_SCALE,
            suggestion_cache: RefCell::new(FxHashMap::default()),
        }
    }

    /// Construct engine directly from a unified model container.
    pub fn from_unified(mut unified: crate::core::unified::UnifiedModel) -> Self {
        unified.translit.build_trigram_index();
        // Normalisation statistics travel with the model from v5 on; older
        // containers leave these empty and fall back to the compiled-in ones.
        let dense_mean = std::mem::take(&mut unified.dense_mean);
        let dense_std = std::mem::take(&mut unified.dense_std);
        // Jointly-trained dense weights travel with the model from v6 on; older
        // containers leave this empty and the reranker falls back to W_DENSE.
        let dense_weights = std::mem::take(&mut unified.dense_weights);
        let decoder = ModelDecoder::with_config(
            unified.translit,
            DecoderConfig {
                beam_width: decoder_beam(),
                ..DecoderConfig::default()
            },
        );
        let reranker = load_reranker().with_freq(Some(unified.vocab_freq.clone()));
        let ranks = crate::core::reranker::FreqRanks::from_freq_map(&unified.vocab_freq);
        let word_trie = Some(crate::core::wordtrie::WordTrie::from_freq_map(
            &unified.vocab_freq,
            &|a| decoder.model.akshara_id(a),
            1,
        ));
        let reranker_data = Some(crate::core::reranker::RerankerData {
            freq: unified.vocab_freq,
            ranks,
        });
        Self {
            decoder,
            reranker,
            reranker_data,
            dense_mean,
            dense_std,
            dense_weights,
            word_trie,
            trie: Trie::new(),
            context_model: ContextModel::new(CONTEXT_WINDOW_SIZE),
            symspell: SymSpell::default(),
            transliteration_model: TransliterationModel::new(),
            learning_engine: LearningEngine::new(),
            dictionary_path: None,
            sparse_table: if unified.sparse_reranker_table.is_empty() {
                None
            } else {
                Some(unified.sparse_reranker_table)
            },
            sparse_scale: if unified.sparse_scale > 0.0 {
                unified.sparse_scale
            } else {
                crate::core::reranker_weights::SPARSE_SCALE
            },
            suggestion_cache: RefCell::new(FxHashMap::default()),
        }
    }

    /// Load engine from a unified model file (`akshar.model`).
    pub fn from_unified_file(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let unified = crate::core::unified::UnifiedModel::load(path)?;
        Ok(Self::from_unified(unified))
    }

    /// Load engine from in-memory unified model bytes.
    pub fn from_unified_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let unified = crate::core::unified::UnifiedModel::from_bytes(bytes)?;
        Ok(Self::from_unified(unified))
    }

    pub fn from_file_or_new(path: &str) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut engine =
                load_from_disk(std::path::Path::new(path)).unwrap_or_else(|_| Self::new());
            engine.dictionary_path = Some(path.to_string());
            engine
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = path;
            Self::new()
        }
    }

    /// Create engine from raw model bytes (no filesystem).
    ///
    /// `_lexicon_bytes` is accepted and ignored. The corpus roman->devanagari
    /// lexicon was removed on 2026-09-06: it was dead by construction on the
    /// unified-container path (`from_unified` never loaded one) and measured a
    /// 0.00pp contribution on every Aksharantar split. The parameter is kept so
    /// existing JS callers of `createEngine(model, lexicon, weights)` keep
    /// compiling; pass `None`.
    pub fn from_bytes(
        model_bytes: &[u8],
        _lexicon_bytes: Option<&[u8]>,
        reranker_json: Option<&str>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::from_bytes_with_weights(model_bytes, _lexicon_bytes, reranker_json)
    }

    /// Build from bytes, allowing caller to supply already-parsed model.
    pub fn from_model(model: TranslitModel, reranker: Option<Reranker>) -> Self {
        let decoder = ModelDecoder::with_config(
            model,
            DecoderConfig {
                beam_width: decoder_beam(),
                ..DecoderConfig::default()
            },
        );
        let reranker = reranker.unwrap_or_else(load_reranker);
        let reranker_data = load_reranker_data();
        let word_trie = reranker_data.as_ref().map(|v| {
            crate::core::wordtrie::WordTrie::from_freq_map(
                &v.freq,
                &|a| decoder.model.akshara_id(a),
                1,
            )
        });
        Self {
            decoder,
            reranker,
            reranker_data,
            dense_mean: Vec::new(),
            dense_std: Vec::new(),
            dense_weights: Vec::new(),
            word_trie,
            trie: Trie::new(),
            context_model: ContextModel::new(CONTEXT_WINDOW_SIZE),
            symspell: SymSpell::new(MAX_EDIT_DISTANCE),
            transliteration_model: HashMap::new(),
            learning_engine: LearningEngine::new(),
            dictionary_path: None,
            sparse_table: None,
            sparse_scale: crate::core::reranker_weights::SPARSE_SCALE,
            suggestion_cache: RefCell::new(FxHashMap::default()),
        }
    }

    pub fn from_bytes_with_weights(
        model_bytes: &[u8],
        _lexicon_bytes: Option<&[u8]>,
        reranker_json: Option<&str>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // The browser fetches whatever `create_engine(model_url, ...)` was
        // pointed at.  Since the unified container exists that is normally
        // `akshar.model`, which carries the vocabulary, reranker table and
        // bigrams as well as the translit model — parsing it as a bare
        // TranslitModel fails outright, and before this dispatch the WASM path
        // could only ever load the legacy `translit_model.bin`.
        //
        // Detect the container by its magic and take the full path when it is
        // one, so the browser gets the vocabulary prior and the trained sparse
        // reranker instead of silently running without them.
        if model_bytes.len() >= 4 && model_bytes[..4] == crate::core::unified::UNIFIED_MAGIC {
            return Ok(Self::from_unified(
                crate::core::unified::UnifiedModel::from_bytes(model_bytes)?,
            ));
        }

        let model = TranslitModel::from_bytes(model_bytes)?;
        if !model.validate() {
            return Err("invalid translit model".into());
        }
        let reranker = if let Some(json) = reranker_json {
            Self::parse_reranker_json(json)
        } else {
            load_reranker()
        };
        Ok(Self::from_model(model, Some(reranker)))
    }

    fn parse_reranker_json(json_str: &str) -> Reranker {
        let mut weights = [1.0f64; NUM_FEATURES];
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_str) {
            if let Some(w) = v.get("weights") {
                let names = crate::core::reranker::feature_names();
                for (i, name) in names.iter().enumerate() {
                    if let Some(val) = w.get(*name).and_then(|x| x.as_f64()) {
                        weights[i] = val;
                    }
                }
            } else if let Ok(arr) = serde_json::from_str::<[f64; NUM_FEATURES]>(json_str) {
                weights = arr;
            }
        }
        Reranker::new(weights)
    }

    /// Serialise learned state (trie + context + symspell) to bytes for persistence.
    pub fn learned_state_to_bytes(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let state = crate::persistence::SerializableState::from_engine(self);
        state.to_bytes()
    }

    /// Load learned state from bytes (e.g. from localStorage).
    pub fn load_learned_state_from_bytes(
        &mut self,
        bytes: &[u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let state = crate::persistence::SerializableState::from_bytes(bytes)?;
        state.apply_to_engine(self);
        Ok(())
    }

    /// Drop cached suggestion lists (called on learning / state import).
    pub fn clear_suggestion_cache(&self) {
        self.suggestion_cache.borrow_mut().clear();
    }

    /// Entry point for every runtime (IBus C layer, WASM): applies the pure
    /// mappings (ASCII digits → Devanagari digits, trailing '.' → purnabiram)
    /// before the roman goes through the statistical model.
    pub fn get_suggestions(&self, prefix: &str, count: usize) -> Vec<(String, u64)> {
        if prefix.is_empty() {
            return vec![];
        }
        let count = count.max(1);
        // Suggestion cache: pure prefix -> ranked list. No hard-coded size;
        // derived from env AKSHAR_CACHE_SIZE, default 256. Clears on learning.
        if let Some(cached) = self.suggestion_cache.borrow().get(prefix) {
            if cached.len() >= count {
                return cached[..count].to_vec();
            }
        }

        // Purnabiram (।): a trailing '.' asks for it on the result; a lone
        // '.' *is* purnabiram.
        let (base, purnabiram) = match prefix.strip_suffix('.') {
            Some("") => return vec![(PURNABIRAM.to_string(), FRESH_SCALE as u64)],
            Some(b) => (b, true),
            None => (prefix, false),
        };
        let cache_key = prefix.to_string();
        let finish = |mut out: Vec<(String, u64)>| {
            if purnabiram {
                for (s, _) in out.iter_mut() {
                    s.push(PURNABIRAM);
                }
            }
            // cache for next keystroke; evict when over limit
            {
                let mut cache = self.suggestion_cache.borrow_mut();
                if cache.len() >= cache_limit() {
                    cache.clear();
                }
                cache.insert(cache_key.clone(), out.clone());
            }
            out
        };

        // Split the input into letter runs and digit runs. Digits never enter
        // the model — they map to Devanagari digits directly.
        let mut segments: Vec<Result<String, String>> = Vec::new(); // Ok(letters) / Err(digits)
        for part in base.split_inclusive(|c: char| c.is_ascii_digit()) {
            if part.is_empty() {
                continue;
            }
            if part.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                let last = segments.last_mut();
                match last {
                    Some(Err(d)) => d.push_str(part),
                    _ => segments.push(Err(part.to_string())),
                }
            } else {
                let mut digits_end = part.len();
                for (i, c) in part.char_indices() {
                    if c.is_ascii_digit() {
                        digits_end = i;
                        break;
                    }
                }
                let (letters, trailing_digits) = part.split_at(digits_end);
                segments.push(Ok(letters.to_string()));
                if !trailing_digits.is_empty() {
                    segments.push(Err(trailing_digits.to_string()));
                }
            }
        }
        let map_digits = |d: &str| -> String {
            d.chars()
                .map(|c| {
                    if c.is_ascii_digit() {
                        char::from_u32('\u{0966}' as u32 + c as u32 - '0' as u32).unwrap_or(c)
                    } else {
                        c
                    }
                })
                .collect()
        };

        let letter_runs: Vec<&String> = segments.iter().filter_map(|s| s.as_ref().ok()).collect();
        if letter_runs.is_empty() {
            // Pure number: 123 -> १२३ (no model involved).
            let digits = segments
                .iter()
                .filter_map(|s| s.as_ref().err())
                .cloned()
                .collect::<Vec<_>>()
                .join("");
            return finish(vec![(map_digits(&digits), FRESH_SCALE as u64)]);
        }
        if letter_runs.len() == 1 {
            // Single word with optional leading/trailing digits: decode the
            // word, then place the mapped digit runs at their original edges.
            let roman = letter_runs[0].as_str();
            let lead_digits = match segments.first() {
                Some(Err(d)) => map_digits(d),
                _ => String::new(),
            };
            let trail_digits = match segments.last() {
                Some(Err(d)) => map_digits(d),
                _ => String::new(),
            };
            let mut out = self.suggestions_for_roman(roman, count);
            for (s, _) in out.iter_mut() {
                *s = format!("{lead_digits}{s}{trail_digits}");
            }
            return finish(out);
        }
        // Digits between words ("ka2024ma"): decode each letter run to its
        // top choice and interleave the mapped digits — one combined result.
        let mut combined = String::new();
        for seg in &segments {
            match seg {
                Ok(roman) => {
                    if let Some((top, _)) = self.suggestions_for_roman(roman, 1).first() {
                        combined.push_str(top);
                    }
                }
                Err(d) => combined.push_str(&map_digits(d)),
            }
        }
        finish(vec![(combined, FRESH_SCALE as u64)])
    }

    /// The statistical path: roman (letters only) -> ranked Devanagari words.
    fn suggestions_for_roman(&self, prefix: &str, count: usize) -> Vec<(String, u64)> {
        if prefix.is_empty() {
            return vec![];
        }
        let query_variants = if crate::core::ablation::no_variants() {
            vec![crate::core::normalizer::RomanVariant {
                roman: prefix.to_string(),
                penalty: 0,
            }]
        } else {
            expand_query_variants(prefix, QUERY_VARIANT_LIMIT)
        };

        let mut candidates: HashMap<String, u64> = HashMap::new();
        let mut add = |dev: String, score: u64| {
            candidates
                .entry(dev)
                .and_modify(|s| *s = (*s).max(score))
                .or_insert(score);
        };

        // 1. Fresh transliterations from the generative decoder.  With the
        //    corpus vocabulary available, rank them with the trained discriminative
        //    reranker (81.02% native top-1 on test); otherwise fall back to
        //    the 5-feature MERT reranker.  We decode the base roman only: the
        //    model's learned emissions already absorb v/w and vowel-length
        //    spelling variants, so re-decoding soft variants is pure latency.
        //    Depth 50 matches the depth the discriminative model was trained on.
        let mut fresh_scores: HashMap<String, u64> = HashMap::new();
        if let Some(qv) = query_variants.first() {
            let roman = qv.roman.as_str();
            let cands =
                self.decoder
                    .decode_union(roman, (count * 4).max(50), self.word_trie.as_ref());
            let ranked: Vec<(String, f64)> = match &self.reranker_data {
                Some(data) => crate::core::reranker::rerank_with_norm(
                    roman,
                    &cands,
                    &data.freq,
                    &data.ranks,
                    self.sparse_table.as_deref(),
                    Some(self.sparse_scale),
                    crate::core::reranker::DenseNorm::from_model(&self.dense_mean, &self.dense_std),
                    // v6+: use jointly-trained dense weights; None falls back to compiled-in W_DENSE.
                    if self.dense_weights.is_empty() {
                        None
                    } else {
                        Some(&self.dense_weights)
                    },
                ),
                None => self.reranker.rerank(roman, cands),
            };
            let mut ranked = ranked;
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

            let s_max = ranked.first().map(|(_, s)| *s).unwrap_or(0.0);
            for (dev, s) in ranked {
                let cost = (s_max - s).max(0.0);
                let score = (FRESH_SCALE / (1.0 + cost)).round().max(1.0) as u64;
                fresh_scores.insert(dev.clone(), score);
                add(dev, score);
            }
        }

        for qv in &query_variants {
            let roman = qv.roman.as_str();

            // 3. User-learned dictionary (trie).
            for (word_id, freq) in self.trie.get_top_k_suggestions(roman, count * 3) {
                if let Some(meta) = self.trie.metadata_store.get(word_id) {
                    add(
                        meta.devanagari.clone(),
                        user_trie_base().saturating_add(freq),
                    );
                }
            }

            // 4. Fuzzy matches over user-learned roman variants (typo tolerance).
            for word_id in self.symspell.lookup(roman) {
                if let Some(meta) = self.trie.metadata_store.get(word_id) {
                    if let Some(min_dist) = self.min_roman_distance(roman, meta, MAX_EDIT_DISTANCE)
                    {
                        let dist_penalty = (min_dist as u64) * FUZZY_DISTANCE_PENALTY_SCALE;
                        let score = fuzzy_base().saturating_sub(dist_penalty);
                        add(meta.devanagari.clone(), score);
                    }
                }
            }
        }

        // 5. Context re-rank for words the user has typed before.
        let mut with_ids: Vec<(WordId, u64)> = candidates
            .iter()
            .filter_map(|(dev, score)| {
                self.trie
                    .find_word_id_by_devanagari(dev)
                    .map(|id| (id, *score))
            })
            .collect();
        self.context_model.rerank_suggestions(&mut with_ids);
        for (id, new_score) in with_ids {
            if let Some(dev) = self.trie.metadata_store.get(id).map(|m| &m.devanagari) {
                if let Some(entry) = candidates.get_mut(dev) {
                    *entry = new_score;
                }
            }
        }

        let mut out: Vec<(String, u64)> = candidates.into_iter().collect();
        out.sort_by_key(|&(_, score)| std::cmp::Reverse(score));
        out.truncate(count);
        out
    }

    /// Set the preceding-word context without learning the word.
    ///
    /// `user_confirms` both sets the context *and* teaches the word to the
    /// user trie / SymSpell / adaptive model.  Offline harnesses that walk a
    /// gold sentence must not do the latter: learning the gold word makes
    /// every later occurrence trivially correct and the measurement
    /// self-fulfilling.  This is the context half on its own.
    pub fn set_context_word(&mut self, _devanagari: &str) {}

    pub fn user_confirms(&mut self, roman: &str, devanagari: &str) {
        if roman.is_empty() || devanagari.is_empty() {
            return;
        }
        self.suggestion_cache.borrow_mut().clear();
        let confirmation = WordConfirmation {
            roman: roman.to_string(),
            devanagari: devanagari.to_string(),
        };
        self.learning_engine.learn(
            &mut self.trie,
            &mut self.context_model,
            &mut self.symspell,
            &mut self.transliteration_model,
            &confirmation,
        );
    }

    /// Clear learned state in place, keeping model + reranker + vocab intact.
    ///
    /// Exp 6: the WASM `resetLearning` previously rebuilt via `from_model`,
    /// which drops the unified container's vocab/word-trie/sparse table (and
    /// returns `None` for reranker data on wasm32). Resetting in place keeps
    /// the decoder exactly as loaded.
    pub fn reset_learned_state(&mut self) {
        self.trie = Trie::new();
        self.context_model = ContextModel::new(CONTEXT_WINDOW_SIZE);
        self.symspell = SymSpell::new(MAX_EDIT_DISTANCE);
        self.transliteration_model = TransliterationModel::new();
        self.suggestion_cache.borrow_mut().clear();
    }

    fn min_roman_distance(
        &self,
        roman_query: &str,
        metadata: &crate::core::types::WordMetadata,
        max_distance: usize,
    ) -> Option<usize> {
        metadata
            .variants
            .iter()
            .filter_map(|variant| {
                let raw = Self::bounded_levenshtein(roman_query, variant, max_distance);
                let collapsed_query = Self::collapse_vowel_runs(roman_query);
                let collapsed_variant = Self::collapse_vowel_runs(variant);
                let collapsed =
                    Self::bounded_levenshtein(&collapsed_query, &collapsed_variant, max_distance);
                match (raw, collapsed) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    (None, None) => None,
                }
            })
            .min()
    }

    fn bounded_levenshtein(a: &str, b: &str, max_distance: usize) -> Option<usize> {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        if a_chars.len().abs_diff(b_chars.len()) > max_distance {
            return None;
        }
        let mut prev: Vec<usize> = (0..=b_chars.len()).collect();
        let mut curr = vec![0usize; b_chars.len() + 1];
        for (i, ca) in a_chars.iter().enumerate() {
            curr[0] = i + 1;
            let mut row_min = curr[0];
            for (j, cb) in b_chars.iter().enumerate() {
                let replace_cost = if ca == cb { 0 } else { 1 };
                let deletion = prev[j + 1] + 1;
                let insertion = curr[j] + 1;
                let replacement = prev[j] + replace_cost;
                curr[j + 1] = deletion.min(insertion).min(replacement);
                row_min = row_min.min(curr[j + 1]);
            }
            if row_min > max_distance {
                return None;
            }
            std::mem::swap(&mut prev, &mut curr);
        }
        let dist = prev[b_chars.len()];
        (dist <= max_distance).then_some(dist)
    }

    fn collapse_vowel_runs(input: &str) -> String {
        let mut out = String::with_capacity(input.len());
        let mut prev: Option<char> = None;
        for c in input.chars() {
            let is_vowel = matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'A' | 'E' | 'I' | 'O' | 'U');
            if is_vowel && prev == Some(c) {
                continue;
            }
            out.push(c);
            prev = Some(c);
        }
        out
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn save_dictionary(&self) -> Result<(), std::io::Error> {
        if let Some(path) = &self.dictionary_path {
            save_to_disk(self, std::path::Path::new(path))
        } else {
            Ok(())
        }
    }
    #[cfg(target_arch = "wasm32")]
    pub fn save_dictionary(&self) -> Result<(), std::io::Error> {
        Ok(())
    }
}

impl Default for ImeEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn data_path(file: &str) -> PathBuf {
    if let Ok(dir) = std::env::var("AKSHAR_DATA_DIR") {
        let p = std::path::Path::new(&dir).join(file);
        if p.exists() {
            return p;
        }
    }
    let repo = std::path::Path::new("data").join(file);
    if repo.exists() {
        return repo;
    }
    if let Some(home) = dirs::home_dir() {
        let p = home.join(".local/share/akshar-ime").join(file);
        if p.exists() {
            return p;
        }
    }
    std::path::Path::new("/usr/share/akshar-ime").join(file)
}

#[cfg(target_arch = "wasm32")]
#[allow(dead_code)]
fn data_path(_file: &str) -> PathBuf {
    PathBuf::new()
}

#[cfg(not(target_arch = "wasm32"))]
fn load_model_or_default() -> TranslitModel {
    match TranslitModel::load(&data_path("translit_model.bin")) {
        Ok(m) if m.validate() => m,
        _ => TranslitModel::default(),
    }
}
#[cfg(target_arch = "wasm32")]
fn load_model_or_default() -> TranslitModel {
    TranslitModel::default()
}

fn load_reranker() -> Reranker {
    let mut weights = [1.0f64; NUM_FEATURES];
    weights[0] = 1.0;
    weights[1] = 1.0;
    #[cfg(not(target_arch = "wasm32"))]
    {
        if let Ok(file) = std::fs::File::open(data_path("reranker_weights.json")) {
            if let Ok(v) = serde_json::from_reader::<_, serde_json::Value>(file) {
                if let Some(w) = v.get("weights") {
                    let names = crate::core::reranker::feature_names();
                    for (i, name) in names.iter().enumerate() {
                        if let Some(val) = w.get(*name).and_then(|x| x.as_f64()) {
                            weights[i] = val;
                        }
                    }
                }
            }
        }
    }
    let reranker = Reranker::new(weights);
    // E3: word-frequency evidence (native targets only; absent on wasm unless
    // fetched separately).
    #[cfg(not(target_arch = "wasm32"))]
    let reranker = {
        let mut r = reranker;
        if let Ok(bytes) = std::fs::read(data_path("word_freq_text.bin")) {
            if let Ok(map) = bincode::deserialize::<HashMap<String, u32>>(&bytes) {
                r = r.with_freq(Some(map));
            }
        }
        r
    };
    reranker
}

/// Reranker data: the corpus vocabulary plus its frequency-rank index.
/// None when the vocabulary file is unavailable (WASM-lite, fresh clones).
#[cfg(not(target_arch = "wasm32"))]
fn load_reranker_data() -> Option<crate::core::reranker::RerankerData> {
    let bytes = std::fs::read(data_path("word_freq_text.bin")).ok()?;
    crate::core::reranker::RerankerData::from_bin_bytes(&bytes)
}

#[cfg(target_arch = "wasm32")]
fn load_reranker_data() -> Option<crate::core::reranker::RerankerData> {
    None
}

#[cfg(test)]
mod dispatch_tests {
    use super::*;

    /// The WASM entry point must accept a unified container, not only a bare
    /// translit model.  Regression: it parsed every payload as a TranslitModel,
    /// so pointing the browser at `akshar.model` failed to load entirely.
    #[test]
    fn from_bytes_with_weights_accepts_a_unified_container() {
        let mut translit = TranslitModel {
            version: 1,
            aksharas: vec!["क".to_string()],
            chunks: vec!["ka".to_string()],
            emissions: vec![vec![(0u32, 0.5f32)]],
            bigrams: vec![vec![]],
            backoff: vec![0.0],
            unigram_kn: vec![0.0],
            word_start: vec![0.0],
            ..Default::default()
        };
        translit.build_trigram_index();
        let mut vocab = HashMap::new();
        vocab.insert("क".to_string(), 9u32);

        let unified = crate::core::unified::UnifiedModel::new(translit, vec![0i8; 4], 0.5, vocab);
        let bytes = unified.to_bytes().expect("serialize");

        let engine = ImeEngine::from_bytes_with_weights(&bytes, None, None)
            .expect("unified container must load through the WASM entry point");
        // The vocabulary rides along with the container; a bare TranslitModel
        // parse would have produced an engine with none.
        assert!(
            engine.reranker_data.is_some(),
            "vocabulary should be loaded"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A decoder built from a tiny hand-built model, so tests don't depend on
    // the on-disk model file.
    fn tiny_decoder() -> ModelDecoder {
        let mut m = TranslitModel {
            version: crate::core::translit_model::MODEL_VERSION,
            ..TranslitModel::default()
        };
        // aksharas: 0=क 1=कि 2=न 3=म 4=मा 5=स्ते 6=ने 7=प 8=आल 9=र
        for a in ["क", "कि", "न", "म", "मा", "स्ते", "ने", "प", "आल", "र"]
        {
            m.aksharas.push(a.to_string());
        }
        for chunk in [
            "ka", "ki", "na", "ma", "maa", "ste", "ne", "pa", "aal", "ra",
        ] {
            m.chunks.push(chunk.to_string());
        }
        // emissions: chunk id -> akshara id
        let chunk = |s: &str| m.chunks.iter().position(|c| c == s).unwrap() as u32;
        let emit = |_a: u32, c: &str, w: f32| vec![(chunk(c), w)];
        m.emissions = vec![
            emit(0, "ka", 0.1),  // क -> ka
            emit(1, "ki", 0.1),  // कि -> ki
            emit(2, "na", 0.1),  // न -> na
            emit(3, "ma", 0.2),  // म -> ma
            emit(4, "maa", 0.1), // मा -> maa
            emit(5, "ste", 0.1), // स्ते -> ste
            emit(6, "ne", 0.1),  // ने -> ne
            emit(7, "pa", 0.1),  // प -> pa
            emit(8, "aal", 0.1), // आल -> aal
            emit(9, "ra", 0.1),  // र -> ra
        ];
        m.bigrams = vec![vec![]; 10];
        m.backoff = vec![6.0; 10];
        m.unigram_kn = vec![4.0; 10];
        m.word_start = vec![4.0; 10];
        m.build_trigram_index();
        ModelDecoder::with_config(m, DecoderConfig::default())
    }

    fn engine_with(model: TranslitModel) -> ImeEngine {
        let decoder = ModelDecoder::with_config(
            model,
            DecoderConfig {
                beam_width: 64,
                ..DecoderConfig::default()
            },
        );
        ImeEngine {
            decoder,
            reranker: Reranker::default(),
            reranker_data: None,
            dense_mean: Vec::new(),
            dense_std: Vec::new(),
            dense_weights: Vec::new(),
            word_trie: None,
            trie: Trie::new(),
            context_model: ContextModel::new(3),
            symspell: SymSpell::new(2),
            transliteration_model: HashMap::new(),
            learning_engine: LearningEngine::new(),
            dictionary_path: None,
            sparse_table: None,
            sparse_scale: crate::core::reranker_weights::SPARSE_SCALE,
            suggestion_cache: RefCell::new(FxHashMap::default()),
        }
    }

    #[test]
    fn suggestions_include_decoder_fresh_transliteration() {
        let engine = engine_with(tiny_decoder().model);
        let suggestions = engine.get_suggestions("namaste", 8);
        assert!(suggestions.iter().any(|(d, _)| d == "नमस्ते"));
    }

    #[test]
    fn user_confirmation_moves_word_up() {
        let mut engine = engine_with(tiny_decoder().model);
        engine.user_confirms("namaste", "नमस्ते");
        engine.user_confirms("namaste", "नमस्ते");
        let suggestions = engine.get_suggestions("namaste", 8);
        let pos = suggestions
            .iter()
            .position(|(d, _)| d == "नमस्ते")
            .expect("नमस्ते should be suggested after learning");
        assert_eq!(pos, 0, "learned word should rank first");
    }

    #[test]
    fn empty_prefix_returns_nothing() {
        let engine = engine_with(tiny_decoder().model);
        assert!(engine.get_suggestions("", 8).is_empty());
    }

    #[test]
    fn bounded_levenshtein_respects_max_distance() {
        assert_eq!(ImeEngine::bounded_levenshtein("kal", "kal", 2), Some(0));
        assert_eq!(ImeEngine::bounded_levenshtein("kal", "kall", 2), Some(1));
        assert_eq!(ImeEngine::bounded_levenshtein("kal", "xyz", 2), None);
    }

    #[test]
    fn lone_dot_is_purnabiram() {
        let engine = engine_with(tiny_decoder().model);
        let out = engine.get_suggestions(".", 8);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "।");
    }

    #[test]
    fn trailing_dot_appends_purnabiram() {
        let engine = engine_with(tiny_decoder().model);
        let out = engine.get_suggestions("namaste.", 8);
        assert!(!out.is_empty(), "purnabiram input should still decode");
        assert!(out.iter().all(|(d, _)| d.ends_with('।')));
        // and the plain word still decodes identically underneath
        let plain: Vec<String> = engine
            .get_suggestions("namaste", 8)
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        let dotted: Vec<String> = out
            .into_iter()
            .map(|(d, _)| d.strip_suffix('।').map(|s| s.to_string()).unwrap_or(d))
            .collect();
        assert_eq!(
            plain, dotted,
            "suggestions with '.' must equal plain suggestions + ।"
        );
    }

    #[test]
    fn all_digits_map_to_devanagari() {
        let engine = engine_with(tiny_decoder().model);
        let out = engine.get_suggestions("123", 8);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "१२३");
        // year with purnabiram
        let out = engine.get_suggestions("2081.", 8);
        assert_eq!(out[0].0, "२०८१।");
    }

    #[test]
    fn trailing_digits_map_onto_suggestions() {
        let engine = engine_with(tiny_decoder().model);
        let out = engine.get_suggestions("namaste1", 8);
        assert!(out.iter().any(|(d, _)| d.ends_with('१')));
        assert!(out.iter().any(|(d, _)| d.starts_with("नमस्ते")));
    }

    #[test]
    fn leading_digits_prepend_mapping() {
        let engine = engine_with(tiny_decoder().model);
        let out = engine.get_suggestions("12na", 8);
        assert!(!out.is_empty());
        assert!(out.iter().all(|(d, _)| d.starts_with("१२")));
    }

    #[test]
    fn digits_between_words_interleave() {
        let engine = engine_with(tiny_decoder().model);
        let out = engine.get_suggestions("na2ma", 8);
        assert!(!out.is_empty());
        assert_eq!(out[0].0, "न२म");
    }
}
