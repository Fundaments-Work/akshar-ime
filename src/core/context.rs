// File: src/core/context.rs
use crate::core::types::WordId;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextModel {
    window_size: usize,
    history: VecDeque<WordId>,
    /// Maps (prev_word_id, current_word_id) -> frequency
    bigrams: HashMap<(WordId, WordId), u64>,
}

impl ContextModel {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            history: VecDeque::with_capacity(window_size),
            bigrams: HashMap::new(),
        }
    }

    /// Adds a confirmed word to the context history and updates bigram counts.
    /// O(1) amortized complexity.
    pub fn add_word(&mut self, word_id: WordId) {
        if let Some(&prev_word_id) = self.history.back() {
            *self.bigrams.entry((prev_word_id, word_id)).or_insert(0) += 1;
        }

        if self.history.len() == self.window_size {
            self.history.pop_front();
        }
        self.history.push_back(word_id);
    }

    /// Re-ranks a list of suggestions based on the current context.
    /// Suggestions that form common bigrams with the previous word get a score boost.
    pub fn rerank_suggestions(&self, suggestions: &mut [(WordId, u64)]) {
        if let Some(&prev_word_id) = self.history.back() {
            for (word_id, score) in suggestions.iter_mut() {
                if let Some(&bigram_count) = self.bigrams.get(&(prev_word_id, *word_id)) {
                    // Simple boost: add a factor of the bigram count, scaled to
                    // the engine's u64 score range (FRESH_SCALE ~ 1e6).
                    let boost = (bigram_count as f64).log2() * 10_000.0;
                    *score += boost as u64;
                }
            }
            // Re-sort the suggestions based on the new boosted scores
            suggestions.sort_by_key(|&(_, score)| std::cmp::Reverse(score));
        }
    }
}

/// Corpus bigram context: P(cur | prev) from running text, string-keyed so it
/// needs no trie WordId (corpus words are not user-learned words).
///
/// Used as a *margin blend*, never an override: the probe
/// (examples/context_headroom.rs) showed bigram comparison alone misleads in
/// ~40-50% of its flips at every count threshold, so the bonus is scaled by
/// `CTX_W` (tuned offline) and applied pre-squash where the 800k decoder-band
/// cap holds structurally. Absent table or empty prev = byte-identical off.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CorpusCtx {
    /// (prev, cur) -> count, pruned at build time (`--threshold`).
    pub table: HashMap<(String, String), u32>,
    /// cur -> unigram count, for backoff when (prev, cur) is unseen.
    pub uni: HashMap<String, u32>,
    /// Sum of `uni` (precomputed: scoring must not sum 470k entries per keystroke).
    #[serde(skip, default)]
    pub uni_total: u64,
}

impl CorpusCtx {
    pub fn new(table: HashMap<(String, String), u32>, uni: HashMap<String, u32>) -> Self {
        let uni_total = uni.values().map(|&c| c as u64).sum();
        Self {
            table,
            uni,
            uni_total,
        }
    }

    /// Recompute `uni_total` after deserialization (`#[serde(skip)]`).
    /// Call once after loading a table from disk.
    pub fn finalize(&mut self) {
        self.uni_total = self.uni.values().map(|&c| c as u64).sum();
    }

    /// Load a builder-written sidecar (`build_corpus_bigrams --out`).
    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        let mut ctx: Self = bincode::deserialize(&bytes)?;
        ctx.finalize();
        Ok(ctx)
    }
}

/// Blend weight for the corpus-context log bonus. Tuned offline on the
/// sentence harness with predicted (not oracle) context; `AKSHAR_CTX_W`
/// overrides for sweeps.
pub const CTX_W: f64 = 0.5;

impl CorpusCtx {
    /// Smoothed bigram score: observed count plus an add-0.5 backoff toward
    /// the unigram. Only *differences* between candidates matter (see
    /// `bonus`): for two candidates sharing `prev`, the bonus reduces to the
    /// log-ratio of their smoothed counts, so no per-prev normaliser is
    /// needed and unseen pairs fall back to their (smoothed) unigrams.
    pub fn log_prob(&self, prev: &str, cur: &str) -> f64 {
        let c = self
            .table
            .get(&(prev.to_string(), cur.to_string()))
            .copied()
            .unwrap_or(0);
        let u = self.uni.get(cur).copied().unwrap_or(0) as f64;
        let backoff = (u + 0.5) / (self.uni_total.max(1) as f64 + 0.5);
        (c as f64 + 0.5 * backoff).max(1e-12).ln()
    }

    /// Bonus for `cur` relative to the current top candidate: the log-ratio.
    /// Positive means context prefers `cur` over `top`.
    ///
    /// ABSTENTION RULE: returns 0 unless the table observes `(prev, cur)`
    /// or `(prev, top)`. Without it, two unseen pairs would fall back to
    /// their unigrams and the bonus would re-rank by unigram ratio —
    /// double-counting the frequency prior the heuristic already contains
    /// and measurably regressing sentence word@1 (87.58% -> 87.04% at W=0.5
    /// with the rule off). Context votes only where it has observations.
    pub fn bonus(&self, prev: &str, cur: &str, top: &str, w: f64) -> f64 {
        if w == 0.0 || cur == top {
            return 0.0;
        }
        let key_c = (prev.to_string(), cur.to_string());
        let key_t = (prev.to_string(), top.to_string());
        if !self.table.contains_key(&key_c) && !self.table.contains_key(&key_t) {
            return 0.0;
        }
        w * (self.log_prob(prev, cur) - self.log_prob(prev, top))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy() -> CorpusCtx {
        let table: HashMap<(String, String), u32> = [
            (("म".to_string(), "नाम".to_string()), 100),
            (("म".to_string(), "काम".to_string()), 10),
        ]
        .into_iter()
        .collect();
        let uni: HashMap<String, u32> = [("नाम".to_string(), 500), ("काम".to_string(), 400)]
            .into_iter()
            .collect();
        CorpusCtx::new(table, uni)
    }

    #[test]
    fn bonus_prefers_frequent_continuation() {
        let ctx = toy();
        // c(म,नाम)=100 vs c(म,काम)=10: bonus for नाम over काम is positive.
        assert!(ctx.bonus("म", "नाम", "काम", 1.0) > 0.0);
        assert!(ctx.bonus("म", "काम", "नाम", 1.0) < 0.0);
    }

    #[test]
    fn unseen_prev_abstains() {
        let ctx = toy();
        // Unseen prev: 0, NOT the unigram ratio — see `bonus` docs.
        assert_eq!(ctx.bonus("नदेखिएको", "नाम", "काम", 1.0), 0.0);
    }

    #[test]
    fn empty_table_never_moves_anything() {
        let ctx = CorpusCtx::default();
        assert_eq!(ctx.bonus("म", "नाम", "काम", 1.0), 0.0);
    }

    #[test]
    fn weight_zero_disables() {
        let ctx = toy();
        assert_eq!(ctx.bonus("म", "नाम", "काम", 0.0), 0.0);
    }

    #[test]
    fn abstains_without_observations() {
        let ctx = toy();
        // Neither (नदेखिएको, नाम) nor (नदेखिएको, काम) observed: 0, even
        // though the unigrams differ (500 vs 400). No unigram double-count.
        assert_eq!(ctx.bonus("नदेखिएको", "नाम", "काम", 1.0), 0.0);
        // One side observed: votes.
        assert!(ctx.bonus("म", "नयाँ", "काम", 1.0) < 0.0);
    }

    #[test]
    fn sidecar_round_trip_preserves_scores() {
        let ctx = toy();
        let bytes = bincode::serialize(&ctx).expect("serialize");
        let mut back: CorpusCtx = bincode::deserialize(&bytes).expect("deserialize");
        assert_eq!(back.uni_total, 0, "skipped field starts at 0");
        back.finalize();
        assert_eq!(
            back.bonus("म", "नाम", "काम", 1.0),
            ctx.bonus("म", "नाम", "काम", 1.0)
        );
    }
}
