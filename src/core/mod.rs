// File: src/core/mod.rs
pub mod akshara;
pub mod alignment;
pub mod codec;
pub mod context;
pub mod decoder;
pub mod em_trainer;
pub mod engine;
pub mod holdout;
pub mod normalizer;
pub mod reranker;
pub mod reranker_weights;
pub mod translit_model;
pub mod trie;
pub mod types;
pub mod unified;
pub mod wordtrie;

/// Ablation switches, read once from the environment.
///
/// Every component of the pipeline should be able to justify itself with a
/// measurement.  These flags exist so that an ablation table can be produced
/// from the shipped binary rather than from a series of patched builds, and so
/// that a component that contributes nothing can be found and deleted.
///
/// | variable | effect |
/// | :-- | :--- |
/// | `AKSHAR_NO_TRIE_UNION` | skip the trie-constrained decode pass |
/// | `AKSHAR_NO_SPARSE` | drop the 2^20 sparse reranker table |
/// | `AKSHAR_NO_RERANK` | rank by raw decoder score, skipping the rerank stage |
/// | `AKSHAR_GAMMA` | override the dense/heuristic blend (0.0 = heuristic only) |
/// | `AKSHAR_NO_TRIGRAM` | force the LM to back off to bigrams |
/// | `AKSHAR_NO_VARIANTS` | decode the raw query only, no normalizer variants |
/// | `AKSHAR_CACHE_SIZE` | suggestion-cache size (default 256) |
/// | `AKSHAR_DATA_DIR` | model directory override |
/// | `AKSHAR_USER_TRIE_BASE` | learned-word base score (default 900000) |
/// | `AKSHAR_FUZZY_BASE` | fuzzy-match base score (default 50000) |
/// | `AKSHAR_KN_FIXED_DISCOUNT` | single-discount LM ablation |
pub mod ablation {
    use std::sync::OnceLock;

    fn flag(name: &'static str) -> bool {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = name;
            false
        }
        #[cfg(not(target_arch = "wasm32"))]
        std::env::var(name).is_ok_and(|v| v == "1")
    }

    macro_rules! cached_flag {
        ($fn_name:ident, $env:literal) => {
            pub fn $fn_name() -> bool {
                static V: OnceLock<bool> = OnceLock::new();
                *V.get_or_init(|| flag($env))
            }
        };
    }

    cached_flag!(no_trie_union, "AKSHAR_NO_TRIE_UNION");
    // trie_only: decode ONLY inside the word trie, skipping the free lattice
    // beam.  The trie-constrained pass is far cheaper because most akshara
    // sequences are not word prefixes, so the beam collapses immediately --
    // but coverage costs 15.6pp of AK-Freq top-1, so it is a diagnostic only.
    cached_flag!(trie_only, "AKSHAR_TRIE_ONLY");
    cached_flag!(no_sparse, "AKSHAR_NO_SPARSE");
    // Return candidates in raw decoder order (emit + lm), skipping the whole
    // rerank stage -- heuristic, dense features and sparse table alike. This
    // is the baseline the rerank stage as a whole must beat.
    cached_flag!(no_rerank, "AKSHAR_NO_RERANK");
    cached_flag!(no_trigram, "AKSHAR_NO_TRIGRAM");
    cached_flag!(no_variants, "AKSHAR_NO_VARIANTS");

    /// Dense/heuristic blend override; `None` means use the compiled GAMMA.
    pub fn gamma() -> Option<f64> {
        static V: OnceLock<Option<f64>> = OnceLock::new();
        *V.get_or_init(|| {
            #[cfg(target_arch = "wasm32")]
            {
                None
            }
            #[cfg(not(target_arch = "wasm32"))]
            std::env::var("AKSHAR_GAMMA")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
        })
    }
}
