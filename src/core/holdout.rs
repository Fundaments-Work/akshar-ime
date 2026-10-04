// File: src/core/holdout.rs
//
// Deterministic corpus holdout split, shared by the trainer and the
// sentence-level evaluator so that held-out text can never leak into
// vocabulary counts, word bigrams, or the EM model.
//
// The split is a stable hash of the sentence itself, not a line range: the
// corpus is a concatenation of Wikipedia, CC100 and a news crawl, so a
// tail slice would sample one source only.  Hashing the text keeps the
// held-out set spread across every source and keeps the decision stable
// across rebuilds, reorderings and languages.

/// Default: 1 sentence in 200 (~32k of the 6.43M in data/nepali_corpus.txt).
pub const DEFAULT_HOLDOUT_DENOM: u32 = 200;

/// FNV-1a 64. Chosen over `DefaultHasher` because that one is explicitly
/// not stable across Rust releases, and this decision must be reproducible.
#[inline]
pub fn stable_hash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in s.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// True when this sentence belongs to the held-out evaluation split.
///
/// `denom == 0` disables the split entirely (nothing is held out), which is
/// what callers pass when they deliberately want to train on everything.
#[inline]
pub fn is_holdout(line: &str, denom: u32) -> bool {
    if denom == 0 {
        return false;
    }
    stable_hash(line.trim()).is_multiple_of(denom as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn denom_zero_holds_nothing_out() {
        assert!(!is_holdout("के तपाईंलाई थाहा छ", 0));
    }

    #[test]
    fn split_is_deterministic_and_whitespace_insensitive() {
        let s = "सबै पाठ जिएनयु नि शुल्क";
        let a = is_holdout(s, 200);
        assert_eq!(a, is_holdout(&format!("  {s}  "), 200));
        assert_eq!(a, is_holdout(s, 200));
    }

    #[test]
    fn split_rate_is_close_to_one_in_denom() {
        let n = 200_000;
        let held = (0..n)
            .filter(|i| is_holdout(&format!("वाक्य संख्या {i}"), 200))
            .count();
        // Expect ~1000; allow generous slack for hash noise.
        assert!((700..1400).contains(&held), "held={held} of {n}");
    }
}
