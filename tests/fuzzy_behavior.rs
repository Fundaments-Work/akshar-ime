//! What the fuzzy path does after the 2026-09-06 removal of the corpus-wide
//! SymSpell source.
//!
//! Two sources of typo tolerance remain, and they are deliberately different:
//!
//!   1. `expand_query_variants` (`normalizer.rs`) rewrites the *query* before
//!      decoding — vowel-length alternations (`ee->ii`, `oo->uu`), v/w, and
//!      separators. This is cold-start: it needs no learning, applies to every
//!      user, and feeds the decoder a principled cost per rewrite.
//!
//!   2. The user SymSpell (`engine.symspell`) covers arbitrary edits within
//!      distance 2, but only over words the user has confirmed. Every hit is
//!      distance-verified against the stored roman variants before it scores.
//!
//! The removed third source indexed a romanization of the top 100k corpus
//! words and returned unverified delete-set intersections above the decoder's
//! band. These tests pin the behaviour of what replaced it.

use akshar_ime::ImeEngine;
use std::path::Path;

fn engine() -> Option<ImeEngine> {
    Path::new("data/akshar.model").exists().then(ImeEngine::new)
}

fn tops(e: &ImeEngine, q: &str, n: usize) -> Vec<String> {
    e.get_suggestions(q, n)
        .into_iter()
        .map(|(s, _)| s)
        .collect()
}

#[test]
fn exact_query_returns_the_learned_word_first() {
    let Some(mut e) = engine() else { return };
    e.user_confirms("kathmandu", "काठमाडौँ");
    let got = tops(&e, "kathmandu", 5);
    assert_eq!(
        got.first().map(String::as_str),
        Some("काठमाडौँ"),
        "exact re-query did not return the learned word first; got {got:?}"
    );
}

/// Documents a defect (D18), it does not assert desired behaviour.
///
/// The user SymSpell path scores `fuzzy_base() - 12_000 * distance`, i.e.
/// 38,000 at distance 1, while the decoder's candidates occupy
/// `800_000 / (1 + cost)` and its 8th-ranked candidate still scores ~250,000.
/// A learned word reached only through a typo therefore cannot appear anywhere
/// near the top of the list: the path is structurally unreachable, not merely
/// conservative.
///
/// This is the same defect class as the corpus-SymSpell source removed on
/// 2026-09-06 — an arbitrary u64 band compared against a squashed reranker
/// score — in the opposite direction. Raising the constant is how the 30.8pp
/// regression happened, so the fix is the log-linear fusion in phase C4 of
/// `docs/plans/archive/2026-09-06-repair-and-path-to-90.md`, not a bigger number.
#[test]
fn learned_word_should_be_recoverable_from_a_typo() {
    let Some(mut e) = engine() else { return };
    e.user_confirms("kathmandu", "काठमाडौँ");
    for typo in ["kathmandau", "kathmndu"] {
        let got = tops(&e, typo, 10);
        assert!(
            got.iter().any(|s| s == "काठमाडौँ"),
            "typo {typo:?} did not recover the learned word; got {got:?}"
        );
    }
}

/// The scoring bands are far enough apart that the fuzzy path is dominated.
/// Pinned so the gap is visible in CI rather than discovered again later.
#[test]
fn user_fuzzy_band_is_below_the_decoder_band() {
    let Some(mut e) = engine() else { return };
    e.user_confirms("kathmandu", "काठमाडौँ");
    let scored = e.get_suggestions("kathmandau", 10);
    let eighth = scored.get(7).map(|(_, s)| *s).unwrap_or(0);
    // 50_000 - 12_000 = 38_000 is the best a distance-1 fuzzy hit can score.
    assert!(
        eighth > 38_000,
        "decoder tail ({eighth}) no longer dominates the fuzzy band; \
         D18 may have been fixed -- unignore learned_word_should_be_recoverable_from_a_typo"
    );
}

#[test]
fn fuzzy_does_not_outrank_an_exact_decode() {
    let Some(mut e) = engine() else { return };
    // Teach a word that is one edit away from a different, unrelated query.
    e.user_confirms("pani", "पानी");
    // A different query that the decoder handles exactly must not be hijacked
    // by the learned neighbour: this is the failure mode that cost 30.8pp when
    // the corpus fuzzy source scored above the decoder band.
    let got = tops(&e, "sarkar", 5);
    assert!(
        !got.is_empty() && got[0] != "पानी",
        "fuzzy match outranked the exact decode; got {got:?}"
    );
}

#[test]
fn vowel_length_variants_need_no_learning() {
    let Some(e) = engine() else { return };
    // Cold start, nothing confirmed: doubled-vowel spellings and the canonical
    // spelling should land on the same word via the query normalizer, not via
    // any corpus-wide fuzzy index.
    let canonical = tops(&e, "sathi", 10);
    let doubled = tops(&e, "saathee", 10);
    assert!(!canonical.is_empty() && !doubled.is_empty());
    let overlap = doubled.iter().filter(|d| canonical.contains(d)).count();
    assert!(
        overlap > 0,
        "no shared candidate between 'sathi' {canonical:?} and 'saathee' {doubled:?}"
    );
}

#[test]
fn unlearned_typo_falls_through_to_the_decoder() {
    let Some(e) = engine() else { return };
    // With nothing learned there is no fuzzy source at all any more, so a typo
    // is simply decoded as written. It must still produce output rather than
    // an empty list — silently returning nothing would be worse than a wrong
    // guess for an IME.
    assert!(!tops(&e, "kthmndu", 5).is_empty());
}
