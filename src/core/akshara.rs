// File: src/core/akshara.rs
//
// Devanagari akshara (syllabic unit) segmenter.
//
// An akshara is the orthographic syllable of Brahmic scripts. It groups a
// sequence of Unicode codepoints into a single pronounceable unit:
//
//     akshara := (consonant halanta)* consonant? (matra | independent-vowel)
//                (anusvara | visarga | chandrabindu | nukta)*
//
// Examples:
//   "नमस्ते"   -> ["न", "म", "स्ते"]
//   "क्ष्त्र"  -> ["क्ष्त्र"]      (a single conjunct akshara)
//   "अग"       -> ["अ", "ग"]
//   "आई"       -> ["आ", "ई"]      (two independent-vowel akshara)

/// True for Devanagari consonants (the `Consonant` class below, including
/// nukta forms). Used by the morphology strip's preceding-consonant gate.
pub fn is_consonant(ch: char) -> bool {
    matches!(classify(ch), AksharaClass::Consonant)
}

/// Categorise a single Devanagari codepoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AksharaClass {
    /// An independent vowel (अ, आ, इ, ...). Always starts a new akshara.
    IndependentVowel,
    /// A consonant (क, ख, ...). Starts a new akshara unless preceded by halanta.
    Consonant,
    /// Virama / halanta (्). Glues the following consonant into the current akshara.
    Halanta,
    /// A dependent vowel sign / matra (ा, ि, ी, ...). Attaches to current akshara.
    Matra,
    /// A combining mark: anusvara (ं), visarga (ः), chandrabindu (ँ), nukta (़).
    CombiningMark,
    /// Anything else: digits, punctuation, spaces, non-Devanagari.
    Other,
}

fn classify(ch: char) -> AksharaClass {
    let cp = ch as u32;
    // Halanta / virama.
    if cp == 0x094D {
        return AksharaClass::Halanta;
    }
    // Independent vowels: U+0904..U+0914, plus ॠ U+0960, ॡ U+0961.
    if (0x0904..=0x0914).contains(&cp) || cp == 0x0960 || cp == 0x0961 {
        return AksharaClass::IndependentVowel;
    }
    // Consonants: U+0915..U+0939, nukta-formed U+0958..U+095F, extended U+0978..U+097A,
    // plus a couple of stray consonants (ऱ U+0931, ऴ U+0934).
    if (0x0915..=0x0939).contains(&cp)
        || (0x0958..=0x095F).contains(&cp)
        || (0x0978..=0x097A).contains(&cp)
        || cp == 0x0931
        || cp == 0x0934
    {
        return AksharaClass::Consonant;
    }
    // Matras / dependent vowel signs: U+093E..U+094C, plus U+093A, U+093B, U+094E,
    // U+094F, U+0962, U+0963.
    if (0x093E..=0x094C).contains(&cp)
        || cp == 0x093A
        || cp == 0x093B
        || cp == 0x094E
        || cp == 0x094F
        || cp == 0x0962
        || cp == 0x0963
    {
        return AksharaClass::Matra;
    }
    // Combining marks: anusvara U+0902, visarga U+0903, chandrabindu U+0901,
    // nukta U+093C, and the stress/cantillation marks U+0951..U+0957.
    if cp == 0x0902
        || cp == 0x0903
        || cp == 0x0901
        || cp == 0x093C
        || (0x0951..=0x0957).contains(&cp)
    {
        return AksharaClass::CombiningMark;
    }
    AksharaClass::Other
}

use unicode_normalization::UnicodeNormalization;

/// Segment a Devanagari string into akshara (syllabic) units.
///
/// The input is NFC-normalized first (via `unicode-normalization`) so that
/// decomposed matra / nukta sequences are classified identically to their
/// composed forms. Classification below is still the engine's own IP.
///
/// Each returned `String` is one akshara: a consonant cluster (possibly with
/// internal halanta conjuncts) plus its vowel sign and trailing combining marks,
/// or a standalone independent vowel, or a lone non-Devanagari character.
pub fn segment(dev: &str) -> Vec<String> {
    let norm: String = dev.nfc().collect();
    let mut units: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut prev_was_halanta = false;

    for ch in norm.chars() {
        match classify(ch) {
            AksharaClass::IndependentVowel => {
                // An independent vowel is always the nucleus of a fresh akshara.
                if !current.is_empty() {
                    units.push(std::mem::take(&mut current));
                }
                current.push(ch);
                prev_was_halanta = false;
            }
            AksharaClass::Consonant => {
                // A consonant starts a new akshara unless it directly continues a
                // conjunct (i.e. the previous codepoint was a halanta).
                if !current.is_empty() && !prev_was_halanta {
                    units.push(std::mem::take(&mut current));
                }
                current.push(ch);
                prev_was_halanta = false;
            }
            AksharaClass::Halanta => {
                current.push(ch);
                prev_was_halanta = true;
            }
            AksharaClass::Matra | AksharaClass::CombiningMark => {
                // Matras and combining marks attach to the current akshara. If there
                // is no current akshara (stray mark at the start), it becomes its own
                // unit so we never silently drop data.
                if current.is_empty() {
                    units.push(ch.to_string());
                } else {
                    current.push(ch);
                }
                prev_was_halanta = false;
            }
            AksharaClass::Other => {
                if !current.is_empty() {
                    units.push(std::mem::take(&mut current));
                }
                units.push(ch.to_string());
                prev_was_halanta = false;
            }
        }
    }

    if !current.is_empty() {
        units.push(current);
    }
    units
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_simple_cv_sequence() {
        assert_eq!(segment("नमस्ते"), vec!["न", "म", "स्ते"]);
    }

    #[test]
    fn segments_independent_vowels_separately() {
        assert_eq!(segment("अग"), vec!["अ", "ग"]);
        assert_eq!(segment("आई"), vec!["आ", "ई"]);
    }

    #[test]
    fn keeps_conjunct_as_one_akshara() {
        // क्ष = क + ् + ष  -> a single conjunct akshara.
        assert_eq!(segment("क्ष"), vec!["क्ष"]);
        // त्र = त + ् + र
        assert_eq!(segment("त्र"), vec!["त्र"]);
    }

    #[test]
    fn segments_word_with_conjunct_and_matra() {
        // कृपया -> क + ृ + प + या  => ["कृ", "प", "या"]
        assert_eq!(segment("कृपया"), vec!["कृ", "प", "या"]);
    }

    #[test]
    fn attaches_combining_marks() {
        // नमस्ते + anusvara on last: नमस्तें -> last akshara keeps the anusvara.
        assert_eq!(segment("नमस्तें"), vec!["न", "म", "स्तें"]);
        // अं is one akshara (independent vowel + anusvara).
        assert_eq!(segment("अं"), vec!["अं"]);
    }

    #[test]
    fn handles_final_halanta() {
        // Word ending in explicit halanta: "क्" stays as one akshara with the halanta.
        assert_eq!(segment("क्"), vec!["क्"]);
    }

    #[test]
    fn passes_through_non_devanagari() {
        assert_eq!(segment("a"), vec!["a"]);
        assert_eq!(segment("क a"), vec!["क", " ", "a"]);
        assert_eq!(segment("१२"), vec!["१", "२"]);
    }

    #[test]
    fn empty_input_yields_no_units() {
        assert!(segment("").is_empty());
    }

    #[test]
    fn segments_long_word() {
        // काठमाडौं -> का, ठ, मा, डौं
        assert_eq!(segment("काठमाडौं"), vec!["का", "ठ", "मा", "डौं"]);
    }
}
