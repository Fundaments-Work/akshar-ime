// File: src/bin/build_wordfreq_text.rs
//
// M4-real: count Devanagari word frequencies from raw running text
// (e.g. a Wikipedia dump for the target language) and serialise
// HashMap<String, u32> -> data/word_freq_text.bin.
//
// Usage: build_wordfreq_text <text-file> [more files...]

use std::collections::HashMap;
use std::io::{BufRead, BufReader};

/// Allowed code points for a vocabulary word: Devanagari letters, matras,
/// nukta consonants, vocalics, anusvara/visarga (U+0900..=U+0963). This
/// deliberately EXCLUDES danda/double-danda (U+0964-0965), Devanagari digits
/// (U+0966-096F), and abbreviation signs (U+0970+) — running text glues all
/// of these onto words ("पुगे।", "२०७८साल", "सोमाली,"), which both pollutes
/// the vocabulary and splits a real word's count across glued variants.
fn is_word_char(c: char) -> bool {
    matches!(c, '\u{0900}'..='\u{0963}')
}

/// Trim non-word characters glued to the token edges, then require the
/// remainder to be pure Devanagari. Returns None for numbers, URLs, English,
/// dates, and glued phrases (ZWJ/ZWNJ joiners are not word chars either).
fn clean_token(word: &str) -> Option<String> {
    let trimmed = word.trim_matches(|c: char| !is_word_char(c));
    let chars: Vec<char> = trimmed.chars().collect();
    if chars.is_empty() || chars.len() > 24 {
        return None;
    }
    if chars.iter().all(|c| is_word_char(*c)) {
        Some(trimmed.to_string())
    } else {
        None
    }
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: build_wordfreq_text <raw-text...>");
        std::process::exit(2);
    }
    let mut freq: HashMap<String, u32> = HashMap::new();
    let mut total = 0u64;
    for path in &paths {
        let Ok(f) = std::fs::File::open(path) else {
            eprintln!("WARNING: cannot open {path}");
            continue;
        };
        for line in BufReader::new(f).lines().map_while(Result::ok) {
            for word in line.split_whitespace() {
                if let Some(clean) = clean_token(word) {
                    *freq.entry(clean).or_insert(0) += 1;
                    total += 1;
                }
            }
        }
    }
    // Prune hapax-ish tail to keep the artifact sane.
    freq.retain(|_, &mut c| c >= 3);
    let out = std::path::Path::new("data/word_freq_text.bin");
    let bytes = bincode::serialize(&freq).expect("serialize");
    std::fs::write(out, &bytes).expect("write");
    eprintln!(
        "{total} tokens -> {} words (count>=3) -> {} ({:.1} MB)",
        freq.len(),
        out.display(),
        bytes.len() as f64 / 1e6
    );
}
