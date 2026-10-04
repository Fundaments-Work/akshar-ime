// File: src/bin/build/build_lexicon.rs
//
// Count words per Devanagari language and build the lexicon automaton
// (src/core/lexicon.rs) the ranker and the dictionary-constrained decoder use.
//
//   cargo run --release --bin build_lexicon -- \
//       [--corpus data/nepali_corpus.txt] \
//       [--min-count 2] [--max-words 300000] [--out data/lexicon.bin]
//
// Input: the Nepali running-text corpus (--corpus), streamed line by line.
// The engine ships Nepali-only priors, so this builds a single-language
// lexicon; the container still reserves MAX_LANGS levels per word.
// --corpus honours the shared 1-in-200 sentence holdout, so the
// sentence-level evaluation text never reaches the counts.
//
// A token is a maximal run of Devanagari letters and signs (U+0900..U+0963)
// after trimming punctuation, NFC-normalised, 1..=24 characters -- the same
// rule as the trainer's vocabulary.  Per language, words seen fewer than
// --min-count times are dropped and only the --max-words most frequent kept:
// the cap bounds the automaton (300k per language: 1.16M words, ~10 MB)
// while keeping 85-97% of validation gold words for Hindi and Marathi.
//
// Output: bincode (LexiconData, tokens counted per language).

use akshar_ime::core::holdout::{is_holdout, DEFAULT_HOLDOUT_DENOM};
use akshar_ime::core::lexicon::{quantise, Levels, Lexicon, MAX_LANGS};
use anyhow::{Context, Result};
use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use unicode_normalization::UnicodeNormalization;

/// The engine ships Nepali-only priors.  `Lexicon` still stores up to
/// `MAX_LANGS` levels per word (src/core/lexicon.rs) so the container format is
/// unchanged, but the pipeline populates exactly this one language.
const LANG: &str = "nep";

fn is_word_char(c: char) -> bool {
    matches!(c, '\u{0900}'..='\u{0963}')
}

fn clean_token(raw: &str) -> Option<String> {
    let nfc: String = raw.nfc().collect();
    let t = nfc.trim_matches(|c: char| !is_word_char(c));
    let n = t.chars().count();
    ((1..=24).contains(&n) && t.chars().all(is_word_char)).then(|| t.to_string())
}

#[derive(Default)]
struct Counts {
    words: HashMap<String, u64>,
    tokens: u64,
}

impl Counts {
    fn add_line(&mut self, line: &str) {
        for raw in line.split_whitespace() {
            if let Some(w) = clean_token(raw) {
                *self.words.entry(w).or_insert(0) += 1;
                self.tokens += 1;
            }
        }
    }
}

/// Count one running-text corpus, dropping the shared sentence holdout so the
/// sentence-level evaluation text never reaches the counts.
fn count_corpus(path: &Path, counts: &mut Counts) -> Result<()> {
    let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    for line in BufReader::new(f).lines() {
        let line = line?;
        if !is_holdout(&line, DEFAULT_HOLDOUT_DENOM) {
            counts.add_line(&line);
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut corpus = PathBuf::from("data/nepali_corpus.txt");
    let mut min_count: u64 = 2;
    let mut max_words: usize = 300_000;
    let mut out = PathBuf::from("data/lexicon.bin");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().with_context(|| format!("value for {a}"));
        match a.as_str() {
            "--corpus" => corpus = PathBuf::from(val()?),
            "--min-count" => min_count = val()?.parse()?,
            "--max-words" => max_words = val()?.parse()?,
            "--out" => out = PathBuf::from(val()?),
            "-h" | "--help" => {
                println!(
                    "build_lexicon [--corpus data/nepali_corpus.txt] [--min-count 2] \
                     [--max-words 300000] [--out path]\n\n\
                     Counts the Nepali running-text corpus and builds the lexicon \
                     automaton ({LANG}-only priors)."
                );
                return Ok(());
            }
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }

    let lang = LANG;
    let files: BTreeMap<&'static str, Vec<PathBuf>> = BTreeMap::from([(lang, vec![corpus])]);
    let langs: Vec<&'static str> = files.keys().copied().collect();
    anyhow::ensure!(langs.len() <= MAX_LANGS, "more than {MAX_LANGS} languages");

    // Count every language in parallel.
    let counted: Vec<(&str, Counts)> = std::thread::scope(|s| {
        let handles: Vec<_> = langs
            .iter()
            .map(|&lang| {
                let files = &files[lang];
                s.spawn(move || -> Result<(&str, Counts)> {
                    let mut counts = Counts::default();
                    for path in files {
                        count_corpus(path, &mut counts)?;
                    }
                    Ok((lang, counts))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().map_err(|_| anyhow::anyhow!("counter panicked"))?)
            .collect::<Result<Vec<_>>>()
    })?;

    // Per language: the most frequent words above the floor.  Union them.
    let mut union: HashMap<String, Levels> = HashMap::new();
    let mut totals = Vec::with_capacity(langs.len());
    println!(
        "{:<5} {:>12} {:>10} {:>10}",
        "lang", "tokens", "distinct", "kept"
    );
    for (l, (lang, counts)) in counted.iter().enumerate() {
        let mut ranked: Vec<(&String, u64)> = counts
            .words
            .iter()
            .filter(|(_, &c)| c >= min_count)
            .map(|(w, &c)| (w, c))
            .collect();
        ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        ranked.truncate(max_words);
        for (word, c) in &ranked {
            union.entry((*word).clone()).or_insert([0; MAX_LANGS])[l] = quantise(*c, counts.tokens);
        }
        println!(
            "{lang:<5} {:>12} {:>10} {:>10}",
            counts.tokens,
            counts.words.len(),
            ranked.len()
        );
        totals.push(counts.tokens);
    }
    let lang_names: Vec<String> = langs.iter().map(|l| l.to_string()).collect();
    let lexicon = Lexicon::build(lang_names, union).context("build automaton")?;
    let bytes = bincode::serialize(&(lexicon.to_data(), totals))?;
    std::fs::write(&out, &bytes).with_context(|| format!("write {}", out.display()))?;
    println!(
        "{} words in {} languages -> {} ({:.2} MB)",
        lexicon.len(),
        langs.len(),
        out.display(),
        lexicon.byte_size() as f64 / 1e6
    );
    Ok(())
}
