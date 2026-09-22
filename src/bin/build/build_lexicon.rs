// File: src/bin/build/build_lexicon.rs
//
// Count words per Devanagari language and build the lexicon automaton
// (src/core/lexicon.rs) the ranker and the dictionary-constrained decoder use.
//
//   cargo run --release --bin build_lexicon -- \
//       [--raw data/raw/indiccorp-v2-sample] [--extra nep=data/store/corpus_clean.txt] \
//       [--min-count 2] [--max-words 300000] [--out data/lexicon.bin]
//
// Input: the IndicCorp v2 sample fetched by `make data-fetch` -- files
// `<code>.<NN>.txt` (evenly spaced 25 MiB slices; their first and last lines
// are cut mid-sentence and dropped) or `<code>.txt` (whole files).  IndicCorp
// codes map to the ISO 639-3 codes used everywhere else.  --extra adds more
// running text for one language (Nepali keeps the 86M-token corpus the
// engine was built on); it honours the shared 1-in-200 sentence holdout, so
// the sentence-level evaluation text never reaches the counts.
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

/// IndicCorp v2 file code -> ISO 639-3.
fn language_of(code: &str) -> Option<&'static str> {
    Some(match code {
        "hi" => "hin",
        "mr" => "mar",
        "ne" => "nep",
        "sa" => "san",
        "gom" => "kok",
        "mai" => "mai",
        "bd" => "brx",
        "dg" => "doi",
        _ => return None,
    })
}

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

/// Count one IndicCorp file; slices lose their (partial) first and last line.
fn count_file(path: &Path, sliced: bool, counts: &mut Counts) -> Result<()> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let text = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = text.lines().collect();
    let body = if sliced && lines.len() >= 2 {
        &lines[1..lines.len() - 1]
    } else {
        &lines[..]
    };
    for line in body {
        counts.add_line(line);
    }
    Ok(())
}

fn count_extra(path: &Path, counts: &mut Counts) -> Result<()> {
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
    let mut raw = PathBuf::from("data/raw/indiccorp-v2-sample");
    let mut extra: Vec<(String, PathBuf)> = Vec::new();
    let mut min_count: u64 = 2;
    let mut max_words: usize = 300_000;
    let mut out = PathBuf::from("data/lexicon.bin");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().with_context(|| format!("value for {a}"));
        match a.as_str() {
            "--raw" => raw = PathBuf::from(val()?),
            "--extra" => {
                let v = val()?;
                let (lang, path) = v.split_once('=').context("--extra wants lang=path")?;
                extra.push((lang.to_string(), PathBuf::from(path)));
            }
            "--min-count" => min_count = val()?.parse()?,
            "--max-words" => max_words = val()?.parse()?,
            "--out" => out = PathBuf::from(val()?),
            "-h" | "--help" => {
                println!(
                    "build_lexicon [--raw dir] [--extra lang=path]... [--min-count 2] \
                     [--max-words 300000] [--out path]"
                );
                return Ok(());
            }
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }

    // Group input files by language.
    let mut files: BTreeMap<&'static str, Vec<(PathBuf, bool)>> = BTreeMap::new();
    for entry in std::fs::read_dir(&raw).with_context(|| format!("read {}", raw.display()))? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some(stem) = name.strip_suffix(".txt") else {
            continue;
        };
        // "hi-1.03" / "mr.11" (slices) or "mai" (whole file).
        let (code, sliced) = match stem.split_once('.') {
            Some((c, _)) => (c.split('-').next().unwrap_or(c), true),
            None => (stem, false),
        };
        if let Some(lang) = language_of(code) {
            files.entry(lang).or_default().push((path.clone(), sliced));
        }
    }
    let langs: Vec<&'static str> = files.keys().copied().collect();
    anyhow::ensure!(
        !langs.is_empty(),
        "no IndicCorp files under {}",
        raw.display()
    );
    anyhow::ensure!(langs.len() <= MAX_LANGS, "more than {MAX_LANGS} languages");

    // Count every language in parallel.
    let counted: Vec<(&str, Counts)> = std::thread::scope(|s| {
        let handles: Vec<_> = langs
            .iter()
            .map(|&lang| {
                let files = &files[lang];
                let extra = &extra;
                s.spawn(move || -> Result<(&str, Counts)> {
                    let mut counts = Counts::default();
                    for (path, sliced) in files {
                        count_file(path, *sliced, &mut counts)?;
                    }
                    for (l, path) in extra {
                        if l == lang {
                            count_extra(path, &mut counts)?;
                        }
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
