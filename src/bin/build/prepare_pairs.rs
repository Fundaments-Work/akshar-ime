// File: src/bin/build/prepare_pairs.rs
//
// Build the per-language word-pair splits every trainer and evaluation
// harness reads, from the verified Aksharantar downloads (`make data-fetch`,
// extracted by `make data-prepare`).
//
//   cargo run --release --bin prepare_pairs -- [--raw data/raw/aksharantar]
//                                              [--out data/pairs] [--seed 7]
//
// Input:  <raw>/<lang>/<lang>_{train,valid,test}.json, or the flat
//         <raw>/<lang>_{split}.json layout of the cleaned copies in
//         data/aksharantar/ (which omit `source`; see --assume-source).
// Output: <out>/{train,valid,test}.jsonl, one object per line:
//         {"english word": .., "native word": .., "source": .., "lang": ..}
//
// Normalisation (every split, so training and scoring share one convention):
//   * roman: trimmed and ASCII-lowercased (Dogri ships capitalised names);
//   * native: trimmed and NFC-normalised.  Aksharantar stores eyelash-ra as
//     र + nukta in 11.5k Marathi and 4k Konkani rows; NFC makes that the one
//     code point ऱ, so a correct answer can never be scored wrong for its
//     byte encoding.
//
// Train rows are additionally cleaned and de-duplicated:
//   * the roman must be a-z only and the native must be Devanagari letters
//     and signs only (U+0900..=U+0963: no danda, digits, abbreviation sign);
//   * exact duplicate pairs are kept once — Hindi and Nepali share ~108k;
//   * a pair whose native word appears in ANY language's valid or test split
//     is dropped, so no evaluation word is ever trained on.  Aksharantar
//     already de-duplicates across languages except Dogri, whose test set
//     overlaps other languages' training words.
//
// Valid and test rows are only normalised, never dropped, so their counts
// match the published splits and results stay comparable with the paper.
//
// Train is shuffled with a fixed seed: the upstream files are one language
// after another, and a "first N rows" sample of a language-sorted file is a
// sample of one language (which is exactly how the reranker once ended up
// trained on Hindi only).

use anyhow::{Context, Result};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use unicode_normalization::UnicodeNormalization;

/// The one language the engine ships priors for.  Aksharantar publishes eight
/// Devanagari languages, but only the Nepali split is vendored under
/// data/aksharantar/ and only Nepali has a running-text corpus, so the
/// pipeline is Nepali-only.  `lang` stays on every record because the
/// container and the eval harness are keyed by it.
const LANG: &str = "nep";

#[derive(Deserialize)]
struct RawRecord {
    #[serde(rename = "english word")]
    roman: String,
    #[serde(rename = "native word")]
    native: String,
    #[serde(default)]
    source: Option<String>,
}

#[derive(Serialize)]
struct Pair<'a> {
    #[serde(rename = "english word")]
    roman: &'a str,
    #[serde(rename = "native word")]
    native: &'a str,
    source: &'a str,
    lang: &'a str,
}

struct Row {
    roman: String,
    native: String,
    source: String,
    lang: &'static str,
}

fn normalise(rec: RawRecord, lang: &'static str) -> Row {
    // Upstream spells one source both "AK-Freq" and "Ak-Freq" (Dogri).
    let source = match rec.source.as_deref().unwrap_or("unknown") {
        "Ak-Freq" => "AK-Freq".to_string(),
        s => s.to_string(),
    };
    Row {
        roman: rec.roman.trim().to_ascii_lowercase(),
        native: rec.native.trim().nfc().collect(),
        source,
        lang,
    }
}

/// 128-bit identity of a (roman, native) pair for de-duplication: two SipHash
/// passes over differently-prefixed input.  Holding ~8M hashes instead of ~8M
/// string pairs keeps this step under a quarter of a gigabyte.
fn pair_key(roman: &str, native: &str) -> u128 {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let half = |salt: u8| {
        let mut h = DefaultHasher::new();
        salt.hash(&mut h);
        roman.hash(&mut h);
        native.hash(&mut h);
        h.finish()
    };
    (u128::from(half(0)) << 64) | u128::from(half(1))
}

fn is_clean_train_row(r: &Row) -> bool {
    !r.roman.is_empty()
        && r.roman.bytes().all(|b| b.is_ascii_lowercase())
        && !r.native.is_empty()
        && r.native
            .chars()
            .all(|c| matches!(c, '\u{0900}'..='\u{0963}'))
}

/// Locate one split, probing both on-disk layouts.
///
/// Upstream ships `data/raw/aksharantar/<lang>/<lang>_<split>.json`.  The
/// cleaned, metadata-stripped copies committed under `data/aksharantar/` are
/// flat and carry only `english word` / `native word`, so `source` arrives as
/// `None` and every row would land in eval_langs' "mined" stratum.  Probing both
/// keeps the pipeline runnable against whichever copy is present.
fn split_path(raw: &Path, lang: &str, split: &str) -> Option<PathBuf> {
    let nested = raw.join(lang).join(format!("{lang}_{split}.json"));
    if nested.exists() {
        return Some(nested);
    }
    let flat = raw.join(format!("{lang}_{split}.json"));
    flat.exists().then_some(flat)
}

/// Read one split.  Returns the rows plus how many had `source` filled in from
/// `assume` -- the cleaned copies do not carry the field, and the stratum
/// assignment in `eval_langs` is derived from it, so the count is reported
/// rather than applied silently.
fn read_split(
    raw: &Path,
    lang: &'static str,
    split: &str,
    assume: Option<&str>,
) -> Result<(Vec<Row>, usize)> {
    let Some(path) = split_path(raw, lang, split) else {
        // Dogri publishes no valid split.
        return Ok((Vec::new(), 0));
    };
    let f = File::open(&path).with_context(|| format!("open {}", path.display()))?;
    let mut rows = Vec::new();
    let mut assumed = 0usize;
    for (i, line) in BufReader::new(f).lines().enumerate() {
        let line = line.with_context(|| format!("read {}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let rec: RawRecord = serde_json::from_str(&line)
            .with_context(|| format!("{}:{}: bad record", path.display(), i + 1))?;
        let mut row = normalise(rec, lang);
        if row.source == "unknown" {
            if let Some(a) = assume {
                row.source = a.to_string();
                assumed += 1;
            }
        }
        rows.push(row);
    }
    Ok((rows, assumed))
}

fn write_split(path: &Path, rows: &[Row]) -> Result<()> {
    let mut w =
        BufWriter::new(File::create(path).with_context(|| format!("create {}", path.display()))?);
    for r in rows {
        let p = Pair {
            roman: &r.roman,
            native: &r.native,
            source: &r.source,
            lang: r.lang,
        };
        serde_json::to_writer(&mut w, &p)?;
        w.write_all(b"\n")?;
    }
    w.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let mut raw = PathBuf::from("data/aksharantar");
    let mut out = PathBuf::from("data/pairs");
    let mut seed: u64 = 7;
    let mut assume_source: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--raw" => raw = PathBuf::from(args.next().context("value for --raw")?),
            "--out" => out = PathBuf::from(args.next().context("value for --out")?),
            "--seed" => seed = args.next().context("value for --seed")?.parse()?,
            "--assume-source" => {
                assume_source = Some(args.next().context("value for --assume-source")?)
            }
            "-h" | "--help" => {
                println!(
                    "prepare_pairs [--raw data/aksharantar] [--out data/pairs] [--seed 7]\n\
                     \x20              [--assume-source AK-Freq]\n\n\
                     Reads the vendored Nepali splits, <raw>/nep_{{train,valid,test}}.json.\n\
                     --assume-source fills the `source` field those copies dropped, which\n\
                     the eval harness needs to assign a stratum; the count is reported."
                );
                return Ok(());
            }
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }

    let lang = LANG;
    anyhow::ensure!(
        ["train", "valid", "test"]
            .iter()
            .any(|s| split_path(&raw, lang, s).is_some()),
        "no Nepali splits under {} (expected {lang}_{{train,valid,test}}.json)",
        raw.display()
    );
    let assume = assume_source.as_deref();

    std::fs::create_dir_all(&out)?;

    // Evaluation splits first: their native words define the hygiene filter.
    let mut valid = Vec::new();
    let mut test = Vec::new();
    let mut assumed_eval = 0usize;
    {
        let (v, av) = read_split(&raw, lang, "valid", assume)?;
        valid.extend(v);
        assumed_eval += av;
        let (t, at) = read_split(&raw, lang, "test", assume)?;
        test.extend(t);
        assumed_eval += at;
    }
    let held_out: HashSet<&str> = valid
        .iter()
        .chain(&test)
        .map(|r| r.native.as_str())
        .collect();

    #[derive(Default)]
    struct Tally {
        read: usize,
        unclean: usize,
        duplicate: usize,
        evaluation_word: usize,
        kept: usize,
    }
    let mut tally = Tally::default();
    let mut seen: HashSet<u128> = HashSet::new();
    let mut train: Vec<Row> = Vec::new();
    let mut assumed_train = 0usize;
    {
        let (rows, assumed) = read_split(&raw, lang, "train", assume)?;
        assumed_train += assumed;
        for r in rows {
            tally.read += 1;
            if !is_clean_train_row(&r) {
                tally.unclean += 1;
            } else if held_out.contains(r.native.as_str()) {
                tally.evaluation_word += 1;
            } else if !seen.insert(pair_key(&r.roman, &r.native)) {
                tally.duplicate += 1;
            } else {
                tally.kept += 1;
                train.push(r);
            }
        }
    }
    drop(held_out);

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    train.shuffle(&mut rng);

    write_split(&out.join("train.jsonl"), &train)?;
    write_split(&out.join("valid.jsonl"), &valid)?;
    write_split(&out.join("test.jsonl"), &test)?;

    println!(
        "{:<5} {:>10} {:>9} {:>10} {:>11} {:>10} {:>8} {:>8}",
        "lang", "train-read", "unclean", "duplicate", "eval-word", "train", "valid", "test"
    );
    println!(
        "{lang:<5} {:>10} {:>9} {:>10} {:>11} {:>10} {:>8} {:>8}",
        tally.read,
        tally.unclean,
        tally.duplicate,
        tally.evaluation_word,
        tally.kept,
        valid.len(),
        test.len()
    );
    if let Some(a) = assume {
        let total = assumed_train + assumed_eval;
        if total > 0 {
            println!(
                "source field absent on {total} rows (train {assumed_train}, \
                 valid+test {assumed_eval}); assumed \"{a}\"."
            );
        }
    }
    println!(
        "wrote {} train / {} valid / {} test rows to {} (train shuffled, seed {seed})",
        train.len(),
        valid.len(),
        test.len(),
        out.display()
    );
    Ok(())
}
