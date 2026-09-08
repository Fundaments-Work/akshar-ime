// File: src/bin/evaluate/eval_ime.rs
//
// Intuitive IME evaluation: human-readable summary + machine JSON.
//
// Human view (typist language):
//   - Correct First Time (top-1): right without touching arrows.
//   - Visible in Top-5: right answer on screen.
//   - Keystrokes Saved: prefix simulation; shortest roman prefix whose top-5
//     contains a target, as 1 - prefix_len/roman_len (0 if never).
//   - Speed/size line + stability note (bootstrap CI width).
// Machine view (--out eval.json): same numbers + CIs + CER + latency, for
// docs generation. One run, two views; never hand-copy numbers.
//
// Library notes: clap for args, rand_chacha for seeded bootstrap,
// unicode-segmentation graphemes for CER, fst::Set for the roman-key index
// (dedup check + prefix statistics), smallvec for hot grapheme buffers.

use akshar_ime::ImeEngine;
use clap::Parser;
use fst::Set;
use rand::{Rng as _, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;
use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug, Parser)]
struct Args {
    /// Test set (Aksharantar JSONL or roman<TAB>target1|target2 TSV).
    #[arg(long, default_value = "data/aksharantar/test_devanagari.jsonl")]
    dataset: PathBuf,
    /// Suggestions to request per query.
    #[arg(long, default_value_t = 5)]
    suggestions: usize,
    /// Only evaluate first <n> cases (debug).
    #[arg(long)]
    limit: Option<usize>,
    /// Explicit unified-model path.
    #[arg(long)]
    model: Option<PathBuf>,
    /// Bootstrap resamples for stability note (0 disables).
    #[arg(long, default_value_t = 1000)]
    resamples: usize,
    /// RNG seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Write machine JSON here (also used by docs generation).
    #[arg(long)]
    out: Option<PathBuf>,
    /// Show up to <n> misses with diffs.
    #[arg(long, default_value_t = 10)]
    show_misses: usize,
}

#[derive(Debug, Clone)]
struct EvalCase {
    roman: String,
    targets: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Record<'a> {
    #[serde(rename = "english word", borrow)]
    english: &'a str,
    #[serde(rename = "native word", borrow)]
    native: &'a str,
}

#[derive(Debug, Clone, Serialize)]
struct EvalJson {
    dataset: String,
    cases: usize,
    correct_first_time: f64,
    visible_top5: f64,
    mrr: f64,
    keystrokes_saved: f64,
    cer_grapheme: f64,
    avg_latency_ms: f64,
    top1_ci95: (f64, f64),
    duplicates_dropped: usize,
}

fn main() {
    let mut args = Args::parse();
    if !args.dataset.exists() {
        for cand in [
            "data/aksharantar/test_devanagari.jsonl",
            "data/aksharantar/nep_test.json",
        ] {
            if std::path::Path::new(cand).exists() {
                args.dataset = PathBuf::from(cand);
                break;
            }
        }
    }
    let cases = parse_dataset(&args.dataset).unwrap_or_else(|e| {
        eprintln!("Failed to parse dataset `{}`: {e}", args.dataset.display());
        std::process::exit(1);
    });
    if cases.is_empty() {
        eprintln!("Dataset is empty: {}", args.dataset.display());
        std::process::exit(2);
    }
    let cases: Vec<EvalCase> = if let Some(lim) = args.limit {
        cases.into_iter().take(lim).collect()
    } else {
        cases
    };

    // Immutable fst index over roman keys: proves dedup + enables prefix stats.
    let mut keys: Vec<String> = cases.iter().map(|c| c.roman.clone()).collect();
    keys.sort();
    let total_keys = keys.len();
    keys.dedup();
    let duplicates_dropped = total_keys - keys.len();
    let key_refs: Vec<&str> = keys.iter().map(String::as_str).collect();
    let key_set = Set::from_iter(key_refs.iter().copied()).unwrap_or_else(|e| {
        eprintln!("fst index build failed: {e}");
        std::process::exit(1);
    });

    let engine = match &args.model {
        Some(path) => ImeEngine::from_unified_file(path).unwrap_or_else(|e| {
            eprintln!("cannot load model {}: {e}", path.display());
            std::process::exit(1);
        }),
        None => ImeEngine::new(),
    };

    let n_suggest = args.suggestions.max(1);
    let mut top1_hits = 0usize;
    let mut top5_hits = 0usize;
    let mut rr_sum = 0.0f64;
    let mut saved_sum = 0.0f64;
    let mut cer_sum = 0.0f64;
    let mut total_latency_us = 0u128;
    let mut outcomes_top1: Vec<f64> = Vec::with_capacity(cases.len());
    let mut misses: Vec<(String, Vec<String>, String)> = Vec::new();

    for case in &cases {
        // Sanity: every evaluated key is in the fst index.
        debug_assert!(key_set.contains(&case.roman));
        let t0 = Instant::now();
        let suggestions: Vec<String> = engine
            .get_suggestions(&case.roman, n_suggest)
            .into_iter()
            .map(|(s, _)| s)
            .collect();
        total_latency_us += t0.elapsed().as_micros();

        let rank = suggestions
            .iter()
            .position(|s| case.targets.iter().any(|t| t == s));
        let rr = rank.map(|r| 1.0 / (r + 1) as f64).unwrap_or(0.0);
        rr_sum += rr;
        let hit1 = rank == Some(0);
        let hit5 = rank.is_some_and(|r| r < 5);
        top1_hits += hit1 as usize;
        top5_hits += hit5 as usize;
        outcomes_top1.push(hit1 as u8 as f64);

        // Keystrokes saved: shortest prefix with target visible in top-5.
        saved_sum += keystrokes_saved(&engine, case, n_suggest);

        // Grapheme CER vs best target, scored on top-1 output ("" if none).
        let top: &str = suggestions.first().map(String::as_str).unwrap_or("");
        cer_sum += 1.0 - best_target_grapheme_f1(top, &case.targets);

        if !hit5 && misses.len() < args.show_misses {
            misses.push((case.roman.clone(), suggestions, case.targets.join("|")));
        }
    }

    let n = cases.len() as f64;
    let top1 = top1_hits as f64 / n;
    let top5 = top5_hits as f64 / n;
    let mrr = rr_sum / n;
    let saved = saved_sum / n;
    let cer = cer_sum / n;
    let avg_ms = total_latency_us as f64 / n / 1000.0;
    let ci = bootstrap_ci(&outcomes_top1, args.resamples, args.seed);
    let stable = ci.1 - ci.0 < 0.05;

    // ---- Human view ----
    println!("\nIME evaluation (plain language)");
    println!(
        "Dataset : {} ({} cases)",
        args.dataset.display(),
        cases.len()
    );
    println!("------------------------------------------------------------");
    println!(
        "Out of 100 words: {:>5.1} right on first suggestion",
        top1 * 100.0
    );
    println!(
        "                  {:>5.1} visible in top-{}",
        top5 * 100.0,
        n_suggest.min(5)
    );
    println!(
        "Effort saved:     {:>5.1}% fewer keystrokes (prefix completion)",
        saved * 100.0
    );
    println!("Speed:            {avg_ms:.3} ms/query");
    println!(
        "Stable?           {} (95% CI [{:.1}%, {:.1}%]){}",
        if stable {
            "yes"
        } else {
            "no — needs more cases"
        },
        ci.0 * 100.0,
        ci.1 * 100.0,
        if duplicates_dropped > 0 {
            format!("  [{duplicates_dropped} duplicate keys]")
        } else {
            String::new()
        }
    );
    if !misses.is_empty() {
        println!("\nMisses ({} shown) — roman | got | wanted:", misses.len());
        for (roman, sug, want) in &misses {
            let got = sug.first().map(String::as_str).unwrap_or("(none)");
            println!("  {roman} | {got} | {want}");
        }
    }

    // ---- Machine view ----
    let payload = EvalJson {
        dataset: args.dataset.display().to_string(),
        cases: cases.len(),
        correct_first_time: top1,
        visible_top5: top5,
        mrr,
        keystrokes_saved: saved,
        cer_grapheme: cer,
        avg_latency_ms: avg_ms,
        top1_ci95: ci,
        duplicates_dropped,
    };
    if let Some(path) = &args.out {
        let file = File::create(path).unwrap_or_else(|e| {
            eprintln!("cannot write {}: {e}", path.display());
            std::process::exit(1);
        });
        serde_json::to_writer_pretty(file, &payload).unwrap_or_else(|e| {
            eprintln!("JSON encode failed: {e}");
            std::process::exit(1);
        });
        println!("\nWrote {}", path.display());
    }
}

/// Fraction of keystrokes saved via prefix completion.
/// Shortest strict prefix of `roman` whose top-`k` contains a target.
/// Returns 0 when only the full string hits (no saving) or never.
fn keystrokes_saved(engine: &ImeEngine, case: &EvalCase, k: usize) -> f64 {
    let chars: Vec<char> = case.roman.chars().collect();
    if chars.len() < 2 {
        return 0.0;
    }
    for len in 1..chars.len() {
        let prefix: String = chars[..len].iter().collect();
        let hit = engine
            .get_suggestions(&prefix, k)
            .into_iter()
            .any(|(s, _)| case.targets.iter().any(|t| t == &s));
        if hit {
            return 1.0 - len as f64 / chars.len() as f64;
        }
    }
    0.0
}

/// Grapheme-level similarity F1 between output and best target; CER = 1 - F1.
/// Buffers use SmallVec to avoid allocation on short words.
fn best_target_grapheme_f1(out: &str, targets: &[String]) -> f64 {
    targets
        .iter()
        .map(|t| grapheme_f1(out, t))
        .fold(0.0f64, f64::max)
}

fn grapheme_f1(a: &str, b: &str) -> f64 {
    let ga: SmallVec<[&str; 16]> = a.graphemes(true).collect();
    let gb: SmallVec<[&str; 16]> = b.graphemes(true).collect();
    if ga.is_empty() && gb.is_empty() {
        return 1.0;
    }
    if ga.is_empty() || gb.is_empty() {
        return 0.0;
    }
    let dist = levenshtein(&ga, &gb);
    let max_len = ga.len().max(gb.len()) as f64;
    1.0 - dist as f64 / max_len
}

fn levenshtein(a: &[&str], b: &[&str]) -> usize {
    let mut prev: SmallVec<[usize; 32]> = (0..=b.len()).collect();
    let mut curr: SmallVec<[usize; 32]> = smallvec::smallvec![0; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            curr[j + 1] = (prev[j] + usize::from(ca != cb))
                .min(curr[j] + 1)
                .min(prev[j + 1] + 1);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

fn bootstrap_ci(xs: &[f64], resamples: usize, seed: u64) -> (f64, f64) {
    if xs.is_empty() || resamples == 0 {
        return (0.0, 0.0);
    }
    let n = xs.len();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut means: Vec<f64> = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let mut s = 0.0;
        for _ in 0..n {
            s += xs[rng.gen_range(0..n)];
        }
        means.push(s / n as f64);
    }
    means.sort_by(|a, b| a.total_cmp(b));
    let lo = percentile_sorted(&means, 0.025);
    let hi = percentile_sorted(&means, 0.975);
    (lo, hi)
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = p * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = rank - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

fn parse_dataset(path: &PathBuf) -> Result<Vec<EvalCase>, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "json" | "jsonl" => parse_jsonl(path),
        _ => parse_tsv(path),
    }
}

fn parse_jsonl(path: &PathBuf) -> Result<Vec<EvalCase>, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);
    let mut cases = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let rec: Record =
            serde_json::from_str(trimmed).map_err(|e| format!("line {}: {e}", i + 1))?;
        let roman = rec.english.trim().to_string();
        let native = rec.native.trim().to_string();
        if roman.is_empty() || native.is_empty() {
            continue;
        }
        cases.push(EvalCase {
            roman,
            targets: vec![native],
        });
    }
    Ok(cases)
}

fn parse_tsv(path: &PathBuf) -> Result<Vec<EvalCase>, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);
    let mut cases = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line_no = i + 1;
        let line = line.map_err(|e| e.to_string())?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let (roman, targets_raw) = trimmed
            .split_once('\t')
            .ok_or_else(|| format!("line {line_no}: expected `roman<TAB>target1|target2`"))?;
        let roman = roman.trim().to_string();
        if roman.is_empty() {
            return Err(format!("line {line_no}: empty roman key"));
        }
        let targets: Vec<String> = targets_raw
            .split('|')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
        if targets.is_empty() {
            return Err(format!("line {line_no}: no targets found"));
        }
        cases.push(EvalCase { roman, targets });
    }
    Ok(cases)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levenshtein_identical_is_zero() {
        let a: SmallVec<[&str; 16]> = "नमस्ते".graphemes(true).collect();
        let b: SmallVec<[&str; 16]> = "नमस्ते".graphemes(true).collect();
        assert_eq!(levenshtein(&a, &b), 0);
        assert!((grapheme_f1("नमस्ते", "नमस्ते") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn grapheme_f1_empty_is_zero() {
        assert_eq!(grapheme_f1("", "नमस्ते"), 0.0);
        assert_eq!(grapheme_f1("नमस्ते", ""), 0.0);
    }

    #[test]
    fn percentile_endpoints() {
        let v: Vec<f64> = (0..=10).map(|x| x as f64).collect();
        assert_eq!(percentile_sorted(&v, 0.0), 0.0);
        assert_eq!(percentile_sorted(&v, 1.0), 10.0);
    }
}
