// File: src/bin/evaluate.rs
//
// Evaluation harness for Roman-to-Devanagari transliteration.
//
// Runs the IME engine over a test set and reports:
//   * Top-1 / Top-5 / Top-10 accuracy
//   * Mean Reciprocal Rank (MRR)
//   * Bootstrap 95% confidence intervals (1000 resamples, seedable RNG)
//   * average per-query latency
//
// Two dataset formats are supported (auto-detected by extension):
//   *.json  -> Aksharantar JSONL: {"english word": ..., "native word": ...}
//   *.tsv   -> roman<TAB>target1|target2|...
//
// Usage:
//   cargo run --release --bin evaluate -- [options]
//     --dataset <path>     test set (default: data/aksharantar/nep_test.json)
//     --topk <n>           top-k for the hit metric (default: 10)
//     --suggestions <n>    suggestions to request per query (default: topk)
//     --resamples <n>      bootstrap resamples (default: 1000)
//     --seed <n>           RNG seed (default: 42)
//     --limit <n>          only evaluate first <n> cases (debug)
//     --show-misses <n>    print up to <n> missed cases (default: 0)
//     --full-case          use ImeEngine::from_file_or_new with a user dict

use akshar_ime::ImeEngine;
use clap::Parser;
use rand::Rng as _;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

/// One test case: a roman query and its accepted Devanagari target(s).
#[derive(Debug, Clone)]
struct EvalCase {
    roman: String,
    targets: Vec<String>,
}

/// Per-case outcome used by every metric.
#[derive(Debug, Clone, Copy)]
struct CaseOutcome {
    top1: bool,
    top5: bool,
    top10: bool,
    reciprocal_rank: f64,
}

/// One Aksharantar JSONL record (borrowed from the source line).
#[derive(Deserialize)]
struct Record<'a> {
    #[serde(rename = "english word")]
    english: &'a str,
    #[serde(rename = "native word")]
    native: &'a str,
}

#[derive(Debug, Parser)]
struct Args {
    /// Test set (Aksharantar JSONL or roman<TAB>target1|target2 TSV).
    #[arg(long, default_value = "data/aksharantar/test_devanagari.jsonl")]
    dataset: PathBuf,
    /// Top-k for the hit metric.
    #[arg(long, default_value_t = 10)]
    topk: usize,
    /// Suggestions to request per query (default: topk).
    #[arg(long, default_value_t = 0)]
    suggestions: usize,
    /// Bootstrap resamples (0 disables CIs).
    #[arg(long, default_value_t = 1000)]
    resamples: usize,
    /// RNG seed for bootstrap resampling.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Only evaluate first <n> cases (debug).
    #[arg(long)]
    limit: Option<usize>,
    /// Print up to <n> missed cases.
    #[arg(long, default_value_t = 0)]
    show_misses: usize,
    /// Load a user dictionary (data/user_dictionary.bin).
    #[arg(long, default_value_t = false)]
    full_case: bool,
    /// Explicit unified-model path. Without it the engine resolves
    /// data/akshar.model by its usual search order.
    #[arg(long)]
    model: Option<PathBuf>,
}

fn default_dataset() -> PathBuf {
    for cand in [
        "data/aksharantar/test_devanagari.jsonl",
        "data/aksharantar/nep_test.json",
    ] {
        if std::path::Path::new(cand).exists() {
            return PathBuf::from(cand);
        }
    }
    PathBuf::from("data/aksharantar/test_devanagari.jsonl")
}

fn main() {
    let mut args = Args::parse();
    if !args.dataset.exists() {
        args.dataset = default_dataset();
    }

    let cases = match parse_dataset(&args.dataset) {
        Ok(c) if !c.is_empty() => c,
        Ok(_) => {
            eprintln!("Dataset is empty: {}", args.dataset.display());
            std::process::exit(2);
        }
        Err(e) => {
            eprintln!("Failed to parse dataset `{}`: {e}", args.dataset.display());
            std::process::exit(1);
        }
    };
    let cases: Vec<EvalCase> = if let Some(lim) = args.limit {
        cases.into_iter().take(lim).collect()
    } else {
        cases
    };

    let engine = match (&args.model, args.full_case) {
        (Some(path), _) => ImeEngine::from_unified_file(path).unwrap_or_else(|e| {
            eprintln!("cannot load model {}: {e}", path.display());
            std::process::exit(1);
        }),
        (None, true) => ImeEngine::from_file_or_new("data/user_dictionary.bin"),
        (None, false) => ImeEngine::new(),
    };

    let n_suggest = args.suggestions.max(args.topk).max(1);

    eprintln!(
        "Evaluating {} cases (request {} suggestions, topk={}, resamples={}, seed={})",
        cases.len(),
        n_suggest,
        args.topk,
        args.resamples,
        args.seed
    );

    let mut outcomes: Vec<CaseOutcome> = Vec::with_capacity(cases.len());
    let mut misses: Vec<(EvalCase, Vec<String>)> = Vec::new();
    let mut total_latency_us = 0u128;

    for case in &cases {
        let t0 = Instant::now();
        let suggestions: Vec<String> = engine
            .get_suggestions(&case.roman, n_suggest)
            .into_iter()
            .map(|(s, _)| s)
            .collect();
        total_latency_us += t0.elapsed().as_micros();

        let outcome = score(&suggestions, &case.targets);
        if args.show_misses > 0 && !outcome.top10 && misses.len() < args.show_misses {
            misses.push((case.clone(), suggestions));
        }
        outcomes.push(outcome);
    }

    let n = outcomes.len();
    let top1 = mean(
        &outcomes
            .iter()
            .map(|o| o.top1 as u8 as f64)
            .collect::<Vec<_>>(),
    );
    let top5 = mean(
        &outcomes
            .iter()
            .map(|o| o.top5 as u8 as f64)
            .collect::<Vec<_>>(),
    );
    let top10 = mean(
        &outcomes
            .iter()
            .map(|o| o.top10 as u8 as f64)
            .collect::<Vec<_>>(),
    );
    let mrr = mean(
        &outcomes
            .iter()
            .map(|o| o.reciprocal_rank)
            .collect::<Vec<_>>(),
    );

    let avg_latency_ms = total_latency_us as f64 / n as f64 / 1000.0;

    // Bootstrap confidence intervals.
    let (top1_ci, top5_ci, top10_ci, mrr_ci) = bootstrap(&outcomes, args.resamples, args.seed);

    println!("\nDevanagari Transliteration Evaluation");
    println!("Dataset : {}", args.dataset.display());
    println!("Cases   : {n}");
    println!("Engine  : ImeEngine (generative decoder)");
    println!("------------------------------------------------------------");
    println!(
        "Top-1  accuracy : {:.2}%  CI95 [{:.2}%, {:.2}%]",
        top1 * 100.0,
        top1_ci.0 * 100.0,
        top1_ci.1 * 100.0
    );
    println!(
        "Top-5  accuracy : {:.2}%  CI95 [{:.2}%, {:.2}%]",
        top5 * 100.0,
        top5_ci.0 * 100.0,
        top5_ci.1 * 100.0
    );
    println!(
        "Top-10 accuracy : {:.2}%  CI95 [{:.2}%, {:.2}%]",
        top10 * 100.0,
        top10_ci.0 * 100.0,
        top10_ci.1 * 100.0
    );
    println!(
        "MRR             : {:.4}    CI95 [{:.4}, {:.4}]",
        mrr, mrr_ci.0, mrr_ci.1
    );
    println!("Avg latency/query : {:.3} ms", avg_latency_ms);

    if !misses.is_empty() {
        println!("\nMisses ({} shown):", misses.len());
        for (case, sug) in &misses {
            let shown: Vec<String> = sug.iter().take(args.topk).cloned().collect();
            println!(
                "  roman=`{}` targets={:?} top={:?}",
                case.roman, case.targets, shown
            );
        }
    }
}

/// Score one case against the suggestion list.
fn score(suggestions: &[String], targets: &[String]) -> CaseOutcome {
    let rank = suggestions
        .iter()
        .position(|s| targets.iter().any(|t| t == s));
    let reciprocal_rank = match rank {
        Some(r) => 1.0 / (r + 1) as f64,
        None => 0.0,
    };
    CaseOutcome {
        top1: rank.is_some_and(|r| r == 0),
        top5: rank.is_some_and(|r| r < 5),
        top10: rank.is_some_and(|r| r < 10),
        reciprocal_rank,
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// A (low, high) confidence-interval pair.
type Ci = (f64, f64);

/// Bootstrap resampling for 95% confidence intervals on all four metrics.
/// Uses `rand_chacha::ChaCha8Rng` (seedable, reproducible) instead of the
/// previous hand-rolled SplitMix64.
fn bootstrap(outcomes: &[CaseOutcome], resamples: usize, seed: u64) -> (Ci, Ci, Ci, Ci) {
    let n = outcomes.len();
    if n == 0 || resamples == 0 {
        return ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0));
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut t1 = Vec::with_capacity(resamples);
    let mut t5 = Vec::with_capacity(resamples);
    let mut t10 = Vec::with_capacity(resamples);
    let mut mrr = Vec::with_capacity(resamples);

    for _ in 0..resamples {
        let mut s1 = 0u32;
        let mut s5 = 0u32;
        let mut s10 = 0u32;
        let mut sr = 0.0f64;
        for _ in 0..n {
            let idx = rng.gen_range(0..n);
            let o = &outcomes[idx];
            s1 += o.top1 as u32;
            s5 += o.top5 as u32;
            s10 += o.top10 as u32;
            sr += o.reciprocal_rank;
        }
        t1.push(s1 as f64 / n as f64);
        t5.push(s5 as f64 / n as f64);
        t10.push(s10 as f64 / n as f64);
        mrr.push(sr / n as f64);
    }
    (
        (percentile(&t1, 0.025), percentile(&t1, 0.975)),
        (percentile(&t5, 0.025), percentile(&t5, 0.975)),
        (percentile(&t10, 0.025), percentile(&t10, 0.975)),
        (percentile(&mrr, 0.025), percentile(&mrr, 0.975)),
    )
}

/// Empirical percentile of an unsorted sample via linear interpolation.
fn percentile(sample: &[f64], p: f64) -> f64 {
    if sample.is_empty() {
        return 0.0;
    }
    let mut v = sample.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let rank = p * (v.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        return v[lo];
    }
    let frac = rank - lo as f64;
    v[lo] * (1.0 - frac) + v[hi] * frac
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
    fn score_top1_and_topk() {
        let targets = vec!["नमस्ते".to_string()];
        let sugs = vec!["गलत".to_string(), "नमस्ते".to_string()];
        let o = score(&sugs, &targets);
        assert!(!o.top1);
        assert!(o.top5);
        assert!(o.top10);
        assert_eq!(o.reciprocal_rank, 0.5);
    }

    #[test]
    fn score_miss_yields_zero_rr() {
        let targets = vec!["नमस्ते".to_string()];
        let sugs = vec!["गलत".to_string(), "अन्य".to_string()];
        let o = score(&sugs, &targets);
        assert!(!o.top1 && !o.top5 && !o.top10);
        assert_eq!(o.reciprocal_rank, 0.0);
    }

    #[test]
    fn score_accepts_multiple_targets() {
        let targets = vec!["काठमाडौं".to_string(), "काठमाण्डौ".to_string()];
        let sugs = vec!["काठमाण्डौ".to_string()];
        let o = score(&sugs, &targets);
        assert!(o.top1);
    }

    #[test]
    fn bootstrap_ci_brackets_point_estimate() {
        let outcomes: Vec<CaseOutcome> = (0..200)
            .map(|i| CaseOutcome {
                top1: i % 2 == 0,
                top5: i % 3 != 0,
                top10: true,
                reciprocal_rank: 0.5,
            })
            .collect();
        let (t1, _, _, _) = bootstrap(&outcomes, 500, 12345);
        assert!(t1.0 <= t1.1);
    }

    #[test]
    fn rng_is_deterministic() {
        let mut a = ChaCha8Rng::seed_from_u64(7);
        let mut b = ChaCha8Rng::seed_from_u64(7);
        for _ in 0..10 {
            assert_eq!(a.gen::<u64>(), b.gen::<u64>());
        }
    }

    #[test]
    fn percentile_endpoints() {
        let v: Vec<f64> = (0..=10).map(|x| x as f64).collect();
        assert_eq!(percentile(&v, 0.0), 0.0);
        assert_eq!(percentile(&v, 1.0), 10.0);
        assert!((percentile(&v, 0.5) - 5.0).abs() < 1e-9);
    }
}
