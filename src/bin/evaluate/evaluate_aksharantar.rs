// File: src/bin/evaluate_aksharantar.rs
//
// Product-level benchmark: the IME engine (generative decoder + lexicon +
// learning) against the held-out Aksharantar test split (4,101 pairs).
//
// The split mirrors the Aksharantar paper:
//   native words     = AK-Freq source   (IndicXlit top-1: 80.25)
//   named entities   = AK-NEF + AK-NEI  (IndicXlit top-1: 52.67)
//
// Usage:
//   cargo run --release --bin evaluate_aksharantar -- [options]
//     --dataset <path>   JSONL test split (default: data/aksharantar/nep_test.json)
//     --topk <n>         top-k for the hit metric (default: 5)
//     --show-misses <n>  number of misses to print (default: 15)

use akshar_ime::ImeEngine;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Deserialize)]
struct Record<'a> {
    #[serde(rename = "english word")]
    roman: &'a str,
    #[serde(rename = "native word")]
    target: &'a str,
    /// Absent in the cleaned vendored splits (data/README.md); required
    /// upstream.  Optional so a missing field degrades to one unlabelled
    /// bucket instead of failing every record -- previously this made the
    /// harness silently evaluate 0 cases and still exit 0.
    #[serde(default)]
    source: Option<&'a str>,
}

#[derive(Debug, Clone)]
struct EvalCase {
    roman: String,
    target: String,
    source: String,
}

#[derive(Debug, Clone, Default)]
struct BucketStats {
    total: usize,
    top1: usize,
    topk: usize,
}

impl BucketStats {
    fn add(&mut self, top1_hit: bool, topk_hit: bool) {
        self.total += 1;
        if top1_hit {
            self.top1 += 1;
        }
        if topk_hit {
            self.topk += 1;
        }
    }
}

struct Report {
    name: &'static str,
    stats: HashMap<String, BucketStats>,
    misses: Vec<(String, String, Vec<String>)>,
}

impl Report {
    fn print(&self, topk: usize, show_misses: usize) {
        let total = self.stats.values().map(|s| s.total).sum::<usize>();
        println!("\n=== {} ===", self.name);
        if total == 0 {
            println!("  (no cases)");
            return;
        }
        for (bucket, s) in &self.stats {
            if s.total == 0 {
                continue;
            }
            let t1 = s.top1 as f64 / s.total as f64 * 100.0;
            let tk = s.topk as f64 / s.total as f64 * 100.0;
            println!(
                "  {:<18} top1={:>5.2}%  top{topk}={:>5.2}%  (top1 {}/{}, top5 {}/{})",
                bucket, t1, tk, s.top1, s.total, s.topk, s.total
            );
        }
        if show_misses > 0 {
            println!("    misses (first {show_misses}):");
            for (roman, target, top) in self.misses.iter().take(show_misses) {
                println!("      roman=`{roman}` target=`{target}` top={top:?}");
            }
        }
    }
}

fn main() {
    let mut dataset_path = if std::path::Path::new("data/aksharantar/nep_test.json").exists() {
        "data/aksharantar/nep_test.json".to_string()
    } else if std::path::Path::new("data/aksharantar/test_devanagari.jsonl").exists() {
        "data/aksharantar/test_devanagari.jsonl".to_string()
    } else {
        "data/aksharantar/nep_test.json".to_string()
    };
    let mut topk = 5usize;
    let mut show_misses = 15usize;
    let mut model_path: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dataset" => dataset_path = next_value(&arg, args.next()),
            "--model" => model_path = Some(next_value(&arg, args.next())),
            "--topk" => {
                topk = next_value(&arg, args.next())
                    .parse::<usize>()
                    .expect("--topk <n>")
                    .max(1)
            }
            "--show-misses" => {
                show_misses = next_value(&arg, args.next())
                    .parse()
                    .expect("--show-misses <n>")
            }
            "--help" | "-h" => {
                print_help();
                return;
            }
            other => {
                eprintln!("Unknown argument: {other}");
                print_help();
                std::process::exit(2);
            }
        }
    }

    let cases = load_cases(&dataset_path);
    // A benchmark that scores nothing and exits 0 is worse than one that fails:
    // it reports success in CI while measuring nothing.  Every record failing to
    // deserialise used to land here silently.
    assert!(
        !cases.is_empty(),
        "no usable cases in {dataset_path} (each line needs \"english word\" and \
         \"native word\")"
    );

    let engine = if let Some(ref mp) = model_path {
        ImeEngine::from_unified_file(std::path::Path::new(mp)).expect("load model from --model")
    } else {
        ImeEngine::new()
    };

    let mut report = Report {
        name: "ImeEngine.get_suggestions",
        stats: HashMap::new(),
        misses: Vec::new(),
    };
    for case in &cases {
        let suggestions = engine.get_suggestions(&case.roman, topk.max(8));
        let top: Vec<String> = suggestions.into_iter().map(|(d, _)| d).collect();
        report.stats.entry(case.source.clone()).or_default().add(
            top.first().is_some_and(|t| t == &case.target),
            top.iter().take(topk).any(|t| t == &case.target),
        );
        if !top.iter().take(topk).any(|t| t == &case.target) {
            report
                .misses
                .push((case.roman.clone(), case.target.clone(), top));
        }
    }

    println!("Aksharantar test split: {} cases", cases.len());
    println!("IndicXlit (neural, top-1) reference: native=80.25%, named-entities=52.67%");
    report.print(topk, show_misses);
}

fn load_cases(path: &str) -> Vec<EvalCase> {
    let file = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let reader = BufReader::new(file);
    let mut cases = Vec::new();
    for line in reader.lines().map_while(Result::ok) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let rec: Record = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let roman = rec.roman.trim().to_ascii_lowercase();
        let target = rec.target.trim().to_string();
        if roman.is_empty() || target.is_empty() {
            continue;
        }
        cases.push(EvalCase {
            roman,
            target,
            source: rec.source.unwrap_or("unlabelled").to_string(),
        });
    }
    cases
}

fn next_value(flag: &str, val: Option<String>) -> String {
    val.unwrap_or_else(|| {
        eprintln!("Missing value for {flag}");
        std::process::exit(2);
    })
}

fn print_help() {
    println!("Usage: cargo run --release --bin evaluate_aksharantar -- [options]");
    println!("  --dataset <path>    JSONL test split (default: data/aksharantar/nep_test.json)");
    println!("  --topk <n>          top-k for hit metric (default: 5)");
    println!("  --show-misses <n>   misses to print per baseline (default: 15)");
    println!("  -h, --help          show help");
}
