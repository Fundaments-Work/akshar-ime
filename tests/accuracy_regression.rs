//! Accuracy regression guard.
//!
//! A 30.8-point top-1 regression once shipped through a fully green unit-test
//! suite (the corpus-SymSpell candidate source scored above the decoder band,
//! displacing the reranker's top-1 on every query that produced a fuzzy hit).
//! Unit tests cannot catch that class of defect: every component was individually
//! correct and it was their *composition* that was wrong.  This test measures
//! end-to-end accuracy on a fixed sample of the held-out Aksharantar test
//! split and fails if it drops.
//!
//! Thresholds are set below the measured baseline (2026-09-06: top-1 81.74%,
//! top-5 92.27% on the full 2,108-case AK-Freq split) with room for sampling
//! noise, so this catches regressions without failing on ordinary variation.
//! Raise them when a change genuinely improves the model.
//!
//! Skips (rather than fails) when the model or dataset is absent, so a fresh
//! clone without `data/` still passes CI.

use akshar_ime::ImeEngine;
use std::io::BufRead;
use std::path::Path;

const DATASET: &str = "data/aksharantar/test_devanagari.jsonl";
const SAMPLE: usize = 400;

/// Measured 81.74% on the full AK-Freq split; allow for sampling noise.
const MIN_TOP1: f64 = 76.0;
/// Measured 92.27% on the full AK-Freq split.
const MIN_TOP5: f64 = 88.0;

fn load_native_pairs(limit: usize) -> Vec<(String, String)> {
    let Ok(file) = std::fs::File::open(DATASET) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(limit);
    for line in std::io::BufReader::new(file).lines() {
        let Ok(line) = line else { break };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        // AK-Freq is the native-word split IndicXlit reports 80.25% top-1 on.
        if v["source"].as_str() != Some("AK-Freq") {
            continue;
        }
        if let (Some(r), Some(d)) = (v["english word"].as_str(), v["native word"].as_str()) {
            out.push((r.to_string(), d.to_string()));
        }
        if out.len() >= limit {
            break;
        }
    }
    out
}

#[test]
fn ak_freq_top1_and_top5_do_not_regress() {
    if !Path::new("data/akshar.model").exists() {
        eprintln!("skipping: data/akshar.model not present");
        return;
    }
    let pairs = load_native_pairs(SAMPLE);
    if pairs.is_empty() {
        eprintln!("skipping: {DATASET} not present");
        return;
    }

    let engine = ImeEngine::new();
    let (mut top1, mut top5) = (0usize, 0usize);
    for (roman, gold) in &pairs {
        let suggestions = engine.get_suggestions(roman, 5);
        if suggestions.first().is_some_and(|(s, _)| s == gold) {
            top1 += 1;
        }
        if suggestions.iter().any(|(s, _)| s == gold) {
            top5 += 1;
        }
    }

    let n = pairs.len() as f64;
    let p1 = top1 as f64 / n * 100.0;
    let p5 = top5 as f64 / n * 100.0;
    println!(
        "AK-Freq sample n={}: top-1 {:.2}%  top-5 {:.2}%",
        pairs.len(),
        p1,
        p5
    );

    assert!(
        p1 >= MIN_TOP1,
        "AK-Freq top-1 regressed to {p1:.2}% (floor {MIN_TOP1}%, n={}). \
         A candidate source is most likely outranking the decoder+reranker; \
         check the score bands in engine.rs.",
        pairs.len()
    );
    assert!(
        p5 >= MIN_TOP5,
        "AK-Freq top-5 regressed to {p5:.2}% (floor {MIN_TOP5}%, n={})",
        pairs.len()
    );
}
