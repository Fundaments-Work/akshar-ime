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
//! Reads the vendored Nepali split (`data/aksharantar/nep_test.json`), which is
//! entirely AK-Freq, and whose records carry no `source` field -- see
//! `data/README.md`. The older `test_devanagari.jsonl` path is still accepted
//! when present, for a `data/` populated from an upstream dump.
//!
//! The sample is an even **stride** over the split, not a prefix: the AK-Freq
//! file is ordered, with the first ~2,050 cases averaging 8.7 codepoints and the
//! remainder 6.4, so a prefix sample reads ~22pp high (82.50% top-1 on the
//! first 400, against 60.35% on all 4,101). Striding brings the sample to
//! 58.75% / 76.00%, within sampling noise of the full-split figures, and costs
//! nothing.
//!
//! Thresholds are set below the measured baseline (2026-10-04, Nepali-only
//! container: top-1 60.35%, top-3 73.15% on the full 4,101-case split) with
//! room for sampling noise, so this catches regressions without failing on
//! ordinary variation. Raise them when a change genuinely improves the model.
//!
//! Skips (rather than fails) when the model or dataset is absent, so a fresh
//! clone without `data/` still passes CI.

use akshar_ime::ImeEngine;
use std::io::BufRead;
use std::path::Path;

const SAMPLE: usize = 400;

/// Measured 60.35% on the full AK-Freq split; allow for sampling noise.
const MIN_TOP1: f64 = 56.0;
/// Measured 73.15% on the full AK-Freq split (top-3 of the k=8 list, which is
/// the tightest of the three figures the eval harness reports).
const MIN_TOP5: f64 = 66.0;

/// The first split that exists, in preference order.
fn dataset() -> Option<&'static str> {
    [
        "data/aksharantar/nep_test.json",
        "data/aksharantar/test_devanagari.jsonl",
        "data/pairs/test.jsonl",
    ]
    .into_iter()
    .find(|p| Path::new(p).exists())
}

/// Every `stride`-th usable row, so the sample spans the whole ordered split.
fn load_native_pairs(path: &str, limit: usize) -> Vec<(String, String)> {
    let Ok(file) = std::fs::File::open(path) else {
        return Vec::new();
    };
    let mut all = Vec::new();
    for line in std::io::BufReader::new(file).lines() {
        let Ok(line) = line else { break };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        // The cleaned Nepali split has no `source`; an upstream dump does, and
        // AK-Freq is the native-word stratum IndicXlit reports on.
        match v["source"].as_str() {
            Some(s) if s != "AK-Freq" => continue,
            _ => {}
        }
        if let (Some(r), Some(d)) = (v["english word"].as_str(), v["native word"].as_str()) {
            all.push((r.to_string(), d.to_string()));
        }
    }
    if all.len() <= limit {
        return all;
    }
    let stride = all.len() / limit;
    all.into_iter().step_by(stride).take(limit).collect()
}

#[test]
fn ak_freq_top1_and_top5_do_not_regress() {
    if !Path::new("data/akshar.model").exists() {
        eprintln!("skipping: data/akshar.model not present");
        return;
    }
    let Some(path) = dataset() else {
        eprintln!("skipping: no Aksharantar test split under data/");
        return;
    };
    let pairs = load_native_pairs(path, SAMPLE);
    if pairs.is_empty() {
        eprintln!("skipping: {path} yielded no usable rows");
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
