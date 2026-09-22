// File: src/bin/build/calibrate_blend.rs
//
// Calibrate the reranker's heuristic/learned blend weight (gamma) on the
// VALIDATION split and write it into the model container.
//
//   cargo run --release --bin calibrate_blend -- --model data/akshar.model \
//       [--valid data/pairs/valid.jsonl] [--out <path>] [--threads 6]
//
// The final score blends the fixed frequency heuristic with the learned
// model: (1 - gamma) * heuristic + gamma * learned (both z-scored within the
// candidate list).  The right mix differs by language -- Nepali has a corpus
// frequency list, so its heuristic is strong (best gamma ~0.3); Hindi,
// Sanskrit and Bodo have none and want the learned model (~1.0) -- so gamma
// is chosen per language, plus one value for language-blind ("auto") use.
//
// Objective: native-word top-1 (AK-Freq, AK-Uni, Dakshina) through
// `get_suggestions`, ties broken by MRR@8.  Auto maximises the mean over
// languages with no language set; each language maximises its own score
// with that language set.  Test data is never read.

use akshar_ime::core::unified::UnifiedModel;
use akshar_ime::ImeEngine;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

const GRID: [f64; 11] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
const K: usize = 8;

#[derive(Deserialize)]
struct Record {
    #[serde(rename = "english word")]
    roman: String,
    #[serde(rename = "native word")]
    native: String,
    source: String,
    #[serde(default)]
    lang: String,
}

/// (top-1 hits, reciprocal-rank sum) for one language at one gamma.
#[derive(Default, Clone, Copy)]
struct Score {
    n: usize,
    top1: usize,
    rr: f64,
}

impl Score {
    fn top1(&self) -> f64 {
        self.top1 as f64 / self.n.max(1) as f64
    }
    fn mrr(&self) -> f64 {
        self.rr / self.n.max(1) as f64
    }
}

/// Score every case at every gamma, with or without the case's language.
fn sweep(
    model: &Path,
    cases: &[Record],
    lang_aware: bool,
    threads: usize,
) -> Result<Vec<BTreeMap<String, Score>>> {
    let chunk = cases.len().div_ceil(threads).max(1);
    let parts: Vec<Vec<BTreeMap<String, Score>>> = std::thread::scope(|s| {
        let handles: Vec<_> = cases
            .chunks(chunk)
            .map(|part| {
                s.spawn(move || -> Result<Vec<BTreeMap<String, Score>>> {
                    let mut engine = ImeEngine::from_unified_file(model)
                        .map_err(|e| anyhow::anyhow!("load {}: {e}", model.display()))?;
                    let mut out = Vec::with_capacity(GRID.len());
                    for &g in &GRID {
                        let per_lang = vec![g; engine.languages().len()];
                        engine.set_blend(Some(g), per_lang);
                        let mut scores: BTreeMap<String, Score> = BTreeMap::new();
                        for c in part {
                            engine.set_language(lang_aware.then_some(c.lang.as_str()));
                            let rank = engine
                                .get_suggestions(&c.roman, K)
                                .iter()
                                .position(|(d, _)| *d == c.native);
                            let e = scores.entry(c.lang.clone()).or_default();
                            e.n += 1;
                            if let Some(r) = rank {
                                e.top1 += usize::from(r == 0);
                                e.rr += 1.0 / (r + 1) as f64;
                            }
                        }
                        out.push(scores);
                    }
                    Ok(out)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().map_err(|_| anyhow::anyhow!("worker panicked"))?)
            .collect::<Result<Vec<_>>>()
    })?;
    // Merge the per-thread tallies.
    let mut merged: Vec<BTreeMap<String, Score>> = vec![BTreeMap::new(); GRID.len()];
    for part in parts {
        for (gi, scores) in part.into_iter().enumerate() {
            for (lang, s) in scores {
                let e = merged[gi].entry(lang).or_default();
                e.n += s.n;
                e.top1 += s.top1;
                e.rr += s.rr;
            }
        }
    }
    Ok(merged)
}

/// Index of the best gamma: highest top-1, then highest MRR, then smallest gamma.
fn best(values: &[(f64, f64)]) -> usize {
    let mut bi = 0;
    for (i, v) in values.iter().enumerate() {
        let b = values[bi];
        if v.0 > b.0 + 1e-12 || ((v.0 - b.0).abs() <= 1e-12 && v.1 > b.1 + 1e-12) {
            bi = i;
        }
    }
    bi
}

fn main() -> Result<()> {
    let mut model: Option<PathBuf> = None;
    let mut valid = PathBuf::from("data/pairs/valid.jsonl");
    let mut out: Option<PathBuf> = None;
    let mut threads = 6usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().with_context(|| format!("value for {a}"));
        match a.as_str() {
            "--model" => model = Some(PathBuf::from(val()?)),
            "--valid" => valid = PathBuf::from(val()?),
            "--out" => out = Some(PathBuf::from(val()?)),
            "--threads" => threads = val()?.parse::<usize>()?.max(1),
            "-h" | "--help" => {
                println!("calibrate_blend --model p [--valid p] [--out p] [--threads 6]");
                return Ok(());
            }
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }
    let model = model.context("--model is required")?;
    let out = out.unwrap_or_else(|| model.clone());
    if std::env::var_os("AKSHAR_GAMMA").is_some() {
        anyhow::bail!("unset AKSHAR_GAMMA: it overrides the blend being calibrated");
    }

    let f = std::fs::File::open(&valid).with_context(|| format!("open {}", valid.display()))?;
    let mut cases = Vec::new();
    for line in BufReader::new(f).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let r: Record = serde_json::from_str(&line).context("bad record")?;
        if matches!(r.source.as_str(), "AK-Freq" | "AK-Uni" | "Dakshina") {
            cases.push(r);
        }
    }
    eprintln!("{} native validation cases", cases.len());

    let mut container = UnifiedModel::load(&model).map_err(|e| anyhow::anyhow!("{e}"))?;
    let blind = sweep(&model, &cases, false, threads)?;
    let aware = if container.langs.is_empty() {
        Vec::new()
    } else {
        sweep(&model, &cases, true, threads)?
    };

    // Auto: mean top-1 / MRR over languages, no language set.
    let auto: Vec<(f64, f64)> = blind
        .iter()
        .map(|by_lang| {
            let n = by_lang.len().max(1) as f64;
            (
                by_lang.values().map(Score::top1).sum::<f64>() / n,
                by_lang.values().map(Score::mrr).sum::<f64>() / n,
            )
        })
        .collect();
    let ai = best(&auto);
    println!(
        "auto: gamma={:.1}  macro top-1 {:.2}%",
        GRID[ai],
        100.0 * auto[ai].0
    );
    container.gamma_auto = Some(GRID[ai]);

    // Per language, with the language set.
    let mut gamma_lang = Vec::with_capacity(container.langs.len());
    for lang in &container.langs {
        let values: Vec<(f64, f64)> = aware
            .iter()
            .map(|by_lang| {
                by_lang
                    .get(lang)
                    .map_or((0.0, 0.0), |s| (s.top1(), s.mrr()))
            })
            .collect();
        let n = aware.first().and_then(|m| m.get(lang)).map_or(0, |s| s.n);
        let g = if n == 0 {
            GRID[ai]
        } else {
            GRID[best(&values)]
        };
        let gi = GRID.iter().position(|&x| x == g).unwrap_or(ai);
        println!(
            "{lang}: gamma={g:.1}  top-1 {:.2}%  (n={n}{})",
            100.0 * values[gi].0,
            if n == 0 {
                ", no validation data: auto"
            } else {
                ""
            }
        );
        gamma_lang.push(g);
    }
    container.gamma_lang = gamma_lang;
    container
        .save(&out)
        .map_err(|e| anyhow::anyhow!("save {}: {e}", out.display()))?;
    println!("wrote calibrated blend to {}", out.display());
    Ok(())
}
