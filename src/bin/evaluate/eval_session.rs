// File: src/bin/evaluate/eval_session.rs
//
// eval_langs measures one cold call to `get_suggestions` per test pair: no
// case ever influences another.  That is the right number for a
// reproducible, SOTA-comparable floor, but it is not what a person
// experiences: the engine also calls `user_confirms` on every pick (a
// learned word jumps to the user-trie score band, above the decoder --
// see AGENTS.md), and real typing repeats the same common words constantly.
// A deduplicated test set cannot show that by construction.
//
// This simulates one: draw romanised words for each language with
// replacement, weighted by how often the word actually occurs in that
// language's running text (the model's own lexicon counts), replay them in
// order through a single engine, and -- whenever the gold word is not
// already top-1 -- call `user_confirms` the way the IBus/browser front end
// does when the user picks it.  Reports accuracy split by how many times
// the *session* has already seen that exact romanisation: cold (first
// time) vs. warm (a repeat) -- plus the blended, volume-weighted number,
// which is the one closest to a real session's felt accuracy.
//
//   cargo run --release --bin eval_session -- \
//       [--dataset data/pairs/test.jsonl] [--model data/akshar.model] \
//       [--k 8] [--langs hin,nep] [--draws 20000] [--seed 7] [--lang-aware]

use akshar_ime::core::lexicon::Lexicon;
use akshar_ime::core::unified::UnifiedModel;
use akshar_ime::ImeEngine;
use anyhow::{Context, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use unicode_normalization::UnicodeNormalization;

#[derive(Deserialize, Clone)]
struct Record {
    #[serde(rename = "english word")]
    roman: String,
    #[serde(rename = "native word")]
    native: String,
    #[serde(default)]
    lang: String,
}

/// Encounter buckets within a simulated session: how many times this exact
/// romanisation has already been drawn before this one.
#[derive(Default, Clone, Copy)]
struct Bucket {
    n: usize,
    top1: usize,
}
impl Bucket {
    fn pct(&self) -> f64 {
        100.0 * self.top1 as f64 / self.n.max(1) as f64
    }
}

#[derive(Default, Clone, Copy)]
struct LangResult {
    cold: Bucket,
    warm1: Bucket, // 2nd encounter
    warm2: Bucket, // 3rd-5th encounter
    warm3: Bucket, // 6th+ encounter
    overall: Bucket,
}

fn floor_weight() -> f64 {
    0.05 // a word absent from the lexicon still gets typed occasionally
}

fn main() -> Result<()> {
    let mut dataset = PathBuf::from("data/pairs/test.jsonl");
    let mut model_path = PathBuf::from("data/akshar.model");
    let mut k: usize = 8;
    let mut langs: Option<Vec<String>> = None;
    let mut draws: usize = 20_000;
    let mut seed: u64 = 7;
    let mut lang_aware = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().with_context(|| format!("value for {a}"));
        match a.as_str() {
            "--dataset" => dataset = PathBuf::from(val()?),
            "--model" => model_path = PathBuf::from(val()?),
            "--k" => k = val()?.parse()?,
            "--langs" => langs = Some(val()?.split(',').map(str::to_string).collect()),
            "--draws" => draws = val()?.parse()?,
            "--seed" => seed = val()?.parse()?,
            "--lang-aware" => lang_aware = true,
            "-h" | "--help" => {
                println!(
                    "eval_session [--dataset p] [--model p] [--k 8] [--langs a,b] \
                     [--draws 20000] [--seed 7] [--lang-aware]"
                );
                return Ok(());
            }
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }

    let unified = UnifiedModel::load(&model_path)
        .map_err(|e| anyhow::anyhow!("load {}: {e}", model_path.display()))?;
    let lexicon = unified
        .lexicon
        .clone()
        .context("model has no lexicon (v8+ only)")
        .and_then(|d| Lexicon::from_data(d).map_err(|e| anyhow::anyhow!("{e}")))?;

    let f = std::fs::File::open(&dataset).with_context(|| format!("open {}", dataset.display()))?;
    let mut by_lang: BTreeMap<String, Vec<Record>> = BTreeMap::new();
    for line in BufReader::new(f).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let r: Record = serde_json::from_str(&line).context("bad record")?;
        if langs.as_ref().is_some_and(|l| !l.contains(&r.lang)) {
            continue;
        }
        by_lang.entry(r.lang.clone()).or_default().push(r);
    }
    eprintln!(
        "{} languages, {draws} draws each, seed {seed}",
        by_lang.len()
    );

    let lang_order: Vec<String> = by_lang.keys().cloned().collect();
    let results: Vec<(String, LangResult)> = std::thread::scope(|s| -> Result<_> {
        let handles: Vec<_> = lang_order
            .iter()
            .enumerate()
            .map(|(li, lang)| {
                let cases = &by_lang[lang];
                let model_path = &model_path;
                let lexicon = &lexicon;
                s.spawn(move || -> Result<(String, LangResult)> {
                    let mut engine = ImeEngine::from_unified_file(model_path)
                        .map_err(|e| anyhow::anyhow!("load: {e}"))?;
                    if lang_aware {
                        engine.set_language(Some(lang));
                    }
                    let lang_idx = lexicon.lang_index(lang);
                    let weights: Vec<f64> = cases
                        .iter()
                        .map(|c| lexicon.count(&c.native, lang_idx).max(floor_weight()))
                        .collect();
                    let mut cum = Vec::with_capacity(weights.len());
                    let mut acc = 0.0;
                    for w in &weights {
                        acc += w;
                        cum.push(acc);
                    }
                    let total = acc;

                    let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(li as u64));
                    let mut seen: HashMap<String, u32> = HashMap::new();
                    let mut result = LangResult::default();
                    for _ in 0..draws {
                        let x = rng.gen_range(0.0..total);
                        let idx = cum.partition_point(|&c| c < x).min(cases.len() - 1);
                        let case = &cases[idx];
                        let gold: String = case.native.nfc().collect();
                        let roman = case.roman.to_ascii_lowercase();

                        let times_before = *seen.get(&roman).unwrap_or(&0);
                        seen.insert(roman.clone(), times_before + 1);

                        let list = engine.get_suggestions(&roman, k);
                        let is_gold = |d: &String| d.nfc().eq(gold.chars());
                        let top1 = list.first().is_some_and(|(d, _)| is_gold(d));

                        let bucket = match times_before {
                            0 => &mut result.cold,
                            1 => &mut result.warm1,
                            2..=4 => &mut result.warm2,
                            _ => &mut result.warm3,
                        };
                        bucket.n += 1;
                        bucket.top1 += usize::from(top1);
                        result.overall.n += 1;
                        result.overall.top1 += usize::from(top1);

                        if !top1 {
                            // The user picks the right word one way or another
                            // (from the list, or by typing/pasting it outright)
                            // and the engine learns it, exactly as `user_confirms`
                            // is wired to the front ends.
                            engine.user_confirms(&roman, &gold);
                        }
                    }
                    Ok((lang.clone(), result))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().map_err(|_| anyhow::anyhow!("worker panicked"))?)
            .collect()
    })?;

    println!(
        "\n{:<5} {:>9} {:>9} {:>9} {:>9} {:>9}   {:>9}",
        "lang", "cold", "2nd", "3rd-5th", "6th+", "overall", "n"
    );
    let mut macro_cold = Vec::new();
    let mut macro_overall = Vec::new();
    let mut pooled = LangResult::default();
    for (lang, r) in &results {
        println!(
            "{lang:<5} {:>8.2}% {:>8.2}% {:>8.2}% {:>8.2}% {:>8.2}%   {:>9}",
            r.cold.pct(),
            r.warm1.pct(),
            r.warm2.pct(),
            r.warm3.pct(),
            r.overall.pct(),
            r.overall.n
        );
        macro_cold.push(r.cold.pct());
        macro_overall.push(r.overall.pct());
        pooled.cold.n += r.cold.n;
        pooled.cold.top1 += r.cold.top1;
        pooled.warm1.n += r.warm1.n;
        pooled.warm1.top1 += r.warm1.top1;
        pooled.warm2.n += r.warm2.n;
        pooled.warm2.top1 += r.warm2.top1;
        pooled.warm3.n += r.warm3.n;
        pooled.warm3.top1 += r.warm3.top1;
        pooled.overall.n += r.overall.n;
        pooled.overall.top1 += r.overall.top1;
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    println!(
        "\nmacro: cold {:.2}%  session {:.2}%   pooled n={}: cold {:.2}%  2nd {:.2}%  3rd-5th {:.2}%  6th+ {:.2}%  session {:.2}%",
        mean(&macro_cold),
        mean(&macro_overall),
        pooled.overall.n,
        pooled.cold.pct(),
        pooled.warm1.pct(),
        pooled.warm2.pct(),
        pooled.warm3.pct(),
        pooled.overall.pct(),
    );
    println!(
        "\n\"session\" is volume-weighted by real corpus frequency, so it is what typing actually feels like;\n\"cold\" alone is what eval_langs reports (comparable across models/papers, but a worst case)."
    );
    Ok(())
}
