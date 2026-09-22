// File: src/bin/evaluate/eval_langs.rs
//
// Per-language IME evaluation over every Devanagari language in
// data/pairs (built by `make data-prepare`).
//
// An IME suggests and the user chooses, so the numbers that describe what a
// user experiences are:
//   * top-1      — what space/enter commits with zero extra effort;
//   * in-list@k  — the correct word is somewhere in the k suggestions shown
//                  (IBus asks the engine for 8), so one selection gets it;
//   * MRR@k      — mean reciprocal rank inside that list (0 when absent):
//                  how far the user has to look on average.
// Everything is measured through `ImeEngine::get_suggestions(roman, k)`,
// the exact call the IBus and browser front ends make.
//
// Strata follow Aksharantar's sources.  "native" = AK-Freq, AK-Uni and
// Dakshina (words of the language, romanised by people); named entities are
// AK-NEI/AK-NEF/Wikidata; IndicCorp/Samanantar/Existing are mined pairs.
// Gold and prediction are both compared in NFC.
//
// Scoring is strict (one reference, exact match) -- the number comparable
// with published results.  Next to it, "lenient" also accepts the spellings
// Devanagari readers treat as the same word: with or without nukta
// (कागज़/कागज), chandrabindu or anusvara, and anusvara or the nasal
// consonant + virama before a stop (संत/सन्त).  It is reported alongside
// strict, never instead of it.
//
//   cargo run --release --bin eval_langs -- [--dataset data/pairs/test.jsonl]
//       [--model data/akshar.model] [--k 8] [--langs hin,nep] [--threads 4]
//       [--ceiling 50] [--json out.json]
//
// --ceiling N adds a diagnostic column: is the gold anywhere in the first N
// suggestions?  Misses inside the ceiling are ranking errors; misses outside
// it are generation errors (the engine never produces the word).
//
// --lang-aware tells the engine each case's language (what a user does by
// picking e.g. "Devanagari (Akshar) — Hindi"); without it every case is
// ranked language-blind ("auto").

use akshar_ime::ImeEngine;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use unicode_normalization::UnicodeNormalization;

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

/// Published IndicXlit AK-Freq top-1 (Madhani et al., Aksharantar, Table 13):
/// (without rerank, with IndicCorp word-unigram rerank).  Reference only.
fn indicxlit_ak_freq(lang: &str) -> Option<(f64, f64)> {
    Some(match lang {
        "hin" => (58.61, 67.94),
        "mar" => (74.69, 85.47),
        "nep" => (80.17, 86.62),
        "san" => (81.56, 90.07),
        "kok" => (65.38, 76.29),
        "mai" => (78.65, 87.57),
        "brx" => (74.80, 78.42),
        _ => return None,
    })
}

/// Canonical form under the orthographic equivalences lenient scoring accepts.
fn lenient_key(word: &str) -> String {
    const NUKTA: char = '\u{093C}';
    const VIRAMA: char = '\u{094D}';
    const ANUSVARA: char = '\u{0902}';
    const CHANDRABINDU: char = '\u{0901}';
    let is_nasal = |c: char| matches!(c, 'ङ' | 'ञ' | 'ण' | 'न' | 'म');
    // Stops (क..म, without the nasals themselves): where a nasal cluster and
    // an anusvara are interchangeable spellings.
    let is_stop = |c: char| ('\u{0915}'..='\u{092E}').contains(&c) && !is_nasal(c);
    let chars: Vec<char> = word.nfd().filter(|&c| c != NUKTA).collect();
    let mut out = String::with_capacity(word.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if is_nasal(c)
            && chars.get(i + 1) == Some(&VIRAMA)
            && chars.get(i + 2).is_some_and(|&n| is_stop(n))
        {
            out.push(ANUSVARA);
            i += 2;
            continue;
        }
        out.push(if c == CHANDRABINDU { ANUSVARA } else { c });
        i += 1;
    }
    out.nfc().collect()
}

fn stratum(source: &str) -> &'static str {
    match source {
        "AK-Freq" | "AK-Uni" | "Dakshina" => "native",
        "AK-NEI" | "AK-NEF" | "Wikidata" => "entity",
        _ => "mined",
    }
}

/// Per case: strict rank within k, reachable within the ceiling, lenient rank.
type Outcome = (Option<usize>, bool, Option<usize>);

#[derive(Default, Clone, Copy)]
struct Stats {
    n: usize,
    top1: usize,
    top3: usize,
    in_list: usize,
    rr: f64,
    reachable: usize,
    lenient_top1: usize,
    lenient_in_list: usize,
}

impl Stats {
    fn add(&mut self, (rank, reachable, lenient): Outcome) {
        self.n += 1;
        self.reachable += usize::from(reachable);
        if let Some(r) = lenient {
            self.lenient_top1 += usize::from(r == 1);
            self.lenient_in_list += 1;
        }
        if let Some(r) = rank {
            self.top1 += usize::from(r == 1);
            self.top3 += usize::from(r <= 3);
            self.in_list += 1;
            self.rr += 1.0 / r as f64;
        }
    }
    fn merge(&mut self, o: &Stats) {
        self.n += o.n;
        self.top1 += o.top1;
        self.top3 += o.top3;
        self.in_list += o.in_list;
        self.rr += o.rr;
        self.reachable += o.reachable;
        self.lenient_top1 += o.lenient_top1;
        self.lenient_in_list += o.lenient_in_list;
    }
    fn pct(&self, x: usize) -> f64 {
        100.0 * x as f64 / self.n.max(1) as f64
    }
    fn mrr(&self) -> f64 {
        self.rr / self.n.max(1) as f64
    }
    fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "n": self.n, "top1": self.pct(self.top1), "top3": self.pct(self.top3),
            "in_list": self.pct(self.in_list), "mrr": self.mrr(),
            "reachable": self.pct(self.reachable),
            "lenient_top1": self.pct(self.lenient_top1),
            "lenient_in_list": self.pct(self.lenient_in_list),
        })
    }
}

fn load_engine(model: &Option<PathBuf>) -> Result<ImeEngine> {
    Ok(match model {
        Some(p) => ImeEngine::from_unified_file(p)
            .map_err(|e| anyhow::anyhow!("load {}: {e}", p.display()))?,
        None => ImeEngine::new(),
    })
}

fn main() -> Result<()> {
    let mut dataset = PathBuf::from("data/pairs/test.jsonl");
    let mut model: Option<PathBuf> = None;
    let mut k: usize = 8;
    let mut langs: Option<Vec<String>> = None;
    let mut threads: usize = 4;
    let mut json_out: Option<PathBuf> = None;
    let mut ceiling: Option<usize> = None;
    let mut lang_aware = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().with_context(|| format!("value for {a}"));
        match a.as_str() {
            "--dataset" => dataset = PathBuf::from(val()?),
            "--model" => model = Some(PathBuf::from(val()?)),
            "--k" => k = val()?.parse()?,
            "--langs" => langs = Some(val()?.split(',').map(str::to_string).collect()),
            "--threads" => threads = val()?.parse::<usize>()?.max(1),
            "--json" => json_out = Some(PathBuf::from(val()?)),
            "--ceiling" => ceiling = Some(val()?.parse()?),
            "--lang-aware" => lang_aware = true,
            "-h" | "--help" => {
                println!("eval_langs [--dataset p] [--model p] [--k 8] [--langs a,b] [--threads 4] [--ceiling n] [--lang-aware] [--json p]");
                return Ok(());
            }
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }

    let f = std::fs::File::open(&dataset).with_context(|| format!("open {}", dataset.display()))?;
    let mut cases: Vec<Record> = Vec::new();
    for line in BufReader::new(f).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let r: Record = serde_json::from_str(&line).context("bad record")?;
        if langs.as_ref().is_some_and(|l| !l.contains(&r.lang)) {
            continue;
        }
        cases.push(r);
    }
    eprintln!(
        "{} cases from {} ({} threads)",
        cases.len(),
        dataset.display(),
        threads
    );

    // One engine per thread: the suggestion cache makes ImeEngine !Sync.
    let chunk = cases.len().div_ceil(threads).max(1);
    let per_case: Vec<Outcome> = std::thread::scope(|s| -> Result<Vec<Outcome>> {
        let handles: Vec<_> = cases
            .chunks(chunk)
            .map(|part| {
                let model = model.clone();
                s.spawn(move || -> Result<Vec<Outcome>> {
                    let mut engine = load_engine(&model)?;
                    Ok(part
                        .iter()
                        .map(|c| {
                            if lang_aware {
                                engine.set_language(Some(&c.lang));
                            }
                            let gold: String = c.native.nfc().collect();
                            let roman = c.roman.to_ascii_lowercase();
                            let is_gold = |d: &String| d.nfc().eq(gold.chars());
                            let list = engine.get_suggestions(&roman, k);
                            let rank = list.iter().position(|(d, _)| is_gold(d)).map(|p| p + 1);
                            let gold_key = lenient_key(&gold);
                            let lenient = list
                                .iter()
                                .position(|(d, _)| lenient_key(d) == gold_key)
                                .map(|p| p + 1);
                            let reachable = rank.is_some()
                                || ceiling.is_some_and(|n| {
                                    engine
                                        .get_suggestions(&roman, n)
                                        .iter()
                                        .any(|(d, _)| is_gold(d))
                                });
                            (rank, reachable, lenient)
                        })
                        .collect())
                })
            })
            .collect();
        let mut all: Vec<Outcome> = Vec::with_capacity(cases.len());
        for h in handles {
            all.extend(h.join().map_err(|_| anyhow::anyhow!("worker panicked"))??);
        }
        Ok(all)
    })?;

    let mut by_source: BTreeMap<(String, String), Stats> = BTreeMap::new();
    let mut by_stratum: BTreeMap<(String, &str), Stats> = BTreeMap::new();
    for (c, outcome) in cases.iter().zip(&per_case) {
        by_source
            .entry((c.lang.clone(), c.source.clone()))
            .or_default()
            .add(*outcome);
        by_stratum
            .entry((c.lang.clone(), stratum(&c.source)))
            .or_default()
            .add(*outcome);
    }
    let ceiling_col = |s: &Stats| match ceiling {
        Some(_) => format!(" {:>8.2}%", s.pct(s.reachable)),
        None => String::new(),
    };
    let ceiling_head = ceiling
        .map(|n| format!(" {:>9}", format!("in@{n}")))
        .unwrap_or_default();

    println!("\nPer language and source (k = {k})");
    println!(
        "{:<5} {:<11} {:>6} {:>7} {:>7} {:>9} {:>7}{ceiling_head}",
        "lang", "source", "n", "top1", "top3", "in-list", "MRR"
    );
    for ((lang, src), s) in &by_source {
        println!(
            "{lang:<5} {src:<11} {:>6} {:>6.2}% {:>6.2}% {:>8.2}% {:>7.3}{}",
            s.n,
            s.pct(s.top1),
            s.pct(s.top3),
            s.pct(s.in_list),
            s.mrr(),
            ceiling_col(s)
        );
    }

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let mut strata_json = serde_json::Map::new();
    let mut lang_json = serde_json::Map::new();
    for lang in by_source
        .keys()
        .map(|(l, _)| l.clone())
        .collect::<std::collections::BTreeSet<_>>()
    {
        let mut strata = serde_json::Map::new();
        for ((l2, st2), s2) in &by_stratum {
            if *l2 == lang {
                strata.insert(st2.to_string(), s2.json());
            }
        }
        lang_json.insert(lang, serde_json::Value::Object(strata));
    }

    // Native words (AK-Freq + AK-Uni + Dakshina) and named entities (AK-NEI +
    // AK-NEF + Wikidata) both get a macro/pooled summary: entity accuracy is
    // a real, separate number, not a footnote, and burying it made a stale
    // headline figure easy to miss when the pipeline changed under it.
    for (label, strat, show_reference) in [
        ("Native words", "native", true),
        ("Named entities", "entity", false),
    ] {
        println!("\n{label} by language");
        println!(
            "{:<5} {:>6} {:>7} {:>9} {:>7}{ceiling_head} {:>9} {:>9}{}",
            "lang",
            "n",
            "top1",
            "in-list",
            "MRR",
            "len-top1",
            "len-list",
            if show_reference {
                "   IndicXlit AK-Freq top-1 (plain / +word LM)"
            } else {
                ""
            }
        );
        let mut macro_top1 = Vec::new();
        let mut macro_list = Vec::new();
        let mut pooled = Stats::default();
        for ((lang, s2), s) in &by_stratum {
            if s2 != &strat {
                continue;
            }
            let reference = if show_reference {
                let r = indicxlit_ak_freq(lang)
                    .map(|(a, b)| format!("{a:.1} / {b:.1}"))
                    .unwrap_or_else(|| "—".into());
                format!("   {r}")
            } else {
                String::new()
            };
            println!(
                "{lang:<5} {:>6} {:>6.2}% {:>8.2}% {:>7.3}{} {:>8.2}% {:>8.2}%{reference}",
                s.n,
                s.pct(s.top1),
                s.pct(s.in_list),
                s.mrr(),
                ceiling_col(s),
                s.pct(s.lenient_top1),
                s.pct(s.lenient_in_list)
            );
            macro_top1.push(s.pct(s.top1));
            macro_list.push(s.pct(s.in_list));
            pooled.merge(s);
        }
        println!(
            "macro (mean over {} languages): top1 {:.2}%  in-list {:.2}%   pooled n={}: top1 {:.2}%  in-list {:.2}%  MRR {:.3}  | lenient top1 {:.2}%  in-list {:.2}%",
            macro_top1.len(),
            mean(&macro_top1),
            mean(&macro_list),
            pooled.n,
            pooled.pct(pooled.top1),
            pooled.pct(pooled.in_list),
            pooled.mrr(),
            pooled.pct(pooled.lenient_top1),
            pooled.pct(pooled.lenient_in_list)
        );
        strata_json.insert(
            strat.to_string(),
            serde_json::json!({
                "macro": {"top1": mean(&macro_top1), "in_list": mean(&macro_list)},
                "pooled": pooled.json(),
            }),
        );
    }

    if let Some(p) = json_out {
        let doc = serde_json::json!({
            "dataset": dataset.display().to_string(),
            "k": k,
            "languages": lang_json,
            "native_macro": strata_json["native"]["macro"],
            "native_pooled": strata_json["native"]["pooled"],
            "strata": strata_json,
        });
        std::fs::write(&p, serde_json::to_string_pretty(&doc)?)?;
        eprintln!("wrote {}", p.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::lenient_key;

    #[test]
    fn lenient_key_accepts_only_standard_equivalences() {
        // nukta
        assert_eq!(lenient_key("कागज़ों"), lenient_key("कागजों"));
        // chandrabindu ~ anusvara
        assert_eq!(lenient_key("माँ"), lenient_key("मां"));
        // nasal cluster before a stop ~ anusvara
        assert_eq!(lenient_key("सन्त"), lenient_key("संत"));
        assert_eq!(lenient_key("साम्प्रदायिकता"), lenient_key("सांप्रदायिकता"));
        // ... but not before a semivowel (अन्य is not अंय), and vowels differ
        assert_ne!(lenient_key("अन्य"), lenient_key("अंय"));
        assert_ne!(lenient_key("दिन"), lenient_key("दीन"));
    }
}
