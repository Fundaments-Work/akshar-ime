// File: src/bin/analyze_errors.rs
//
// E0 error-analysis harness: where does the engine lose accuracy?
//
// Reports, per source bucket (AK-Freq = native words, AK-NEF/AK-NEI = named
// entities) over the Aksharantar test split:
//   1. Oracle top-k curves for the decoder and the full engine
//      (gold-in-first-k for k = 1,2,3,5,8,10,20,50) -> splits RANKING error
//      (gold generated but not ranked first) from COVERAGE error (gold never
//      generated).
//   2. Engine-vs-decoder top-1 disagreement: when they differ, who is right?
//      Quantifies fusion-layer damage.
//   3. Strict vs multi-reference top-1 (any native observed for the same roman
//      in the training split counts as correct) -> ambiguity ceiling.
//   4. Error taxonomy of top-1 misses -- reported for BOTH the decoder and
//      the full engine (the shipped 81.40% chain): matra-only (vowel length /
//      nasal), halant-only (conjunct), or substantive (coverage).
//   5. Character error rate (CER) of the top-1 candidate.
//   6. W0 collision bound: the accuracy of a system with perfect generation
//      and a perfect corpus-frequency prior.  For each test roman R, A(R) is
//      every native ever paired with R across train + valid + test; the bound
//      counts the cases where the gold IS the most frequent member of A(R).
//      No string-only system with a unigram prior can beat it.
//
// Usage:
//   cargo run --release --bin analyze_errors -- [options]
//     --test <path>   test split (default: test_devanagari.jsonl)
//     --train <path>  train split for multi-reference sets + collision bound
//     --valid <path>  valid split, also folded into the collision bound
//     --beam <n>      decoder beam width (default 256, the benchmark config)
//     --lm-weight <f> decoder LM weight (default 0.85)
//     --vocab-weight <f>  frequency rescoring weight (default 0.75; 0 = off)

use akshar_ime::core::decoder::{DecoderConfig, ModelDecoder};
use akshar_ime::ImeEngine;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Deserialize)]
struct Record<'a> {
    #[serde(rename = "english word")]
    roman: &'a str,
    #[serde(rename = "native word")]
    target: &'a str,
    source: &'a str,
}

struct Case {
    roman: String,
    target: String,
    source: String,
    refs: HashSet<String>,
}

const KS: [usize; 8] = [1, 2, 3, 5, 8, 10, 20, 50];

fn main() {
    let mut test_path = "data/aksharantar/test_devanagari.jsonl".to_string();
    let mut train_path = "data/aksharantar/train_devanagari.jsonl".to_string();
    let mut valid_path = "data/aksharantar/valid_devanagari.jsonl".to_string();
    let mut model_path = "data/akshar.model".to_string();
    let mut beam = 256usize;
    let mut lm_weight = 0.85f64;
    let mut vocab_weight = 0.75f64;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--test" | "--dataset" => test_path = args.next().expect("value"),
            "--model" => model_path = args.next().expect("value"),
            "--train" => train_path = args.next().expect("value"),
            "--valid" => valid_path = args.next().expect("value"),
            "--beam" => beam = args.next().expect("value").parse().expect("--beam <n>"),
            "--lm-weight" => {
                lm_weight = args
                    .next()
                    .expect("value")
                    .parse()
                    .expect("--lm-weight <f>")
            }
            "--vocab-weight" => {
                vocab_weight = args
                    .next()
                    .expect("value")
                    .parse()
                    .expect("--vocab-weight <f>")
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }

    let cases = load_test(&test_path);
    let n = cases.len();
    println!("cases: {n}");

    // Multi-reference sets from train + valid (one streaming pass each).
    let mut refsets = build_refsets(&train_path, &cases);
    for (roman, set) in build_refsets(&valid_path, &cases) {
        refsets.entry(roman).or_default().extend(set);
    }
    let mut cases = cases;
    for c in &mut cases {
        c.refs.insert(c.target.clone());
        if let Some(set) = refsets.get(&c.roman) {
            for r in set {
                c.refs.insert(r.clone());
            }
        }
    }
    let ambiguous = cases.iter().filter(|c| c.refs.len() > 1).count();
    println!(
        "multi-reference romans: {ambiguous}/{n} ({:.1}%)",
        ambiguous as f64 / n as f64 * 100.0
    );

    // Corpus word frequencies: the prior used both for decoder rescoring and
    // for the collision bound.
    // Both the vocabulary and the transliteration model come from the unified
    // container.  The loose `word_freq_text.bin` / `translit_model.bin` files
    // this used to read are no longer produced by the training pipeline.
    let unified = akshar_ime::core::unified::UnifiedModel::load(Path::new(&model_path))
        .unwrap_or_else(|e| panic!("load unified model {model_path}: {e}"));
    let vocab: HashMap<String, u32> = unified.vocab_freq.clone();
    eprintln!("model: {model_path} ({} vocab words)", vocab.len());

    collision_bound(&cases, &vocab);

    let model = unified.translit;
    let decoder = ModelDecoder::with_config(
        model,
        DecoderConfig {
            beam_width: beam,
            lm_weight,
            ..Default::default()
        },
    );
    let engine = ImeEngine::new();
    eprintln!("decoder: beam={beam} lm_weight={lm_weight} vocab_weight={vocab_weight}");

    // Buckets: AK-Freq (native), NE (NEF+NEI), ALL.
    let bucket = |src: &str| match src {
        "AK-Freq" => "AK-Freq",
        "AK-NEF" | "AK-NEI" => "NE",
        _ => "other",
    };

    struct Agg {
        name: &'static str,
        total: usize,
        dec_hits: [usize; KS.len()],    // strict decoder oracle
        dec_hits_mr: [usize; KS.len()], // multi-ref decoder oracle
        eng_hits: [usize; KS.len()],    // strict engine oracle
        dec_top1: usize,
        dec_top1_mr: usize,
        eng_top1: usize,
        eng_top1_mr: usize,
        dis_dec_right: usize,
        dis_eng_right: usize,
        dis_both_wrong: usize,
        tax_matra: usize,
        tax_halant: usize,
        tax_subst: usize,
        etax_matra: usize,
        etax_halant: usize,
        etax_subst: usize,
        cer_num: usize,
        cer_den: usize,
        ecer_num: usize,
    }
    let mk = |name: &'static str| Agg {
        name,
        total: 0,
        dec_hits: [0; KS.len()],
        dec_hits_mr: [0; KS.len()],
        eng_hits: [0; KS.len()],
        dec_top1: 0,
        dec_top1_mr: 0,
        eng_top1: 0,
        eng_top1_mr: 0,
        dis_dec_right: 0,
        dis_eng_right: 0,
        dis_both_wrong: 0,
        tax_matra: 0,
        tax_halant: 0,
        tax_subst: 0,
        etax_matra: 0,
        etax_halant: 0,
        etax_subst: 0,
        cer_num: 0,
        cer_den: 0,
        ecer_num: 0,
    };
    let mut aggs = [mk("ALL"), mk("AK-Freq"), mk("NE")];

    // W4 forensics: for every matra-only engine miss in the AK-Freq bucket,
    // record (freq(gold), freq(pred)).  This decides whether the matra class is
    // winnable by a better PRIOR (gold is the more frequent word, the ranker
    // just failed to use it) or only by CONTEXT (the prior actively prefers the
    // wrong reading).
    let mut matra_gold_freq_higher = 0usize;
    let mut matra_pred_freq_higher = 0usize;
    let mut matra_gold_oov = 0usize;
    let mut matra_both_oov = 0usize;
    let mut matra_in_list = 0usize;
    let mut matra_examples: Vec<(String, String, u32, String, u32, bool)> = Vec::new();

    for case in &cases {
        let b_idx = match bucket(&case.source) {
            "AK-Freq" => 1,
            _ => 2, // fold NE + rare buckets together
        };
        // The benchmark chain: decoder cost + LM, then frequency rescoring --
        // the same heuristic the engine's reranker blends against.
        let mut scored: Vec<(String, f64)> = decoder
            .decode_detailed(&case.roman, 50)
            .into_iter()
            .map(|c| {
                let mut sc = c.emit + lm_weight * c.lm;
                if vocab_weight > 0.0 {
                    if let Some(&f) = vocab.get(&c.dev) {
                        sc -= vocab_weight * (1.0 + f as f64).ln();
                    }
                }
                (c.dev, sc)
            })
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        let dec: Vec<String> = scored.into_iter().map(|(d, _)| d).collect();
        let eng: Vec<String> = engine
            .get_suggestions(&case.roman, 10)
            .into_iter()
            .map(|(d, _)| d)
            .collect();

        for idx in [0usize, b_idx] {
            let agg = &mut aggs[idx];
            agg.total += 1;
            for (i, k) in KS.iter().enumerate() {
                if dec.get(..*k).is_some_and(|p| p.contains(&case.target)) {
                    agg.dec_hits[i] += 1;
                }
                if dec
                    .get(..*k)
                    .is_some_and(|p| p.iter().any(|c| case.refs.contains(c)))
                {
                    agg.dec_hits_mr[i] += 1;
                }
                if eng.get(..*k).is_some_and(|p| p.contains(&case.target)) {
                    agg.eng_hits[i] += 1;
                }
            }
            if dec.first().is_some_and(|c| *c == case.target) {
                agg.dec_top1 += 1;
            }
            if dec.first().is_some_and(|c| case.refs.contains(c)) {
                agg.dec_top1_mr += 1;
            }
            if eng.first().is_some_and(|c| *c == case.target) {
                agg.eng_top1 += 1;
            }
            if eng.first().is_some_and(|c| case.refs.contains(c)) {
                agg.eng_top1_mr += 1;
            }
            // Disagreement analysis.
            let dt = dec.first();
            let et = eng.first();
            if dt != et {
                let d_right = dt.is_some_and(|c| case.refs.contains(c));
                let e_right = et.is_some_and(|c| case.refs.contains(c));
                if d_right && !e_right {
                    agg.dis_dec_right += 1;
                } else if e_right && !d_right {
                    agg.dis_eng_right += 1;
                } else {
                    agg.dis_both_wrong += 1;
                }
            }
            // Taxonomy + CER on decoder top-1 misses (strict).
            let miss = dec.first().is_none_or(|c| *c != case.target);
            if miss {
                if let Some(pred) = dec.first() {
                    match classify(pred, &case.target) {
                        Tax::Matra => agg.tax_matra += 1,
                        Tax::Halant => agg.tax_halant += 1,
                        Tax::Substantive => agg.tax_subst += 1,
                    }
                    agg.cer_num += levenshtein(pred, &case.target);
                }
            }
            // Same taxonomy on the ENGINE's top-1 -- the shipped chain, and the
            // one the W4 arithmetic is sized against.
            if eng.first().is_none_or(|c| *c != case.target) {
                if let Some(pred) = eng.first() {
                    match classify(pred, &case.target) {
                        Tax::Matra => agg.etax_matra += 1,
                        Tax::Halant => agg.etax_halant += 1,
                        Tax::Substantive => agg.etax_subst += 1,
                    }
                    agg.ecer_num += levenshtein(pred, &case.target);
                }
            }
            // AK-Freq matra forensics (once per case, not per bucket).
            if idx == b_idx && b_idx == 1 {
                if let Some(pred) = eng.first() {
                    if *pred != case.target && matches!(classify(pred, &case.target), Tax::Matra) {
                        let fg = vocab.get(&case.target).copied().unwrap_or(0);
                        let fp = vocab.get(pred).copied().unwrap_or(0);
                        // Was the gold anywhere in the engine's own list?
                        let in_list = eng.contains(&case.target);
                        if in_list {
                            matra_in_list += 1;
                        }
                        if fg == 0 && fp == 0 {
                            matra_both_oov += 1;
                        } else if fg == 0 {
                            matra_gold_oov += 1;
                        } else if fg > fp {
                            matra_gold_freq_higher += 1;
                        } else {
                            matra_pred_freq_higher += 1;
                        }
                        if matra_examples.len() < 20 {
                            matra_examples.push((
                                case.roman.clone(),
                                case.target.clone(),
                                fg,
                                pred.clone(),
                                fp,
                                in_list,
                            ));
                        }
                    }
                }
            }
            agg.cer_den += case.target.chars().count();
        }
    }

    // --- W4 forensics report ------------------------------------------------
    let mtot = (matra_gold_freq_higher + matra_pred_freq_higher + matra_gold_oov + matra_both_oov)
        .max(1) as f64;
    println!(
        "\n=== W4 forensics: AK-Freq matra-only engine misses (n={}) ===",
        mtot as usize
    );
    println!(
        "gold is the MORE frequent word (winnable by a better prior/ranker): {} ({:.1}%)",
        matra_gold_freq_higher,
        matra_gold_freq_higher as f64 / mtot * 100.0
    );
    println!(
        "prediction is more frequent (prior actively misleads; needs context/morph): {} ({:.1}%)",
        matra_pred_freq_higher,
        matra_pred_freq_higher as f64 / mtot * 100.0
    );
    println!(
        "gold is OOV in the 470k corpus vocab (needs W2's smoothed prior): {} ({:.1}%)",
        matra_gold_oov,
        matra_gold_oov as f64 / mtot * 100.0
    );
    println!(
        "both OOV (no frequency signal at all): {} ({:.1}%)",
        matra_both_oov,
        matra_both_oov as f64 / mtot * 100.0
    );
    println!(
        "gold was somewhere in the engine's top-10 (pure ranking loss): {} ({:.1}%)",
        matra_in_list,
        matra_in_list as f64 / mtot * 100.0
    );
    println!("  examples (roman / gold:freq / predicted:freq / gold-in-list):");
    for (r, g, fg, p, fp, il) in &matra_examples {
        println!("    {r:<16} {g}:{fg:<8} {p}:{fp:<8} {il}");
    }

    for agg in &aggs {
        let t = agg.total.max(1) as f64;
        println!("\n=== {} (n={}) ===", agg.name, agg.total);
        println!(
            "decoder top-1: strict {:.2}%  multi-ref {:.2}%",
            agg.dec_top1 as f64 / t * 100.0,
            agg.dec_top1_mr as f64 / t * 100.0
        );
        println!(
            "engine  top-1: strict {:.2}%  multi-ref {:.2}%",
            agg.eng_top1 as f64 / t * 100.0,
            agg.eng_top1_mr as f64 / t * 100.0
        );
        print!("decoder oracle k: ");
        for (i, k) in KS.iter().enumerate() {
            print!("{}:{:.1}% ", k, agg.dec_hits[i] as f64 / t * 100.0);
        }
        println!();
        print!("decoder oracle k (multi-ref): ");
        for (i, k) in KS.iter().enumerate() {
            print!("{}:{:.1}% ", k, agg.dec_hits_mr[i] as f64 / t * 100.0);
        }
        println!();
        print!("engine oracle k (depth 10): ");
        for (i, k) in KS.iter().enumerate() {
            if *k > 10 {
                continue; // get_suggestions was called with depth 10; deeper slots are vacuous
            }
            print!("{}:{:.1}% ", k, agg.eng_hits[i] as f64 / t * 100.0);
        }
        println!();
        println!(
            "disagreements (dec_right/eng_right/both_wrong): {}/{}/{}",
            agg.dis_dec_right, agg.dis_eng_right, agg.dis_both_wrong
        );
        println!(
            "decoder-miss taxonomy: matra-only {}  halant-only {}  substantive {}",
            agg.tax_matra, agg.tax_halant, agg.tax_subst
        );
        let etax_tot = (agg.etax_matra + agg.etax_halant + agg.etax_subst).max(1) as f64;
        println!(
            "engine-miss  taxonomy: matra-only {} ({:.1}% of misses)  halant-only {}  substantive {}",
            agg.etax_matra,
            agg.etax_matra as f64 / etax_tot * 100.0,
            agg.etax_halant,
            agg.etax_subst
        );
        // W4's headline: what the engine would score if matra assignment were solved.
        let solved = (agg.eng_top1 + agg.etax_matra) as f64 / t * 100.0;
        println!(
            "  -> engine top-1 with matra class solved: {:.2}%  (now {:.2}%)",
            solved,
            agg.eng_top1 as f64 / t * 100.0
        );
        println!(
            "CER (decoder top-1): {:.2}%",
            agg.cer_num as f64 / agg.cer_den.max(1) as f64 * 100.0
        );
        println!(
            "CER (engine  top-1): {:.2}%",
            agg.ecer_num as f64 / agg.cer_den.max(1) as f64 * 100.0
        );
    }
}

fn load_test(path: &str) -> Vec<Case> {
    let f = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let mut out = Vec::new();
    for line in BufReader::new(f).lines().map_while(Result::ok) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(rec) = serde_json::from_str::<Record>(trimmed) else {
            continue;
        };
        let roman = rec.roman.to_ascii_lowercase();
        let target = rec.target.trim().to_string();
        if roman.is_empty() || target.is_empty() {
            continue;
        }
        out.push(Case {
            roman,
            target,
            source: rec.source.to_string(),
            refs: HashSet::new(),
        });
    }
    out
}

/// Streaming pass over the train split: collect native variants only for the
/// romans that appear in the test set (keeps memory tiny).
fn build_refsets(path: &str, cases: &[Case]) -> HashMap<String, HashSet<String>> {
    let want: HashSet<String> = cases.iter().map(|c| c.roman.clone()).collect();
    let mut map: HashMap<String, HashSet<String>> = HashMap::new();
    let f = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("WARNING: cannot open train split {path}: {e}; multi-ref analysis skipped");
            return map;
        }
    };
    for line in BufReader::new(f).lines().map_while(Result::ok) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(rec) = serde_json::from_str::<Record>(trimmed) else {
            continue;
        };
        let roman = rec.roman.to_ascii_lowercase();
        if want.contains(&roman) {
            map.entry(roman)
                .or_default()
                .insert(rec.target.trim().to_string());
        }
    }
    map
}

/// W0 collision bound.  A(R) = every native ever paired with this roman
/// (train + valid + test).  A system with perfect generation whose only
/// tie-breaker is the corpus unigram prior picks argmax_f over A(R); the gold
/// is recovered only when it is that argmax.  Cases where |A(R)| = 1 are free.
fn collision_bound(cases: &[Case], vocab: &HashMap<String, u32>) {
    let mut n = 0usize;
    let mut single = 0usize;
    let mut hit = 0usize;
    let mut lost_examples: Vec<(String, String, String)> = Vec::new();
    for case in cases {
        // refs already holds train+valid variants plus the gold itself.
        let mut alts: Vec<&String> = case.refs.iter().collect();
        alts.sort();
        n += 1;
        if alts.len() <= 1 {
            single += 1;
            hit += 1;
            continue;
        }
        let freq = |w: &str| vocab.get(w).copied().unwrap_or(0);
        // argmax by frequency, ties broken deterministically by the string.
        let best = alts
            .iter()
            .max_by(|a, b| freq(a).cmp(&freq(b)).then_with(|| b.cmp(a)))
            .unwrap();
        if **best == case.target {
            hit += 1;
        } else if lost_examples.len() < 15 {
            lost_examples.push((case.roman.clone(), case.target.clone(), (*best).clone()));
        }
    }
    let d = n.max(1) as f64;
    println!("\n=== W0 collision bound (perfect generation + perfect unigram prior) ===");
    println!(
        "cases {n}  unambiguous {single} ({:.1}%)  ambiguous {} ({:.1}%)",
        single as f64 / d * 100.0,
        n - single,
        (n - single) as f64 / d * 100.0
    );
    println!("collision bound Acc* = {:.2}%", hit as f64 / d * 100.0);
    println!(
        "  (i.e. {:.2}% of cases are unwinnable for ANY string-only system",
        (d - hit as f64) / d * 100.0
    );
    println!("   whose only prior is corpus unigram frequency)");
    if !lost_examples.is_empty() {
        println!("  examples lost to the prior (roman / gold / frequency-argmax):");
        for (r, g, b) in &lost_examples {
            println!("    {r:<16} {g:<14} -> {b}");
        }
    }
}

enum Tax {
    Matra,
    Halant,
    Substantive,
}

fn is_matra(c: char) -> bool {
    matches!(c, '\u{0901}'..='\u{0903}' | '\u{093C}' | '\u{093E}'..='\u{094C}')
}

fn is_halant(c: char) -> bool {
    c == '\u{094D}'
}

fn strip_matching(s: &str, matra: bool, halant: bool) -> String {
    s.chars()
        .filter(|&c| {
            let m = is_matra(c);
            let h = is_halant(c);
            !((matra && m) || (halant && h))
        })
        .collect()
}

/// Compare a wrong prediction against the gold form.
fn classify(pred: &str, gold: &str) -> Tax {
    if strip_matching(pred, true, true) == strip_matching(gold, true, true) {
        return Tax::Matra;
    }
    if strip_matching(pred, false, true) == strip_matching(gold, false, true) {
        return Tax::Halant;
    }
    Tax::Substantive
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}
