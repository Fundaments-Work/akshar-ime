// File: src/bin/evaluate/evaluate_sentences.rs
//
// Sentence-level evaluation on held-out corpus sentences.

use akshar_ime::core::akshara::segment;
use akshar_ime::core::engine::ImeEngine;
use akshar_ime::core::holdout::{is_holdout, DEFAULT_HOLDOUT_DENOM};
use akshar_ime::core::translit_model::TranslitModel;
use akshar_ime::core::unified::UnifiedModel;
use akshar_ime::fuzzy::grammar::orthographic_skeleton;
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

/// Per-akshara best roman spelling, precomputed once from the EM emissions.
struct Romanizer {
    best: HashMap<String, String>,
}

impl Romanizer {
    fn new(model: &TranslitModel) -> Self {
        let mut best = HashMap::with_capacity(model.aksharas.len());
        for (aid, emissions) in model.emissions.iter().enumerate() {
            let Some(akshara) = model.aksharas.get(aid) else {
                continue;
            };
            let pick = emissions
                .iter()
                .filter_map(|(cid, w)| model.chunks.get(*cid as usize).map(|c| (c, *w)))
                .filter(|(c, _)| !c.is_empty())
                .min_by(|(ca, wa), (cb, wb)| {
                    wa.total_cmp(wb)
                        .then(ca.len().cmp(&cb.len()))
                        .then(ca.cmp(cb))
                });
            if let Some((chunk, _)) = pick {
                best.insert(akshara.clone(), chunk.clone());
            }
        }
        Self { best }
    }

    fn romanize(&self, word: &str) -> Option<String> {
        let mut out = String::with_capacity(word.len());
        let units = segment(word);
        if units.is_empty() {
            return None;
        }
        for unit in units {
            out.push_str(self.best.get(&unit)?);
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }
}

#[derive(Default)]
struct Tally {
    words: usize,
    top1: usize,
    topk: usize,
    rank_sum: usize,
    ranked: usize,
    sentences: usize,
    sentences_exact: usize,
    skeleton_misses: usize,
    skeleton_in_list: usize,
    skeleton_misses_same_roman: usize,
}

impl Tally {
    fn pct(n: usize, d: usize) -> f64 {
        if d == 0 {
            0.0
        } else {
            n as f64 / d as f64 * 100.0
        }
    }

    fn report(&self, label: &str, k: usize) {
        println!(
            "  {label:<18} word@1={:>6.2}%  word@{k}={:>6.2}%  sentence-exact={:>6.2}%  \
             mean-gold-rank={:.2} (recall={:.2}%)\n  {:<18} spelling-variant top-1 misses={:>6.2}% of all words              ({} of {} misses), of which {} are input-ambiguous (same roman) \
             leaving {:.2}pp resolvable; variant present in list for {:>6.2}% of words",
            Self::pct(self.top1, self.words),
            Self::pct(self.topk, self.words),
            Self::pct(self.sentences_exact, self.sentences),
            if self.ranked == 0 {
                f64::NAN
            } else {
                self.rank_sum as f64 / self.ranked as f64
            },
            Self::pct(self.ranked, self.words),
            "",
            Self::pct(self.skeleton_misses, self.words),
            self.skeleton_misses,
            self.words - self.top1,
            self.skeleton_misses_same_roman,
            Self::pct(self.skeleton_misses - self.skeleton_misses_same_roman, self.words),
            Self::pct(self.skeleton_in_list, self.words),
        );
    }
}

fn main() {
    let mut model_path = "data/akshar.model".to_string();
    let mut corpus_path = "data/nepali_corpus.txt".to_string();
    let mut n_sentences = 3000usize;
    let mut skip_sentences = 0usize;
    let mut topk = 5usize;
    let mut holdout_denom = DEFAULT_HOLDOUT_DENOM;
    let mut show_misses = 0usize;
    let mut ctx_path: Option<String> = None;
    let mut ctx_mode = "off".to_string();

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model_path = args.next().expect("value for --model"),
            "--corpus" => corpus_path = args.next().expect("value for --corpus"),
            "--n" => n_sentences = args.next().expect("value for --n").parse().unwrap(),
            "--topk" => topk = args.next().expect("value for --topk").parse().unwrap(),
            "--holdout-denom" => {
                holdout_denom = args
                    .next()
                    .expect("value for --holdout-denom")
                    .parse()
                    .unwrap()
            }
            "--show-misses" => {
                show_misses = args
                    .next()
                    .expect("value for --show-misses")
                    .parse()
                    .unwrap()
            }
            "--skip" => {
                skip_sentences = args.next().expect("value").parse().unwrap();
            }
            "--ctx" => ctx_path = Some(args.next().expect("value for --ctx")),
            "--ctx-mode" => ctx_mode = args.next().expect("value for --ctx-mode"),
            "--help" | "-h" => {
                println!("evaluate_sentences — in-context word accuracy on held-out sentences");
                println!("  --model <path>          unified model (default data/akshar.model)");
                println!("  --corpus <path>         nepali_corpus.txt");
                println!("  --n <count>             held-out sentences to score (default 3000)");
                println!("  --topk <k>              top-k cutoff (default 5)");
                println!("  --holdout-denom <d>     hold out 1 sentence in d (default 200)");
                println!("  --show-misses <n>       print n in-context top-1 misses");
                println!("  --skip <k>              skip the first k held-out sentences");
                println!("    (disjoint tuning set: --skip 1000 --n 1000 avoids the measured 0-1000 window)");
                println!("  --ctx <path>            corpus bigram sidecar (enables context)");
                println!("  --ctx-mode <m>          off|oracle|predicted (default off)");
                println!("    oracle feeds the gold prev word (ceiling); predicted feeds the");
                println!("    engine's own top-1 pick (realistic; errors propagate).");
                return;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }

    let t0 = Instant::now();
    let file = std::fs::File::open(&corpus_path).unwrap_or_else(|e| {
        eprintln!("cannot open corpus {corpus_path}: {e}");
        std::process::exit(1);
    });
    let mut sentences: Vec<Vec<String>> = Vec::with_capacity(n_sentences);
    let mut skipped = 0usize;
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        if sentences.len() >= n_sentences {
            break;
        }
        let line = line.trim();
        if !is_holdout(line, holdout_denom) {
            continue;
        }
        let words: Vec<String> = line.split_whitespace().map(str::to_string).collect();
        if words.len() >= 2 {
            if skipped < skip_sentences {
                skipped += 1;
                continue;
            }
            sentences.push(words);
        }
    }
    if sentences.is_empty() {
        eprintln!("no held-out sentences found (denom={holdout_denom}) — nothing to score");
        std::process::exit(1);
    }
    eprintln!(
        "Collected {} held-out sentences (1-in-{}) in {:.1?}",
        sentences.len(),
        holdout_denom,
        t0.elapsed()
    );

    let unified = UnifiedModel::load(Path::new(&model_path)).unwrap_or_else(|e| {
        eprintln!("cannot load model {model_path}: {e}");
        std::process::exit(1);
    });
    let romanizer = Romanizer::new(&unified.translit);
    eprintln!(
        "Indexed best roman chunk for {} aksharas",
        romanizer.best.len()
    );
    let mut engine = ImeEngine::from_unified(unified);
    if !matches!(ctx_mode.as_str(), "off" | "oracle" | "predicted") {
        eprintln!("unknown --ctx-mode {ctx_mode} (want off|oracle|predicted)");
        std::process::exit(2);
    }
    if let Some(path) = &ctx_path {
        match akshar_ime::core::context::CorpusCtx::load(Path::new(path)) {
            Ok(ctx) => {
                eprintln!(
                    "corpus context: {} bigrams from {path} (mode {ctx_mode})",
                    ctx.table.len()
                );
                engine.set_corpus_ctx(Some(ctx));
            }
            Err(e) => {
                eprintln!("cannot load ctx sidecar {path}: {e}");
                std::process::exit(1);
            }
        }
    } else if ctx_mode != "off" {
        eprintln!("--ctx-mode {ctx_mode} needs --ctx <sidecar>");
        std::process::exit(2);
    }

    let mut misses: Vec<(String, String, Vec<String>, String)> = Vec::new();
    let mut t = Tally::default();
    let run_t0 = Instant::now();
    let mut skipped_words = 0usize;

    for words in &sentences {
        engine.set_context_word("");
        let mut sentence_perfect = true;
        let mut scored_any = false;

        for (i, gold) in words.iter().enumerate() {
            let Some(roman) = romanizer.romanize(gold) else {
                skipped_words += 1;
                // Unscored (aksharas unknown to the model): the user still
                // typed the word, so oracle continuity uses gold. Predicted
                // mode has no pick; gold is the only honest fallback.
                if ctx_mode != "off" {
                    engine.set_context_word(gold);
                }
                continue;
            };
            scored_any = true;
            t.words += 1;

            let suggestions = engine.get_suggestions(&roman, topk.max(8));
            let rank = suggestions.iter().position(|(dev, _)| dev == gold);

            if rank != Some(0) {
                let gold_skel = orthographic_skeleton(gold);
                if suggestions
                    .first()
                    .is_some_and(|(dev, _)| orthographic_skeleton(dev) == gold_skel)
                {
                    t.skeleton_misses += 1;
                    if suggestions
                        .first()
                        .and_then(|(dev, _)| romanizer.romanize(dev))
                        .is_some_and(|r| r == roman)
                    {
                        t.skeleton_misses_same_roman += 1;
                    }
                }
                if suggestions
                    .iter()
                    .any(|(dev, _)| orthographic_skeleton(dev) == gold_skel)
                {
                    t.skeleton_in_list += 1;
                }
            }
            match rank {
                Some(0) => {
                    t.top1 += 1;
                    t.topk += 1;
                    t.rank_sum += 0;
                    t.ranked += 1;
                }
                Some(r) => {
                    if r < topk {
                        t.topk += 1;
                    }
                    t.rank_sum += r;
                    t.ranked += 1;
                    sentence_perfect = false;
                }
                None => sentence_perfect = false,
            }

            if rank != Some(0) && misses.len() < show_misses {
                misses.push((
                    roman.clone(),
                    gold.clone(),
                    suggestions.iter().take(4).map(|(d, _)| d.clone()).collect(),
                    words.get(i.wrapping_sub(1)).cloned().unwrap_or_default(),
                ));
            }

            // Advance the context: oracle feeds gold (ceiling), predicted feeds
            // the engine's own top-1 (realistic; errors propagate), off feeds
            // nothing (isolated words — the pre-context baseline).
            match ctx_mode.as_str() {
                "oracle" => engine.set_context_word(gold),
                "predicted" => engine
                    .set_context_word(suggestions.first().map(|(d, _)| d.as_str()).unwrap_or("")),
                _ => {}
            }
        }

        if scored_any {
            t.sentences += 1;
            if sentence_perfect {
                t.sentences_exact += 1;
            }
        }
    }

    println!(
        "\n=== Sentence-level evaluation ({} sentences, held out 1-in-{}) ===",
        sentences.len(),
        holdout_denom
    );
    println!("model: {model_path}");
    if skipped_words > 0 {
        println!("skipped {skipped_words} words with aksharas unknown to the model");
    }
    t.report("sentence", topk);
    eprintln!("    ({} words in {:.1?})", t.words, run_t0.elapsed());

    if !misses.is_empty() {
        println!("\n  in-context top-1 misses (first {}):", misses.len());
        for (roman, gold, top, prev) in &misses {
            println!("    prev=`{prev}` roman=`{roman}` gold=`{gold}` top={top:?}");
        }
    }
}
