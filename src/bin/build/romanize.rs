// File: src/bin/romanize.rs
//
// High-Performance Multi-Threaded Cycle-Consistent Romanizer for Akshar-IME.
//
// Generates high-quality synthetic training pairs (Roman, Devanagari) from vocabulary:
//   1. Canonicalize the input Devanagari word using the orthography rules
//      (resolves common errors: सहिद -> शहीद, बिकास -> विकास, गरीन्छ -> गरिन्छ, etc.).
//   2. Backward romanization: segment into aksharas and generate Roman chunks from EM emissions,
//      plus natural user phonetic typing mutations (b/v/w, ee/i, oo/u, s/sh, ch/chh).
//   3. Forward transliteration (IME Decoder): decode each Roman candidate back to Devanagari.
//   4. Round-trip cycle consistency & grammar check: verify that forward predictions match
//      the canonical Devanagari word or its verified orthographic equivalence class.
//   5. Output deduplicated weighted training pairs for the EM trainer.
//
// Usage:
//   cargo run --release --bin romanize -- \
//     [--vocab data/store/word_freq.csv] \
//     [--model data/translit_model.bin] \
//     [--ref-vocab data/word_freq_text.bin] \
//     [--out data/store/synthetic.jsonl] \
//     [--max-combos 6] [--top 3] [--max-weight 9.0] [--min-freq 2] [--threads 12]

use akshar_ime::core::akshara::segment;
use akshar_ime::core::decoder::ModelDecoder;
use akshar_ime::core::translit_model::TranslitModel;
use akshar_ime::fuzzy::grammar::{
    generate_roman_phonetic_variants, orthographic_skeleton, GrammarCanonicalizer,
};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let mut vocab_path = "data/store/word_freq.csv".to_string();
    let mut model_path = "data/translit_model.bin".to_string();
    let mut ref_vocab_path = "data/word_freq_text.bin".to_string();
    let mut out_path = "data/store/synthetic.jsonl".to_string();
    let mut max_combos = 6usize;
    let mut top = 3usize;
    let mut max_weight = 9.0f32;
    let mut min_freq = 2u64;
    let mut verify_roundtrip = true;
    let mut num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--vocab" => vocab_path = args.next().expect("vocab path"),
            "--model" => model_path = args.next().expect("model path"),
            "--ref-vocab" => ref_vocab_path = args.next().expect("ref vocab path"),
            "--out" => out_path = args.next().expect("out path"),
            "--max-combos" => max_combos = args.next().expect("max combos").parse().unwrap(),
            "--top" => top = args.next().expect("top").parse().unwrap(),
            "--max-weight" => max_weight = args.next().expect("max weight").parse().unwrap(),
            "--min-freq" => min_freq = args.next().expect("min freq").parse().unwrap(),
            "--threads" => num_threads = args.next().expect("threads").parse().unwrap(),
            "--verify-roundtrip" => {
                let v = args.next().expect("bool");
                verify_roundtrip = v == "true" || v == "1";
            }
            "--no-roundtrip" => verify_roundtrip = false,
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }

    let t0 = Instant::now();
    eprintln!("=== Akshar-IME Parallel Cycle-Consistent Romanizer ===");

    // 1. Load Transliteration Model & Decoder
    let model = TranslitModel::load(Path::new(&model_path)).expect("load transliteration model");
    let decoder = ModelDecoder::new(model.clone());

    // 2. Load Grammar Canonicalizer & Reference Vocab
    let mut canonicalizer = GrammarCanonicalizer::new();
    let ref_path = Path::new(&ref_vocab_path);
    if ref_path.exists() {
        match canonicalizer.load_reference_vocab(ref_path) {
            Ok(n) => eprintln!(
                "Loaded {n} reference words for grammar validation from {}",
                ref_path.display()
            ),
            Err(e) => eprintln!(
                "Warning: reference vocab failed to load ({e}), using rule-based grammar engine"
            ),
        }
    } else {
        let fallback = Path::new("data/word_freq.bin");
        if fallback.exists() {
            match canonicalizer.load_reference_vocab(fallback) {
                Ok(n) => eprintln!(
                    "Loaded {n} reference words from fallback {}",
                    fallback.display()
                ),
                Err(e) => eprintln!("Warning: fallback failed ({e})"),
            }
        } else {
            eprintln!(
                "No reference vocabulary found, running with pure grammar transformation engine"
            );
        }
    }

    // 3. Pre-index akshara candidates sorted by emission weight
    let mut cands: HashMap<u32, Vec<(String, f32)>> = HashMap::new();
    for (a, list) in model.emissions.iter().enumerate() {
        let mut v: Vec<(String, f32)> = list
            .iter()
            .filter_map(|(cid, w)| model.chunks.get(*cid as usize).map(|s| (s.clone(), *w)))
            .filter(|(_, w)| *w <= max_weight && *w > 0.0)
            .collect();
        v.sort_by(|x, y| x.1.total_cmp(&y.1));
        v.truncate(top);
        if !v.is_empty() {
            cands.insert(a as u32, v);
        }
    }
    eprintln!(
        "Indexed candidates for {} aksharas (top={top}, max_weight={max_weight:.1})",
        cands.len()
    );

    // 4. Load all qualifying vocabulary words
    let f = std::fs::File::open(&vocab_path).unwrap_or_else(|e| {
        eprintln!("Error: cannot open vocabulary file {vocab_path}: {e}");
        std::process::exit(1);
    });

    let mut words: Vec<(String, u64)> = Vec::new();
    for line in BufReader::new(f).lines().map_while(Result::ok) {
        let line = line.trim();
        if line.is_empty() || line.starts_with("word,") {
            continue;
        }
        let Some((raw_word, freq_str)) = line.rsplit_once(',') else {
            continue;
        };
        let Ok(freq) = freq_str.parse::<u64>() else {
            continue;
        };
        if freq >= min_freq {
            words.push((raw_word.to_string(), freq));
        }
    }
    let total_words = words.len();
    eprintln!(
        "Loaded {total_words} vocabulary words (freq >= {min_freq}) across {num_threads} CPU worker threads"
    );

    let progress = Arc::new(AtomicUsize::new(0));
    let work_index = Arc::new(AtomicUsize::new(0));
    let t_start_work = Instant::now();

    // 5. Multi-Threaded Parallel Romanization & Verification
    let results = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_threads);

        for _ in 0..num_threads {
            let work_index = Arc::clone(&work_index);
            let progress = Arc::clone(&progress);
            let words = &words;
            let canonicalizer = &canonicalizer;
            let model = &model;
            let decoder = &decoder;
            let cands = &cands;

            handles.push(s.spawn(move || {
                let batch_size = 128usize;
                let mut local_weighted: HashMap<(String, String), u32> = HashMap::new();
                let mut local_canonicalized = 0usize;
                let mut local_rejected = 0usize;
                let mut local_pairs = 0usize;

                loop {
                    let start_idx = work_index.fetch_add(batch_size, Ordering::Relaxed);
                    if start_idx >= total_words {
                        break;
                    }
                    let end_idx = (start_idx + batch_size).min(total_words);
                    let batch = &words[start_idx..end_idx];

                    for (raw_word, freq) in batch {
                        let freq = *freq;

                        // STEP 1: Grammar Canonicalization
                        let canon_word = canonicalizer.canonicalize(raw_word);
                        if canon_word != *raw_word {
                            local_canonicalized += 1;
                        }
                        let canon_skel = orthographic_skeleton(&canon_word);

                        // STEP 2: Backward Romanization on the Canonical Word
                        let aks = segment(&canon_word);
                        if aks.is_empty() {
                            continue;
                        }

                        let mut lists: Vec<&Vec<(String, f32)>> = Vec::with_capacity(aks.len());
                        let mut ok = true;
                        for a in &aks {
                            match model.akshara_id(a).and_then(|id| cands.get(&id)) {
                                Some(c) => lists.push(c),
                                None => {
                                    ok = false;
                                    break;
                                }
                            }
                        }
                        if !ok {
                            continue;
                        }

                        // Combine lowest-weight joint spellings
                        let mut combos: Vec<(String, f32)> = vec![(String::new(), 0.0)];
                        for list in &lists {
                            let mut next: Vec<(String, f32)> = Vec::new();
                            for (s, w) in &combos {
                                for (c, cw) in list.iter().take(top) {
                                    next.push((format!("{s}{c}"), w + cw));
                                }
                            }
                            next.sort_by(|a, b| a.1.total_cmp(&b.1));
                            next.truncate(max_combos);
                            combos = next;
                        }

                        // Expand Roman spellings with natural typing variants
                        let mut roman_candidates: Vec<(String, f32)> = Vec::new();
                        for (base_roman, base_w) in &combos {
                            let base_lower = base_roman.to_lowercase();
                            roman_candidates.push((base_lower.clone(), *base_w));
                            let mut variants = generate_roman_phonetic_variants(&base_lower, 3);
                            variants.retain(|v| v != &base_lower);
                            for v in variants {
                                roman_candidates.push((v, base_w + 1.2));
                            }
                        }

                        let mut seen_roman = std::collections::HashSet::new();
                        roman_candidates.retain(|(r, _)| seen_roman.insert(r.clone()));
                        roman_candidates.truncate(max_combos.saturating_mul(2));

                        // STEP 3: Forward Transliteration & Cycle-Consistency Verification
                        for (roman, w) in &roman_candidates {
                            let mut cycle_ok = true;
                            let mut rank_multiplier = 1.0f64;

                            if verify_roundtrip {
                                let decodings = decoder.decode(roman, 4);
                                if decodings.is_empty() {
                                    cycle_ok = false;
                                } else {
                                    let mut found_rank = None;
                                    for (rank, (cand, _score)) in decodings.iter().enumerate() {
                                        if cand == &canon_word
                                            || orthographic_skeleton(cand) == canon_skel
                                            || canonicalizer.are_equivalent(cand, &canon_word)
                                        {
                                            found_rank = Some(rank);
                                            break;
                                        }
                                    }

                                    match found_rank {
                                        Some(0) => rank_multiplier = 1.0,
                                        Some(1) => rank_multiplier = 0.85,
                                        Some(2) | Some(3) => rank_multiplier = 0.70,
                                        Some(_) => rank_multiplier = 0.50,
                                        None => {
                                            cycle_ok = false;
                                            local_rejected += 1;
                                        }
                                    }
                                }
                            }

                            if !cycle_ok {
                                continue;
                            }

                            let emit_discount = (-*w as f64).exp() * 3.0;
                            let base =
                                (freq as f64) * rank_multiplier * emit_discount.clamp(0.2, 1.0);
                            let weight = (base.round() as u64).clamp(1, 25) as u32;

                            let key = (roman.clone(), canon_word.clone());
                            *local_weighted.entry(key).or_insert(0) += weight;
                            local_pairs += 1;
                        }
                    }

                    let prev = progress.fetch_add(batch.len(), Ordering::Relaxed);
                    let cur = prev + batch.len();
                    if cur / 2500 > prev / 2500 || cur >= total_words {
                        let elapsed = t_start_work.elapsed().as_secs_f64();
                        let rate = if elapsed > 0.0 {
                            cur as f64 / elapsed
                        } else {
                            0.0
                        };
                        let pct = (cur * 100).checked_div(total_words).unwrap_or(100);
                        eprintln!(
                            "[{pct:3}%] Processed {cur}/{total_words} words ({rate:.0} words/s)..."
                        );
                    }
                }

                (
                    local_weighted,
                    local_canonicalized,
                    local_rejected,
                    local_pairs,
                )
            }));
        }

        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });

    // 6. Merge thread-local results
    let mut weighted: HashMap<(String, String), u32> = HashMap::new();
    let mut n_canonicalized = 0usize;
    let mut n_rejected_cycle = 0usize;
    let mut n_pairs = 0usize;

    for (lw, lc, lr, lp) in results {
        for (k, v) in lw {
            *weighted.entry(k).or_insert(0) += v;
        }
        n_canonicalized += lc;
        n_rejected_cycle += lr;
        n_pairs += lp;
    }

    // 7. Write out results in JSONL format
    let mut out = std::io::BufWriter::new(std::fs::File::create(&out_path).unwrap_or_else(|e| {
        eprintln!("Error: cannot create output file {out_path}: {e}");
        std::process::exit(1);
    }));

    for ((roman, native), weight) in &weighted {
        writeln!(
            out,
            "{{\"english word\": {}, \"native word\": {}, \"weight\": {}}}",
            serde_json::to_string(roman).unwrap(),
            serde_json::to_string(native).unwrap(),
            weight
        )
        .expect("write jsonl line");
    }
    out.flush().expect("flush output file");

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!("\n=== Romanization & Verification Summary ===");
    eprintln!("Total Vocabulary Words Processed: {total_words}");
    eprintln!("Orthographically Canonicalized:   {n_canonicalized}");
    eprintln!("Cycle-Inconsistent Candidates:    {n_rejected_cycle}");
    eprintln!("Total Clean Synthetic Pairs:     {n_pairs}");
    eprintln!("Deduplicated Weighted Rows:       {}", weighted.len());
    eprintln!("Output File:                      {out_path}");
    eprintln!("Completed in:                     {elapsed:.2}s");
}
