// File: src/bin/prune_model.rs
//
// Phonetic-model pruning: after training emissions on all Devanagari data
// (including pairs from other languages sharing the script), drop the emission
// rows of aksharas that never appear in the target vocabulary — they cannot be
// correct outputs for this IME,
// and their peaked emissions crowd real candidates out of the reverse index.
//
// Usage: prune_model --model data/translit_model.bin \
//                    --vocab data/word_freq_text.bin \
//                    --min-count 1 [--out data/translit_model.bin]

use akshar_ime::core::akshara::segment;
use akshar_ime::core::translit_model::TranslitModel;
use std::collections::HashSet;
use std::path::Path;

fn main() {
    let mut model_path = "data/translit_model.bin".to_string();
    let mut vocab_path = "data/word_freq_text.bin".to_string();
    let mut out_path = "data/translit_model.bin".to_string();
    let mut min_count = 1u32;
    let mut no_trigrams = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model_path = args.next().expect("v"),
            "--vocab" => vocab_path = args.next().expect("v"),
            "--out" => out_path = args.next().expect("v"),
            "--min-count" => min_count = args.next().expect("v").parse().unwrap(),
            "--no-trigrams" => no_trigrams = true,
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }

    let mut model = TranslitModel::load(Path::new(&model_path)).expect("load model");
    // Aksharas that occur in the target vocabulary (the IME's output words).
    let map: std::collections::HashMap<String, u32> =
        bincode::deserialize(&std::fs::read(&vocab_path).expect("vocab")).expect("deserialize");
    let mut seen: HashSet<u32> = HashSet::new();
    for (word, &count) in &map {
        if count < min_count {
            continue;
        }
        for a in segment(word) {
            if let Some(id) = model.akshara_id(&a) {
                seen.insert(id);
            }
        }
    }
    let before_em: usize = model.emissions.iter().filter(|e| !e.is_empty()).count();
    let before_bi: usize = model.bigrams.iter().map(|b| b.len()).sum();
    let before_tri: usize = model.trigrams.iter().map(|t| t.len()).sum();

    // 1. Prune emissions
    for (a, em) in model.emissions.iter_mut().enumerate() {
        if !seen.contains(&(a as u32)) {
            em.clear();
        }
    }

    // 2. Prune bigrams
    for (a, bi) in model.bigrams.iter_mut().enumerate() {
        if !seen.contains(&(a as u32)) {
            bi.clear();
        } else {
            bi.retain(|(next_id, _)| seen.contains(next_id));
        }
    }

    // 3. Prune trigrams
    if no_trigrams {
        model.trigram_keys.clear();
        model.trigrams.clear();
        model.trigram_backoff.clear();
        model.build_trigram_index();
    } else {
        let mut new_keys = Vec::new();
        let mut new_trigrams = Vec::new();
        let mut new_backoff = Vec::new();
        for (i, &(a, b)) in model.trigram_keys.iter().enumerate() {
            if seen.contains(&a) && seen.contains(&b) {
                let mut list = model.trigrams[i].clone();
                list.retain(|(c, _)| seen.contains(c));
                if !list.is_empty() {
                    new_keys.push((a, b));
                    new_trigrams.push(list);
                    new_backoff.push(model.trigram_backoff.get(i).copied().unwrap_or(0.0));
                }
            }
        }
        model.trigram_keys = new_keys;
        model.trigrams = new_trigrams;
        model.trigram_backoff = new_backoff;
        model.build_trigram_index();
    }

    let after_em: usize = model.emissions.iter().filter(|e| !e.is_empty()).count();
    let after_bi: usize = model.bigrams.iter().map(|b| b.len()).sum();
    let after_tri: usize = model.trigrams.iter().map(|t| t.len()).sum();

    model.save(Path::new(&out_path)).expect("save");
    eprintln!(
        "Pruning stats:\n  Emissions: {} -> {} aksharas\n  Bigrams  : {} -> {} transitions\n  Trigrams : {} -> {} transitions across {} contexts",
        before_em, after_em,
        before_bi, after_bi,
        before_tri, after_tri,
        model.trigram_keys.len()
    );
}
