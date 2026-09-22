// Diagnostic: is the shipped sparse reranker table populated, and does the
// reranker training loop actually reduce loss?
use akshar_ime::core::decoder::{DecoderConfig, ModelDecoder};
use akshar_ime::core::reranker::{
    extract_dense_features, extract_sparse_features, rank_candidates, FreqRanks, DENSE_DIM,
};
use akshar_ime::core::reranker_weights::{HASH_SIZE, MEAN_DENSE, STD_DENSE, W_DENSE};
use akshar_ime::core::unified::UnifiedModel;
use std::io::BufRead;

fn main() {
    let u = UnifiedModel::load(std::path::Path::new("data/akshar.model")).expect("load");
    let t = &u.sparse_reranker_table;
    let nz = t.iter().filter(|&&w| w != 0).count();
    println!("=== shipped sparse table ===");
    println!(
        "slots {} (HASH_SIZE {}), non-zero {} ({:.4}%), scale {:.3e}",
        t.len(),
        HASH_SIZE,
        nz,
        nz as f64 / t.len().max(1) as f64 * 100.0,
        u.sparse_scale
    );
    println!("min {:?} max {:?}", t.iter().min(), t.iter().max());

    // Replay the training loop on a small sample and watch the loss.
    let vocab_freq = u.vocab_freq.clone();
    let ranks = FreqRanks::from_freq_map(&vocab_freq);
    let decoder = ModelDecoder::with_config(u.translit.clone(), DecoderConfig::default());

    let f = std::fs::File::open("data/aksharantar/valid_devanagari.jsonl").expect("valid set");
    let mut pairs: Vec<(String, String)> = Vec::new();
    for line in std::io::BufReader::new(f).lines().take(4000) {
        let l = line.unwrap();
        let v: serde_json::Value = match serde_json::from_str(&l) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let (Some(r), Some(d)) = (v["english word"].as_str(), v["native word"].as_str()) {
            pairs.push((r.to_string(), d.to_string()));
        }
        if pairs.len() >= 2000 {
            break;
        }
    }
    println!(
        "\n=== replaying reranker training on {} valid pairs ===",
        pairs.len()
    );

    struct Item {
        target_idx: usize,
        base: Vec<f64>,
        sparse: Vec<Vec<usize>>,
    }
    let mut samples: Vec<Item> = Vec::new();
    let mut empty_sparse = 0usize;
    let mut identical_sparse = 0usize;
    let counts = akshar_ime::core::reranker::VocabCounts {
        freq: &vocab_freq,
        ranks: &ranks,
    };
    for (roman, gold) in &pairs {
        let cands = decoder.decode_union(roman, 50, None);
        let (order, heur, heur_rank) = rank_candidates(&cands, &counts);
        if let Some(target_idx) = order.iter().position(|c| &c.dev == gold) {
            let sparse: Vec<Vec<usize>> = order
                .iter()
                .map(|c| {
                    let aks = akshar_ime::core::akshara::segment(&c.dev);
                    extract_sparse_features(&c.dev, roman, c.akshara_count, &aks)
                })
                .collect();
            if sparse.iter().any(|s| s.is_empty()) {
                empty_sparse += 1;
            }
            if sparse.len() > 1 && sparse.iter().all(|s| *s == sparse[0]) {
                identical_sparse += 1;
            }
            let base: Vec<f64> = order
                .iter()
                .enumerate()
                .map(|(idx, c)| {
                    let dense =
                        extract_dense_features(c, idx, heur[idx], heur_rank[idx], roman, &counts);
                    (0..DENSE_DIM)
                        .map(|k| W_DENSE[k] * ((dense[k] - MEAN_DENSE[k]) / STD_DENSE[k]))
                        .sum()
                })
                .collect();
            samples.push(Item {
                target_idx,
                base,
                sparse,
            });
        }
    }
    println!("samples with gold in top-50: {}", samples.len());
    println!(
        "samples where some candidate has NO sparse features: {}",
        empty_sparse
    );
    println!(
        "samples where ALL candidates share identical sparse features: {}",
        identical_sparse
    );

    let mut table = vec![0.0f32; HASH_SIZE];
    let mut grad_sq = vec![0.0f32; HASH_SIZE];
    let mut lr: f64 = 0.05;
    for ep in 1..=8 {
        let (mut loss, mut hits) = (0.0f64, 0usize);
        for s in &samples {
            let mut scores = s.base.clone();
            for (idx, sf) in s.sparse.iter().enumerate() {
                for &h in sf {
                    scores[idx] += f64::from(table[h]);
                }
            }
            let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_s: Vec<f64> = scores.iter().map(|&sc| (sc - max_s).exp()).collect();
            let sum_exp: f64 = exp_s.iter().sum();
            let probs: Vec<f64> = exp_s.iter().map(|&e| e / (sum_exp + 1e-12)).collect();
            loss += -probs[s.target_idx].max(1e-12).ln();
            if probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .is_some_and(|(i, _)| i == s.target_idx)
            {
                hits += 1;
            }
            for (idx, p) in probs.iter().enumerate() {
                let grad = if idx == s.target_idx { *p - 1.0 } else { *p };
                if grad.abs() > 1e-5 {
                    for &h in &s.sparse[idx] {
                        let g = grad as f32;
                        grad_sq[h] += g * g;
                        let eff = (lr as f32) / (grad_sq[h].sqrt() + 1e-4);
                        table[h] -= eff * g;
                    }
                }
            }
        }
        let nz = table.iter().filter(|&&w| w != 0.0).count();
        println!(
            "  epoch {}: loss={:.4} top1={:.2}%  lr={:.2e}  nonzero_slots={}",
            ep,
            loss / samples.len() as f64,
            hits as f64 / samples.len() as f64 * 100.0,
            lr,
            nz
        );
        lr *= 0.8;
    }
}
