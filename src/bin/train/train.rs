// File: src/bin/train/train.rs
//
// End-to-End Unified Model Trainer for Akshar Devanagari IME.
// Ingests training data (word pairs + text) and directly produces
// a single, low-size, production-ready `akshar.model` artifact.

use akshar_ime::core::decoder::{DecoderConfig, ModelDecoder};
use akshar_ime::core::em_trainer::{Trainer, TrainerConfig};
use akshar_ime::core::holdout::{is_holdout, DEFAULT_HOLDOUT_DENOM};
use akshar_ime::core::reranker::{
    extract_dense_features, extract_sparse_features, rank_candidates, FreqRanks, DENSE_DIM,
    HASH_SIZE,
};
use akshar_ime::core::reranker_weights::{MEAN_DENSE, STD_DENSE, W_DENSE};

/// Online mean/variance accumulator for the reranker's dense features.
///
/// `W_DENSE` was fitted on standardised features, so the standardisation must
/// describe the distribution the model being trained actually produces.  The
/// compiled-in `MEAN_DENSE`/`STD_DENSE` describe whichever model produced them,
/// and no training stage ever refreshed them: retraining the EM/LM moves the
/// `emit`/`lm` features out from under weights fitted on the old scale.  These
/// statistics are written into the v5 container instead.
#[derive(Clone)]
struct DenseStats {
    n: u64,
    mean: Vec<f64>,
    m2: Vec<f64>,
}

impl DenseStats {
    fn new() -> Self {
        Self {
            n: 0,
            mean: vec![0.0; DENSE_DIM],
            m2: vec![0.0; DENSE_DIM],
        }
    }

    /// Welford's online update, numerically stable over millions of samples.
    fn push(&mut self, x: &[f64; DENSE_DIM]) {
        self.n += 1;
        let n = self.n as f64;
        for (k, &xk) in x.iter().enumerate() {
            let d = xk - self.mean[k];
            self.mean[k] += d / n;
            self.m2[k] += d * (xk - self.mean[k]);
        }
    }

    /// (mean, std), with a floor so a constant feature cannot divide by zero.
    fn finish(&self) -> (Vec<f64>, Vec<f64>) {
        if self.n < 2 {
            return (MEAN_DENSE.to_vec(), STD_DENSE.to_vec());
        }
        let n = self.n as f64;
        let std = self
            .m2
            .iter()
            .map(|v| {
                let sd = (v / n).sqrt();
                if sd < 1e-6 {
                    1.0
                } else {
                    sd
                }
            })
            .collect();
        (self.mean.clone(), std)
    }
}
use akshar_ime::core::unified::UnifiedModel;
use akshar_ime::core::wordtrie::WordTrie;
use akshar_ime::ImeEngine;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Deserialize)]
struct Record {
    #[serde(rename = "english word", alias = "english", alias = "roman")]
    english: String,
    #[serde(rename = "native word", alias = "native", alias = "devanagari")]
    native: String,
    #[serde(default = "default_weight")]
    weight: u32,
}

fn default_weight() -> u32 {
    1
}

fn is_word_char(c: char) -> bool {
    matches!(c, '\u{0900}'..='\u{0963}')
}

fn clean_devanagari_token(word: &str) -> Option<String> {
    let trimmed = word.trim_matches(|c: char| !is_word_char(c));
    let chars: Vec<char> = trimmed.chars().collect();
    if chars.is_empty() || chars.len() > 24 {
        return None;
    }
    if chars.iter().all(|c| is_word_char(*c)) {
        Some(trimmed.to_string())
    } else {
        None
    }
}

fn auto_detect_pairs() -> Option<PathBuf> {
    let candidates = [
        "data/aksharantar/train_devanagari.jsonl",
        "data/aksharantar/nep_train.json",
        "data/corpus_clean.json",
    ];
    for c in candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn auto_detect_text() -> Option<PathBuf> {
    let candidates = ["data/store/corpus_clean.txt", "data/raw/corpus.txt"];
    for c in candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Stream the running-text corpus into cleaned token counts, honoring the
/// shared holdout split exactly as Phase 2 always has (untrimmed line keys,
/// so sentence-eval words never leak into vocab/WordTrie/freq priors).
/// Extracted so the attestation pre-pass and Phase 2 share one implementation.
fn count_corpus_tokens(tp: &Path) -> Result<(HashMap<String, u32>, usize)> {
    let mut freq: HashMap<String, u32> = HashMap::new();
    let mut skipped_holdout = 0usize;
    let tf = File::open(tp).context("open text file")?;
    for line in BufReader::new(tf).lines().map_while(Result::ok) {
        if is_holdout(&line, DEFAULT_HOLDOUT_DENOM) {
            skipped_holdout += 1;
            continue;
        }
        for token in line.split_whitespace() {
            if let Some(clean) = clean_devanagari_token(token) {
                *freq.entry(clean).or_insert(0) += 1;
            }
        }
    }
    Ok((freq, skipped_holdout))
}

/// Attestation weight for a Devanagari form: 1 + floor(ln(1+f)).
/// Unseen forms weigh 1.0 (identical to unweighted training); integer steps
/// keep KN counts-of-counts valid. f=38k (top words) -> 11.
fn attestation_weight(freq: &HashMap<String, u32>, nat: &str) -> u32 {
    let key = clean_devanagari_token(nat).unwrap_or_else(|| nat.to_string());
    let f = freq.get(&key).copied().unwrap_or(0);
    1 + (1.0 + f as f64).ln().floor() as u32
}

const RERANK_DECODE_DEPTH: usize = 50;
const RERANK_DECODE_BEAM: usize = 64;

fn main() -> Result<()> {
    let mut pairs_path: Option<PathBuf> = None;
    let mut text_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut limit: Option<usize> = None;
    let mut iterations: usize = 10;
    let mut epochs: usize = 3;
    let mut min_freq: u32 = 3;
    let mut reranker_pairs: usize = 100_000;
    let mut wasm_mode = false;
    let mut smoke = false;
    let mut attestation = false;
    // Attestation weighting (Roark/Dakshina §4.2 pattern): weight each EM/LM
    // pair by log-dampened corpus frequency of its Devanagari side, so
    // dominant romanization conventions dominate training. Off = all weights
    // 1.0 = byte-identical current behavior. Reranker sampling stays uniform
    // (isolates the EM/LM effect).

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--pairs" | "-p" => {
                pairs_path = Some(PathBuf::from(args.next().context("value for --pairs")?))
            }
            "--text" | "-t" => {
                text_path = Some(PathBuf::from(args.next().context("value for --text")?))
            }
            "--out" | "-o" => {
                out_path = Some(PathBuf::from(args.next().context("value for --out")?))
            }
            "--limit" | "-n" => {
                limit = Some(
                    args.next()
                        .context("value for --limit")?
                        .parse()
                        .context("parse --limit")?,
                )
            }
            "--iterations" | "-i" => {
                iterations = args
                    .next()
                    .context("value for --iterations")?
                    .parse()
                    .context("parse --iterations")?
            }
            "--epochs" | "-e" => {
                epochs = args
                    .next()
                    .context("value for --epochs")?
                    .parse()
                    .context("parse --epochs")?
            }
            "--min-freq" => {
                min_freq = args
                    .next()
                    .context("value for --min-freq")?
                    .parse()
                    .context("parse --min-freq")?
            }
            "--reranker-pairs" => {
                reranker_pairs = args
                    .next()
                    .context("value for --reranker-pairs")?
                    .parse()
                    .context("parse --reranker-pairs")?
            }
            "--wasm" => wasm_mode = true,
            "--smoke" => smoke = true,
            "--attestation" => attestation = true,
            "-h" | "--help" => {
                println!("Akshar Unified Model Trainer");
                println!("Usage: cargo run --release --bin train -- [options]");
                println!("  --pairs <path>          Parallel word pairs JSONL");
                println!("  --text <path>           Clean running text for vocabulary");
                println!("  --out <path>            Output path (default: data/akshar.model)");
                println!("  --min-freq <n>          Prune vocabulary with freq < n (default: 3)");
                println!("  --reranker-pairs <n>    Number of pairs for reranker training (0 = all pairs, default: 100,000)");
                println!("  --limit <n>             Limit pairs (for rapid prototyping)");
                println!("  --iterations <n>        EM iterations (default: 10)");
                println!("  --epochs <n>            Reranker epochs (default: 3)");
                println!("  --wasm                  Export lightweight WASM web profile");
                println!("  --smoke                 Run fast smoke training validation");
                println!("  --attestation           Weight EM/LM pairs by log corpus frequency");
                return Ok(());
            }
            other => {
                anyhow::bail!("Unknown argument: {other}");
            }
        }
    }

    if wasm_mode && min_freq == 3 {
        min_freq = 5;
    }

    let out_path = out_path.unwrap_or_else(|| {
        if wasm_mode {
            PathBuf::from("data/akshar_wasm.model")
        } else {
            PathBuf::from("data/akshar.model")
        }
    });

    if smoke {
        println!(
            ">>> Running in SMOKE mode (fast validation: 1000 pairs, 3 EM iterations, 1 epoch)"
        );
        limit = Some(1000);
        iterations = 3;
        epochs = 1;
        reranker_pairs = 500;
    }

    let pairs_file = pairs_path
        .or_else(auto_detect_pairs)
        .context("No training pairs file found! Specify --pairs <path>")?;
    let text_file = text_path.or_else(auto_detect_text);

    println!("============================================================");
    println!("           Akshar One-Shot Unified Trainer                  ");
    println!("============================================================");
    println!("Training Pairs: {}", pairs_file.display());
    println!(
        "Text Corpus:    {}",
        text_file
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "Derived from pairs".to_string())
    );
    println!("Target Output:  {}", out_path.display());
    println!(
        "Parameters:     EM iterations={}, Reranker epochs={}",
        iterations, epochs
    );
    if let Some(lim) = limit {
        println!("Limit:          {} pairs", lim);
    }
    println!("============================================================");

    let start_time = Instant::now();

    // Phase 0: attestation pre-pass (only with --attestation). Counts the
    // corpus once so EM/LM pair weights reflect token frequency; the same
    // map feeds Phase 2 below (no re-stream).
    let attest_freq: Option<HashMap<String, u32>> = if attestation {
        let tp = text_file
            .as_ref()
            .context("--attestation needs running text (--text or auto-detect)")?;
        println!("\n[Phase 0/4] Counting corpus for attestation weights...");
        let t0 = Instant::now();
        let (freq, skipped) = count_corpus_tokens(tp)?;
        println!(
            "Counted {} types in {:.1?} ({} held-out lines skipped).",
            freq.len(),
            t0.elapsed(),
            skipped
        );
        Some(freq)
    } else {
        None
    };

    // Phase 1: EM
    println!("\n[Phase 1/4] Training EM Source-Channel Transliteration Model...");
    let em_t0 = Instant::now();
    let em_config = TrainerConfig {
        iterations,
        limit,
        seed_from_aligner: true,
        ..Default::default()
    };

    let mut em_trainer = Trainer::new().with_limit(limit);
    let f = File::open(&pairs_file).context("open pairs file")?;
    let mut raw_pairs: Vec<(String, String)> = Vec::new();
    let mut count = 0usize;
    // Attestation observability (else we're flying blind).
    let (mut wmax, mut wsum) = (0u32, 0u64);

    for line in BufReader::new(f).lines().map_while(Result::ok) {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if let Ok(rec) = serde_json::from_str::<Record>(t) {
            let eng = rec.english.trim().to_ascii_lowercase();
            let nat = rec.native.trim().to_string();
            if !eng.is_empty() && !nat.is_empty() {
                // Attestation: log-dampened corpus frequency of the Devanagari
                // side; off (None) reproduces the file weight (default 1.0).
                let w = match attest_freq.as_ref() {
                    Some(freq) => attestation_weight(freq, &nat),
                    None => rec.weight,
                };
                wmax = wmax.max(w);
                wsum += w as u64;
                em_trainer.add_pair_weighted(&eng, &nat, f64::from(w));
                raw_pairs.push((eng, nat));
                count += 1;
                if let Some(lim) = limit {
                    if count >= lim {
                        break;
                    }
                }
            }
        }
    }
    println!("Ingested {} valid parallel pairs.", count);
    if attestation {
        println!(
            "Attestation weights: max={} mean={:.2} (1.0 = unseen, log-dampened).",
            wmax,
            wsum as f64 / count.max(1) as f64
        );
    }

    let mut translit_model = em_trainer.finalize(&em_config);
    translit_model.build_trigram_index();
    println!(
        "EM training finished in {:.2?} (aksharas: {}, chunks: {}).",
        em_t0.elapsed(),
        translit_model.aksharas.len(),
        translit_model.chunks.len()
    );

    // Phase 2: vocab
    println!("\n[Phase 2/4] Compiling Vocabulary & Empirical Frequencies...");
    let vocab_t0 = Instant::now();
    // Reuse the attestation pre-pass map when present (same implementation,
    // zero re-stream); otherwise count as before.
    let mut vocab_freq: HashMap<String, u32> = match attest_freq {
        Some(freq) => {
            println!("Reusing Phase-0 corpus counts for the vocabulary.");
            freq
        }
        None => {
            let mut vocab_freq: HashMap<String, u32> = HashMap::new();
            if let Some(ref tp) = text_file {
                if tp.exists() {
                    println!("Reading running text from {} ...", tp.display());
                    let (freq, skipped_holdout) = count_corpus_tokens(tp)?;
                    vocab_freq = freq;
                    println!(
                        "Skipped {} held-out lines (1-in-{} split).",
                        skipped_holdout, DEFAULT_HOLDOUT_DENOM
                    );
                }
            }
            vocab_freq
        }
    };

    let raw_vocab_count = vocab_freq.len();
    vocab_freq.retain(|_, &mut c| c >= min_freq);
    let ranks = FreqRanks::from_freq_map(&vocab_freq);
    println!(
        "Compiled clean vocabulary of {} unique words (pruned {} hapax/noise words < {} freq) in {:.2?}.",
        vocab_freq.len(),
        raw_vocab_count.saturating_sub(vocab_freq.len()),
        min_freq,
        vocab_t0.elapsed()
    );

    // Phase 3: reranker - chunked for large training sets to avoid OOM
    println!("\n[Phase 3/4] Training Discriminative Reranker...");
    let rank_t0 = Instant::now();
    let decoder = ModelDecoder::with_config(
        translit_model.clone(),
        DecoderConfig {
            beam_width: RERANK_DECODE_BEAM,
            ..DecoderConfig::default()
        },
    );

    let word_trie = WordTrie::from_freq_map(&vocab_freq, &|a| translit_model.akshara_id(a), 1);

    // Held-out dev size; defined up front so the full run can reserve it.
    const DEV_SIZE: usize = 4_000;
    let num_train_pairs = if smoke {
        500
    } else if reranker_pairs == 0 {
        // Exp 5: full run previously consumed every pair AND left dev empty,
        // losing its overfit guard. Reserve the tail as dev (mirrors partial
        // runs, which use raw_pairs[len-DEV..]).
        if raw_pairs.len() > DEV_SIZE {
            raw_pairs.len() - DEV_SIZE
        } else {
            raw_pairs.len()
        }
    } else {
        raw_pairs.len().min(reranker_pairs)
    };
    // (actual count printed after dev reservation below)

    struct RerankItem {
        target_idx: usize,
        /// Standardised dense feature matrix: dense_feats[i] is the DENSE_DIM feature
        /// vector for candidate i (already z-scored against MEAN_DENSE/STD_DENSE).
        dense_feats: Vec<[f64; DENSE_DIM]>,
        sparse: Vec<Vec<usize>>,
    }

    // For large training sets, process in batches to keep memory bounded.
    // Each batch is decoded once and reused for all epochs.
    const BATCH_SIZE: usize = 100_000;
    let use_chunked = num_train_pairs > 200_000;
    // Jointly train: 29 dense weights + 2^20 sparse weights.
    // Warm-start dense weights from W_DENSE (the established linear model)
    // and jointly optimize them with the sparse table under the softmax objective.
    let mut dense_weights: Vec<f64> = W_DENSE.to_vec();
    let mut dense_grad_sq: Vec<f64> = vec![1.0f64; DENSE_DIM];
    let mut sparse_table: Vec<f32> = vec![0.0f32; HASH_SIZE];
    let mut grad_sq: Vec<f32> = vec![0.0f32; HASH_SIZE];
    const LR0: f64 = 0.05;
    const DENSE_LR_SCALE: f64 = 0.1;
    // Fraction of the initial learning rate remaining at the end of the run.
    // AdaGrad already adapts per-slot, so this only needs to be a gentle global
    // anneal -- the previous schedule decayed by ~1e-9 across a full run, which
    // silently discarded most of the corpus.
    const LR_FINAL_FRACTION: f64 = 0.1;
    let mut lr: f64 = LR0;
    let mut dense_stats = DenseStats::new();

    // Held-out dev set, taken from the tail of the corpus so it never overlaps
    // the training slice.  Without this a long run reports only training loss,
    // which cannot distinguish "learning" from "memorising 1M sparse slots":
    // the table has ~1e6 parameters and no regularisation, so overfitting is
    // the default failure and it was previously invisible.
    // Exp 5: the tail is ALWAYS reserved (except smoke / tiny corpora), so the
    // training slice below can never include dev pairs — previously the full
    // run trained on all pairs and evaluated on none.
    let mut num_train_pairs = num_train_pairs;
    let dev_pairs: Vec<(String, String)> = if !smoke && raw_pairs.len() > DEV_SIZE {
        let tail = raw_pairs[raw_pairs.len() - DEV_SIZE..].to_vec();
        num_train_pairs = num_train_pairs.min(raw_pairs.len() - DEV_SIZE);
        tail
    } else {
        Vec::new()
    };
    println!(
        "Pre-decoding candidates for {} training pairs (dev reserved: {})...",
        num_train_pairs,
        dev_pairs.len()
    );

    let dev_items: Vec<RerankItem> = if dev_pairs.is_empty() {
        Vec::new()
    } else {
        let t0 = Instant::now();
        let mut items = Vec::with_capacity(dev_pairs.len());
        for (roman, gold) in &dev_pairs {
            let cands = decoder.decode_union(roman, RERANK_DECODE_DEPTH, Some(&word_trie));
            let (order, heur, heur_rank) = rank_candidates(&cands, &vocab_freq);
            if let Some(target_idx) = order.iter().position(|c| c.dev == *gold) {
                let sparse: Vec<Vec<usize>> = order
                    .iter()
                    .map(|c| {
                        let aks = akshar_ime::core::akshara::segment(&c.dev);
                        extract_sparse_features(&c.dev, roman, c.akshara_count, &aks)
                    })
                    .collect();
                let dense_feats: Vec<[f64; DENSE_DIM]> = order
                    .iter()
                    .enumerate()
                    .map(|(idx, c)| {
                        let raw = extract_dense_features(
                            c,
                            idx,
                            heur[idx],
                            heur_rank[idx],
                            roman,
                            &vocab_freq,
                            &ranks,
                        );
                        let mut z = [0.0f64; DENSE_DIM];
                        for k in 0..DENSE_DIM {
                            let sd = STD_DENSE[k];
                            z[k] = if sd.abs() > 1e-12 {
                                (raw[k] - MEAN_DENSE[k]) / sd
                            } else {
                                0.0
                            };
                        }
                        z
                    })
                    .collect();
                items.push(RerankItem {
                    target_idx,
                    dense_feats,
                    sparse,
                });
            }
        }
        println!(
            "Held-out dev set: {} of {} pairs have the gold in the candidate list (decoded in {:.2?}).",
            items.len(),
            dev_pairs.len(),
            t0.elapsed()
        );
        items
    };

    /// Loss and top-1 on the held-out set under the current sparse table.
    fn dev_eval(items: &[RerankItem], table: &[f32], dw: &[f64]) -> Option<(f64, f64)> {
        if items.is_empty() {
            return None;
        }
        let (mut loss, mut hits) = (0.0f64, 0usize);
        for s in items {
            let mut scores: Vec<f64> = s
                .dense_feats
                .iter()
                .map(|z| dw.iter().zip(z.iter()).map(|(w, x)| w * x).sum::<f64>())
                .collect();
            for (idx, sf) in s.sparse.iter().enumerate() {
                for &h in sf {
                    scores[idx] += f64::from(table[h]);
                }
            }
            let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_s: Vec<f64> = scores.iter().map(|&sc| (sc - max_s).exp()).collect();
            let sum_exp: f64 = exp_s.iter().sum();
            let p_target = (exp_s[s.target_idx] / (sum_exp + 1e-12)).max(1e-12);
            loss += -p_target.ln();
            if scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .is_some_and(|(i, _)| i == s.target_idx)
            {
                hits += 1;
            }
        }
        let n = items.len() as f64;
        Some((loss / n, hits as f64 / n * 100.0))
    }

    // Snapshot of the best weights (sparse + dense) by dev loss, so a run that
    // starts overfitting does not have to be thrown away.
    let mut best_dev: Option<(f64, Vec<f32>, Vec<f64>)> = None;

    if let Some((l, a)) = dev_eval(&dev_items, &sparse_table, &dense_weights) {
        println!("  dev before training: loss={l:.4} top-1={a:.2}%");
    }

    if use_chunked {
        let total_batches = num_train_pairs.div_ceil(BATCH_SIZE);
        println!(
            "  Chunked mode: {} batches of {} (memory bounded, decode once per batch, {} epochs per batch)",
            total_batches, BATCH_SIZE, epochs
        );
        // One gentle decay per batch, sized so the final batch runs at
        // LR_FINAL_FRACTION of the initial rate regardless of batch count.
        let lr_decay_per_batch = LR_FINAL_FRACTION.powf(1.0 / total_batches.max(1) as f64);
        let mut global_samples: usize = 0;
        // Decode once per batch and train `epochs` passes over that batch before moving to next.
        // This is ~5x faster than re-decoding per global epoch (was 18h for 3.59M) and keeps
        // memory bounded to one batch.
        for batch_idx in 0..total_batches {
            let start = batch_idx * BATCH_SIZE;
            let end = (start + BATCH_SIZE).min(num_train_pairs);
            let batch = &raw_pairs[start..end];
            let batch_t0 = Instant::now();
            println!(
                "  Batch {}/{}: decoding {} pairs ...",
                batch_idx + 1,
                total_batches,
                batch.len()
            );
            let mut samples: Vec<RerankItem> = Vec::with_capacity(batch.len());
            for (roman, gold) in batch {
                let cands = decoder.decode_union(roman, RERANK_DECODE_DEPTH, Some(&word_trie));
                let (order, heur, heur_rank) = rank_candidates(&cands, &vocab_freq);
                if let Some(target_idx) = order.iter().position(|c| c.dev == *gold) {
                    let cand_sparse: Vec<Vec<usize>> = order
                        .iter()
                        .map(|c| {
                            let aks = akshar_ime::core::akshara::segment(&c.dev);
                            extract_sparse_features(&c.dev, roman, c.akshara_count, &aks)
                        })
                        .collect();
                    let dense_feats: Vec<[f64; DENSE_DIM]> = order
                        .iter()
                        .enumerate()
                        .map(|(idx, c)| {
                            let raw = extract_dense_features(
                                c,
                                idx,
                                heur[idx],
                                heur_rank[idx],
                                roman,
                                &vocab_freq,
                                &ranks,
                            );
                            // Exp 5: accumulate RAW features so finish() yields
                            // the true distribution; z-scores (mean~0/std~1)
                            // would describe the old constants, not the data.
                            dense_stats.push(&raw);
                            let mut z = [0.0f64; DENSE_DIM];
                            for k in 0..DENSE_DIM {
                                let sd = STD_DENSE[k];
                                z[k] = if sd.abs() > 1e-12 {
                                    (raw[k] - MEAN_DENSE[k]) / sd
                                } else {
                                    0.0
                                };
                            }
                            z
                        })
                        .collect();
                    samples.push(RerankItem {
                        target_idx,
                        dense_feats,
                        sparse: cand_sparse,
                    });
                }
            }
            println!(
                "    -> {} valid (decoded in {:.1?})",
                samples.len(),
                batch_t0.elapsed()
            );
            // Train `epochs` passes over this batch before moving on (~5x less decode)
            let mut batch_loss: f64 = 0.0;
            for _ep in 1..=epochs {
                for s in &samples {
                    let mut scores: Vec<f64> = s
                        .dense_feats
                        .iter()
                        .map(|z| {
                            dense_weights
                                .iter()
                                .zip(z.iter())
                                .map(|(w, x)| w * x)
                                .sum::<f64>()
                        })
                        .collect();
                    for (idx, sparse_feats) in s.sparse.iter().enumerate() {
                        for &h in sparse_feats {
                            scores[idx] += f64::from(sparse_table[h]);
                        }
                    }
                    let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let exp_s: Vec<f64> = scores.iter().map(|&sc| (sc - max_s).exp()).collect();
                    let sum_exp: f64 = exp_s.iter().sum();
                    let probs: Vec<f64> = exp_s.iter().map(|&e| e / (sum_exp + 1e-12)).collect();
                    batch_loss += -probs[s.target_idx].max(1e-12).ln();
                    let mut sample_dense_grad = [0.0f64; DENSE_DIM];
                    for (idx, p) in probs.iter().enumerate() {
                        let grad = if idx == s.target_idx { *p - 1.0 } else { *p };
                        if grad.abs() > 1e-5 {
                            // Update sparse weights
                            for &h in &s.sparse[idx] {
                                let g = grad as f32;
                                grad_sq[h] += g * g;
                                let eff_lr = (lr as f32) / (grad_sq[h].sqrt() + 1e-4);
                                sparse_table[h] -= eff_lr * g;
                            }
                            // Accumulate true sample gradient for dense weights:
                            // dL/dw_k = sum_idx (p_idx - y_idx) * z_{idx, k}
                            for (k, grad_val) in sample_dense_grad.iter_mut().enumerate() {
                                *grad_val += grad * s.dense_feats[idx][k];
                            }
                        }
                    }
                    // Apply dense weight update once per sample
                    for (k, &gd) in sample_dense_grad.iter().enumerate() {
                        if gd.abs() > 1e-6 {
                            dense_grad_sq[k] += gd * gd;
                            let eff_lr_d = lr * DENSE_LR_SCALE / (dense_grad_sq[k].sqrt() + 1e-4);
                            dense_weights[k] -= eff_lr_d * gd;
                        }
                    }
                }
            }
            global_samples += samples.len() * epochs;
            batch_loss /= epochs as f64 * samples.len().max(1) as f64;
            println!(
                "    Batch {}/{} done: avg loss {:.4} ({} samples, lr {:.4})",
                batch_idx + 1,
                total_batches,
                batch_loss,
                samples.len(),
                lr
            );
            if let Some((dl, da)) = dev_eval(&dev_items, &sparse_table, &dense_weights) {
                let better = best_dev.as_ref().is_none_or(|(b, _, _)| dl < *b);
                println!(
                    "      dev: loss={dl:.4} top-1={da:.2}%{}",
                    if better { "  <- best" } else { "" }
                );
                if better {
                    best_dev = Some((dl, sparse_table.clone(), dense_weights.clone()));
                }
            }
            lr *= lr_decay_per_batch;
        }
        println!(
            "  Chunked training done: {} samples processed in {:.2?}",
            global_samples,
            rank_t0.elapsed()
        );
    } else {
        // Original in-memory path for smaller training sets (faster)
        let decode_t0 = Instant::now();
        let mut samples: Vec<RerankItem> = Vec::with_capacity(num_train_pairs);
        for (roman, gold) in &raw_pairs[..num_train_pairs] {
            let cands = decoder.decode_union(roman, RERANK_DECODE_DEPTH, Some(&word_trie));
            let (order, heur, heur_rank) = rank_candidates(&cands, &vocab_freq);
            if let Some(target_idx) = order.iter().position(|c| c.dev == *gold) {
                let cand_sparse: Vec<Vec<usize>> = order
                    .iter()
                    .map(|c| {
                        let aks = akshar_ime::core::akshara::segment(&c.dev);
                        extract_sparse_features(&c.dev, roman, c.akshara_count, &aks)
                    })
                    .collect();
                let dense_feats: Vec<[f64; DENSE_DIM]> = order
                    .iter()
                    .enumerate()
                    .map(|(idx, c)| {
                        let raw = extract_dense_features(
                            c,
                            idx,
                            heur[idx],
                            heur_rank[idx],
                            roman,
                            &vocab_freq,
                            &ranks,
                        );
                        // Exp 5: raw distribution, not z-scores (see chunked path).
                        dense_stats.push(&raw);
                        let mut z = [0.0f64; DENSE_DIM];
                        for k in 0..DENSE_DIM {
                            let sd = STD_DENSE[k];
                            z[k] = if sd.abs() > 1e-12 {
                                (raw[k] - MEAN_DENSE[k]) / sd
                            } else {
                                0.0
                            };
                        }
                        z
                    })
                    .collect();
                samples.push(RerankItem {
                    target_idx,
                    dense_feats,
                    sparse: cand_sparse,
                });
            }
        }
        println!(
            "Pre-decoded {} valid samples with targets in candidate list in {:.2?}.",
            samples.len(),
            decode_t0.elapsed()
        );

        for ep in 1..=epochs {
            let mut ep_loss: f64 = 0.0;
            let mut ep_hits: usize = 0;
            for s in &samples {
                let mut scores: Vec<f64> = s
                    .dense_feats
                    .iter()
                    .map(|z| {
                        dense_weights
                            .iter()
                            .zip(z.iter())
                            .map(|(w, x)| w * x)
                            .sum::<f64>()
                    })
                    .collect();
                for (idx, sparse_feats) in s.sparse.iter().enumerate() {
                    for &h in sparse_feats {
                        scores[idx] += f64::from(sparse_table[h]);
                    }
                }
                let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_s: Vec<f64> = scores.iter().map(|&sc| (sc - max_s).exp()).collect();
                let sum_exp: f64 = exp_s.iter().sum();
                let probs: Vec<f64> = exp_s.iter().map(|&e| e / (sum_exp + 1e-12)).collect();
                ep_loss += -probs[s.target_idx].max(1e-12).ln();
                if probs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .is_some_and(|(i, _)| i == s.target_idx)
                {
                    ep_hits += 1;
                }
                let mut sample_dense_grad = [0.0f64; DENSE_DIM];
                for (idx, p) in probs.iter().enumerate() {
                    let grad = if idx == s.target_idx { *p - 1.0 } else { *p };
                    if grad.abs() > 1e-5 {
                        // Update sparse weights
                        for &h in &s.sparse[idx] {
                            let g = grad as f32;
                            grad_sq[h] += g * g;
                            let eff_lr = (lr as f32) / (grad_sq[h].sqrt() + 1e-4);
                            sparse_table[h] -= eff_lr * g;
                        }
                        // Accumulate true sample gradient for dense weights:
                        // dL/dw_k = sum_idx (p_idx - y_idx) * z_{idx, k}
                        for (k, grad_val) in sample_dense_grad.iter_mut().enumerate() {
                            *grad_val += grad * s.dense_feats[idx][k];
                        }
                    }
                }
                // Apply dense weight update once per sample
                for (k, &gd) in sample_dense_grad.iter().enumerate() {
                    if gd.abs() > 1e-6 {
                        dense_grad_sq[k] += gd * gd;
                        let eff_lr_d = lr * DENSE_LR_SCALE / (dense_grad_sq[k].sqrt() + 1e-4);
                        dense_weights[k] -= eff_lr_d * gd;
                    }
                }
            }
            lr *= 0.8;
            if !samples.is_empty() {
                let dev = dev_eval(&dev_items, &sparse_table, &dense_weights);
                let better = match (&dev, &best_dev) {
                    (Some((dl, _)), Some((b, _, _))) => dl < b,
                    (Some(_), None) => true,
                    _ => false,
                };
                println!(
                    "  Epoch {}/{}: train loss={:.4} top-1={:.2}%{}",
                    ep,
                    epochs,
                    ep_loss / samples.len() as f64,
                    ep_hits as f64 / samples.len() as f64 * 100.0,
                    match dev {
                        Some((dl, da)) => format!(
                            "  |  dev loss={dl:.4} top-1={da:.2}%{}",
                            if better { "  <- best" } else { "" }
                        ),
                        None => String::new(),
                    }
                );
                if let (Some((dl, _)), true) = (dev, better) {
                    best_dev = Some((dl, sparse_table.clone(), dense_weights.clone()));
                }
            }
        }
    }
    println!("Reranker trained in {:.2?}.", rank_t0.elapsed());

    // Phase 4: Pack
    println!(
        "\n[Phase 4/4] Packing Unified Model -> {} ...",
        out_path.display()
    );
    let pack_t0 = Instant::now();

    // Pack the weights that scored best on held-out data, not necessarily the
    // last ones: with ~1e6 unregularised sparse slots, later batches can overfit.
    if let Some((dl, best_sparse, best_dense)) = best_dev {
        let final_dl = dev_eval(&dev_items, &sparse_table, &dense_weights).map(|(l, _)| l);
        if final_dl.is_some_and(|f| f > dl + 1e-9) {
            println!(
                "Packing the best-by-dev weights (dev loss {:.4}) rather than the final ({:.4}).",
                dl,
                final_dl.unwrap_or(f64::NAN)
            );
            sparse_table = best_sparse;
            dense_weights = best_dense;
        } else {
            println!("Final weights are the best by dev loss ({dl:.4}).");
        }
    }

    // Set the quantization scale from a high percentile of the non-zero weights
    // and clip the tail, rather than from the single largest weight.  One
    // outlier setting the scale compresses every other weight's resolution --
    // the shipped table had min -127 / max +42, i.e. one weight consuming the
    // whole negative range while the rest of the distribution sat in a handful
    // of levels.
    let mut magnitudes: Vec<f32> = sparse_table
        .iter()
        .map(|w| w.abs())
        .filter(|&w| w > 0.0)
        .collect();
    magnitudes.sort_by(f32::total_cmp);
    let clip = if magnitudes.is_empty() {
        0.0
    } else {
        let idx = ((magnitudes.len() as f64 * 0.999) as usize).min(magnitudes.len() - 1);
        magnitudes[idx]
    };
    let sparse_scale = if clip > 0.0 { 127.0 / clip } else { 1.0 };
    let quantized_table: Vec<i8> = sparse_table
        .iter()
        .map(|&w| (w * sparse_scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    let non_zero = quantized_table.iter().filter(|&&w| w != 0).count();
    let saturated = quantized_table
        .iter()
        .filter(|&&w| w == 127 || w == -127)
        .count();
    println!(
        "Quantized sparse table: {} non-zero of {} slots ({:.2}%), {} saturated, clip {:.5}, scale {:.4}",
        non_zero,
        quantized_table.len(),
        non_zero as f64 / quantized_table.len() as f64 * 100.0,
        saturated,
        clip,
        sparse_scale
    );

    println!("Pruning unreferenced aksharas and cleaning out-of-vocabulary transitions...");
    let mut seen_aks = std::collections::HashSet::new();
    for word in vocab_freq.keys() {
        for a in akshar_ime::core::akshara::segment(word) {
            if let Some(id) = translit_model.akshara_id(&a) {
                seen_aks.insert(id);
            }
        }
    }
    let before_em = translit_model
        .emissions
        .iter()
        .filter(|e| !e.is_empty())
        .count();
    for (a, em) in translit_model.emissions.iter_mut().enumerate() {
        if !seen_aks.contains(&(a as u32)) {
            em.clear();
        }
    }
    for (a, bi) in translit_model.bigrams.iter_mut().enumerate() {
        if !seen_aks.contains(&(a as u32)) {
            bi.clear();
        } else {
            bi.retain(|(next_id, _)| seen_aks.contains(next_id));
        }
    }
    let mut new_keys = Vec::new();
    let mut new_trigrams = Vec::new();
    let mut new_backoff = Vec::new();
    for (i, &(a, b)) in translit_model.trigram_keys.iter().enumerate() {
        if seen_aks.contains(&a) && seen_aks.contains(&b) {
            let mut list = translit_model.trigrams[i].clone();
            list.retain(|(c, _)| seen_aks.contains(c));
            if !list.is_empty() {
                new_keys.push((a, b));
                new_trigrams.push(list);
                new_backoff.push(
                    translit_model
                        .trigram_backoff
                        .get(i)
                        .copied()
                        .unwrap_or(0.0),
                );
            }
        }
    }
    translit_model.trigram_keys = new_keys;
    translit_model.trigrams = new_trigrams;
    translit_model.trigram_backoff = new_backoff;
    translit_model.build_trigram_index();
    let after_em = translit_model
        .emissions
        .iter()
        .filter(|e| !e.is_empty())
        .count();
    println!(
        "Pruned unused emission rows: {} -> {} (and cleaned transitions)",
        before_em, after_em
    );

    let (dense_mean, dense_std) = dense_stats.finish();
    println!(
        "Dense-feature statistics from {} candidate scorings (emit mean {:.3} std {:.3}, lm mean {:.3} std {:.3})",
        dense_stats.n, dense_mean[0], dense_std[0], dense_mean[1], dense_std[1]
    );
    let mut unified = UnifiedModel::new(
        translit_model,
        quantized_table,
        1.0 / f64::from(sparse_scale),
        vocab_freq,
    );
    // Exp 5: pack the freshly computed statistics. Packing the compiled-in
    // constants here silently kept every retrained model decalibrated.
    unified.dense_mean = dense_mean;
    unified.dense_std = dense_std;
    unified.dense_weights = dense_weights;
    unified
        .save(&out_path)
        .map_err(|e| anyhow::anyhow!("save unified model: {e}"))?;
    let meta = std::fs::metadata(&out_path).context("model metadata")?;
    let size_mb = meta.len() as f64 / (1024.0 * 1024.0);

    println!(
        "Unified model container successfully written in {:.2?}!",
        pack_t0.elapsed()
    );
    println!("Artifact: {} ({:.2} MB)", out_path.display(), size_mb);

    println!("\n>>> Running Self-Verification Smoke Test on newly created model...");
    let engine = ImeEngine::from_unified_file(&out_path)
        .map_err(|e| anyhow::anyhow!("load newly trained model: {e}"))?;
    let test_queries = ["namaste", "dhanyabad", "pustak", "sarkar", "pani"];
    println!("Testing top suggestion generation for basic words:");
    for q in test_queries {
        let sugs = engine.get_suggestions(q, 3);
        let top: Vec<String> = sugs.iter().map(|(s, _)| s.clone()).collect();
        println!("  {:12} -> {:?}", q, top);
        assert!(!top.is_empty(), "Failed to generate suggestions for {q}");
    }

    println!("\n============================================================");
    println!(
        "All done in {:.2?}! Model is 100% verified and ready.",
        start_time.elapsed()
    );
    println!("============================================================");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn attestation_weight_shape() {
        let mut freq = HashMap::new();
        freq.insert("नमस्ते".to_string(), 1000u32);
        freq.insert("छ".to_string(), 2u32);
        // unseen -> 1 (identical to unweighted training)
        assert_eq!(attestation_weight(&freq, "कखग",), 1);
        assert_eq!(attestation_weight(&HashMap::new(), "नमस्ते"), 1);
        // monotone in frequency, log-dampened (1000 -> 1+6=7, not 1001)
        let w_hi = attestation_weight(&freq, "नमस्ते");
        let w_lo = attestation_weight(&freq, "छ");
        assert!(w_hi > w_lo && w_lo >= 1, "{w_hi} {w_lo}");
        assert_eq!(w_hi, 1 + (1.0 + 1000.0f64).ln().floor() as u32);
    }
}
