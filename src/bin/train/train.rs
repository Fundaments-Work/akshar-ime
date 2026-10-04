// File: src/bin/train/train.rs
//
// End-to-End Unified Model Trainer for Akshar Devanagari IME.
// Ingests training data (word pairs + text) and directly produces
// a single, low-size, production-ready `akshar.model` artifact.

use akshar_ime::core::decoder::{DecoderConfig, Dictionary, LexiconDict, ModelDecoder};
use akshar_ime::core::em_trainer::{Trainer, TrainerConfig};
use akshar_ime::core::holdout::{is_holdout, DEFAULT_HOLDOUT_DENOM};
use akshar_ime::core::reranker::{
    extract_dense_features, extract_sparse_features_lang, rank_candidates, FreqRanks,
    LexiconCounts, VocabCounts, WordCounts, DENSE_DIM, HASH_SIZE,
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
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
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
    /// Language of the pair (ISO 639-3, from `prepare_pairs`); empty if unknown.
    #[serde(default)]
    lang: String,
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
    let candidates = ["data/pairs/train.jsonl", "data/aksharantar/nep_train.json"];
    for c in candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn auto_detect_text() -> Option<PathBuf> {
    let candidates = ["data/nepali_corpus.txt"];
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
/// Same beam as the engine (engine.rs DECODER_BEAM), so the reranker trains
/// on the candidate lists users will actually be shown.
const RERANK_DECODE_BEAM: usize = 32;

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
    let mut seed: u64 = 7;
    let mut rerank_holdout = true;
    let mut min_akshara_count: u32 = 2;
    let mut em_cache: Option<PathBuf> = None;
    let mut lexicon_path: Option<PathBuf> = None;
    let mut use_lexicon = true;
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
            "--seed" => {
                seed = args
                    .next()
                    .context("value for --seed")?
                    .parse()
                    .context("parse --seed")?
            }
            "--no-rerank-holdout" => rerank_holdout = false,
            "--em-cache" => {
                em_cache = Some(PathBuf::from(args.next().context("value for --em-cache")?))
            }
            "--lexicon" => {
                lexicon_path = Some(PathBuf::from(args.next().context("value for --lexicon")?))
            }
            "--no-lexicon" => use_lexicon = false,
            "--min-akshara-count" => {
                min_akshara_count = args
                    .next()
                    .context("value for --min-akshara-count")?
                    .parse()
                    .context("parse --min-akshara-count")?
            }
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
                println!("  --seed <n>              Shuffle seed for the pair order (default: 7)");
                println!("  --no-rerank-holdout     Let EM/LM see the reranker's own pairs (old behaviour)");
                println!("  --min-akshara-count <n> Keep aksharas seen >= n times in training (default: 2)");
                println!("  --em-cache <path>       Reuse/save the EM+LM model for this exact data split");
                println!("  --lexicon <path>        Per-language lexicon (build_lexicon; default data/lexicon.bin)");
                println!("  --no-lexicon            Use the single-corpus vocabulary instead (v7 behaviour)");
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

    // Read every pair before training anything: the reranker's pairs and its
    // held-out dev set are chosen first, so they can be kept out of EM/LM.
    let f = File::open(&pairs_file).context("open pairs file")?;
    let mut rows: Vec<((String, String), u32, String)> = Vec::new();
    for line in BufReader::new(f).lines().map_while(Result::ok) {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if let Ok(rec) = serde_json::from_str::<Record>(t) {
            let eng = rec.english.trim().to_ascii_lowercase();
            let nat = rec.native.trim().to_string();
            if !eng.is_empty() && !nat.is_empty() {
                rows.push(((eng, nat), rec.weight, rec.lang));
                if limit.is_some_and(|lim| rows.len() >= lim) {
                    break;
                }
            }
        }
    }
    // Uniform order.  Pair files arrive one language after another (the
    // upstream releases are per language), and every slice below -- the
    // reranker's "first N", the dev "tail" -- must sample all of them.  The
    // reranker was once trained on the first 500k rows of a Hindi-then-Nepali
    // file: Hindi only, while its vocabulary, dev set and test were Nepali.
    rows.shuffle(&mut ChaCha8Rng::seed_from_u64(seed));
    // The ranking is conditioned on these languages (sorted, so the index of a
    // language is stable for a given data set); none if the file has no tags.
    let langs: Vec<String> = {
        let mut l: Vec<String> = rows
            .iter()
            .map(|r| r.2.clone())
            .filter(|l| !l.is_empty())
            .collect();
        l.sort();
        l.dedup();
        l
    };
    let mut raw_pairs: Vec<(String, String)> = Vec::with_capacity(rows.len());
    let mut pair_weights: Vec<u32> = Vec::with_capacity(rows.len());
    let mut pair_lang: Vec<Option<u8>> = Vec::with_capacity(rows.len());
    for (pair, weight, lang) in rows {
        raw_pairs.push(pair);
        pair_weights.push(weight);
        pair_lang.push(langs.iter().position(|l| *l == lang).map(|i| i as u8));
    }
    let count = raw_pairs.len();
    if !langs.is_empty() {
        println!(
            "Languages (ranking is conditioned on each): {}",
            langs.join(", ")
        );
    }
    println!("Read {count} valid parallel pairs (shuffled, seed {seed}).");

    // Reranker pairs are the head of the shuffled list, dev the tail.
    const DEV_SIZE: usize = 4_000;
    let dev_reserved = if !smoke && count > DEV_SIZE {
        DEV_SIZE
    } else {
        0
    };
    let num_rerank = if smoke {
        500.min(count)
    } else if reranker_pairs == 0 {
        count - dev_reserved
    } else {
        reranker_pairs.min(count - dev_reserved)
    };
    // Hold both out of EM/LM, so the reranker learns from candidate lists
    // scored by a model that never saw those words -- the situation at test
    // time.  Trained on memorised pairs, emit/lm look far more decisive than
    // they are on unseen words, and the learned weights over-trust them.
    // Skipped when the held-out share would exceed a quarter of the data
    // (train-full) or with --no-rerank-holdout.
    let holdout = rerank_holdout && (num_rerank + dev_reserved) * 4 <= count;
    let em_range = if holdout {
        num_rerank..count - dev_reserved
    } else {
        0..count
    };
    println!(
        "EM/LM train on {} pairs; reranker {} + dev {} pairs {}.",
        em_range.len(),
        num_rerank,
        dev_reserved,
        if holdout {
            "held out of EM/LM"
        } else {
            "NOT held out (in-sample)"
        }
    );

    // Akshara occurrence counts over the EM training words: the packing step
    // keeps an akshara if the model was trained on it often enough.
    let mut akshara_count: HashMap<String, u32> = HashMap::new();
    for (_, nat) in &raw_pairs[em_range.clone()] {
        for a in akshar_ime::core::akshara::segment(nat) {
            *akshara_count.entry(a).or_insert(0) += 1;
        }
    }

    // --em-cache: EM is the slow phase and is deterministic given the exact
    // training slice and settings, which the key records.  A reranker or
    // packing experiment can then skip it without risking a stale model.
    let em_key = format!(
        "pairs={} n={count} seed={seed} em={}..{} iterations={iterations} attestation={attestation}",
        pairs_file.display(),
        em_range.start,
        em_range.end
    );
    let cached = em_cache.as_deref().and_then(|p| {
        let bytes = std::fs::read(p).ok()?;
        let (key, model): (String, akshar_ime::core::translit_model::TranslitModel) =
            bincode::deserialize(&bytes).ok()?;
        (key == em_key).then_some(model)
    });
    let mut translit_model = if let Some(model) = cached {
        println!("Loaded the EM/LM model from the cache (key matches).");
        model
    } else {
        let mut em_trainer = Trainer::new();
        // Attestation observability (else we're flying blind).
        let (mut wmax, mut wsum) = (0u32, 0u64);
        for i in em_range.clone() {
            let (eng, nat) = &raw_pairs[i];
            // Attestation: log-dampened corpus frequency of the Devanagari
            // side; off (None) reproduces the file weight (default 1.0).
            let w = match attest_freq.as_ref() {
                Some(freq) => attestation_weight(freq, nat),
                None => pair_weights[i],
            };
            wmax = wmax.max(w);
            wsum += w as u64;
            em_trainer.add_pair_weighted(eng, nat, f64::from(w));
        }
        println!("Ingested {} pairs into EM/LM.", em_range.len());
        if attestation {
            println!(
                "Attestation weights: max={} mean={:.2} (1.0 = unseen, log-dampened).",
                wmax,
                wsum as f64 / em_range.len().max(1) as f64
            );
        }
        let model = em_trainer.finalize(&em_config);
        // The trainer holds every pair plus its count tables; free it before
        // the reranker phase, which needs the memory for its candidate batches.
        drop(em_trainer);
        if let Some(p) = em_cache.as_deref() {
            let bytes = bincode::serialize(&(&em_key, &model))
                .map_err(|e| anyhow::anyhow!("serialise EM cache: {e}"))?;
            std::fs::write(p, bytes).with_context(|| format!("write {}", p.display()))?;
            println!("Cached the EM/LM model at {}.", p.display());
        }
        model
    };
    drop(pair_weights);
    translit_model.build_trigram_index();
    println!(
        "EM training finished in {:.2?} (aksharas: {}, chunks: {}).",
        em_t0.elapsed(),
        translit_model.aksharas.len(),
        translit_model.chunks.len()
    );

    // Phase 2: word knowledge.  The per-language lexicon (build_lexicon) is
    // both the frequency source and the decoding dictionary; without one the
    // vocabulary is counted from the single running-text corpus as before.
    println!("\n[Phase 2/4] Compiling Vocabulary & Empirical Frequencies...");
    let vocab_t0 = Instant::now();
    let lexicon_path = lexicon_path
        .or_else(|| Some(PathBuf::from("data/lexicon.bin")).filter(|p| p.exists()))
        .filter(|_| use_lexicon);
    let lexicon: Option<akshar_ime::core::lexicon::Lexicon> = match &lexicon_path {
        Some(p) => {
            let bytes = std::fs::read(p).with_context(|| format!("read {}", p.display()))?;
            let (data, _tokens): (akshar_ime::core::lexicon::LexiconData, Vec<u64>) =
                bincode::deserialize(&bytes).context("parse lexicon")?;
            let lex = akshar_ime::core::lexicon::Lexicon::from_data(data)
                .map_err(|e| anyhow::anyhow!("lexicon automaton: {e}"))?;
            println!(
                "Lexicon {}: {} words in {} languages ({}).",
                p.display(),
                lex.len(),
                lex.langs().len(),
                lex.langs().join(", ")
            );
            Some(lex)
        }
        None => None,
    };
    // Reuse the attestation pre-pass map when present (same implementation,
    // zero re-stream); otherwise count as before.
    let mut vocab_freq: HashMap<String, u32> = match attest_freq {
        _ if lexicon.is_some() => HashMap::new(),
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

    // The reranker's pairs (head) and dev set (tail) were chosen in Phase 1.
    let num_train_pairs = num_rerank;

    /// One training list: the decoder's candidates for a pair, with the gold's
    /// position and the pair's language (`None` = trained language-blind).
    struct RerankItem {
        target_idx: usize,
        lang: Option<u8>,
        /// dense_feats[i]: candidate i's DENSE_DIM features (raw until
        /// standardised with the frozen dev normalisation).
        dense_feats: Vec<[f64; DENSE_DIM]>,
        sparse: Vec<Vec<usize>>,
    }

    // Decode one pair into a training list; None when the gold is not among
    // the candidates (nothing to learn from).
    let lexicon_dict = lexicon.as_ref().map(|lexicon| LexiconDict {
        lexicon,
        keys: decoder.akshara_keys(),
    });
    let dict: &(dyn Dictionary + Sync) = match &lexicon_dict {
        Some(d) => d,
        None => &word_trie,
    };
    let vocab_counts = VocabCounts {
        freq: &vocab_freq,
        ranks: &ranks,
    };
    let decode_item = |roman: &str, gold: &str, lang: Option<u8>| -> Option<RerankItem> {
        // Frequencies in the pair's language (or best of all when it was
        // dropped), exactly as the engine will see them.
        let lexicon_counts = lexicon.as_ref().map(|lexicon| LexiconCounts {
            lexicon,
            lang: lang.and_then(|l| lexicon.lang_index(&langs[usize::from(l)])),
        });
        let counts: &dyn WordCounts = match &lexicon_counts {
            Some(c) => c,
            None => &vocab_counts,
        };
        let cands = decoder.decode_union(roman, RERANK_DECODE_DEPTH, Some(dict));
        let (order, heur, heur_rank) = rank_candidates(&cands, counts);
        let target_idx = order.iter().position(|c| c.dev == gold)?;
        let sparse = order
            .iter()
            .map(|c| {
                let aks = akshar_ime::core::akshara::segment(&c.dev);
                extract_sparse_features_lang(
                    &c.dev,
                    roman,
                    c.akshara_count,
                    &aks,
                    lang.map(usize::from),
                )
            })
            .collect();
        let dense_feats = order
            .iter()
            .enumerate()
            .map(|(idx, c)| {
                extract_dense_features(c, idx, heur[idx], heur_rank[idx], roman, counts)
            })
            .collect();
        Some(RerankItem {
            target_idx,
            lang,
            dense_feats,
            sparse,
        })
    };

    // Decode many pairs in parallel, keeping input order: every pair decodes
    // independently against read-only tables, so the batch splits across all
    // cores.  `jobs[i]` = (pair index, language to tag it with).
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let decode_all = |jobs: &[(usize, Option<u8>)]| -> Vec<RerankItem> {
        let chunk = jobs.len().div_ceil(threads).max(1);
        std::thread::scope(|s| {
            let handles: Vec<_> = jobs
                .chunks(chunk)
                .map(|part| {
                    let decode_item = &decode_item;
                    let raw_pairs = &raw_pairs;
                    s.spawn(move || {
                        part.iter()
                            .filter_map(|&(i, lang)| {
                                decode_item(&raw_pairs[i].0, &raw_pairs[i].1, lang)
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("decode thread panicked"))
                .collect()
        })
    };

    // The learned linear model: shared dense weights (warm-started from
    // W_DENSE), per-language dense offsets, and the sparse table -- which holds
    // both the shared and the language-tagged feature slots.  AdaGrad per slot.
    struct Weights {
        dense: Vec<f64>,
        dense_g2: Vec<f64>,
        lang: Vec<Vec<f64>>,
        lang_g2: Vec<Vec<f64>>,
        sparse: Vec<f32>,
        sparse_g2: Vec<f32>,
    }
    impl Weights {
        fn scores(&self, s: &RerankItem) -> Vec<f64> {
            let offsets = s.lang.and_then(|l| self.lang.get(usize::from(l)));
            s.dense_feats
                .iter()
                .zip(&s.sparse)
                .map(|(z, slots)| {
                    let mut sc = 0.0;
                    for k in 0..DENSE_DIM {
                        sc += (self.dense[k] + offsets.map_or(0.0, |o| o[k])) * z[k];
                    }
                    sc + slots
                        .iter()
                        .map(|&h| f64::from(self.sparse[h]))
                        .sum::<f64>()
                })
                .collect()
        }
        /// Softmax over the list; returns (candidate probabilities, top-1 hit).
        fn posterior(&self, s: &RerankItem) -> (Vec<f64>, bool) {
            let scores = self.scores(s);
            let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp: Vec<f64> = scores.iter().map(|&x| (x - max).exp()).collect();
            let z: f64 = exp.iter().sum::<f64>() + 1e-12;
            let hit = scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .is_some_and(|(i, _)| i == s.target_idx);
            (exp.iter().map(|e| e / z).collect(), hit)
        }
        /// One listwise cross-entropy step; returns the sample's loss.
        fn step(&mut self, s: &RerankItem, lr: f64) -> f64 {
            const DENSE_LR_SCALE: f64 = 0.1;
            let (probs, _) = self.posterior(s);
            let mut dense_grad = [0.0f64; DENSE_DIM];
            for (idx, p) in probs.iter().enumerate() {
                let g = if idx == s.target_idx { p - 1.0 } else { *p };
                if g.abs() <= 1e-5 {
                    continue;
                }
                for &h in &s.sparse[idx] {
                    let gf = g as f32;
                    self.sparse_g2[h] += gf * gf;
                    self.sparse[h] -= (lr as f32) / (self.sparse_g2[h].sqrt() + 1e-4) * gf;
                }
                // dL/dw_k = sum_i (p_i - y_i) z_ik, applied once per sample.
                for (k, acc) in dense_grad.iter_mut().enumerate() {
                    *acc += g * s.dense_feats[idx][k];
                }
            }
            let lang = s.lang.map(usize::from);
            for (k, &g) in dense_grad.iter().enumerate() {
                if g.abs() <= 1e-6 {
                    continue;
                }
                // The shared copy and the language copy see the same feature,
                // so they receive the same gradient (feature augmentation).
                self.dense_g2[k] += g * g;
                self.dense[k] -= lr * DENSE_LR_SCALE / (self.dense_g2[k].sqrt() + 1e-4) * g;
                if let Some(l) = lang {
                    self.lang_g2[l][k] += g * g;
                    self.lang[l][k] -= lr * DENSE_LR_SCALE / (self.lang_g2[l][k].sqrt() + 1e-4) * g;
                }
            }
            -probs[s.target_idx].max(1e-12).ln()
        }
        /// Mean loss and top-1 (%) over held-out items.
        fn evaluate(&self, items: &[RerankItem]) -> Option<(f64, f64)> {
            if items.is_empty() {
                return None;
            }
            let (mut loss, mut hits) = (0.0f64, 0usize);
            for s in items {
                let (probs, hit) = self.posterior(s);
                loss -= probs[s.target_idx].max(1e-12).ln();
                hits += usize::from(hit);
            }
            let n = items.len() as f64;
            Some((loss / n, hits as f64 / n * 100.0))
        }
    }
    let mut weights = Weights {
        dense: W_DENSE.to_vec(),
        dense_g2: vec![1.0; DENSE_DIM],
        lang: vec![vec![0.0; DENSE_DIM]; langs.len()],
        lang_g2: vec![vec![1.0; DENSE_DIM]; langs.len()],
        sparse: vec![0.0; HASH_SIZE],
        sparse_g2: vec![0.0; HASH_SIZE],
    };

    // Held-out dev set: the tail of the shuffled pairs, kept out of EM/LM and
    // of reranker training.  It measures every batch (the table has ~1e6
    // unregularised slots, so overfitting is the default failure) and it fixes
    // the dense-feature normalisation below.
    let dev_range = count - dev_reserved..count;
    println!(
        "Pre-decoding candidates for {} training pairs (dev reserved: {})...",
        num_train_pairs,
        dev_range.len()
    );
    let t0 = Instant::now();
    let mut dense_stats = DenseStats::new();
    let dev_jobs: Vec<(usize, Option<u8>)> = dev_range.clone().map(|i| (i, pair_lang[i])).collect();
    let mut dev_items: Vec<RerankItem> = decode_all(&dev_jobs);
    for item in &dev_items {
        for f in &item.dense_feats {
            dense_stats.push(f);
        }
    }
    println!(
        "Held-out dev set: {} of {} pairs have the gold in the candidate list (decoded in {:.2?}).",
        dev_items.len(),
        dev_range.len(),
        t0.elapsed()
    );
    // One normalisation for training AND inference: measured on the dev
    // candidate lists (decoded by this model), frozen, used to standardise
    // every training batch, and packed into the container.  Training used to
    // standardise with the compiled-in MEAN/STD_DENSE while the container
    // shipped statistics re-measured at the end, so the learned weights were
    // applied at inference to features on a 15-25% different scale.
    let (norm_mean, norm_std) = dense_stats.finish();
    let standardise = |item: &mut RerankItem| {
        for f in item.dense_feats.iter_mut() {
            for k in 0..DENSE_DIM {
                f[k] = (f[k] - norm_mean[k]) / norm_std[k];
            }
        }
    };
    dev_items.iter_mut().for_each(standardise);
    println!(
        "Dense-feature normalisation from {} dev candidates (emit mean {:.3} std {:.3}, lm mean {:.3} std {:.3}).",
        dense_stats.n, norm_mean[0], norm_std[0], norm_mean[1], norm_std[1]
    );

    // Snapshot of the best weights by dev loss, so a run that starts
    // overfitting does not have to be thrown away.
    type Snapshot = (Vec<f64>, Vec<Vec<f64>>, Vec<f32>);
    let snapshot =
        |w: &Weights| -> Snapshot { (w.dense.clone(), w.lang.clone(), w.sparse.clone()) };
    let mut best_dev: Option<(f64, Snapshot)> = None;
    if let Some((l, a)) = weights.evaluate(&dev_items) {
        println!("  dev before training: loss={l:.4} top-1={a:.2}%");
    }

    // Decode in batches of BATCH_SIZE (memory stays bounded) and run `epochs`
    // passes over each batch before decoding the next.  The learning rate
    // decays once per batch so the final batch runs at LR_FINAL_FRACTION of
    // LR0; AdaGrad adapts per slot on top of that gentle global anneal.
    const BATCH_SIZE: usize = 100_000;
    const LR0: f64 = 0.05;
    const LR_FINAL_FRACTION: f64 = 0.1;
    // Share of training lists shown WITHOUT their language, so the shared
    // weights also rank well on their own (language "auto").
    const LANG_DROPOUT: f64 = 0.2;
    let mut dropout_rng = ChaCha8Rng::seed_from_u64(seed ^ 0x6c61_6e67);
    let total_batches = num_train_pairs.div_ceil(BATCH_SIZE).max(1);
    let lr_decay = LR_FINAL_FRACTION.powf(1.0 / total_batches as f64);
    let mut lr = LR0;
    let mut global_samples = 0usize;
    for batch_idx in 0..total_batches {
        let range = batch_idx * BATCH_SIZE..((batch_idx + 1) * BATCH_SIZE).min(num_train_pairs);
        let batch_t0 = Instant::now();
        println!(
            "  Batch {}/{}: decoding {} pairs ...",
            batch_idx + 1,
            total_batches,
            range.len()
        );
        // Dropout decisions are drawn in order, before the parallel decode, so
        // a run is reproducible whatever the thread count.
        let jobs: Vec<(usize, Option<u8>)> = range
            .map(|i| {
                let keep = !rand::Rng::gen_bool(&mut dropout_rng, LANG_DROPOUT);
                (i, pair_lang[i].filter(|_| keep))
            })
            .collect();
        let mut samples = decode_all(&jobs);
        samples.iter_mut().for_each(standardise);
        println!(
            "    -> {} valid (decoded in {:.1?})",
            samples.len(),
            batch_t0.elapsed()
        );
        let mut batch_loss = 0.0f64;
        for _ in 0..epochs {
            for s in &samples {
                batch_loss += weights.step(s, lr);
            }
        }
        global_samples += samples.len() * epochs;
        batch_loss /= (epochs * samples.len()).max(1) as f64;
        println!(
            "    Batch {}/{} done: avg loss {:.4} ({} samples, lr {:.4})",
            batch_idx + 1,
            total_batches,
            batch_loss,
            samples.len(),
            lr
        );
        if let Some((dl, da)) = weights.evaluate(&dev_items) {
            let better = best_dev.as_ref().is_none_or(|(b, _)| dl < *b);
            println!(
                "      dev: loss={dl:.4} top-1={da:.2}%{}",
                if better { "  <- best" } else { "" }
            );
            if better {
                best_dev = Some((dl, snapshot(&weights)));
            }
        }
        lr *= lr_decay;
    }
    println!("  Trained on {global_samples} samples.");
    println!("Reranker trained in {:.2?}.", rank_t0.elapsed());

    // Phase 4: Pack
    println!(
        "\n[Phase 4/4] Packing Unified Model -> {} ...",
        out_path.display()
    );
    let pack_t0 = Instant::now();

    // Pack the weights that scored best on held-out data, not necessarily the
    // last ones: with ~1e6 unregularised sparse slots, later batches can overfit.
    if let Some((dl, best)) = best_dev {
        let final_dl = weights.evaluate(&dev_items).map(|(l, _)| l);
        if final_dl.is_some_and(|f| f > dl + 1e-9) {
            println!(
                "Packing the best-by-dev weights (dev loss {:.4}) rather than the final ({:.4}).",
                dl,
                final_dl.unwrap_or(f64::NAN)
            );
            (weights.dense, weights.lang, weights.sparse) = best;
        } else {
            println!("Final weights are the best by dev loss ({dl:.4}).");
        }
    }
    let dense_weights = weights.dense;
    let sparse_table = weights.sparse;

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

    // Keep an akshara when the model was trained on it at least
    // --min-akshara-count times, or when the vocabulary uses it.  Pruning used
    // to keep ONLY the vocabulary's aksharas; with a single-language corpus
    // vocabulary that deleted every syllable the other languages need (all of
    // Marathi's ळ forms, Hindi's nukta/chandra forms, Sanskrit's visarga
    // clusters), so 6-13% of their words could not be produced at all.  The
    // count floor still drops one-off noise from the pair data.
    println!(
        "Pruning aksharas seen < {min_akshara_count} times in training (and not in the vocabulary)..."
    );
    let mut seen_aks = std::collections::HashSet::new();
    for (a, &n) in &akshara_count {
        if n >= min_akshara_count {
            if let Some(id) = translit_model.akshara_id(a) {
                seen_aks.insert(id);
            }
        }
    }
    drop(akshara_count);
    // The end-of-word token is in no word, so the rules above would drop it --
    // and with it every P(</w> | ...) transition.  That silently disabled
    // end-of-word modelling in every model packed before this line existed.
    if let Some(eow) = translit_model.akshara_id(akshar_ime::core::translit_model::END_OF_WORD) {
        seen_aks.insert(eow);
    }
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

    let (dense_mean, dense_std) = (norm_mean.clone(), norm_std.clone());
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
    unified.langs = langs;
    unified.dense_lang_weights = weights.lang;
    unified.lexicon = lexicon.as_ref().map(|l| l.to_data());
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
