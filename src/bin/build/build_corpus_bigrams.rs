// File: src/bin/build/build_corpus_bigrams.rs
//
// Builds the pruned corpus bigram sidecar for the sentence-context blend
// (engine step 1b, `CorpusCtx`). Streams the running-text corpus once,
// keeping (prev, cur) pairs with count >= --threshold whose *current* word
// is in the model vocabulary (only in-vocab words can be reranked; the
// previous word is unrestricted so names and OOV still condition).
//
// The threshold sweep (examples/context_headroom.rs, 1000 held-out
// sentences) sized this: >=20 keeps 436k entries (~2.6 MB raw) at NET +264
// (+0.96pp oracle-override); >=50 keeps 170k (~1 MB) at NET +218. Default 20.
//
// Usage:
//   cargo run --release --bin build_corpus_bigrams -- \
//     --corpus data/nepali_corpus.txt --model data/akshar.model \
//     --threshold 20 --out data/corpus_bigrams.bin
//
// `data/` is gitignored: the sidecar is built locally, never committed.

use akshar_ime::core::context::CorpusCtx;
use akshar_ime::core::unified::UnifiedModel;
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

fn main() {
    let mut corpus_path = "data/nepali_corpus.txt".to_string();
    let mut model_path = "data/akshar.model".to_string();
    let mut threshold = 20u32;
    let mut out_path = "data/corpus_bigrams.bin".to_string();

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--corpus" => corpus_path = args.next().expect("value for --corpus"),
            "--model" => model_path = args.next().expect("value for --model"),
            "--threshold" => {
                threshold = args
                    .next()
                    .expect("value")
                    .parse()
                    .expect("--threshold <n>")
            }
            "--out" => out_path = args.next().expect("value for --out"),
            "--help" | "-h" => {
                println!("build_corpus_bigrams — pruned (prev, cur) count sidecar");
                println!("  --corpus <path>     running text (default data/nepali_corpus.txt)");
                println!("  --model <path>      unified model for the vocab filter");
                println!("  --threshold <n>     keep pairs with count >= n (default 20)");
                println!("  --out <path>        output sidecar (default data/corpus_bigrams.bin)");
                return;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }

    let unified = UnifiedModel::load(Path::new(&model_path))
        .unwrap_or_else(|e| panic!("load unified model {model_path}: {e}"));
    let vocab = &unified.vocab_freq;
    eprintln!("vocab: {} words", vocab.len());

    let mut bi: HashMap<(String, String), u32> = HashMap::new();
    let mut uni: HashMap<String, u32> = HashMap::new();
    let file = std::fs::File::open(&corpus_path)
        .unwrap_or_else(|e| panic!("open corpus {corpus_path}: {e}"));
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let mut prev = "<s>";
        for w in line.split_whitespace() {
            *uni.entry(w.to_string()).or_insert(0) += 1;
            // Only in-vocab current words can ever be reranked; count the
            // pair only then (prev stays unrestricted).
            if vocab.contains_key(w) {
                *bi.entry((prev.to_string(), w.to_string())).or_insert(0) += 1;
            }
            prev = w;
        }
    }
    let raw = bi.len();
    bi.retain(|_, &mut c| c >= threshold);
    // Unigrams only for words that survive as a current word (backoff needs
    // nothing else); drop the rest to keep the sidecar small.
    let kept_cur: std::collections::HashSet<&String> = bi.keys().map(|(_, c)| c).collect();
    uni.retain(|w, _| kept_cur.contains(w));
    let ctx = CorpusCtx::new(bi, uni);
    let bytes = bincode::serialize(&ctx).expect("serialize sidecar");
    std::fs::write(&out_path, &bytes).unwrap_or_else(|e| panic!("write {out_path}: {e}"));
    println!(
        "pairs: {raw} raw -> {} kept (threshold {threshold}); unigrams {}; sidecar {:.2} MB -> {out_path}",
        ctx.table.len(),
        ctx.uni.len(),
        bytes.len() as f64 / 1e6
    );
}
