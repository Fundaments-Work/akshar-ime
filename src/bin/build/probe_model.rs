// File: src/bin/probe_model.rs
//
// Diagnostic: dump the decoder's behaviour on a single roman word.
//
// Usage: cargo run --release --bin probe_model -- <word> [--model <path>]

use akshar_ime::core::decoder::ModelDecoder;
use akshar_ime::core::translit_model::TranslitModel;
use std::path::Path;

fn main() {
    let mut word: Option<String> = None;
    let mut model_path = "data/akshar.model".to_string();
    let mut inspect_mode = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model_path = args.next().unwrap_or_else(|| model_path.clone()),
            "--inspect" => inspect_mode = true,
            "-h" | "--help" => {
                println!("usage: probe_model [word] [--model path] [--inspect]");
                return;
            }
            other => word = Some(other.to_string()),
        }
    }

    if inspect_mode {
        let p = Path::new(&model_path);
        // Check if it is a unified model or translit model
        if let Ok(unified) = akshar_ime::core::unified::UnifiedModel::load(p) {
            println!("=== Unified Model Inspection: {} ===", p.display());
            let m_bytes = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
            println!(
                "Total file size: {:.2} MB ({} bytes)",
                m_bytes as f64 / (1024.0 * 1024.0),
                m_bytes
            );

            // Measure sections as they are ACTUALLY STORED, not as they sit in
            // memory.  Since v3 the n-grams, vocabulary and chunk list are
            // re-encoded on save (codec: CSR, delta varints, 8-bit codebooks),
            // so `serialized_size` of the decoded field describes a layout the
            // file does not use and the parts stop summing to the whole.
            use akshar_ime::core::codec;
            let tr = &unified.translit;
            let ak_sz = bincode::serialized_size(&tr.aksharas).unwrap_or(0);
            let ch_sz = codec::encode_chunks(&tr.chunks).len() as u64;
            let em_sz = codec::encode_adjacency(&tr.emissions).len() as u64;
            let bi_sz = codec::encode_adjacency(&tr.bigrams).len() as u64;
            let tri_k_sz = codec::encode_pairs(&tr.trigram_keys).len() as u64;
            let tri_v_sz = codec::encode_adjacency(&tr.trigrams).len() as u64;
            let tri_b_sz = codec::encode_weights(&tr.trigram_backoff).len() as u64;
            let misc_sz = codec::encode_weights(&tr.backoff).len() as u64
                + codec::encode_weights(&tr.unigram_kn).len() as u64
                + codec::encode_weights(&tr.word_start).len() as u64;
            let t_sz = ak_sz + ch_sz + em_sz + bi_sz + tri_k_sz + tri_v_sz + tri_b_sz + misc_sz;
            let s_sz = bincode::serialized_size(&unified.sparse_reranker_table).unwrap_or(0);
            let v_sz = codec::encode_vocab(&unified.vocab_freq, &tr.aksharas).len() as u64;

            println!(
                "  Translit Model    : {:>7.2} MB ({:>5.1}%)",
                t_sz as f64 / (1024.0 * 1024.0),
                (t_sz as f64 / m_bytes as f64) * 100.0
            );
            println!(
                "  Sparse Reranker   : {:>7.2} MB ({:>5.1}%)",
                s_sz as f64 / (1024.0 * 1024.0),
                (s_sz as f64 / m_bytes as f64) * 100.0
            );
            println!(
                "  Vocab Frequency   : {:>7.2} MB ({:>5.1}%) [{} words]",
                v_sz as f64 / (1024.0 * 1024.0),
                (v_sz as f64 / m_bytes as f64) * 100.0,
                unified.vocab_freq.len()
            );
            println!(
                "\n--- Dense Reranker Weights ({} weights) ---",
                unified.dense_weights.len()
            );
            for (k, w) in unified.dense_weights.iter().enumerate() {
                let mean = unified.dense_mean.get(k).copied().unwrap_or(0.0);
                let std = unified.dense_std.get(k).copied().unwrap_or(1.0);
                println!(
                    "    feat[{:02}]: w={:>10.6}  (mean={:.3}, std={:.3})",
                    k, w, mean, std
                );
            }

            println!("\n--- Translit Sub-components (as encoded) ---");
            let other_sz = misc_sz;

            println!(
                "    Aksharas list   : {:>6.2} MB ({} aksharas)",
                ak_sz as f64 / (1024.0 * 1024.0),
                unified.translit.aksharas.len()
            );
            println!(
                "    Chunks list     : {:>6.2} MB ({} chunks)",
                ch_sz as f64 / (1024.0 * 1024.0),
                unified.translit.chunks.len()
            );
            let em_count: usize = unified.translit.emissions.iter().map(|e| e.len()).sum();
            println!(
                "    Emissions       : {:>6.2} MB ({} entries across {} aksharas)",
                em_sz as f64 / (1024.0 * 1024.0),
                em_count,
                unified.translit.emissions.len()
            );
            let bi_count: usize = unified.translit.bigrams.iter().map(|b| b.len()).sum();
            println!(
                "    Bigram LM       : {:>6.2} MB ({} transitions)",
                bi_sz as f64 / (1024.0 * 1024.0),
                bi_count
            );
            let tri_count: usize = unified.translit.trigrams.iter().map(|t| t.len()).sum();
            println!(
                "    Trigram LM      : {:>6.2} MB ({} contexts, {} transitions)",
                (tri_k_sz + tri_v_sz + tri_b_sz) as f64 / (1024.0 * 1024.0),
                unified.translit.trigram_keys.len(),
                tri_count
            );
            println!(
                "    Other fields    : {:>6.2} MB",
                other_sz as f64 / (1024.0 * 1024.0)
            );
            return;
        }

        let model = TranslitModel::load(p).expect("load model");
        let m_bytes = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        println!("=== Translit Model Inspection: {} ===", p.display());
        println!(
            "Total file size: {:.2} MB ({} bytes)",
            m_bytes as f64 / (1024.0 * 1024.0),
            m_bytes
        );
        let ak_sz = bincode::serialized_size(&model.aksharas).unwrap_or(0);
        let ch_sz = bincode::serialized_size(&model.chunks).unwrap_or(0);
        let em_sz = bincode::serialized_size(&model.emissions).unwrap_or(0);
        let bi_sz = bincode::serialized_size(&model.bigrams).unwrap_or(0);
        let tri_k_sz = bincode::serialized_size(&model.trigram_keys).unwrap_or(0);
        let tri_v_sz = bincode::serialized_size(&model.trigrams).unwrap_or(0);
        let tri_b_sz = bincode::serialized_size(&model.trigram_backoff).unwrap_or(0);
        let other_sz =
            m_bytes.saturating_sub(ak_sz + ch_sz + em_sz + bi_sz + tri_k_sz + tri_v_sz + tri_b_sz);

        println!(
            "  Aksharas list   : {:>6.2} MB ({} aksharas)",
            ak_sz as f64 / (1024.0 * 1024.0),
            model.aksharas.len()
        );
        println!(
            "  Chunks list     : {:>6.2} MB ({} chunks)",
            ch_sz as f64 / (1024.0 * 1024.0),
            model.chunks.len()
        );
        let em_count: usize = model.emissions.iter().map(|e| e.len()).sum();
        println!(
            "  Emissions       : {:>6.2} MB ({} entries across {} aksharas)",
            em_sz as f64 / (1024.0 * 1024.0),
            em_count,
            model.emissions.len()
        );
        let bi_count: usize = model.bigrams.iter().map(|b| b.len()).sum();
        println!(
            "  Bigram LM       : {:>6.2} MB ({} transitions)",
            bi_sz as f64 / (1024.0 * 1024.0),
            bi_count
        );
        let tri_count: usize = model.trigrams.iter().map(|t| t.len()).sum();
        println!(
            "  Trigram LM      : {:>6.2} MB ({} contexts, {} transitions)",
            (tri_k_sz + tri_v_sz + tri_b_sz) as f64 / (1024.0 * 1024.0),
            model.trigram_keys.len(),
            tri_count
        );
        println!(
            "  Other fields    : {:>6.2} MB",
            other_sz as f64 / (1024.0 * 1024.0)
        );
        return;
    }

    let word = word.unwrap_or_else(|| "holi".to_string());

    let model = if let Ok(u) = akshar_ime::core::unified::UnifiedModel::load(Path::new(&model_path))
    {
        u.translit
    } else {
        TranslitModel::load(Path::new(&model_path)).expect("load model")
    };
    let dec = ModelDecoder::new(model);

    let roman = word.to_ascii_lowercase();
    let bytes = roman.as_bytes();

    // Dump reverse-index candidates per position.
    for i in 0..roman.len() {
        for l in 1..=4.min(roman.len() - i) {
            let chunk = &roman[i..i + l];
            let list = dec.chunk_candidates(chunk);
            let shown: Vec<String> = list
                .iter()
                .take(8)
                .map(|(a, w)| format!("{}({:.2})", dec.model.aksharas[*a as usize], w))
                .collect();
            println!("pos {i} chunk `{chunk}` -> {}", shown.join(" "));
        }
    }

    println!("\nTop 10 decodings:");
    for (d, w) in dec.decode(&roman, 10) {
        println!("  {d}  (weight {w:.3})");
    }

    println!("\nImeEngine Suggestions for `{}`:", roman);
    let engine = akshar_ime::ImeEngine::new();
    let sugs = engine.get_suggestions(&roman, 10);
    for (s, sc) in sugs {
        println!("  {s}  (score {sc})");
    }

    println!("\nKN stats:");
    for aks in ["हो", "ली", "ओः", "ऊः", "ग्को", "को", "ल"] {
        let Some(aid) = dec.model.akshara_id(aks) else {
            println!("  {aks}: not in vocab");
            continue;
        };
        let uni = dec.model.unigram_kn[aid as usize];
        let backoff = dec.model.backoff[aid as usize];
        let n_bi = dec.model.bigrams[aid as usize].len();
        println!("  {aks}: id={aid} unigram_kn={uni:.3} backoff={backoff:.3} bigrams={n_bi}");
    }
    let _ = bytes;
}
