// Where does a keystroke's time actually go?
//
//   cargo run --release --example profile_decode -- [data/akshar.model]
//
// Profiles the phases of one suggestion query on 1,000 real test romans:
// the free beam, the dictionary-constrained beam (lexicon automaton for v8
// models, word trie before), their union, the reranker, and the engine end
// to end.  Beam and dictionary match what the engine uses.
use akshar_ime::core::decoder::{DecoderConfig, Dictionary, LexiconDict, ModelDecoder};
use akshar_ime::core::lexicon::Lexicon;
use akshar_ime::core::reranker::{
    rerank_with_norm, DenseNorm, FreqRanks, LexiconCounts, VocabCounts, WordCounts,
};
use akshar_ime::core::unified::UnifiedModel;
use akshar_ime::core::wordtrie::WordTrie;
use akshar_ime::ImeEngine;
use std::io::BufRead;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let path = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "data/akshar.model".into()),
    );
    let mut u = UnifiedModel::load(&path).expect("load model");
    let lexicon = u
        .lexicon
        .take()
        .map(|d| Lexicon::from_data(d).expect("lexicon"));
    let vocab = std::mem::take(&mut u.vocab_freq);
    let ranks = FreqRanks::from_freq_map(&vocab);
    let mut m = u.translit.clone();
    m.build_trigram_index();
    let trie = lexicon
        .is_none()
        .then(|| WordTrie::from_freq_map(&vocab, &|a| m.akshara_id(a), 1));
    let beam = std::env::var("AKSHAR_BEAM")
        .ok()
        .and_then(|b| b.parse().ok())
        .unwrap_or(32);
    let dec = ModelDecoder::with_config(
        m,
        DecoderConfig {
            beam_width: beam,
            ..DecoderConfig::default()
        },
    );
    let lexicon_dict = lexicon.as_ref().map(|lexicon| LexiconDict {
        lexicon,
        keys: dec.akshara_keys(),
    });
    let dict: &dyn Dictionary = match (&lexicon_dict, &trie) {
        (Some(d), _) => d,
        (None, Some(t)) => t,
        _ => unreachable!(),
    };
    let lexicon_counts = lexicon.as_ref().map(|lexicon| LexiconCounts {
        lexicon,
        lang: None,
    });
    let vocab_counts = VocabCounts {
        freq: &vocab,
        ranks: &ranks,
    };
    let counts: &dyn WordCounts = match &lexicon_counts {
        Some(c) => c,
        None => &vocab_counts,
    };

    // Real queries, not synthetic ones: length distribution drives beam cost.
    let f = std::fs::File::open("data/aksharantar/test_devanagari.jsonl").unwrap();
    let mut qs: Vec<String> = Vec::new();
    for line in std::io::BufReader::new(f).lines().take(4000) {
        let l = line.unwrap();
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&l) {
            if let Some(r) = v["english word"].as_str() {
                qs.push(r.to_string());
            }
        }
        if qs.len() >= 1000 {
            break;
        }
    }
    let n = qs.len() as f64;
    println!(
        "{}: {} queries, mean length {:.1} chars, beam {beam}, dictionary: {}\n",
        path.display(),
        qs.len(),
        qs.iter().map(|q| q.len()).sum::<usize>() as f64 / n,
        if lexicon.is_some() {
            "lexicon automaton"
        } else {
            "word trie"
        }
    );

    macro_rules! phase {
        ($label:expr, $body:expr) => {{
            let t = Instant::now();
            let r = $body;
            let ms = t.elapsed().as_secs_f64() * 1000.0 / n;
            println!("  {:<38} {:>7.3} ms/query", $label, ms);
            (r, ms)
        }};
    }

    let (_, t_free) = phase!("decode_detailed (free beam)", {
        let mut acc = 0usize;
        for q in &qs {
            acc += dec.decode_detailed(q, 50).len();
        }
        acc
    });
    let (_, t_trie) = phase!("decode_in_words_detailed (dictionary)", {
        let mut acc = 0usize;
        for q in &qs {
            acc += dec.decode_in_words_detailed(q, 50, dict).len();
        }
        acc
    });
    let (cands, t_union) = phase!("decode_union (both + merge)", {
        let mut v = Vec::new();
        for q in &qs {
            v.push(dec.decode_union(q, 50, Some(dict)));
        }
        v
    });
    let (_, t_rr) = phase!("rerank_with_norm", {
        let mut acc = 0usize;
        for (q, c) in qs.iter().zip(&cands) {
            acc += rerank_with_norm(
                q,
                c,
                counts,
                Some(&u.sparse_reranker_table),
                Some(u.sparse_scale),
                DenseNorm::from_model(&u.dense_mean, &u.dense_std),
                (!u.dense_weights.is_empty()).then_some(u.dense_weights.as_slice()),
                None,
                u.gamma_auto,
            )
            .len();
        }
        acc
    });

    let engine = ImeEngine::from_unified_file(&path).expect("engine");
    let (_, t_all) = phase!("ImeEngine::get_suggestions (end to end)", {
        let mut acc = 0usize;
        for q in &qs {
            acc += engine.get_suggestions(q, 8).len();
        }
        acc
    });

    println!(
        "\n  decode_union share of end-to-end : {:.0}%",
        t_union / t_all * 100.0
    );
    println!(
        "  rerank      share of end-to-end : {:.0}%",
        t_rr / t_all * 100.0
    );
    println!("  free beam vs dictionary beam    : {t_free:.2} / {t_trie:.2} ms");
    println!(
        "  unaccounted (engine overhead)   : {:.3} ms",
        (t_all - t_union - t_rr).max(0.0)
    );
}
