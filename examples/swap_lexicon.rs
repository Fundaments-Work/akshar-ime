// Swap a model's lexicon for another one, keeping the phonetic model and
// reranker weights (EM, KN LM, dense/sparse, dense_lang_weights, gamma)
// byte-identical. An ablation tool: does the vocabulary a language *shares*
// with others in one automaton change its accuracy, holding everything else
// constant? Build a reduced lexicon with `build_lexicon --raw <dir with only
// some languages' files>`, swap it in here, and diff `eval_langs` against the
// unswapped model.
//
//   cargo run --release --example swap_lexicon -- <model-in> <lexicon.bin> <model-out>

use akshar_ime::core::lexicon::{Lexicon, LexiconData};
use akshar_ime::core::unified::UnifiedModel;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [model_in, lexicon_path, model_out] = args.as_slice() else {
        eprintln!("usage: swap_lexicon <model-in> <lexicon.bin> <model-out>");
        std::process::exit(2);
    };

    let mut m = UnifiedModel::load(&PathBuf::from(model_in)).expect("load model");
    let bytes = std::fs::read(lexicon_path).expect("read lexicon");
    let (data, _tokens): (LexiconData, Vec<u64>) =
        bincode::deserialize(&bytes).expect("parse lexicon");
    let lex = Lexicon::from_data(data.clone()).expect("lexicon automaton");
    println!(
        "swapping in {}: {} words, languages [{}]",
        lexicon_path,
        lex.len(),
        lex.langs().join(", ")
    );
    m.lexicon = Some(data);
    m.save(&PathBuf::from(model_out)).expect("save model");
    let size = std::fs::metadata(model_out).unwrap().len();
    println!("wrote {model_out} ({:.2} MB)", size as f64 / 1e6);
}
