// File: src/core/lexicon.rs
//
// Per-language word knowledge in one minimal automaton.
//
// The ranker's strongest signal is how often a candidate occurs as a word in
// running text of the language being typed.  Every Devanagari language has
// its own counts, but their words overlap heavily, so all of them live in ONE
// minimal acyclic finite-state map (an FST: shared prefixes and suffixes are
// stored once).  The same automaton is the dictionary the decoder intersects
// its lattice with, so real words of every language can be generated.
//
// * Key: the word, one byte per character.  Every character of a word is a
//   Devanagari letter or sign (U+0900..U+097F), so byte = code point - 0x880;
//   UTF-8 would spend three bytes per character and triple the automaton.
// * Value: a palette index.  Each word has one frequency level per language,
//
//       level = round(8 * log2(1 + c)),   c = count per 100M tokens
//
//   (0 = the language does not use the word).  Normalising by corpus size
//   makes levels comparable across languages; eighth-octave steps (~9%) are
//   far finer than count noise.  The distinct level vectors are few (~65k
//   for 1.2M words), so each is stored once in `palette`, most frequent
//   first, and the automaton carries its small index.
//
// Lookups and walks run on the serialised bytes: no per-word allocation, no
// hashing, a few MB of memory where a HashMap of the same words took ~100.

use fst::raw::{CompiledAddr, Fst};
use fst::Map;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Languages one lexicon can hold.
pub const MAX_LANGS: usize = 8;

/// Frequency levels of one word, one byte per language (0 = absent).
pub type Levels = [u8; MAX_LANGS];

/// Tokens the common frequency scale is expressed in (per 100M tokens).
const SCALE_TOKENS: f64 = 1e8;

/// Quantise `count` occurrences in a corpus of `total_tokens` to a level
/// byte; 0 only for words never seen.
pub fn quantise(count: u64, total_tokens: u64) -> u8 {
    if count == 0 || total_tokens == 0 {
        return 0;
    }
    let per_scale = count as f64 * SCALE_TOKENS / total_tokens as f64;
    (8.0 * (1.0 + per_scale).log2()).round().clamp(1.0, 255.0) as u8
}

/// Occurrences per 100M tokens a level stands for (0 for absent words).
pub fn dequantise(level: u8) -> f64 {
    if level == 0 {
        0.0
    } else {
        (f64::from(level) / 8.0).exp2() - 1.0
    }
}

/// The automaton's key for a word: one byte per Devanagari character, `None`
/// if the word has any other character (it cannot be in the lexicon).
pub fn encode_key(word: &str) -> Option<Vec<u8>> {
    word.chars().map(encode_char).collect()
}

fn encode_char(c: char) -> Option<u8> {
    let u = c as u32;
    (0x0900..=0x097F).contains(&u).then(|| (u - 0x0880) as u8)
}

fn decode_key(key: &[u8]) -> String {
    key.iter()
        .filter_map(|&b| char::from_u32(u32::from(b) + 0x0880))
        .collect()
}

/// Serialised form: languages, palette, automaton bytes.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct LexiconData {
    pub langs: Vec<String>,
    pub palette: Vec<Levels>,
    pub fst: Vec<u8>,
}

/// A position in the automaton while spelling a word out akshara by akshara.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LexState {
    addr: CompiledAddr,
    out: u64,
}

impl LexState {
    /// The state as two integers (for callers that store it opaquely).
    pub fn to_raw(self) -> (u64, u64) {
        (self.addr as u64, self.out)
    }
    pub fn from_raw(addr: u64, out: u64) -> Self {
        Self {
            addr: addr as CompiledAddr,
            out,
        }
    }
}

/// Word -> per-language frequency levels.
pub struct Lexicon {
    map: Map<Vec<u8>>,
    palette: Vec<Levels>,
    langs: Vec<String>,
}

impl Lexicon {
    /// Build from `(word, levels)` pairs (any order; words that cannot be
    /// encoded or have no non-zero level are skipped).
    pub fn build(
        langs: Vec<String>,
        entries: impl IntoIterator<Item = (String, Levels)>,
    ) -> Result<Self, fst::Error> {
        assert!(langs.len() <= MAX_LANGS, "at most {MAX_LANGS} languages");
        let mut keyed: Vec<(Vec<u8>, Levels)> = entries
            .into_iter()
            .filter(|(_, lv)| lv.iter().any(|&x| x > 0))
            .filter_map(|(w, lv)| encode_key(&w).map(|k| (k, lv)))
            .collect();
        keyed.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        keyed.dedup_by(|a, b| a.0 == b.0);
        // Palette, most frequent level vector first: small indices are the
        // common case, and the automaton stores them in fewer bytes.
        let mut uses: HashMap<Levels, usize> = HashMap::new();
        for (_, lv) in &keyed {
            *uses.entry(*lv).or_default() += 1;
        }
        let mut palette: Vec<(Levels, usize)> = uses.into_iter().collect();
        palette.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let index: HashMap<Levels, u64> = palette
            .iter()
            .enumerate()
            .map(|(i, (lv, _))| (*lv, i as u64))
            .collect();
        let mut builder = fst::MapBuilder::memory();
        for (key, lv) in &keyed {
            builder.insert(key, index[lv])?;
        }
        Ok(Self {
            map: Map::new(builder.into_inner()?)?,
            palette: palette.into_iter().map(|(lv, _)| lv).collect(),
            langs,
        })
    }

    pub fn from_data(data: LexiconData) -> Result<Self, fst::Error> {
        Ok(Self {
            map: Map::new(data.fst)?,
            palette: data.palette,
            langs: data.langs,
        })
    }

    pub fn to_data(&self) -> LexiconData {
        LexiconData {
            langs: self.langs.clone(),
            palette: self.palette.clone(),
            fst: self.map.as_fst().as_bytes().to_vec(),
        }
    }

    /// Serialised size in bytes (automaton + palette).
    pub fn byte_size(&self) -> usize {
        self.map.as_fst().as_bytes().len() + self.palette.len() * MAX_LANGS
    }

    /// Languages, in the order of their level bytes.
    pub fn langs(&self) -> &[String] {
        &self.langs
    }

    /// Index of a language code in this lexicon.
    pub fn lang_index(&self, code: &str) -> Option<usize> {
        self.langs.iter().position(|l| l == code)
    }

    /// Number of distinct words.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    fn palette_levels(&self, index: u64) -> Levels {
        self.palette
            .get(index as usize)
            .copied()
            .unwrap_or([0; MAX_LANGS])
    }

    /// All level bytes of `word` (all zero when it is unknown).
    pub fn levels(&self, word: &str) -> Levels {
        encode_key(word)
            .and_then(|k| self.map.get(k))
            .map_or([0; MAX_LANGS], |i| self.palette_levels(i))
    }

    /// The level of `word` in language `lang`, or -- with no language -- its
    /// highest level in any language.
    pub fn level(&self, word: &str, lang: Option<usize>) -> u8 {
        pick(self.levels(word), lang)
    }

    /// Occurrences per 100M tokens (see [`Self::level`]).
    pub fn count(&self, word: &str, lang: Option<usize>) -> f64 {
        dequantise(self.level(word, lang))
    }

    /// Every (word, levels) pair, in key order.
    pub fn entries(&self) -> Vec<(String, Levels)> {
        use fst::Streamer;
        let mut out = Vec::with_capacity(self.len());
        let mut stream = self.map.stream();
        while let Some((k, v)) = stream.next() {
            out.push((decode_key(k), self.palette_levels(v)));
        }
        out
    }

    // --- Walking the automaton (lattice x dictionary intersection) ---

    fn fst(&self) -> &Fst<Vec<u8>> {
        self.map.as_fst()
    }

    /// The empty prefix.
    pub fn root(&self) -> LexState {
        LexState {
            addr: self.fst().root().addr(),
            out: 0,
        }
    }

    /// Extend a prefix by the key bytes of one akshara; `None` if no word
    /// continues that way.
    pub fn step(&self, state: LexState, key_bytes: &[u8]) -> Option<LexState> {
        let fst = self.fst();
        let mut node = fst.node(state.addr);
        let mut out = state.out;
        for &b in key_bytes {
            let t = node.transition(node.find_input(b)?);
            out += t.out.value();
            node = fst.node(t.addr);
        }
        Some(LexState {
            addr: node.addr(),
            out,
        })
    }

    /// The levels of the word spelled so far, if the prefix is a whole word.
    pub fn word_levels(&self, state: LexState) -> Option<Levels> {
        let node = self.fst().node(state.addr);
        node.is_final()
            .then(|| self.palette_levels(state.out + node.final_output().value()))
    }
}

/// The level for one language, or the highest level of any language.
pub fn pick(levels: Levels, lang: Option<usize>) -> u8 {
    match lang {
        Some(l) => levels.get(l).copied().unwrap_or(0),
        None => levels.into_iter().max().unwrap_or(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantisation_is_monotone_and_round_trips_within_a_step() {
        assert_eq!(quantise(0, 1_000), 0);
        let total = 50_000_000;
        let mut last = 0;
        for count in [1, 2, 5, 50, 5_000, 500_000] {
            let q = quantise(count, total);
            assert!(q > last, "levels must grow with frequency");
            last = q;
            let back = dequantise(q) * total as f64 / SCALE_TOKENS;
            let rel = (back - count as f64).abs() / count as f64;
            assert!(rel < 0.1, "count {count} -> level {q} -> {back:.1}");
        }
    }

    fn tiny() -> Lexicon {
        Lexicon::build(
            vec!["hin".to_string(), "nep".to_string()],
            vec![
                ("घर".to_string(), [100, 90, 0, 0, 0, 0, 0, 0]),
                ("घरमा".to_string(), [0, 80, 0, 0, 0, 0, 0, 0]),
                ("छ".to_string(), [0, 120, 0, 0, 0, 0, 0, 0]),
                ("english".to_string(), [50, 0, 0, 0, 0, 0, 0, 0]),
            ],
        )
        .unwrap()
    }

    #[test]
    fn looks_up_per_language_and_best_of_all() {
        let lex = tiny();
        assert_eq!(lex.len(), 3, "a non-Devanagari key is not a word");
        assert_eq!(lex.level("घर", Some(0)), 100);
        assert_eq!(lex.level("छ", Some(0)), 0);
        assert_eq!(lex.level("छ", None), 120);
        assert_eq!(lex.level("नहीं", None), 0);
        let again = Lexicon::from_data(lex.to_data()).unwrap();
        assert_eq!(again.levels("घरमा"), lex.levels("घरमा"));
        assert_eq!(again.entries().len(), 3);
        assert_eq!(again.entries()[0].0, "घर");
    }

    #[test]
    fn walking_by_akshara_finds_words_and_prefixes() {
        let lex = tiny();
        let key = |s: &str| encode_key(s).unwrap();
        let s = lex.step(lex.root(), &key("घ")).unwrap();
        assert!(lex.word_levels(s).is_none(), "घ is only a prefix");
        let s = lex.step(s, &key("र")).unwrap();
        assert_eq!(lex.word_levels(s), Some([100, 90, 0, 0, 0, 0, 0, 0]));
        let s = lex.step(s, &key("मा")).unwrap();
        assert_eq!(lex.word_levels(s), Some([0, 80, 0, 0, 0, 0, 0, 0]));
        assert!(lex.step(s, &key("को")).is_none());
    }
}
