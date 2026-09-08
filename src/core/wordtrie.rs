// File: src/core/wordtrie.rs
//
// M4-real: a trie over real Devanagari words, keyed by akshara id, with
// corpus frequencies at terminals.  Built from running-text counts (e.g. the
// a Wikipedia dump).  Intersecting the transliteration lattice with this trie
// restricts decoding to actual words — combinatorial ambiguity reduction, no
// probability required for the pruning step.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WordTrie {
    /// children[node] = { akshara_id -> child_node }
    pub children: Vec<HashMap<u32, usize>>,
    /// terminal[node] = frequency of the word ending exactly here.
    pub terminal: HashMap<usize, u32>,
    /// Number of words inserted.
    pub words: usize,
}

impl WordTrie {
    pub fn new() -> Self {
        Self {
            children: vec![HashMap::new()],
            terminal: HashMap::new(),
            words: 0,
        }
    }

    /// Insert one word given its akshara-id sequence and frequency.
    pub fn insert(&mut self, aks: &[u32], freq: u32) {
        let mut node = 0usize;
        for &a in aks {
            let next = self.children[node].get(&a).copied();
            node = match next {
                Some(n) => n,
                None => {
                    let n = self.children.len();
                    self.children.push(HashMap::new());
                    self.children[node].insert(a, n);
                    n
                }
            };
        }
        // Keep the max frequency if the same akshara sequence recurs.
        let e = self.terminal.entry(node).or_insert(0);
        *e = (*e).max(freq);
        self.words += 1;
    }

    #[inline]
    pub fn child(&self, node: usize, a: u32) -> Option<usize> {
        self.children[node].get(&a).copied()
    }

    #[inline]
    pub fn freq(&self, node: usize) -> Option<u32> {
        self.terminal.get(&node).copied()
    }

    /// Build from a bincode HashMap<String, u32> of word frequencies, mapping
    /// each word through the akshara segmenter and the model's akshara table.
    /// Words containing aksharas unknown to the model are skipped.
    pub fn from_freq_map(
        freq: &HashMap<String, u32>,
        akshara_id: &dyn Fn(&str) -> Option<u32>,
        min_count: u32,
    ) -> Self {
        let mut trie = Self::new();
        for (word, &count) in freq {
            if count < min_count {
                continue;
            }
            let mut aks = Vec::with_capacity(8);
            let mut ok = true;
            for unit in crate::core::akshara::segment(word) {
                match akshara_id(&unit) {
                    Some(id) => aks.push(id),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok && !aks.is_empty() {
                trie.insert(&aks, count);
            }
        }
        trie
    }
}
