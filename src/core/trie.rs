// File: src/core/trie.rs
use crate::core::types::{WordId, WordMetadata};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

#[derive(Clone, Serialize, Deserialize)]
struct Node {
    children: HashMap<u8, usize>,
    word_id: Option<WordId>,
    max_freq_in_subtree: u64,
}

impl Node {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            word_id: None,
            max_freq_in_subtree: 0,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Trie {
    nodes: Vec<Node>,
    pub metadata_store: Vec<WordMetadata>,
    /// O(1) Devanagari -> WordId index. Skipped in serialization and rebuilt
    /// on load so the on-disk format is unchanged; fixes the previous O(n)
    /// linear scan in `find_word_id_by_devanagari`.
    #[serde(skip, default = "HashMap::new")]
    id_by_devanagari: HashMap<String, WordId>,
}

impl Default for Trie {
    fn default() -> Self {
        Self::new()
    }
}

impl Trie {
    pub fn new() -> Self {
        Self {
            nodes: vec![Node::new()],
            metadata_store: Vec::new(),
            id_by_devanagari: HashMap::new(),
        }
    }

    pub fn find_word_id_by_devanagari(&self, devanagari: &str) -> Option<WordId> {
        self.id_by_devanagari.get(devanagari).copied()
    }

    /// Rebuild the in-memory index after deserialization (format unchanged).
    pub fn rebuild_index(&mut self) {
        self.id_by_devanagari.clear();
        for (id, meta) in self.metadata_store.iter().enumerate() {
            self.id_by_devanagari
                .entry(meta.devanagari.clone())
                .or_insert(id);
        }
    }

    pub fn get_or_create_metadata(&mut self, devanagari: &str) -> WordId {
        if let Some(id) = self.find_word_id_by_devanagari(devanagari) {
            id
        } else {
            let new_meta = WordMetadata {
                devanagari: devanagari.to_string(),
                frequency: 0,
                variants: HashSet::new(),
            };
            self.metadata_store.push(new_meta);
            let id = self.metadata_store.len() - 1;
            self.id_by_devanagari.insert(devanagari.to_string(), id);
            id
        }
    }

    pub fn insert(&mut self, key: &str, word_id: WordId, _frequency: u64) {
        let mut node_idx = 0;
        let mut path = vec![0];

        for &byte in key.as_bytes() {
            // Check and modify in separate steps to avoid holding a mutable
            // borrow of a node while touching the parent `nodes` vector.
            let next_idx = if let Some(&child_idx) = self.nodes[node_idx].children.get(&byte) {
                child_idx
            } else {
                let new_node_idx = self.nodes.len();
                self.nodes.push(Node::new());
                self.nodes[node_idx].children.insert(byte, new_node_idx);
                new_node_idx
            };
            // --- END OF FIX ---
            node_idx = next_idx;
            path.push(node_idx);
        }
        self.nodes[node_idx].word_id = Some(word_id);

        for &idx in path.iter().rev() {
            let current_node_freq = self.nodes[idx]
                .word_id
                .map_or(0, |id| self.metadata_store[id].frequency);

            let max_child_freq = self.nodes[idx]
                .children
                .values()
                .map(|&child_idx| self.nodes[child_idx].max_freq_in_subtree)
                .max()
                .unwrap_or(0);

            let new_max_freq = current_node_freq.max(max_child_freq);

            if new_max_freq == self.nodes[idx].max_freq_in_subtree {
                break;
            }
            self.nodes[idx].max_freq_in_subtree = new_max_freq;
        }
    }

    pub fn get_top_k_suggestions(&self, prefix: &str, k: usize) -> Vec<(WordId, u64)> {
        let mut node_idx = 0;
        for &byte in prefix.as_bytes() {
            if let Some(&next_idx) = self.nodes[node_idx].children.get(&byte) {
                node_idx = next_idx;
            } else {
                return vec![];
            }
        }

        // Min-heap via Reverse so peek() is the kth-largest (smallest kept).
        // A plain max-heap here kept the wrong k (e.g. {3,1} instead of {3,2}).
        let mut heap: BinaryHeap<Reverse<(u64, WordId)>> = BinaryHeap::with_capacity(k + 1);
        self.dfs_pruning_search(node_idx, k, &mut heap);

        heap.into_iter().map(|r| (r.0 .1, r.0 .0)).collect()
    }

    fn dfs_pruning_search(
        &self,
        node_idx: usize,
        k: usize,
        heap: &mut BinaryHeap<Reverse<(u64, WordId)>>,
    ) {
        if k == 0 {
            return;
        }
        let node = &self.nodes[node_idx];

        if let Some(id) = node.word_id {
            let freq = self.metadata_store[id].frequency;
            if freq > 0 {
                if heap.len() < k {
                    heap.push(Reverse((freq, id)));
                } else if freq > heap.peek().unwrap().0 .0 {
                    heap.pop();
                    heap.push(Reverse((freq, id)));
                }
            }
        }

        let min_freq_in_heap = if heap.len() == k {
            heap.peek().unwrap().0 .0
        } else {
            0
        };

        for &child_idx in node.children.values() {
            if self.nodes[child_idx].max_freq_in_subtree > min_freq_in_heap {
                self.dfs_pruning_search(child_idx, k, heap);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trie_with_freqs(freqs: &[u64]) -> Trie {
        let mut t = Trie::new();
        for (i, &f) in freqs.iter().enumerate() {
            let dev = format!("w{i}");
            let id = t.get_or_create_metadata(&dev);
            t.metadata_store[id].frequency = f;
            t.insert(&format!("k{i}"), id, f);
        }
        t
    }

    #[test]
    fn top_k_returns_largest_not_smallest() {
        // Exp 1 (H1): freqs 1,2,3 k=2 must yield {3,2}, not {3,1}.
        let t = trie_with_freqs(&[1, 2, 3]);
        let mut got: Vec<u64> = t
            .get_top_k_suggestions("k", 2)
            .into_iter()
            .map(|(_, f)| f)
            .collect();
        got.sort_unstable();
        assert_eq!(got, vec![2, 3]);
    }

    #[test]
    fn top_k_k_zero_returns_empty() {
        let t = trie_with_freqs(&[5]);
        assert!(t.get_top_k_suggestions("k", 0).is_empty());
    }
}
