// File: src/learning.rs
use crate::core::{context::ContextModel, trie::Trie, types::TransliterationModel};
use crate::fuzzy::symspell::SymSpell;

pub struct LearningEngine {
    frequency_increment: u64,
}

pub struct WordConfirmation {
    pub roman: String,
    pub devanagari: String,
}

impl Default for LearningEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl LearningEngine {
    pub fn new() -> Self {
        Self {
            frequency_increment: 1,
        }
    }

    pub fn learn(
        &self,
        trie: &mut Trie,
        context_model: &mut ContextModel,
        symspell: &mut SymSpell,
        transliteration_model: &mut TransliterationModel,
        confirmation: &WordConfirmation,
    ) {
        let word_id = trie.get_or_create_metadata(&confirmation.devanagari);

        let metadata = &mut trie.metadata_store[word_id];
        metadata.frequency += self.frequency_increment;

        // Track P(roman|word) evidence from each confirmation event.
        *transliteration_model
            .entry((confirmation.roman.clone(), word_id))
            .or_insert(0) += self.frequency_increment;

        // Only add the variant if it's new, to avoid bloating the metadata store
        if metadata.variants.insert(confirmation.roman.clone()) {
            // OPTIMIZATION: Only add the primary Roman variant and the Devanagari word itself to the
            // fuzzy index. This keeps the SymSpell dictionary much smaller and faster than
            // indexing every single user-typed variant.
            symspell.add_word(&confirmation.roman, word_id);
            if metadata.variants.len() == 1 {
                // First time we see this word, add its Devanagari form too
                symspell.add_word(&confirmation.devanagari, word_id);
            }
        }

        let updated_freq = metadata.frequency;

        trie.insert(&confirmation.roman, word_id, updated_freq);

        context_model.add_word(word_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{context::ContextModel, trie::Trie};
    use std::collections::HashMap;

    #[test]
    fn learn_updates_transliteration_pair_counts() {
        let learner = LearningEngine::new();
        let mut trie = Trie::new();
        let mut context_model = ContextModel::new(3);
        let mut symspell = SymSpell::new(2);
        let mut transliteration_model = HashMap::new();

        learner.learn(
            &mut trie,
            &mut context_model,
            &mut symspell,
            &mut transliteration_model,
            &WordConfirmation {
                roman: "ram".to_string(),
                devanagari: "राम".to_string(),
            },
        );
        learner.learn(
            &mut trie,
            &mut context_model,
            &mut symspell,
            &mut transliteration_model,
            &WordConfirmation {
                roman: "ram".to_string(),
                devanagari: "राम".to_string(),
            },
        );
        learner.learn(
            &mut trie,
            &mut context_model,
            &mut symspell,
            &mut transliteration_model,
            &WordConfirmation {
                roman: "raam".to_string(),
                devanagari: "राम".to_string(),
            },
        );

        let word_id = trie
            .find_word_id_by_devanagari("राम")
            .expect("word id missing");
        assert_eq!(trie.metadata_store[word_id].frequency, 3);
        assert_eq!(
            transliteration_model.get(&("ram".to_string(), word_id)),
            Some(&2)
        );
        assert_eq!(
            transliteration_model.get(&("raam".to_string(), word_id)),
            Some(&1)
        );
    }
}
