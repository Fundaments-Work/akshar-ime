// File: src/persistence.rs
use crate::core::engine::ImeEngine;
use crate::core::trie::Trie;
use crate::core::types::TransliterationModel;
use crate::fuzzy::symspell::SymSpell;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SerializableState {
    pub trie: Trie,
    pub context_model: crate::core::context::ContextModel,
    pub symspell: SymSpell,
    #[serde(default)]
    pub transliteration_model: TransliterationModel,
}

impl SerializableState {
    pub fn from_engine(engine: &ImeEngine) -> Self {
        Self {
            trie: engine.trie.clone(),
            context_model: engine.context_model.clone(),
            symspell: engine.symspell.clone(),
            transliteration_model: engine.transliteration_model.clone(),
        }
    }
    pub fn apply_to_engine(self, engine: &mut ImeEngine) {
        let mut trie = self.trie;
        trie.rebuild_index();
        engine.trie = trie;
        engine.context_model = self.context_model;
        engine.symspell = self.symspell;
        engine.transliteration_model = self.transliteration_model;
        // Exp 6: ranked lists cached under the old trie must not survive import.
        engine.clear_suggestion_cache();
    }
    pub fn to_bytes(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        Ok(bincode::serialize(self)?)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(bincode::deserialize(bytes)?)
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod native_fs {
    use super::*;
    use std::fs::{self, File};
    use std::io::{BufReader, BufWriter, Error};
    use std::path::Path;
    use tempfile::NamedTempFile;

    pub fn save_to_disk(engine: &ImeEngine, path: &Path) -> Result<(), Error> {
        let parent_dir = path.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent_dir)?;

        let state = SerializableState::from_engine(engine);

        let temp_file = NamedTempFile::new_in(parent_dir)?;
        let writer = BufWriter::new(&temp_file);

        bincode::serialize_into(writer, &state).map_err(std::io::Error::other)?;

        temp_file.persist(path)?;
        Ok(())
    }

    pub fn load_from_disk(path: &Path) -> Result<ImeEngine, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let state: SerializableState = bincode::deserialize_from(reader)?;

        let mut engine = ImeEngine::new();
        state.apply_to_engine(&mut engine);

        Ok(engine)
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native_fs::{load_from_disk, save_to_disk};
