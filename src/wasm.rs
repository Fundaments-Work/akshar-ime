// File: src/wasm.rs
// WASM bindings for akshar-ime — runs the full transliteration engine in the browser.
//
// Design goals:
// * Small wasm binary (~300-600KB gzipped); model is fetched separately, not baked in.
// * No filesystem; persistence via localStorage.
// * Works with any <input>, <textarea>, or contenteditable via the JS helper.

use crate::core::engine::ImeEngine;
use wasm_bindgen::prelude::*;

// Optional: better panic messages in browser console
#[wasm_bindgen]
pub fn init_panic_hook() {
    #[cfg(feature = "wasm")]
    console_error_panic_hook();
}

#[cfg(feature = "wasm")]
fn console_error_panic_hook() {
    // Use std panic hook that logs to console.error via wasm-bindgen
    std::panic::set_hook(Box::new(|info| {
        web_sys::console::error_1(&JsValue::from_str(&format!("{}", info)));
    }));
}

// ---------------------------------------------------------------------------
// WasmEngine — the main exported class
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct WasmEngine {
    inner: ImeEngine,
}

#[wasm_bindgen]
impl WasmEngine {
    /// Create an engine from raw bytes already fetched in JS.
    ///
    /// `model_bytes` is the `translit_model.bin` file as Uint8Array.
    /// `lexicon_bytes` is accepted and IGNORED. The corpus lexicon was removed
    /// in v1.1.0 (it measured 0.00pp on every split and was dead by
    /// construction on the unified-container path); the parameter is retained
    /// so existing `createEngine(model, lexicon, weights)` callers keep
    /// working. Pass `null`.
    /// `reranker_json` may be null/undefined for default weights.
    #[wasm_bindgen(constructor)]
    pub fn from_bytes(
        model_bytes: &[u8],
        lexicon_bytes: Option<Vec<u8>>,
        reranker_json: Option<String>,
    ) -> Result<WasmEngine, JsValue> {
        let lexicon_slice = lexicon_bytes.as_deref();
        let json_slice = reranker_json.as_deref();
        let engine = ImeEngine::from_bytes_with_weights(model_bytes, lexicon_slice, json_slice)
            .map_err(|e| JsValue::from_str(&format!("failed to load model: {}", e)))?;
        let mut w = WasmEngine { inner: engine };
        // Try to restore learned state from localStorage (silent if missing)
        let _ = w.restore_from_storage();
        Ok(w)
    }

    /// Create an engine with an empty (untrained) model — useful for testing without downloading.
    /// Will only do dictionary + fuzzy matching, no transliteration.
    #[wasm_bindgen]
    pub fn empty() -> WasmEngine {
        WasmEngine {
            inner: ImeEngine::new(),
        }
    }

    /// Get Devanagari suggestions for a roman prefix.
    /// Returns JS array of strings (up to `count`, default 8).
    #[wasm_bindgen(js_name = getSuggestions)]
    pub fn get_suggestions(&self, prefix: &str, count: Option<usize>) -> Vec<JsValue> {
        let n = count.unwrap_or(8).clamp(1, 20);
        self.inner
            .get_suggestions(prefix, n)
            .into_iter()
            .map(|(s, _)| JsValue::from_str(&s))
            .collect()
    }

    /// Get suggestions with scores: returns JSON string `[{"text":"नमस्ते","score":123}, ...]`
    #[wasm_bindgen(js_name = getSuggestionsWithScores)]
    pub fn get_suggestions_with_scores(&self, prefix: &str, count: Option<usize>) -> String {
        let n = count.unwrap_or(8).clamp(1, 20);
        let suggestions = self.inner.get_suggestions(prefix, n);
        let arr: Vec<serde_json::Value> = suggestions
            .into_iter()
            .map(|(text, score)| serde_json::json!({"text": text, "score": score}))
            .collect();
        serde_json::to_string(&arr).unwrap_or_else(|_| "[]".into())
    }

    /// Top transliteration (single string) or empty if none.
    #[wasm_bindgen(js_name = transliterate)]
    pub fn transliterate(&self, roman: &str) -> String {
        self.inner
            .get_suggestions(roman, 1)
            .into_iter()
            .next()
            .map(|(s, _)| s)
            .unwrap_or_default()
    }

    /// Record that user confirmed `roman -> devanagari`. Learning is immediate and persisted.
    #[wasm_bindgen(js_name = confirm)]
    pub fn confirm(&mut self, roman: &str, devanagari: &str) {
        self.inner.user_confirms(roman, devanagari);
        let _ = self.save_to_storage();
    }

    /// Rank suggestions for one language: an ISO 639-3 code ("hin", "nep",
    /// ...) or ISO 639-1 alias ("hi"); null, "" or "auto" for language-blind.
    /// Returns false if the model does not support it (auto is used).
    #[wasm_bindgen(js_name = setLanguage)]
    pub fn set_language(&mut self, code: Option<String>) -> bool {
        self.inner.set_language(code.as_deref())
    }

    /// Languages the model can rank for (ISO 639-3 codes).
    #[wasm_bindgen(js_name = languages)]
    pub fn languages(&self) -> Vec<JsValue> {
        self.inner
            .languages()
            .iter()
            .map(|l| JsValue::from_str(l))
            .collect()
    }

    /// Returns true if model is loaded and valid (not empty fallback).
    #[wasm_bindgen(js_name = isReady)]
    pub fn is_ready(&self) -> bool {
        // Model is valid if it has at least one akshara
        !self.inner.decoder.model.aksharas.is_empty()
    }

    /// Number of aksharas in model vocabulary (diagnostic).
    #[wasm_bindgen(js_name = vocabSize)]
    pub fn vocab_size(&self) -> usize {
        self.inner.decoder.model.aksharas.len()
    }

    /// Export learned state as base64 string for backup / sync.
    #[wasm_bindgen(js_name = exportState)]
    pub fn export_state(&self) -> Result<String, JsValue> {
        let bytes = self
            .inner
            .learned_state_to_bytes()
            .map_err(|e| JsValue::from_str(&format!("export failed: {}", e)))?;
        Ok(base64_encode(&bytes))
    }

    /// Import learned state from base64 string.
    #[wasm_bindgen(js_name = importState)]
    pub fn import_state(&mut self, b64: &str) -> Result<(), JsValue> {
        let bytes = base64_decode(b64).map_err(|e| JsValue::from_str(&e))?;
        self.inner
            .load_learned_state_from_bytes(&bytes)
            .map_err(|e| JsValue::from_str(&format!("import failed: {}", e)))?;
        self.save_to_storage().map_err(|e| JsValue::from_str(&e))?;
        Ok(())
    }

    /// Clear learned dictionary and persist.
    #[wasm_bindgen(js_name = resetLearning)]
    pub fn reset_learning(&mut self) -> Result<(), JsValue> {
        // Exp 6: reset in place so the unified vocab/word-trie/sparse table
        // survive (from_model would drop them on wasm32).
        self.inner.reset_learned_state();
        self.save_to_storage().map_err(|e| JsValue::from_str(&e))?;
        // Also clear storage key
        if let Some(storage) = local_storage() {
            let _ = storage.remove_item(STORAGE_KEY);
        }
        Ok(())
    }

    /// How many learned words are stored.
    #[wasm_bindgen(js_name = learnedCount)]
    pub fn learned_count(&self) -> usize {
        self.inner.trie.metadata_store.len()
    }
}

// ---------------------------------------------------------------------------
// Async fetch helpers — fetch model/lexicon/weights from URLs
// ---------------------------------------------------------------------------

const STORAGE_KEY: &str = "akshar-ime-state-v1";

fn local_storage() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok()?
}

impl WasmEngine {
    fn save_to_storage(&self) -> Result<(), String> {
        let storage = local_storage().ok_or("no localStorage")?;
        let bytes = self
            .inner
            .learned_state_to_bytes()
            .map_err(|e| e.to_string())?;
        // localStorage has ~5MB limit; bincode of learned trie is tiny (<100KB for thousands of words)
        let b64 = base64_encode(&bytes);
        storage
            .set_item(STORAGE_KEY, &b64)
            .map_err(|_| "localStorage set failed".to_string())?;
        Ok(())
    }

    fn restore_from_storage(&mut self) -> Result<(), String> {
        let storage = local_storage().ok_or("no storage")?;
        let b64 = storage
            .get_item(STORAGE_KEY)
            .map_err(|_| "get failed".to_string())?
            .ok_or("no saved state")?;
        if b64.is_empty() {
            return Err("empty".into());
        }
        let bytes = base64_decode(&b64)?;
        self.inner
            .load_learned_state_from_bytes(&bytes)
            .map_err(|e| e.to_string())?;
        Ok(())
    }
}

/// Fetch bytes from URL using browser fetch API.
async fn fetch_bytes(url: &str) -> Result<Vec<u8>, JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;
    let resp_value = wasm_bindgen_futures::JsFuture::from(window.fetch_with_str(url)).await?;
    let resp: web_sys::Response = resp_value
        .dyn_into()
        .map_err(|_| JsValue::from_str("not a Response"))?;
    if !resp.ok() {
        return Err(JsValue::from_str(&format!(
            "fetch {} failed: {} {}",
            url,
            resp.status(),
            resp.status_text()
        )));
    }
    let buf = wasm_bindgen_futures::JsFuture::from(
        resp.array_buffer()
            .map_err(|_| JsValue::from_str("arrayBuffer failed"))?,
    )
    .await?;
    let u8arr = js_sys::Uint8Array::new(&buf);
    let mut bytes = vec![0u8; u8arr.length() as usize];
    u8arr.copy_to(&mut bytes);
    Ok(bytes)
}

async fn fetch_text(url: &str) -> Result<String, JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;
    let resp_value = wasm_bindgen_futures::JsFuture::from(window.fetch_with_str(url)).await?;
    let resp: web_sys::Response = resp_value
        .dyn_into()
        .map_err(|_| JsValue::from_str("not a Response"))?;
    if !resp.ok() {
        return Err(JsValue::from_str(&format!(
            "fetch {} failed: {} {}",
            url,
            resp.status(),
            resp.status_text()
        )));
    }
    let text = wasm_bindgen_futures::JsFuture::from(
        resp.text()
            .map_err(|_| JsValue::from_str("text() failed"))?,
    )
    .await?;
    text.as_string()
        .ok_or_else(|| JsValue::from_str("not a string"))
}

/// Async factory: fetch model (+ optional lexicon/weights) from URLs and create engine.
///
/// Usage from JS:
/// ```js
/// const engine = await createEngine("https://cdn.example.com/translit_model.bin");
/// // with lexicon:
/// const engine = await createEngine(url, lexiconUrl, weightsUrl);
/// ```
#[wasm_bindgen(js_name = createEngine)]
pub async fn create_engine(
    model_url: String,
    lexicon_url: Option<String>,
    reranker_url: Option<String>,
) -> Result<WasmEngine, JsValue> {
    let model_bytes = fetch_bytes(&model_url).await?;
    let lexicon_bytes = if let Some(u) = lexicon_url {
        if u.is_empty() {
            None
        } else {
            Some(fetch_bytes(&u).await?)
        }
    } else {
        None
    };
    let reranker_json = if let Some(u) = reranker_url {
        if u.is_empty() {
            None
        } else {
            Some(fetch_text(&u).await?)
        }
    } else {
        None
    };
    WasmEngine::from_bytes(&model_bytes, lexicon_bytes, reranker_json)
}

/// Create engine from model URL only (convenience, lexicon optional).
#[wasm_bindgen(js_name = createEngineFromModelUrl)]
pub async fn create_engine_from_model_url(model_url: String) -> Result<WasmEngine, JsValue> {
    create_engine(model_url, None, None).await
}

// ---------------------------------------------------------------------------
// Simple free functions for non-class usage + diagnostics
// ---------------------------------------------------------------------------

/// Returns the engine version (crate version).
#[wasm_bindgen(js_name = getVersion)]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Transliterate without needing an engine instance, using an empty model — mainly for testing wiring.
#[wasm_bindgen(js_name = quickTransliterate)]
pub fn quick_transliterate(roman: &str) -> String {
    let e = ImeEngine::new();
    e.get_suggestions(roman, 1)
        .into_iter()
        .next()
        .map(|(s, _)| s)
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Tiny base64 (no extra crate — keeps wasm small)
// ---------------------------------------------------------------------------

const B64_TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64_TABLE[((n >> 18) & 63) as usize] as char);
        out.push(B64_TABLE[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            B64_TABLE[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            B64_TABLE[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    let mut table = [255u8; 256];
    for (i, &c) in B64_TABLE.iter().enumerate() {
        table[c as usize] = i as u8;
    }
    let bytes = s.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return Err("invalid base64 length".into());
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks(4) {
        let mut n = 0u32;
        let mut pad = 0;
        for (i, &c) in chunk.iter().enumerate() {
            if c == b'=' {
                pad += 1;
                if i < 2 {
                    return Err("invalid padding".into());
                }
            } else {
                let v = table[c as usize];
                if v == 255 {
                    return Err(format!("invalid base64 char {}", c as char));
                }
                n |= (v as u32) << (18 - i * 6);
            }
        }
        out.push((n >> 16) as u8);
        if pad < 2 {
            out.push((n >> 8) as u8);
        }
        if pad < 1 {
            out.push(n as u8);
        }
        if pad > 0 && chunk[3] != b'=' {
            // invalid but ignore
        }
    }
    Ok(out)
}

#[cfg(test)]
mod wasm_tests {
    use super::*;

    #[test]
    fn b64_roundtrip() {
        for s in [b"hello".as_slice(), b"", b"a", b"ab", b"abc", b"Man"] {
            let enc = base64_encode(s);
            let dec = base64_decode(&enc).unwrap();
            assert_eq!(dec, s);
        }
    }
}
