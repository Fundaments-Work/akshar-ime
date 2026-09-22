// File: src/c_api.rs
#![cfg(not(target_arch = "wasm32"))]
use crate::ImeEngine;
use std::ffi::c_char;
use std::ffi::{CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

// Exp 3: the old `static mut *mut ImeEngine` was a data race on concurrent
// read (get_suggestions) + write (confirm_word). A process-wide locked
// singleton is still single-instance for IBus but sound.
static IME_ENGINE: OnceLock<Mutex<Option<ImeEngine>>> = OnceLock::new();

fn engine_cell() -> &'static Mutex<Option<ImeEngine>> {
    IME_ENGINE.get_or_init(|| Mutex::new(None))
}

// Exp 3 / P0-6: `dirs::config_dir()` can be absent and the path can be
// non-UTF8. Return None instead of panicking or collapsing to "".
fn get_dictionary_path() -> Option<PathBuf> {
    let mut path = dirs::config_dir()?;
    path.push("akshar-devanagari");
    path.push("user_dictionary.bin");
    Some(path)
}

#[no_mangle]
pub extern "C" fn akshar_ime_engine_init() {
    let result = catch_unwind(|| {
        let cell = engine_cell();
        let Ok(mut guard) = cell.lock() else {
            return;
        };
        if guard.is_some() {
            return;
        }
        let engine = match get_dictionary_path() {
            Some(dict_path) => {
                if let Some(parent) = dict_path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                // Non-UTF8 paths: fall back to in-memory engine rather than
                // persisting to "" (which always failed silently later).
                match dict_path.to_str() {
                    Some(s) => ImeEngine::from_file_or_new(s),
                    None => {
                        eprintln!("[akshar] non-UTF8 config path; using in-memory engine");
                        ImeEngine::new()
                    }
                }
            }
            None => {
                eprintln!("[akshar] no config dir; using in-memory engine");
                ImeEngine::new()
            }
        };
        *guard = Some(engine);
    });
    if result.is_err() {
        eprintln!("[Rust FATAL] A panic occurred during IME engine initialization.");
    }
}

#[no_mangle]
pub extern "C" fn akshar_ime_engine_destroy() {
    let _ = catch_unwind(|| {
        let cell = engine_cell();
        let Ok(mut guard) = cell.lock() else {
            return;
        };
        if let Some(engine) = guard.take() {
            // Best-effort persist; log instead of silently discarding.
            if let Err(e) = engine.save_dictionary() {
                eprintln!("[akshar] save_dictionary failed on destroy: {e}");
            }
        }
    });
}

/// Returns JSON-encoded Devanagari suggestions for the given roman prefix.
///
/// # Safety
///
/// `prefix` must be a valid NUL-terminated UTF-8 C string, or NULL (which
/// yields `"[]"`). The returned pointer must be released with
/// [`akshar_ime_free_string`].
#[no_mangle]
pub unsafe extern "C" fn akshar_ime_get_suggestions(prefix: *const c_char) -> *mut c_char {
    if prefix.is_null() {
        return CString::new("[]")
            .unwrap_or_else(|_| CString::new("").unwrap())
            .into_raw();
    }
    let roman_prefix = unsafe { CStr::from_ptr(prefix) }
        .to_str()
        .unwrap_or("")
        .to_string();
    let result = catch_unwind(AssertUnwindSafe(|| {
        let cell = engine_cell();
        let Ok(guard) = cell.lock() else {
            return "[]".to_string();
        };
        if let Some(engine) = guard.as_ref() {
            let suggestions = engine.get_suggestions(&roman_prefix, 8);
            let json_suggestions: Vec<String> = suggestions.into_iter().map(|(s, _)| s).collect();
            return serde_json::to_string(&json_suggestions).unwrap_or_else(|_| "[]".to_string());
        }
        "[]".to_string()
    }));
    let json_string = result.unwrap_or_else(|_| "[]".to_string());
    // `json_string` never contains NUL (serde_json escapes it), but stay
    // infallible: never `unwrap()` across an `extern "C"` boundary.
    CString::new(json_string)
        .unwrap_or_else(|_| CString::new("[]").unwrap())
        .into_raw()
}

/// Records a confirmed roman → Devanagari pair so the engine learns it.
///
/// # Safety
///
/// `roman` and `devanagari` must each be a valid NUL-terminated UTF-8 C
/// string, or NULL (which is treated as an empty string and ignored).
#[no_mangle]
pub unsafe extern "C" fn akshar_ime_confirm_word(roman: *const c_char, devanagari: *const c_char) {
    if roman.is_null() || devanagari.is_null() {
        return;
    }
    let roman_str = unsafe { CStr::from_ptr(roman) }.to_str().unwrap_or("");
    let devanagari_str = unsafe { CStr::from_ptr(devanagari) }.to_str().unwrap_or("");
    if roman_str.is_empty() || devanagari_str.is_empty() {
        return;
    }
    let roman_owned = roman_str.to_string();
    let dev_owned = devanagari_str.to_string();
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let cell = engine_cell();
        let Ok(mut guard) = cell.lock() else {
            return;
        };
        if let Some(engine) = guard.as_mut() {
            engine.user_confirms(&roman_owned, &dev_owned);
            // Exp 3/P0-4: previously learning lived only in RAM until
            // destroy, so a kill/crash lost the whole session. Persist
            // best-effort on every confirm (commit-frequency, not keystroke).
            if let Err(e) = engine.save_dictionary() {
                eprintln!("[akshar] save_dictionary failed on confirm: {e}");
            }
        }
    }));
}

/// Chooses the language suggestions are ranked for: an ISO 639-3 code
/// (`"hin"`, `"nep"`, ...) or its ISO 639-1 alias (`"hi"`), or NULL / `""` /
/// `"auto"` for language-blind ranking.  Returns 1 when the model supports
/// the language (or auto was requested), 0 when it fell back to auto.
///
/// # Safety
///
/// `code` must be a valid NUL-terminated UTF-8 C string, or NULL.
#[no_mangle]
pub unsafe extern "C" fn akshar_ime_set_language(code: *const c_char) -> i32 {
    let code = if code.is_null() {
        None
    } else {
        unsafe { CStr::from_ptr(code) }
            .to_str()
            .ok()
            .map(str::to_string)
    };
    catch_unwind(AssertUnwindSafe(|| {
        let Ok(mut guard) = engine_cell().lock() else {
            return 0;
        };
        match guard.as_mut() {
            Some(engine) => i32::from(engine.set_language(code.as_deref())),
            None => 0,
        }
    }))
    .unwrap_or(0)
}

/// Frees a string previously returned by [`akshar_ime_get_suggestions`].
///
/// # Safety
///
/// `s` must be a pointer returned by [`akshar_ime_get_suggestions`] that has
/// not been freed already, or NULL.
#[no_mangle]
pub unsafe extern "C" fn akshar_ime_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}
