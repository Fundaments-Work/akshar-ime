/**
 * Akshar IME - JS wrapper for easy website integration
 *
 * Drop-in Devanagari transliteration for any <input>, <textarea>, or contenteditable.
 *
 * Usage (ES Module):
 *   import { AksharIME } from './js/akshar-ime.js';
 *   await AksharIME.init({ modelUrl: '/data/akshar_wasm.model' });
 *   AksharIME.attach(document.querySelector('input'));
 *   // or auto-attach all [data-akshar] elements:
 *   AksharIME.autoAttach();
 *
 * Usage (script tag):
 *   <script type="module">
 *     import { AksharIME } from 'https://cdn.example.com/akshar-ime/js/akshar-ime.js';
 *   </script>
 *
 * All Roman -> Devanagari conversion runs locally in WASM (no server, no tracking).
 * Learned words persist in localStorage.
 */

import initWasm, { WasmEngine, createEngine, getVersion } from '../wasm/pkg/akshar_ime.js';

// ----- public API -----
export const AksharIME = {
  version: null,
  engine: null,
  _ready: null,
  _readyResolve: null,
  _styleInjected: false,

  /**
   * Initialize the WASM engine.
   * @param {Object} opts
   * @param {string} opts.modelUrl - URL to akshar_wasm.model, the unified
   *   container (required). A bare translit_model.bin is still accepted.
   * @param {string} [opts.lexiconUrl] - URL to roman_lexicon.bin (optional, +25MB gzipped)
   * @param {string} [opts.rerankerUrl] - URL to reranker_weights.json (optional)
   * @param {string} [opts.wasmUrl] - Override wasm pkg URL (default: auto)
   * @returns {Promise<WasmEngine>}
   */
  async init(opts = {}) {
    if (this.engine) return this.engine;
    if (this._ready) return this._ready;

    const modelUrl = opts.modelUrl || opts.model_url;
    if (!modelUrl) {
      throw new Error('AksharIME.init({ modelUrl }) is required. Example: AksharIME.init({ modelUrl: "/data/akshar_wasm.model" })');
    }

    this._ready = (async () => {
      // init wasm module first (loads .wasm file relative to pkg js)
      if (opts.wasmUrl) {
        await initWasm({ module_or_path: opts.wasmUrl });
      } else {
        await initWasm();
      }
      try { const { init_panic_hook } = await import('../wasm/pkg/akshar_ime.js'); init_panic_hook(); } catch {}

      const engine = await createEngine(
        modelUrl,
        opts.lexiconUrl || opts.lexicon_url || null,
        opts.rerankerUrl || opts.reranker_url || null
      );
      this.engine = engine;
      this.version = getVersion();
      console.info(`[AksharIME] ready v${this.version} vocab=${engine.vocabSize()} learned=${engine.learnedCount()}`);
      return engine;
    })();

    return this._ready;
  },

  /** Ensure engine is ready; throws if init() not called */
  ensureReady() {
    if (!this.engine) throw new Error('AksharIME not initialized. Call await AksharIME.init({ modelUrl }) first.');
    return this.engine;
  },

  /** Transliterate a single word (no UI) */
  transliterate(roman) {
    return this.ensureReady().transliterate(roman);
  },

  /** Get suggestions array */
  getSuggestions(prefix, count = 8) {
    return this.ensureReady().getSuggestions(prefix, count);
  },

  /**
   * Attach IME to an input/textarea/contenteditable element.
   * @param {HTMLElement} el
   * @param {Object} [opts]
   * @param {number} [opts.suggestions=5] - max suggestions
   * @param {string} [opts.triggerRegex] - regex for word boundary (default: last word)
   * @param {boolean} [opts.autoConfirmOnSpace=true]
   * @param {(roman:string, dev:string)=>void} [opts.onCommit] - callback
   * @returns {{destroy:()=>void}}
   */
  attach(el, opts = {}) {
    if (!el) throw new Error('attach() requires an element');
    const engine = this.ensureReady();
    const state = {
      el,
      opts: { suggestions: 5, autoConfirmOnSpace: true, ...opts },
      selected: 0,
      suggestions: [],
      lastWord: '',
      lastWordRange: null,
    };

    this._injectStyles();

    // Create suggestion box (shared per element)
    const box = document.createElement('div');
    box.className = 'akshar-ime-box';
    box.style.display = 'none';
    document.body.appendChild(box);

    const updatePosition = () => {
      if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
        // For inputs, position below the element
        const r = el.getBoundingClientRect();
        box.style.left = `${r.left + window.scrollX}px`;
        box.style.top = `${r.bottom + window.scrollY + 4}px`;
        box.style.minWidth = `${Math.max(160, r.width * 0.6)}px`;
      } else {
        // contenteditable: try caret position
        const sel = window.getSelection();
        if (sel && sel.rangeCount) {
          const range = sel.getRangeAt(0).cloneRange();
          const rect = range.getBoundingClientRect();
          if (rect.width || rect.height) {
            box.style.left = `${rect.left + window.scrollX}px`;
            box.style.top = `${rect.bottom + window.scrollY + 4}px`;
          }
        }
      }
    };

    const render = () => {
      if (!state.suggestions.length) { box.style.display = 'none'; return; }
      box.innerHTML = '';
      state.suggestions.forEach((text, i) => {
        const item = document.createElement('div');
        item.className = 'akshar-ime-item' + (i === state.selected ? ' selected' : '');
        item.textContent = `${i + 1}. ${text}`;
        item.dataset.index = i;
        item.addEventListener('mousedown', (e) => {
          e.preventDefault();
          commit(i);
        });
        box.appendChild(item);
      });
      const romanHint = document.createElement('div');
      romanHint.className = 'akshar-ime-hint';
      romanHint.textContent = state.lastWord;
      box.appendChild(romanHint);
      box.style.display = 'block';
      updatePosition();
    };

    const hide = () => { box.style.display = 'none'; state.suggestions = []; };

    const getWordInfo = () => {
      if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
        const val = el.value;
        const pos = el.selectionStart ?? val.length;
        const left = val.slice(0, pos);
        const m = left.match(/([a-zA-Z]+)$/);
        if (!m) return null;
        const word = m[1];
        return { word, start: pos - word.length, end: pos, value: val, pos };
      } else {
        // contenteditable
        const sel = window.getSelection();
        if (!sel || !sel.rangeCount) return null;
        const range = sel.getRangeAt(0);
        const text = range.startContainer.textContent || '';
        const offset = range.startOffset;
        const left = text.slice(0, offset);
        const m = left.match(/([a-zA-Z]+)$/);
        if (!m) return null;
        const word = m[1];
        return { word, start: offset - word.length, end: offset, text, range };
      }
    };

    const commit = (index) => {
      const dev = state.suggestions[index];
      const roman = state.lastWord;
      if (!dev || !roman) return;
      const info = getWordInfo();
      if (!info) return;

      if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
        const before = info.value.slice(0, info.start);
        const after = info.value.slice(info.end);
        el.value = before + dev + after;
        const newPos = before.length + dev.length;
        el.setSelectionRange(newPos, newPos);
        el.dispatchEvent(new Event('input', { bubbles: true }));
      } else {
        // contenteditable: replace text node slice
        const node = window.getSelection().getRangeAt(0).startContainer;
        if (node.nodeType === Node.TEXT_NODE) {
          const before = node.textContent.slice(0, info.start);
          const after = node.textContent.slice(info.end);
          node.textContent = before + dev + after;
          const newOffset = before.length + dev.length;
          const r = document.createRange();
          r.setStart(node, newOffset);
          r.collapse(true);
          const sel = window.getSelection();
          sel.removeAllRanges();
          sel.addRange(r);
        }
      }
      engine.confirm(roman.toLowerCase(), dev);
      if (state.opts.onCommit) state.opts.onCommit(roman, dev);
      hide();
      el.focus();
    };

    const onInput = () => {
      const info = getWordInfo();
      if (!info || info.word.length < 1) { hide(); return; }
      state.lastWord = info.word;
      const lower = info.word.toLowerCase();
      const sugs = engine.getSuggestions(lower, state.opts.suggestions);
      if (!sugs.length) { hide(); return; }
      state.suggestions = sugs;
      state.selected = 0;
      render();
    };

    const onKeyDown = (e) => {
      if (box.style.display === 'none' || !state.suggestions.length) {
        // allow space to still trigger learning if needed, but not navigation
        return;
      }
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        state.selected = (state.selected + 1) % state.suggestions.length;
        render();
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        state.selected = (state.selected - 1 + state.suggestions.length) % state.suggestions.length;
        render();
      } else if (e.key === 'Enter' || e.key === 'Tab') {
        if (state.suggestions.length) {
          e.preventDefault();
          commit(state.selected);
        }
      } else if (e.key === 'Escape') {
        e.preventDefault();
        hide();
      } else if (e.key === ' ' && state.opts.autoConfirmOnSpace) {
        // Space confirms top suggestion automatically? Optional behavior:
        // We do NOT auto-commit on space by default for predictability; only Enter/Tab commits.
        // But if user typed "namaste " and suggestion is there, they can press Enter.
        // Hide after space so next word starts fresh.
        hide();
      } else if (/^[1-9]$/.test(e.key)) {
        const idx = parseInt(e.key, 10) - 1;
        if (idx < state.suggestions.length) {
          e.preventDefault();
          commit(idx);
        }
      }
    };

    const onBlur = () => {
      // delay hide to allow click on suggestion
      setTimeout(() => hide(), 180);
    };

    el.addEventListener('input', onInput);
    el.addEventListener('keyup', (e) => {
      // also catch typing that doesn't fire input (IME edge)
      if (e.key.length === 1 && /[a-zA-Z]/.test(e.key)) onInput();
    });
    el.addEventListener('keydown', onKeyDown);
    el.addEventListener('blur', onBlur);
    el.addEventListener('click', onInput);

    // expose destroy
    const destroy = () => {
      el.removeEventListener('input', onInput);
      el.removeEventListener('keydown', onKeyDown);
      el.removeEventListener('blur', onBlur);
      el.removeEventListener('click', onInput);
      box.remove();
    };

    // Mark element
    el.dataset.aksharAttached = '1';
    return { destroy, box, engine };
  },

  /**
   * Auto-attach to all elements matching selector (default: [data-akshar])
   * Waits for init() to complete.
   */
  autoAttach(selector = '[data-akshar]', opts = {}) {
    const els = document.querySelectorAll(selector);
    const handles = [];
    els.forEach(el => {
      if (el.dataset.aksharAttached) return;
      handles.push(this.attach(el, opts));
    });
    // Also watch for dynamically added elements
    const obs = new MutationObserver(() => {
      document.querySelectorAll(selector).forEach(el => {
        if (!el.dataset.aksharAttached) handles.push(this.attach(el, opts));
      });
    });
    obs.observe(document.body, { childList: true, subtree: true });
    return { handles, observer: obs };
  },

  _injectStyles() {
    if (this._styleInjected) return;
    this._styleInjected = true;
    const css = `
.akshar-ime-box {
  position: absolute;
  z-index: 999999;
  background: #fff;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.12), 0 2px 6px rgba(0,0,0,0.08);
  padding: 4px;
  font-family: system-ui, -apple-system, sans-serif;
  max-width: 320px;
}
.akshar-ime-item {
  padding: 6px 10px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 15px;
  line-height: 1.4;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.akshar-ime-item:hover { background: #f3f4f6; }
.akshar-ime-item.selected { background: #111827; color: #fff; }
.akshar-ime-hint {
  font-size: 11px;
  color: #9ca3af;
  padding: 4px 8px 2px;
  border-top: 1px solid #f3f4f6;
  margin-top: 4px;
  font-style: italic;
}
@media (prefers-color-scheme: dark) {
  .akshar-ime-box { background: #1f2937; border-color: #374151; }
  .akshar-ime-item:hover { background: #374151; }
  .akshar-ime-item.selected { background: #f9fafb; color: #111827; }
  .akshar-ime-hint { color: #6b7280; border-color: #374151; }
}
`;
    const style = document.createElement('style');
    style.textContent = css;
    document.head.appendChild(style);
  }
};

// Also expose globally for script-tag usage
if (typeof window !== 'undefined') window.AksharIME = AksharIME;
