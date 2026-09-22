/* tslint:disable */
/* eslint-disable */
/**
 * Create engine from model URL only (convenience, lexicon optional).
 */
export function createEngineFromModelUrl(model_url: string): Promise<WasmEngine>;
/**
 * Transliterate without needing an engine instance, using an empty model — mainly for testing wiring.
 */
export function quickTransliterate(roman: string): string;
export function init_panic_hook(): void;
/**
 * Async factory: fetch model (+ optional lexicon/weights) from URLs and create engine.
 *
 * Usage from JS:
 * ```js
 * const engine = await createEngine("https://cdn.example.com/translit_model.bin");
 * // with lexicon:
 * const engine = await createEngine(url, lexiconUrl, weightsUrl);
 * ```
 */
export function createEngine(model_url: string, lexicon_url?: string | null, reranker_url?: string | null): Promise<WasmEngine>;
/**
 * Returns the engine version (crate version).
 */
export function getVersion(): string;
export class WasmEngine {
  free(): void;
  /**
   * Create an engine from raw bytes already fetched in JS.
   *
   * `model_bytes` is the `translit_model.bin` file as Uint8Array.
   * `lexicon_bytes` is accepted and IGNORED. The corpus lexicon was removed
   * in v1.1.0 (it measured 0.00pp on every split and was dead by
   * construction on the unified-container path); the parameter is retained
   * so existing `createEngine(model, lexicon, weights)` callers keep
   * working. Pass `null`.
   * `reranker_json` may be null/undefined for default weights.
   */
  constructor(model_bytes: Uint8Array, lexicon_bytes?: Uint8Array | null, reranker_json?: string | null);
  /**
   * Number of aksharas in model vocabulary (diagnostic).
   */
  vocabSize(): number;
  /**
   * Export learned state as base64 string for backup / sync.
   */
  exportState(): string;
  /**
   * Import learned state from base64 string.
   */
  importState(b64: string): void;
  /**
   * How many learned words are stored.
   */
  learnedCount(): number;
  /**
   * Top transliteration (single string) or empty if none.
   */
  transliterate(roman: string): string;
  /**
   * Clear learned dictionary and persist.
   */
  resetLearning(): void;
  /**
   * Get Devanagari suggestions for a roman prefix.
   * Returns JS array of strings (up to `count`, default 8).
   */
  getSuggestions(prefix: string, count?: number | null): any[];
  /**
   * Get suggestions with scores: returns JSON string `[{"text":"नमस्ते","score":123}, ...]`
   */
  getSuggestionsWithScores(prefix: string, count?: number | null): string;
  /**
   * Create an engine with an empty (untrained) model — useful for testing without downloading.
   * Will only do dictionary + fuzzy matching, no transliteration.
   */
  static empty(): WasmEngine;
  /**
   * Record that user confirmed `roman -> devanagari`. Learning is immediate and persisted.
   */
  confirm(roman: string, devanagari: string): void;
  /**
   * Returns true if model is loaded and valid (not empty fallback).
   */
  isReady(): boolean;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_wasmengine_free: (a: number, b: number) => void;
  readonly createEngine: (a: number, b: number, c: number, d: number, e: number, f: number) => any;
  readonly createEngineFromModelUrl: (a: number, b: number) => any;
  readonly getVersion: () => [number, number];
  readonly quickTransliterate: (a: number, b: number) => [number, number];
  readonly wasmengine_confirm: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly wasmengine_empty: () => number;
  readonly wasmengine_exportState: (a: number) => [number, number, number, number];
  readonly wasmengine_from_bytes: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
  readonly wasmengine_getSuggestions: (a: number, b: number, c: number, d: number) => [number, number];
  readonly wasmengine_getSuggestionsWithScores: (a: number, b: number, c: number, d: number) => [number, number];
  readonly wasmengine_importState: (a: number, b: number, c: number) => [number, number];
  readonly wasmengine_isReady: (a: number) => number;
  readonly wasmengine_learnedCount: (a: number) => number;
  readonly wasmengine_resetLearning: (a: number) => [number, number];
  readonly wasmengine_transliterate: (a: number, b: number, c: number) => [number, number];
  readonly wasmengine_vocabSize: (a: number) => number;
  readonly init_panic_hook: () => void;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __externref_table_alloc: () => number;
  readonly __wbindgen_export_2: WebAssembly.Table;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_export_5: WebAssembly.Table;
  readonly __externref_table_dealloc: (a: number) => void;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __externref_drop_slice: (a: number, b: number) => void;
  readonly closure80_externref_shim: (a: number, b: number, c: any) => void;
  readonly closure94_externref_shim: (a: number, b: number, c: any, d: any) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
