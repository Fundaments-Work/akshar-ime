# Akshar IME — WASM engine package

Browser runtime for the Akshar Devanagari IME: compiled WASM + JS factory
(`pkg/`), consumed through the shared wrapper in `../engine-js/`.

See [`apps/playground/`](../../apps/playground/) for the deployable demo
and [`docs/MANUAL.md`](../../docs/MANUAL.md) (deployment chapter) for
integration docs.

## Build

```bash
./packages/engine-wasm/build.sh   # or: make wasm
# outputs to packages/engine-wasm/pkg/ (akshar_ime.js + akshar_ime_bg.wasm + d.ts)
```

Requires `wasm-bindgen-cli` (installed automatically if missing) and the
`wasm32-unknown-unknown` target. Pinned: `wasm-bindgen-cli 0.2.100
--locked` against `wasm-bindgen =0.2.100`, `web-sys =0.3.77` — bump together
or not at all.

## Use

```html
<input data-akshar />
<script type="module">
  import { AksharIME } from '../engine-js/akshar-ime.js';
  await AksharIME.init({ modelUrl: './models/akshar_wasm.model' });
  AksharIME.autoAttach();
</script>
```

The wrapper resolves the WASM module relative to `packages/engine-wasm/pkg/`;
the playground build copies both into one servable tree so the relative
layout is preserved (see `apps/playground/build.sh`).

## Publish

```bash
npm publish ./packages/engine-wasm
```
