# Akshar playground — deployable Devanagari editor

Type Roman, write Devanagari. Fully offline once loaded: WASM engine +
model + on-device learning, no keystrokes leave the page.

## Local dev

```bash
make playground-build   # assembles apps/playground/public/ (vendor + config + model)
make wasm-serve          # serves it at http://localhost:PORT/
```

`build.sh` also runs standalone (`MODEL_URL=… MODEL_SRC=…`). Generated
files (`public/vendor/`, `public/models/`, `public/config.js`) are
gitignored — only `public/index.html` is tracked.

## Deploy (Cloudflare)

The model (~9 MB) is NOT bundled — upload it once to R2, then bake its URL:

```bash
# one-time: R2 bucket + upload (needs wrangler login)
wrangler r2 bucket create akshar-models
wrangler r2 object put akshar-models/akshar_wasm.model \
  --file=data/akshar_wasm.model --content-type=application/octet-stream
# enable public access (r2.dev) or a custom domain in the dashboard,
# then build with the public URL:
MODEL_URL=https://pub-XXX.r2.dev/akshar_wasm.model apps/playground/build.sh
```

Workers (static assets, recommended):

```bash
wrangler deploy --config apps/playground/wrangler.toml
```

Pages (equivalent, no config needed):

```bash
wrangler pages deploy apps/playground/public --project-name akshar-playground
```

Custom domain: attach in the dashboard (Workers → Custom Domains, or
Pages → Custom domains). Both paths serve the same static tree.

## Updating

Engine/model changes ship by rebuilding in order:

```bash
make wasm            # rebuild packages/engine-wasm/pkg/
make playground-build [MODEL_URL=…]   # re-assemble public/
# redeploy (workers or pages command above)
```

No step touches the Rust hot path; `apps/playground/` is a thin shell over
`packages/engine-js` + `packages/engine-wasm`.
