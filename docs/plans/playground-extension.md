# Playground + Browser Extension Plan (2026-09-22)

Status: PLAN ONLY — no code. Training (`train-mid-att`) is running;
nothing here touches the engine, the model, or any hot path.

## Non-negotiable principle

One engine, three shells. `wasm/pkg` + `js/akshar-ime.js` is the single
client core; the playground and the extension are thin shells around it.
No forks, no per-surface scoring logic, no model variants per surface
(except the existing `akshar_wasm.model` browser profile). A bug fixed in
the core fixes all three surfaces; a surface never patches around the core.

Privacy is the selling point and a constraint: everything runs locally
(WASM + local model + on-device learning). No keystroke telemetry, no
server calls, no accounts — in all three surfaces. Say it on every page.

## Repo reorg (do first, small, mechanical)

Current `web/` + `js/` + `wasm/` split confuses hosting from packaging:

```
apps/
  playground/     # Cloudflare Worker + static editor (moves web/index.html)
  extension/      # Chrome MV3 (new)
packages/
  engine-js/      # js/akshar-ime.js (moved, versioned)
  engine-wasm/    # wasm/ build + pkg output (moved)
docs/plans/       # this file lives on
```

Move, don't rewrite: git-mv preserves history. `Makefile` targets
(`wasm`, `wasm-serve`) gain path updates only. `web/` becomes a redirect
stub for one release, then deleted.

## A. Playground on Cloudflare (experiment + write)

Purpose: a URL where anyone can type Devanagari in the browser —
experiment with transliteration and actually write text. Marketing,
dogfooding, and the eval sandbox for future work.

- Hosting: Workers Static Assets (or Pages — decide at build; both free).
  Static shell + editor; zero server compute per keystroke (WASM local).
- Model delivery: R2 bucket (`akshar_wasm.model` + Brotli pre-compressed
  variant), long-cache headers, range-request support, custom domain later
  (`playground.akshar.example`). ~9 MB raw / ~5 MB Brotli — R2 free tier
  covers it 1000× over.
- Editor v1: full-page writing surface (contenteditable doc, not just two
  demo boxes), candidate bar via `AksharIME.attach`, word/commit counters,
  model-info footer (version, vocab size, profile). v2: side-by-side
  roman/Devanagari, export (txt/md/clipboard), shareable read-only links
  (URL-encoded text, capped length), dark mode (already in demo CSS).
- Explicit non-goals v1: accounts, saved docs server-side, collaboration,
  mobile keyboard (OS territory), analytics beyond privacy-safe aggregates
  (Cloudflare Web Analytics, cookieless — optional).
- Config: `wrangler.toml` (assets + R2 binding, even if binding unused at
  first), `apps/playground/README.md` with deploy steps
  (`wrangler deploy`), preview URLs per PR.
- Cost: $0 (Workers + R2 free tiers). Risk: slow networks — mitigate with
  Brotli + lazy model load + progress UI (demo already shows loading state).

## B. Chrome extension (MV3)

Purpose: type Devanagari in ANY text field on the web (Gmail, Docs, X…).

- Shape: content script matching text inputs/textareas/contenteditables
  (all-frames false, opt-in per site via action popup); background service
  worker only for lifecycle; popup for enable/disable + learning reset;
  options page for depth/suggestion-count (maps to existing attach opts).
- Bundling: ship `engine-wasm` (wasm+JS, ~2 MB) + `akshar_wasm.model`
  (~9 MB) INSIDE the package (~11 MB total, far under store limits).
  Rationale: MV3 bans remote code and restricts remote WASM; bundled =
  no host permissions for the model, works offline, store-review friendly.
  CSP needs `wasm-unsafe-eval` for the bundled module — standard, allowed.
- Learning: extension `storage.local` instead of page localStorage
  (per-site localStorage fragments learning across sites — the extension
  exists precisely to unify it). Reuse `exportState`/`import_state`
  bytes; quota ~10 MB is plenty for learned words.
- Permissions minimal: `storage`, `activeTab` + `scripting` (inject on
  demand) — NOT `<all_urls>` at install. Request host access at first use
  per site (action click), which users trust and reviewers prefer.
- Packaging: `apps/extension/` with `manifest.json`, icons (derive from
  `Akshar.svg`), `npm run build` (esbuild copy, no framework), zip for
  the Web Store (Firefox port later: same code + `browser` polyfill).
- Explicit non-goals v1: Gmail-compose deep integration, mobile (Kiwi/Yandex
  only), sync across devices (needs accounts — violates privacy stance
  until designed), Safari (paid developer program + Xcode build).
- Store listing: name/screenshots/privacy page ("no data leaves device")
  prepared alongside code; publish checklist in the dir README.

## Phasing

1. Reorg moves (git-mv, Makefile paths, CI paths). No behavior change.
2. Extension skeleton (manifest + inject + popup + bundled model) → local
   load-unpacked test → store draft. Independent of playground.
3. Playground shell move + R2 deploy + editor v1. Independent of extension.
4. Shared hardening driven by both: versioned `packages/engine-js`
   releases, model URL convention (`/models/<version>/akshar_wasm.model`
   + `.br`), cross-surface smoke test (type `namaste` → `नमस्ते` in each).

## What this needs from you (decisions, not labor)

- Domain(s): `akshar.example`? Playground subdomain + R2 custom domain.
- Store account: Chrome Web Store developer registration ($5 one-time).
- Name/branding lock: "Akshar" + the existing SVG, or new marks?
- Sequencing: extension-first (daily-use value) or playground-first
  (visibility value)? Plan supports either; pick one to staff first.

## What this deliberately defers

- Firefox/Safari ports (same core, later).
- Accounts/sync (privacy architecture decision first).
- Mobile keyboards (separate OS-input projects, not web tech).
- `make web-model` changes: none needed — both surfaces consume the
  existing browser profile as-is.
