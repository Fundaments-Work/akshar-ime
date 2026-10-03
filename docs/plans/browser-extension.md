# Browser Extension Plan (Chrome + Firefox) — live, opened 2026-10-03

Status: **plan only, nothing built.** This document supersedes section B
("Chrome extension (MV3) — TODO") of
`docs/plans/archive/playground-extension.md` (2026-09-22), which assumed a
single-browser, in-repo extension. Both the repository decision and the
target-browser set have changed. Section A of that document (the playground,
shipped to `Fundaments-Work/akshar-playground`) is unaffected and remains
the record of that work.

Nothing here touches the engine's scoring, the model, or any hot path. The
single exception is the storage seam in Phase 0, which is additive and
defaults to today's behaviour.

## Decisions locked

| Decision | Choice | Basis |
| :--- | :--- | :--- |
| Repository | **Separate repo**, `Fundaments-Work/akshar-extension` | See "Why a separate repo" |
| Target browsers | **Chrome + Firefox**, MV3, from first release | Firefox diverges on background model; see "Firefox is not a port" |
| Model delivery | **Bundled inside the extension** | Preserves the "nothing leaves your device" claim with zero network code |
| Engine coupling | **Pinned GitHub release tag**, consumed at build time | Reproducible builds; no vendored binaries; no submodule |
| Remote code | **None.** WASM and model both bundled | MV3 RHC policy |

## Decisions still open

- **Repo visibility.** Recommend *public*: for a privacy-positioned
  extension, auditable source is the credibility argument, and both stores
  accept it. The playground precedent is private, but that is an editor, not
  something installed into someone's browser.
- **Whether Phase 0 lands before the extension repo is created.** Recommend
  yes, so the extension pins one stable engine version from its first commit
  instead of tracking a moving target.
- **Chrome Web Store registration** ($5 one-time) and the AMO listing
  account. Administrative, but both block submission.
- **Domain/hosting for the required privacy policy page.** Chrome's Limited
  Use policy requires a public web page carrying the affirmative compliance
  statement; there is no privacy-policy-hosting exemption.

## Verified platform facts (checked 2026-10-03, not assumed)

These replace several assumptions in the archived plan. Sources are in
"References".

1. **WASM is permitted in MV3 extensions in both browsers.**
   `wasm-unsafe-eval` is the *enforced minimum* CSP for Chrome's
   `extension_pages` — it cannot be opted out of — and is one of only three
   values Firefox accepts in MV3 (`'self'`, `'none'`, `'wasm-unsafe-eval'`).
   Isolated-world content scripts receive an extension CSP that already
   includes `wasm-unsafe-eval`. **No sandboxed-iframe or `unsafe-eval`
   workaround is needed.** This was the largest open risk and it is clean.
2. **The bundled model is data, not remote-hosted code.** Chrome's RHC
   definition covers "JavaScript and WASM" and explicitly "does not include
   data or things like JSON or CSS". The `.wasm` must be bundled; the
   `.model` would have been *permitted* to be fetched. We bundle it anyway
   (decision above), but this matters: it means the privacy posture is a
   product choice, not a platform constraint, and should be described that
   way honestly rather than as a limitation.
3. **Package size is a non-issue.** Chrome's documented maximum is **2 GB**
   (long-standing third-party guidance still says 128 MB; that is wrong).
   Our bundle is ~24 MB.
4. **Firefox does not support `background.service_worker` at all**
   ([bug 1573659](https://bugzil.la/1573659)). MDN documents the supported
   cross-browser pattern: specify **both** `scripts` and `service_worker` in
   `background`; each browser picks the one it implements. Chrome ignores
   `scripts` from 121; Firefox ignores `service_worker`.
5. **`storage.local` quotas are irrelevant here.** Chrome 10 MB (5 MB before
   113), Firefox IndexedDB-tier with a global disk cap under
   `unlimitedStorage`. Learned state is <100 KB (comment at
   `src/wasm.rs:204`). Not a constraint.
6. **Chrome Web Store Limited Use policy has been tightened, enforced since
   2026-08-01.** Data must be strictly necessary to the disclosed single
   purpose, and collection must be prominently disclosed regardless of
   relation to that purpose. Our posture (collect nothing, transmit nothing)
   satisfies this comfortably — but the disclosure page is mandatory.
7. **Firefox requires data-collection disclosure in the listing itself**
   (enforced for new submissions from 2025-11-03) and a privacy policy in
   full text. AMO's review queue is publicly documented as backlogged;
   plan for days, not hours, and do not make a submission depend on a
   deadline.

## Why a separate repo

Not "extension code is different" — the substantive reasons:

- **The toolchain is disjoint.** A cross-browser extension is a
  TypeScript/JS project needing esbuild (content scripts declared in the
  manifest cannot be ES modules), manifest templating for two targets, and
  `web-ext` for the `.xpi`. None of that belongs in a crate that also builds
  an IBus engine and trains models.
- **Two release cadences and two review queues.** Chrome and Firefox
  versions, `.zip` vs `.xpi`, independent review cycles. CWS minor updates
  run 1–3 days; AMO is slower and variable.
- **The coupling is one artifact.** wasm pkg + model, both already
  published as release artifacts. A pinned tag resolves it.
- **Precedent.** The playground already moved to
  `Fundaments-Work/akshar-playground` for the same class of reason.

The honest cost: an engine fix reaches users only after publish → bump pin
→ rebuild → **re-review**. Mitigation is to pin the engine version and not
chase `main`; the extension is a shell, and the shell changes rarely.

Rejected: `git submodule` (permanently confusing repo, breaks clean clone
and CI, buys sync we do not want since we want version pinning).

## Firefox is not a port

The archived plan said "same code + `browser` polyfill". That undersells it.
Three real divergences:

| | Chrome | Firefox |
| :--- | :--- | :--- |
| Background context | `service_worker`, ~30 s idle teardown | `scripts` event page; **no service worker support** |
| MV3 CSP floor | enforced `wasm-unsafe-eval` | same three allowed values |
| Data disclosure | privacy-policy web page | listing field **and** full-text policy |

The background difference is architectural, not cosmetic: a service worker
that dies every 30 seconds cannot hold a 24 MB model in memory, while a
Firefox event page has different lifetime guarantees again. **Where the
engine instance lives must be designed for both, not bolted on later.**

Candidate designs, none chosen yet:
- *Per-tab engine in the content script* — simplest, but re-parses the model
  per tab and multiplies memory by open tabs.
- *Engine in the background context* — single instance, but must survive
  teardown on Chrome and needs a state-restore path.
- *Engine in an offscreen document* — Chrome-only, so it cannot be the
  primary design if Firefox is first-class.

This decision needs a measurement, not an opinion. Phase 1 exists partly to
produce it.

## The engine work this requires (Phase 0)

One genuine engine change, and it is a defect fix rather than a feature.

`src/wasm.rs:193` persists learning via `web_sys::window()?.local_storage()`.
Inside a content script's isolated world that `localStorage` belongs to the
**host page**, not the extension. Consequences:

- **Learning fragments per site** — precisely the problem the extension
  exists to solve, as the archived plan correctly identified but did not
  trace to this line.
- It **throws silently** where `localStorage` is unavailable (sandboxed
  iframes, strict privacy settings, some `file:` contexts). `confirm()`
  discards the error with `let _ =` at `src/wasm.rs:111`.

Fix: a settable persistence backend, defaulting to the current
`localStorage` implementation so desktop behaviour is bit-identical, with
the extension injecting a `storage.local`-backed shim. Small, additive,
independently testable, and useful beyond the extension (the playground has
the same per-origin fragmentation).

## Phase 0 — engine prerequisites (this repo)

1. Storage-backend seam in `src/wasm.rs` behind the existing wasm API
   surface; default path unchanged. Cover with a test that the default still
   round-trips.
2. **Publish the browser profile.** Blocking, and currently broken: neither
   `release.yml` nor `make release-upload` emits `akshar_wasm.model`.
   `release-upload` ships `data/akshar.model` (24.4 MB **desktop** profile),
   while the extension needs the browser profile. Add it to the release
   artifacts.
3. **Fix the `make web-model` discrepancy.** `docs/MANUAL.md` §10.4 and
   line 1849 document a `make web-model` target that does not exist in the
   `Makefile`. The browser profile is actually produced by
   `train --wasm` (`src/bin/train/train.rs:255`, which redirects `out_path`
   to `data/akshar_wasm.model`). Per `AGENTS.md`, code wins over prose:
   reconcile the manual to the code, not the reverse.
4. Cut a tagged release so the extension has something to pin.

## Phase 1 — extension skeleton (new repo)

- TypeScript, esbuild, one source tree → two manifests from a template.
- Background declared with **both** `scripts` and `service_worker`.
- `extension_pages` CSP carries `wasm-unsafe-eval` on both targets.
- Content script: intercept keys, render the existing candidate UI (the
  logic in `packages/engine-js/akshar-ime.js` is the reference behaviour;
  do not fork scoring logic), persist learning through the Phase 0 seam into
  `storage.local`.
- Model + wasm copied in at build time from the pinned release tag, verified
  by checksum.
- CI matrix builds `.zip` (CWS) and `.xpi` (AMO). **No automated store
  submission** initially — submission is a human decision with review
  consequences, and AMO queues are long.
- Permissions: `storage`, `activeTab`, `scripting`; host access granted
  per-site on demand. **Not** `<all_urls>` at install.

## Phase 2 — site compatibility (the real work)

The honest risk, and the part most likely to be underestimated. Ranked by
expected difficulty:

| Surface | Problem | Likely fix |
| :--- | --- | :--- |
| Plain `<input>`/`<textarea>` | — | works; this is the baseline |
| React controlled inputs | Framework overwrites the DOM value; needs the native value-setter trick before dispatching `input` | known, solvable |
| `contenteditable` (Gmail-style) | Range/offset bookkeeping across node splits; the wrapper's existing logic is a starting point, not a solution | per-site work |
| Google Docs | **Canvas-rendered.** No DOM text nodes to target | likely not viable without site-specific integration effort |
| Sandboxed / cross-origin iframes | Content script may not inject at all | accept the gap, document it |

**Measure this before promising it.** Phase 1 should produce a support
matrix across real sites (Gmail, Docs, X/Twitter, Reddit, LinkedIn,
WhatsApp Web, GitHub, ChatGPT, Notion) before the store listing claims
anything. Discovering Google Docs is canvas-bound *after* writing the
listing is an expensive way to learn it.

Per-site quirks belong in a site-adapter layer keyed by hostname, never as
branches inside the core transliteration path.

## Explicitly deferred

- Google Docs / Gmail deep integration (see Phase 2 — likely a separate
  research question, not a feature).
- Safari (paid developer programme, Xcode build, `preferred_environment`).
- Mobile keyboards — separate OS-input projects, not web tech.
- Cross-device sync — requires accounts, which contradicts the privacy
  stance until deliberately designed.
- Virtual on-screen keyboard. Out of scope; this is a transliteration layer.

## Risks

| Risk | Impact | Mitigation |
| :--- | --- | :--- |
| Chrome "Blue Argon" RHC rejection | Release blocked | Both artifacts bundled; no `http(s)://` in shipped code except the policy page URL. Audit the *built* zip, not the source. |
| Model/engine version skew after a pinned bump | Silent accuracy change in the extension | Pin exactly; surface engine version in the options page; bump deliberately |
| 24 MB model in a content script | Multiplied memory across tabs, slow first keystroke | Drives the Phase 1 architecture decision; measure before committing |
| AMO review latency | Slower iteration than expected | Submit early and often; don't gate on deadlines |
| Store disclosure drift | Takedown risk under the Aug 2026 Limited Use rules | Single source of truth for the privacy statement, referenced by both listings |

## Definition of done

- Extension installs unpacked in **both** browsers and transliterates in a
  plain `<input>` on both.
- `namaste` → `नमस्ते` in Chrome, Firefox, and the playground — the
  cross-surface smoke test the archived plan called for and never built.
- Learned words typed on site A are recognised on site B (the Phase 0 seam,
  proven end to end).
- No network request leaves the browser during typing. Verifiable in devtools.

## References

- Chrome, [Manifest V3 content security policy](https://developer.chrome.com/docs/extensions/reference/manifest/content-security-policy)
- Chrome, [Content scripts](https://developer.chrome.com/docs/extensions/develop/concepts/content-scripts)
- Chrome, [Deal with remote hosted code violations](https://developer.chrome.com/docs/extensions/develop/migrate/remote-hosted-code)
- Chrome, [Publish in the Chrome Web Store](https://developer.chrome.com/docs/webstore/publish) (2 GB limit)
- Chrome, [chrome.storage](https://developer.chrome.com/docs/extensions/reference/api/storage) (10 MB)
- Chrome, [Chrome Web Store policy updates: 2026](https://developer.chrome.com/blog/cws-policy-updates-2026) (Aug 1 enforcement)
- Chrome, [Limited Use policy](https://developer.chrome.com/docs/webstore/program-policies/limited-use)
- MDN, [background](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/manifest.json/background) (cross-browser MV3 pattern)
- MDN, [content_security_policy](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/manifest.json/content_security_policy)
- MDN, [storage.local](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/storage/local)
- [Firefox bug 1573659](https://bugzil.la/1573659) (no extension service worker)
