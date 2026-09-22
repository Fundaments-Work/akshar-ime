#!/usr/bin/env bash
# Build WASM package for Akshar IME
# Usage: ./packages/engine-wasm/build.sh (or: make wasm)
# Outputs to packages/engine-wasm/pkg/ (ESM, ready for web)
set -euo pipefail
DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$DIR"

echo ">> Building WASM (release, wasm feature)..."
cargo build --target wasm32-unknown-unknown --features wasm --release

WB="$HOME/.cargo/bin/wasm-bindgen"
if [ ! -x "$WB" ]; then
  echo "wasm-bindgen not found. Installing..."
  cargo install wasm-bindgen-cli --version 0.2.100 --locked
  WB="$HOME/.cargo/bin/wasm-bindgen"
fi

echo ">> Generating JS bindings..."
mkdir -p packages/engine-wasm/pkg
"$WB" --target web --out-dir packages/engine-wasm/pkg --out-name akshar_ime target/wasm32-unknown-unknown/release/akshar_ime.wasm

echo ">> Done. Files in packages/engine-wasm/pkg/:"
ls -lh packages/engine-wasm/pkg/
echo ""
echo "WASM sizes:"
echo "  raw:  $(du -h packages/engine-wasm/pkg/akshar_ime_bg.wasm | cut -f1)"
echo "  gzip: $(gzip -c packages/engine-wasm/pkg/akshar_ime_bg.wasm | wc -c | awk '{printf "%.0fK", $1/1024}')"
echo "  brotli: $(( $(brotli -c packages/engine-wasm/pkg/akshar_ime_bg.wasm 2>/dev/null | wc -c || echo 0) / 1024 ))K (if brotli installed)"
echo ""
echo "Next: assemble + serve the playground:"
echo "  make playground-build   # copies wrapper + pkg + model into apps/playground/public/"
echo "  make wasm-serve         # serves it at http://localhost:PORT/"
