#!/usr/bin/env bash
# Download the public datasets Akshar trains and evaluates on.
#
# Every file is pinned to an upstream revision and a SHA-256 in
# scripts/data-manifest.tsv.  Downloads resume after interruption, are
# verified before use, and land in data/raw/<set>/<name>.  A verified file
# gets a stamp (<name>.sha256) so later runs skip it without re-hashing.
# A row with a byte range (7th column, "start-end") fetches just that slice
# of the upstream file; its SHA-256 is the slice's.
#
#   scripts/fetch-data.sh                          # everything
#   scripts/fetch-data.sh aksharantar              # one set
#   scripts/fetch-data.sh indiccorp-v2 ne.txt      # one file of a set
#
# Environment: AKSHAR_RAW_DIR overrides the destination (default data/raw).
# Uses aria2c (parallel connections) when installed, curl otherwise.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="$root/scripts/data-manifest.tsv"
raw_dir="${AKSHAR_RAW_DIR:-$root/data/raw}"
want_set="${1:-}"
want_name="${2:-}"

human() { numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "$1 bytes"; }

verify() { # file sha256 -> 0 when the file matches
	[[ -f "$1" ]] && [[ "$(sha256sum "$1" | cut -d' ' -f1)" == "$2" ]]
}

fetch_one() { # set name bytes sha256 url [range]
	local set=$1 name=$2 bytes=$3 sha=$4 url=$5 range=${6:-}
	local dir="$raw_dir/$set" file="$raw_dir/$set/$name" stamp="$raw_dir/$set/$name.sha256"
	mkdir -p "$dir"
	if [[ -f "$stamp" && -f "$file" && "$(cat "$stamp")" == "$sha" ]]; then
		echo "ok      $set/$name (verified earlier)"
		return 0
	fi
	echo "fetch   $set/$name ($(human "$bytes"))"
	if [[ -n "$range" ]]; then
		# A slice: small enough to re-fetch whole rather than resume.
		curl -fsSL --retry 20 --retry-delay 10 -r "$range" -o "$file.part" "$url"
		mv -f "$file.part" "$file"
	elif command -v aria2c >/dev/null 2>&1; then
		aria2c --quiet=false --console-log-level=warn --summary-interval=120 \
			--continue=true --auto-file-renaming=false --allow-overwrite=true \
			--file-allocation=none --max-connection-per-server=8 --split=8 \
			--min-split-size=20M --max-tries=20 --retry-wait=10 \
			--checksum="sha-256=$sha" --dir="$dir" --out="$name" "$url"
	else
		curl -fL --retry 20 --retry-delay 10 -C - -o "$file" "$url"
	fi
	echo "verify  $set/$name"
	if ! verify "$file" "$sha"; then
		echo "FAILED  $set/$name: SHA-256 mismatch (expected $sha)" >&2
		echo "        delete $file and run again" >&2
		return 1
	fi
	echo "$sha" >"$stamp"
	echo "ok      $set/$name"
}

total=0
while IFS=$'\t' read -r set name bytes sha _licence url range; do
	[[ -z "$set" || "$set" == \#* ]] && continue
	[[ -n "$want_set" && "$set" != "$want_set" ]] && continue
	[[ -n "$want_name" && "$name" != "$want_name" ]] && continue
	fetch_one "$set" "$name" "$bytes" "$sha" "$url" "$range" </dev/null
	total=$((total + 1))
done <"$manifest"

if [[ $total -eq 0 ]]; then
	echo "nothing matched '${want_set}${want_name:+ $want_name}' in $manifest" >&2
	exit 1
fi
echo "done: $total file(s) under $raw_dir"
