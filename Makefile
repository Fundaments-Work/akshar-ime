# ==============================================================================
# Makefile for Akshar Devanagari IME
# ==============================================================================

# --- Variables ---
RUST_LIB_NAME := libakshar_ime.so
C_ENGINE_NAME := devanagari-smart
TARGET_DIR    := target/release

# Compiler and Linker Flags (discovered via pkg-config for portability)
CFLAGS   := $(shell pkg-config --cflags ibus-1.0 jansson) -fPIC -O2
LDFLAGS  := $(shell pkg-config --libs ibus-1.0 jansson)

# System Paths
PREFIX            ?= /usr
LIB_DIR           := $(PREFIX)/lib
IBUS_ENGINE_DIR   := $(PREFIX)/lib/ibus/engines
IBUS_COMPONENT_DIR:= $(PREFIX)/share/ibus/component
DATA_DIR          := $(PREFIX)/share/akshar-ime

# Relative-entropy pruning threshold for the browser model.  Higher = smaller
# and less accurate; see docs/MANUAL.md "Browser profile" for the measured
# sizes.  3e-2 keeps 47%
# of trigram transitions and yields 8.91 MB / 4.94 MB Brotli (verified
# 2026-09-08).  Its accuracy has
# not been re-measured since the 2026-09-06 engine changes.
TRIGRAM_THRESHOLD ?= 3e-2
.PHONY: all release debug test install uninstall reinstall clean reset-learning \
        restart-ibus help wasm wasm-clean release-upload pack web-model \
        train train-quick train-mid train-full eval eval-full eval-ime eval-errors \
        ablate manual docs check check-native check-wasm release-check data-fetch \
        data-prepare eval-langs
# --- Main Targets ---

all: release  ## Build the engine for release (default).

release: rust_lib c_engine  ## Build the Rust library and C engine in release mode.

debug:  ## Build the Rust library in debug mode.
	@echo "Building Rust library in debug mode..."
	@cargo build

test:  ## Run the Rust test suite (includes the accuracy regression guard).
	@echo "Running Rust tests..."
	@cargo test --release

# --- Build Steps ---

rust_lib:
	@echo "Building Rust library in release mode..."
	@cargo build --release

c_engine: rust_lib
	@echo "Building C engine against release library..."
	@$(CC) $(CFLAGS) -o $(TARGET_DIR)/$(C_ENGINE_NAME) src/ibus_engine.c \
		-L$(TARGET_DIR) -lakshar_ime $(LDFLAGS) -Wl,-rpath,$(LIB_DIR)


install:  ## Compile (if needed) and install the engine to system directories.
	@if [ ! -f $(TARGET_DIR)/libakshar_ime.so ] || [ ! -f $(TARGET_DIR)/$(C_ENGINE_NAME) ]; then \
		echo "  > Building release artifacts first..."; \
		$(MAKE) release; \
	fi
	@echo "Installing Akshar Devanagari IME..."
	@echo "  > Creating system directories..."
	@sudo mkdir -p $(IBUS_ENGINE_DIR)
	@sudo mkdir -p $(IBUS_COMPONENT_DIR)
	@sudo mkdir -p $(DATA_DIR)
	@echo "  > Installing engine binary and library..."
	@sudo install -m 755 $(TARGET_DIR)/$(C_ENGINE_NAME) $(IBUS_ENGINE_DIR)/$(C_ENGINE_NAME).new
	@sudo mv -f $(IBUS_ENGINE_DIR)/$(C_ENGINE_NAME).new $(IBUS_ENGINE_DIR)/$(C_ENGINE_NAME)
	@sudo install -m 755 $(TARGET_DIR)/$(RUST_LIB_NAME) $(LIB_DIR)/$(RUST_LIB_NAME).new
	@sudo mv -f $(LIB_DIR)/$(RUST_LIB_NAME).new $(LIB_DIR)/$(RUST_LIB_NAME)
	@echo "  > Installing IBus component file..."
	@sudo cp devanagari-smart.xml $(IBUS_COMPONENT_DIR)/
	@echo "  > Installing model artifacts..."
	@if [ -f data/akshar.model ]; then \
		echo "    * Installing unified model (data/akshar.model)..."; \
		sudo cp data/akshar.model $(DATA_DIR)/akshar.model; \
		echo "    * Purging legacy model files from $(DATA_DIR)..."; \
		sudo rm -f $(DATA_DIR)/translit_model.bin $(DATA_DIR)/word_freq_text.bin $(DATA_DIR)/reranker_weights_sparse.bin $(DATA_DIR)/reranker_weights.json $(DATA_DIR)/crf_model.bin $(DATA_DIR)/roman_lexicon.bin; \
	else \
		echo "    * Installing legacy model artifacts..."; \
		for f in data/translit_model.bin data/word_freq_text.bin data/reranker_weights_sparse.bin data/roman_lexicon.bin; do \
			if [ -f "$$f" ]; then sudo cp "$$f" $(DATA_DIR)/; fi; \
		done; \
	fi
	@echo "  > Updating linker cache..."
	@sudo ldconfig
	@echo "\nInstallation complete. Run 'make restart-ibus' (no sudo) to reload IBus,"
	@echo "then add 'Devanagari (Akshar)' in Settings > Keyboard > Input Sources."

restart-ibus:  ## Restart the user's IBus daemon (run WITHOUT sudo).
	@echo "Restarting IBus..."
	@-timeout 5 ibus restart 2>/dev/null || true
	@rm -f ~/.cache/ibus/bus/* 2>/dev/null || true
	@echo "Done. If the input source still doesn't appear, log out and back in."


uninstall:  ## Remove the engine from the system.
	@echo "Uninstalling Akshar Devanagari IME..."
	@echo "  > Removing system files..."
	@sudo rm -f $(IBUS_ENGINE_DIR)/$(C_ENGINE_NAME)
	@sudo rm -f $(LIB_DIR)/$(RUST_LIB_NAME)
	@sudo rm -f $(IBUS_COMPONENT_DIR)/devanagari-smart.xml
	@sudo rm -rf $(DATA_DIR)
	@echo "  > Updating linker cache..."
	@sudo ldconfig
	@echo "\nUninstallation complete. Run 'make restart-ibus' (no sudo) to reload IBus."

reinstall: uninstall install  ## Run uninstall and then install.

# --- Data ---------------------------------------------------------------------
#
# Public datasets, pinned by upstream revision and SHA-256 in
# scripts/data-manifest.tsv, downloaded resumably into data/raw/.  SET narrows
# to one set, NAME to one file of it.  The Aksharantar word pairs for every
# Devanagari language (~229 MB) train the engine and measure it per language;
# only the trained model ships, never the downloaded data.

data-fetch:  ## Download + verify the public datasets (SET=..., NAME=... to narrow).
	@bash scripts/fetch-data.sh $(SET) $(NAME)

data-prepare:  ## Build data/pairs/{train,valid,test}.jsonl (all Devanagari languages).
	@for z in data/raw/aksharantar/*.zip; do \
		l=$$(basename $$z .zip); mkdir -p data/raw/aksharantar/$$l; \
		unzip -o -q -j $$z -d data/raw/aksharantar/$$l; \
	done
	@cargo run --release --bin prepare_pairs

# --- Training -----------------------------------------------------------------
#
# NOTE: --reranker-pairs sizes the DISCRIMINATIVE RERANKER's training set only.
# The EM emission model and the Kneser-Ney syllable LM always ingest all
# 3,588,793 parallel pairs, and the vocabulary always comes from the full
# corpus, regardless of which target you run.  These targets differ only in how
# much ranking supervision the sparse table sees.
#
# Chunked mode (and therefore the per-batch learning-rate schedule) engages
# above 200k pairs, so train-quick does NOT exercise it -- use train-mid to
# validate a trainer change before committing to an overnight run.

train-quick:  ## Reranker on 100k pairs, ~10min. Does NOT exercise chunked mode.
	@cargo run --release --bin train -- --reranker-pairs 100000 --epochs 3 --iterations 12

train-mid:  ## Reranker on 500k pairs (5 batches, ~40min). Validation gate for train-full.
	@cargo run --release --bin train -- --reranker-pairs 500000 --epochs 5 --iterations 12

train-mid-att:  ## train-mid + attestation-weighted EM/LM (tests convention dominance).
	@cargo run --release --bin train -- --reranker-pairs 500000 --epochs 5 --iterations 12 \
		--attestation --out data/akshar_attested.model

train: train-mid  ## Alias for train-mid.

train-full:  ## Reranker on all 3.59M pairs (36 batches, ~4h). Watch the dev loss.
	@cargo run --release --bin train -- --reranker-pairs 0 --epochs 5 --iterations 12

# --- Evaluation ---------------------------------------------------------------

# Every Devanagari language, IME metrics (top-1 / in-list@8 / MRR) per
# language and source.  Tune on SPLIT=valid; read test once per decision.
# MODEL=path evaluates a specific container instead of the installed lookup.
SPLIT ?= test
eval-langs:  ## Per-language IME metrics on data/pairs/$(SPLIT).jsonl (needs data-prepare).
	@cargo run --release --bin eval_langs -- --dataset data/pairs/$(SPLIT).jsonl \
		$(if $(MODEL),--model $(MODEL),) $(if $(JSON),--json $(JSON),)

eval:  ## Aksharantar accuracy by split (AK-Freq / AK-NEF / AK-NEI).
	@cargo run --release --bin evaluate_aksharantar -- \
		--dataset data/aksharantar/test_devanagari.jsonl --topk 5 --show-misses 0

eval-full:  ## Accuracy with bootstrap 95% CIs and per-query latency.
	@cargo run --release --bin evaluate -- --resamples 1000

eval-errors:  ## Oracle curves, error taxonomy, CER and the collision bound.
	@cargo run --release --bin analyze_errors -- \
		--dataset data/aksharantar/test_devanagari.jsonl --beam 256

ctx-model:  ## Pruned corpus bigram sidecar for sentence context (data/corpus_bigrams.bin).
	@cargo run --release --bin build_corpus_bigrams -- \
		--corpus data/store/corpus_clean.txt --model data/akshar.model \
		--threshold 20 --out data/corpus_bigrams.bin

eval-sentences:  ## In-context word accuracy on held-out sentences (needs ctx-model).
	@cargo run --release --bin evaluate_sentences -- --n 3000 --topk 5 \
		--ctx data/corpus_bigrams.bin --ctx-mode predicted

eval-ime:  ## Plain-language IME report + machine JSON for docs (docs/generated/eval.json).
	@mkdir -p docs/generated
	@cargo run --release --bin eval_ime -- \
		--dataset data/aksharantar/test_devanagari.jsonl \
		--out docs/generated/eval.json

ablate:  ## Component ablation: what each part of the pipeline contributes.
	@cargo build --release --bin evaluate_aksharantar 2>/dev/null
	@printf '%-30s' "full system"; ./target/release/evaluate_aksharantar \
		--dataset data/aksharantar/test_devanagari.jsonl --topk 5 --show-misses 0 \
		| grep AK-Freq | sed 's/.*top1/top1/'
	@for v in NO_TRIGRAM NO_TRIE_UNION NO_SPARSE NO_VARIANTS; do \
		printf '%-30s' "  -$$v"; \
		env AKSHAR_$$v=1 ./target/release/evaluate_aksharantar \
			--dataset data/aksharantar/test_devanagari.jsonl --topk 5 --show-misses 0 \
			| grep AK-Freq | sed 's/.*top1/top1/'; \
	done
	@printf '%-30s' "  -dense (gamma=0)"; AKSHAR_GAMMA=0.0 ./target/release/evaluate_aksharantar \
		--dataset data/aksharantar/test_devanagari.jsonl --topk 5 --show-misses 0 \
		| grep AK-Freq | sed 's/.*top1/top1/'

profile:  ## Per-phase latency breakdown of a suggestion query.
	@cargo run --release --example profile_decode

# --- Documentation ------------------------------------------------------------
#
# The manual contains Devanagari examples, so the body font must cover the
# Devanagari block. FreeSerif/FreeSans do; most Latin-only faces render tofu.
# Override on the command line if you have something better installed.
MANUAL_SERIF ?= FreeSerif
MANUAL_SANS  ?= FreeSans
MANUAL_MONO  ?= Liberation Mono

manual: docs/AksharIME-Manual.pdf  ## Build the source manual as a PDF (needs pandoc + xelatex).

docs/AksharIME-Manual.pdf: docs/MANUAL.md
	@command -v pandoc >/dev/null || { echo "pandoc not found: install pandoc and a LaTeX engine"; exit 1; }
	@echo "Building $@ ..."
	@pandoc docs/MANUAL.md -o $@ \
		--pdf-engine=xelatex \
		--toc --toc-depth=3 --number-sections \
		--syntax-highlighting=tango \
		-V mainfont="$(MANUAL_SERIF)" \
		-V sansfont="$(MANUAL_SANS)" \
		-V monofont="$(MANUAL_MONO)"
	@echo "Wrote $@ ($$(du -h $@ | cut -f1))"

docs: manual  ## Alias for manual.

# --- Release checks -----------------------------------------------------------

check: check-native check-wasm  ## Format, clippy, tests, and the wasm target.
	@echo "All checks passed."

check-native:  ## Format check, clippy with warnings denied, and the test suite.
	@echo "==> cargo fmt --check"
	@cargo fmt --check || { echo "run 'cargo fmt' to fix"; exit 1; }
	@echo "==> cargo clippy -D warnings"
	@cargo clippy --release --all-targets -- -D warnings
	@echo "==> cargo test"
	@cargo test --release

# The wasm target compiles a different set of cfg branches, so a change can pass
# every native check and still break the browser build -- which is exactly how a
# broken wasm build shipped in v1.1.0. This target is not optional.
check-wasm:  ## Compile-check and lint the wasm32 target (catches cfg-gated breakage).
	@if ! rustup target list --installed 2>/dev/null | grep -q wasm32-unknown-unknown; then \
		echo "==> wasm32 target not installed; skipping (rustup target add wasm32-unknown-unknown)"; \
		exit 0; \
	fi
	@echo "==> cargo check --features wasm --target wasm32-unknown-unknown"
	@cargo check --features wasm --target wasm32-unknown-unknown
	@echo "==> cargo clippy (wasm) -D warnings"
	@cargo clippy --features wasm --target wasm32-unknown-unknown -- -D warnings

release-check: check manual  ## Everything a release needs: checks, accuracy, manual.
	@echo "==> accuracy"
	@$(MAKE) --no-print-directory eval
	@echo "==> artefacts"
	@ls -lh data/akshar.model data/akshar_wasm.model docs/AksharIME-Manual.pdf 2>/dev/null || true
	@echo "Release checks complete."

pack:  ## Pack model binaries into unified data/akshar.model container.
	@cargo run --release --bin pack_model

web-model:  ## Build the compact browser model (data/akshar_wasm.model, ~4.9 MB Brotli).
	@echo "Pruning the syllable trigram LM (half the model's bytes)..."
	@cargo run --release --bin prune_lm -- \
		--model data/akshar.model \
		--trigram-threshold $(TRIGRAM_THRESHOLD) \
		--out data/akshar_wasm.model
	@if command -v brotli >/dev/null 2>&1; then \
		brotli -q 11 -c data/akshar_wasm.model | wc -c \
			| awk '{printf "Brotli wire size: %.2f MB\n", $$1/1048576}'; \
	fi

clean:  ## Remove all build artifacts and temporary files.
	@echo "Cleaning build artifacts..."
	@cargo clean
	@rm -f data/*.tmp data/smoke.model data/akshar_pruned.model data/akshar_quantized.model

reset-learning:  ## Delete the user's learned dictionary (start fresh).
	@echo "Removing user learning data..."
	@rm -f $${XDG_CONFIG_HOME:-$$HOME/.config}/akshar-devanagari/user_dictionary.bin
	@echo "Done."

wasm:  ## Build WASM package (packages/engine-wasm/pkg).
	@echo "Building WASM package..."
	@bash packages/engine-wasm/build.sh

wasm-clean:  ## Remove WASM build artifacts.
	@echo "Cleaning WASM artifacts..."
	@rm -rf packages/engine-wasm/pkg

release-upload:  ## Upload locally built model artifacts to a GitHub release (TAG=vX.Y.Z required).
	@if [ -z "$(TAG)" ]; then echo "usage: make release-upload TAG=vX.Y.Z"; exit 1; fi
	@if [ -f data/akshar.model ]; then \
		echo "Uploading unified model data/akshar.model..."; \
		gh release view $(TAG) >/dev/null 2>&1 || gh release create $(TAG) --generate-notes --verify-tag; \
		gh release upload $(TAG) data/akshar.model --clobber; \
	elif [ -f data/translit_model.bin ] && [ -f data/word_freq_text.bin ]; then \
		echo "Uploading legacy model binaries..."; \
		gh release view $(TAG) >/dev/null 2>&1 || gh release create $(TAG) --generate-notes --verify-tag; \
		gh release upload $(TAG) data/translit_model.bin data/word_freq_text.bin --clobber; \
	else \
		echo "No model artifacts found in data/. Run 'make train' or 'make pack' first."; exit 1; \
	fi
	@echo "Uploaded model artifacts to release $(TAG)."

# --- Help ---

help:  ## Show this help.
	@echo "Akshar Devanagari IME Makefile"
	@echo "-------------------------"
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'