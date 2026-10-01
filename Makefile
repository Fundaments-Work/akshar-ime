# ==============================================================================
# Makefile for Akshar Devanagari IME
# Universal Phonetic Input Method Engine for the Devanagari Script
# ==============================================================================

# --- Variables ---
RUST_LIB_NAME := libakshar_ime.so
C_ENGINE_NAME := devanagari-smart
TARGET_DIR    := target/release

# Compiler and Linker Flags (discovered via pkg-config for portability)
CFLAGS   := $(shell pkg-config --cflags ibus-1.0 jansson 2>/dev/null) -fPIC -O2
LDFLAGS  := $(shell pkg-config --libs ibus-1.0 jansson 2>/dev/null)

# System Installation Paths
PREFIX             ?= /usr
LIB_DIR            := $(PREFIX)/lib
IBUS_ENGINE_DIR    := $(PREFIX)/lib/ibus/engines
IBUS_COMPONENT_DIR := $(PREFIX)/share/ibus/component
DATA_DIR           := $(PREFIX)/share/akshar-ime

# Trigram pruning threshold for fitting size budget (25MB desktop budget)
DESKTOP_TRIGRAM_THRESHOLD ?= 3e-2

.PHONY: all release debug test check check-native check-wasm \
        install uninstall reinstall restart-ibus \
        data-fetch data-prepare lexicon train calibrate promote model eval \
        wasm wasm-clean clean reset-learning release-upload help

# --- Build Targets ---

all: release  ## Build the Rust library and C engine for release (default).

release: rust_lib c_engine  ## Build release library and C IBus binary.

debug:  ## Build the Rust library in debug mode.
	@echo "Building Rust library in debug mode..."
	@cargo build

test:  ## Run Rust tests and regression suites.
	@echo "Running tests..."
	@cargo test --release

rust_lib:
	@echo "Building Rust library in release mode..."
	@cargo build --release

c_engine: rust_lib
	@echo "Building C engine against release library..."
	@$(CC) $(CFLAGS) -o $(TARGET_DIR)/$(C_ENGINE_NAME) src/ibus_engine.c \
		-L$(TARGET_DIR) -lakshar_ime $(LDFLAGS) -Wl,-rpath,$(LIB_DIR)

# --- Quality & Lint Checks ---

check: check-native check-wasm  ## Run formatting check, clippy, tests, and wasm compile check.
	@echo "All checks passed successfully."

check-native:  ## Check formatting, run clippy with warnings denied, and execute test suite.
	@echo "==> cargo fmt --check"
	@cargo fmt --check || { echo "Run 'cargo fmt' to fix formatting."; exit 1; }
	@echo "==> cargo clippy -D warnings"
	@cargo clippy --release --all-targets -- -D warnings
	@echo "==> cargo test"
	@cargo test --release

check-wasm:  ## Check compilation and clippy for the wasm32 target.
	@if ! rustup target list --installed 2>/dev/null | grep -q wasm32-unknown-unknown; then \
		echo "==> wasm32-unknown-unknown not installed; skipping (run: rustup target add wasm32-unknown-unknown)"; \
		exit 0; \
	fi
	@echo "==> cargo check (wasm32)"
	@cargo check --features wasm --target wasm32-unknown-unknown
	@echo "==> cargo clippy (wasm32) -D warnings"
	@cargo clippy --features wasm --target wasm32-unknown-unknown -- -D warnings

# --- IBus Installation ---

install:  ## Install engine and unified Devanagari model to system directories.
	@if [ ! -f $(TARGET_DIR)/$(RUST_LIB_NAME) ] || [ ! -f $(TARGET_DIR)/$(C_ENGINE_NAME) ]; then \
		echo "Building release artifacts first..."; \
		$(MAKE) release; \
	fi
	@echo "Installing Akshar Devanagari IME..."
	@sudo mkdir -p $(IBUS_ENGINE_DIR) $(IBUS_COMPONENT_DIR) $(DATA_DIR)
	@sudo install -m 755 $(TARGET_DIR)/$(C_ENGINE_NAME) $(IBUS_ENGINE_DIR)/$(C_ENGINE_NAME).new
	@sudo mv -f $(IBUS_ENGINE_DIR)/$(C_ENGINE_NAME).new $(IBUS_ENGINE_DIR)/$(C_ENGINE_NAME)
	@sudo install -m 755 $(TARGET_DIR)/$(RUST_LIB_NAME) $(LIB_DIR)/$(RUST_LIB_NAME).new
	@sudo mv -f $(LIB_DIR)/$(RUST_LIB_NAME).new $(LIB_DIR)/$(RUST_LIB_NAME)
	@sudo cp devanagari-smart.xml $(IBUS_COMPONENT_DIR)/
	@if [ -f data/akshar.model ]; then \
		echo "Installing model: data/akshar.model -> $(DATA_DIR)/akshar.model"; \
		sudo cp data/akshar.model $(DATA_DIR)/akshar.model; \
	fi
	@sudo ldconfig
	@echo "\nInstallation complete. Run 'make restart-ibus' (without sudo) to reload IBus."

restart-ibus:  ## Restart user IBus daemon (run WITHOUT sudo).
	@echo "Restarting IBus..."
	@-timeout 5 ibus restart 2>/dev/null || true
	@rm -f ~/.cache/ibus/bus/* 2>/dev/null || true
	@echo "Done. If the engine does not appear in keyboard settings, log out and back in."

uninstall:  ## Remove Akshar IME from system directories.
	@echo "Uninstalling Akshar Devanagari IME..."
	@sudo rm -f $(IBUS_ENGINE_DIR)/$(C_ENGINE_NAME)
	@sudo rm -f $(LIB_DIR)/$(RUST_LIB_NAME)
	@sudo rm -f $(IBUS_COMPONENT_DIR)/devanagari-smart.xml
	@sudo rm -rf $(DATA_DIR)
	@sudo ldconfig
	@echo "Uninstallation complete. Run 'make restart-ibus' to finish."

reinstall: uninstall install  ## Reinstall the engine.

# --- Data Pipeline & Training ---

data-fetch:  ## Download and verify pinned Devanagari datasets (SET=... NAME=... optional).
	@bash scripts/fetch-data.sh $(SET) $(NAME)

data-prepare:  ## Build normalized pairs: data/pairs/{train,valid,test}.jsonl.
	@for z in data/raw/aksharantar/*.zip; do \
		[ -f "$$z" ] || continue; \
		l=$$(basename $$z .zip); mkdir -p data/raw/aksharantar/$$l; \
		unzip -o -q -j $$z -d data/raw/aksharantar/$$l; \
	done
	@cargo run --release --bin prepare_pairs

lexicon:  ## Build universal Devanagari FST lexicon (data/lexicon.bin).
	@cargo run --release --bin build_lexicon

PAIRS ?= 500000
EPOCHS ?= 5
train:  ## Train model (PAIRS=500000 by default; PAIRS=0 for all pairs).
	@cargo run --release --bin train -- --reranker-pairs $(PAIRS) --epochs $(EPOCHS) --iterations 12

calibrate:  ## Calibrate heuristic/learned blend weights on validation set.
	@cargo run --release --bin calibrate_blend -- --model data/akshar.model --out data/akshar.model

promote:  ## Prune model to fit 25MB container budget.
	@cargo run --release --bin prune_lm -- --model data/akshar.model \
		--trigram-threshold $(DESKTOP_TRIGRAM_THRESHOLD) --out data/akshar.model.tmp
	@mv data/akshar.model.tmp data/akshar.model
	@ls -lh data/akshar.model

model: calibrate promote  ## Calibrate and prune trained model in data/akshar.model.

# --- Evaluation ---

SPLIT ?= test
eval:  ## Evaluate Devanagari benchmark on data/pairs/$(SPLIT).jsonl (SPLIT=valid|test).
	@cargo run --release --bin eval_langs -- --dataset data/pairs/$(SPLIT).jsonl \
		$(if $(MODEL),--model $(MODEL),) $(if $(JSON),--json $(JSON),)

# --- WebAssembly ---

wasm:  ## Build WASM engine package (packages/engine-wasm/pkg).
	@bash packages/engine-wasm/build.sh

wasm-clean:  ## Clean WASM build artifacts.
	@rm -rf packages/engine-wasm/pkg

# --- Utilities ---

clean:  ## Clean cargo build artifacts and temporary files.
	@cargo clean
	@rm -f data/*.tmp data/smoke.model

reset-learning:  ## Reset user learned dictionary.
	@rm -f $${XDG_CONFIG_HOME:-$$HOME/.config}/akshar-devanagari/user_dictionary.bin
	@echo "User learned dictionary reset."

release-upload:  ## Upload built model artifact to GitHub release (requires TAG=vX.Y.Z).
	@if [ -z "$(TAG)" ]; then echo "Usage: make release-upload TAG=vX.Y.Z"; exit 1; fi
	@if [ -f data/akshar.model ]; then \
		gh release view $(TAG) >/dev/null 2>&1 || gh release create $(TAG) --generate-notes --verify-tag; \
		gh release upload $(TAG) data/akshar.model --clobber; \
		echo "Uploaded data/akshar.model to release $(TAG)."; \
	else \
		echo "Error: data/akshar.model not found. Run 'make train && make model' first."; exit 1; \
	fi

help:  ## Show this help message.
	@echo "Akshar Devanagari IME"
	@echo "====================="
	@echo "Usage: make [target]"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'