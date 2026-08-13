SHELL := /bin/bash

UV ?= uv
UV_CACHE_DIR ?= /tmp/uv-cache
RUNS_PATH ?= runs
TARGET ?= reference
REPORT ?= report.html
TRAIN_TOKENS ?= 624984064

XPROF_VERSION ?= 2.22.3
XPROF_PORT ?= 8791
XPROF_START_STEP ?= 11
XPROF_STEPS ?= 10
PROFILE_ID ?= $(shell date -u +%Y%m%dT%H%M%SZ)
PROFILE_ID := $(PROFILE_ID)
PROFILE_OUTPUT ?= $(CURDIR)/profiles/$(PROFILE_ID)-$(TARGET)
PROFILE_OUTPUT := $(PROFILE_OUTPUT)
TARGET_DIR := $(CURDIR)/submissions/$(TARGET)
TARGET_ENTRY := $(TARGET_DIR)/train.py

UV_BASE = $(UV) --cache-dir "$(UV_CACHE_DIR)"
UV_RUN = $(UV_BASE) run --frozen --no-sync

.PHONY: help prepare require-prepare validate-target preflight run baseline profile report

help:
	@printf '%s\n' \
	  'GPT TPU speedrun workflows:' \
	  '  make prepare   sync the frozen environment and open the setup wizard' \
	  '  make run       validate configured hosts/data and run the reference target' \
	  '  make baseline  compatibility alias for make run' \
	  '  make run TARGET=name  run submissions/name/train.py with its config.yaml' \
	  '  make profile   run a distributed XProf diagnostic, then serve it on :8791' \
	  '  make report    rebuild report.html from integrity-checked run logs' \
	  '' \
	  'Useful overrides: TRAIN_TOKENS=624984064 TARGET=reference PROFILE_OUTPUT=... REPORT=report.html' \
	  'TRAIN_TOKENS sizes prepare-only corpus selection; run contracts remain fixed.'

# This is intentionally interactive: the wizard owns personal paths and defaults.
prepare:
	$(UV_BASE) sync --frozen
	$(UV_RUN) speedrun prepare --training-tokens "$(TRAIN_TOKENS)"

require-prepare:
	@if [[ ! -f "$(CURDIR)/.speedrun.toml" ]] || \
	   ! grep -Eq '^[[:space:]]*default_profile[[:space:]]*=[[:space:]]*"(smoke|dev|official)"[[:space:]]*$$' "$(CURDIR)/.speedrun.toml"; then \
	  printf '%s\n' 'No saved default profile found. Run `make prepare` first.' >&2; \
	  exit 2; \
	fi

validate-target:
	@if [[ ! "$(TARGET)" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$$ ]]; then \
	  printf '%s\n' 'TARGET must be a folder name using letters, digits, dot, underscore, or hyphen.' >&2; \
	  exit 2; \
	fi
	@if [[ ! -f "$(TARGET_ENTRY)" || -L "$(TARGET_ENTRY)" ]]; then \
	  printf '%s\n' 'Missing regular target entry: submissions/$(TARGET)/train.py' >&2; \
	  exit 2; \
	fi
	@if [[ ! -f "$(TARGET_DIR)/config.yaml" || -L "$(TARGET_DIR)/config.yaml" ]]; then \
	  printf '%s\n' 'Missing regular target config: submissions/$(TARGET)/config.yaml' >&2; \
	  exit 2; \
	fi

# Full SHA-256 data checks plus a small BF16 matmul and topology-wide collective.
preflight: require-prepare
	$(UV_RUN) speedrun doctor \
	  --require-tpu \
	  --color always

# The saved profile/track/data/checkpoint defaults come from make prepare. The
# target's complete experiment definition lives in its sibling config.yaml.
run: validate-target preflight
	$(UV_RUN) speedrun run "$(TARGET)" --color always

# Preserve the original user-facing name while `run TARGET=name` is the more
# general upstream interface.
baseline: run

# Diagnostic, not a leaderboard attempt. Compilation precedes tracing; steps 11-20
# are captured by default so the trace stays small while the run still exercises 100
# faithful prefix steps of the selected schedule. Override XPROF_START_STEP/STEPS.
profile: validate-target preflight
	mkdir -p "$(PROFILE_OUTPUT)"
	$(UV_RUN) speedrun profile "$(TARGET)" \
	  --output-dir "$(PROFILE_OUTPUT)" \
	  --steps 100 \
	  --xprof-start-step $(XPROF_START_STEP) \
	  --xprof-steps $(XPROF_STEPS) \
	  --color always
	@printf '\nProfile ready at %s\nOpen http://localhost:%s (Ctrl-C stops the viewer).\n\n' \
	  "$(PROFILE_OUTPUT)/xprof" "$(XPROF_PORT)"
	$(UV_BASE) tool run --from "xprof==$(XPROF_VERSION)" --with "setuptools<70" xprof \
	  --port="$(XPROF_PORT)" "$(PROFILE_OUTPUT)/xprof"

report:
	$(UV_RUN) speedrun report --runs "$(RUNS_PATH)" --output "$(REPORT)"
