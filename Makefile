SHELL := /bin/bash

UV ?= uv
UV_CACHE_DIR ?= /tmp/uv-cache
RUNS_PATH ?= runs
TARGET ?= reference
TIER ?= 125m
REPORT ?= report.html
# Enough distinct train data for the largest 500m × 5-TPP LR-study point.
# The preparation router selects the immutable 4B corpus for this budget.
TRAIN_TOKENS ?= 2600000000

XPROF_VERSION ?= 2.22.3
XPROF_PORT ?= 8791
XPROF_START_STEP ?= 11
XPROF_STEPS ?= 10
PROFILE_ID ?= $(shell date -u +%Y%m%dT%H%M%SZ)
PROFILE_ID := $(PROFILE_ID)
PROFILE_OUTPUT ?= $(CURDIR)/profiles/$(PROFILE_ID)-$(TARGET)
PROFILE_OUTPUT := $(PROFILE_OUTPUT)
LR_SUITE ?= studies/complete_d_p_lr_v1/suite.yaml
LR_RESULTS ?= runs/studies/complete_d_p_lr_v1/results.csv
TARGET_DIR := $(CURDIR)/submissions/$(TARGET)
TARGET_ENTRY := $(TARGET_DIR)/train.py

UV_BASE = $(UV) --cache-dir "$(UV_CACHE_DIR)"
UV_RUN = $(UV_BASE) run --frozen --no-sync

.PHONY: help prepare require-prepare validate-target preflight run baseline profile sweep-lr sweep-lr-60m-extend report

help:
	@printf '%s\n' \
	  'GPT TPU speedrun workflows:' \
	  '  make prepare   sync the frozen environment and open the setup wizard' \
	  '  make run       validate configured hosts/data and run the 125m reference tier' \
	  '  make baseline  compatibility alias for make run' \
	  '  make run TARGET=name TIER=250m  run one tier from a model family' \
	  '  make profile   run a distributed XProf diagnostic, then serve it on :8791' \
	  '  make sweep-lr  run/resume the CSV-first 60m-500m 0.25-OT LR study' \
	  '  make sweep-lr-60m-extend  bracket the 60m optimum above the first grid' \
	  '  make report    rebuild report.html from integrity-checked run logs' \
	  '' \
	  'Useful overrides: TIER=125m TRAIN_TOKENS=2600000000 TARGET=reference PROFILE_OUTPUT=... REPORT=report.html' \
	  'TRAIN_TOKENS selects the immutable corpus prefix used by non-smoke runs.'

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
	@if ! grep -Eq '^schema_version:[[:space:]]*2[[:space:]]*$$' "$(TARGET_DIR)/config.yaml"; then \
	  printf '%s\n' 'TARGET is a legacy fixed-model submission. Clone the current reference to create a tiered family.' >&2; \
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
	$(UV_RUN) speedrun run "$(TARGET)" --tier "$(TIER)" --color always

# Preserve the original user-facing name while `run TARGET=name` is the more
# general upstream interface.
baseline: run

# Preflight performs the expensive integrity/topology check once. Individual
# resumable study points retain header checks but skip repeated full SHA scans.
sweep-lr: validate-target preflight
	$(UV_RUN) python -m speedrun.family_study run \
	  --suite "$(LR_SUITE)" \
	  --results "$(LR_RESULTS)" \
	  --color always

sweep-lr-60m-extend: LR_SUITE := studies/complete_d_p_lr_60m_extension_v1/suite.yaml
sweep-lr-60m-extend: LR_RESULTS := runs/studies/complete_d_p_lr_60m_extension_v1/results.csv
sweep-lr-60m-extend: sweep-lr

# Diagnostic, not a leaderboard attempt. Compilation precedes tracing; steps 11-20
# are captured by default so the trace stays small while the run still exercises 100
# faithful prefix steps of the selected schedule. Override XPROF_START_STEP/STEPS.
profile: validate-target preflight
	mkdir -p "$(PROFILE_OUTPUT)"
	$(UV_RUN) speedrun profile "$(TARGET)" \
	  --tier "$(TIER)" \
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
