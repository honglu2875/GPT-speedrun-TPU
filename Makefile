SHELL := /bin/bash

UV ?= uv
UV_CACHE_DIR ?= /tmp/uv-cache
RUNS_PATH ?= runs
TARGET ?= reference
TIER ?= 125m
# Optional run label. Left empty, `rig run` prompts for one on a terminal.
NAME ?=
REPORT ?= report.html
# Steps kept per layer-snapshot chart. 0 = every recorded step (finest dragger).
LAYER_SNAPSHOTS ?= 0
# Explicit corpus identity and shard prefix (0/empty means the manifest default).
DATASET ?= 8B
TRAIN_SHARDS ?= 79
# Optional scientific overrides. Empty values defer to the recipe profile.
CONTEXT ?=
TPP ?=
BASE_LR ?=
BATCH_SIZE ?=
STOP_AFTER_STEP ?=

XPROF_VERSION ?= 2.22.3
XPROF_PORT ?= 8791
XPROF_START_STEP ?= 11
XPROF_STEPS ?= 10
PROFILE_ID ?= $(shell date -u +%Y%m%dT%H%M%SZ)
PROFILE_ID := $(PROFILE_ID)
PROFILE_OUTPUT ?= $(CURDIR)/profiles/$(PROFILE_ID)-$(TARGET)
PROFILE_OUTPUT := $(PROFILE_OUTPUT)
TARGET_DIR := $(CURDIR)/recipes/$(TARGET)
TARGET_ENTRY := $(TARGET_DIR)/train.py

UV_BASE = $(UV) --cache-dir "$(UV_CACHE_DIR)"
UV_RUN = $(UV_BASE) run --frozen --no-sync

.PHONY: help check prepare require-prepare validate-target preflight run baseline profile report study-export

help:
	@printf '%s\n' \
	  'GPT TPU rig workflows:' \
	  '  make check     run every CPU-only check; no TPU needed (~2 min)' \
	  '  make prepare   sync the frozen environment and open the setup wizard' \
	  '  make run       validate configured hosts/data and run the 125m reference tier' \
	  '  make baseline  compatibility alias for make run' \
	  '  make run TARGET=name TIER=250m  run one tier from a model family' \
	  '  make profile   run a distributed XProf diagnostic, then serve it on :8791' \
	  '  make report    rebuild report.html from integrity-checked run logs' \
	  '' \
	  'Useful overrides: TIER=500m CONTEXT=8k TPP=5 BATCH_SIZE=128 NAME=my-run TARGET=reference' \
	  'Preparation: DATASET=8B TRAIN_SHARDS=79. Empty run overrides use config.yaml.'

# Everything that can be verified without an accelerator. There is no hosted CI,
# so this is the gate: run it before committing and before any TPU time. It is
# CPU-only by construction -- tests/conftest.py pins JAX_PLATFORMS=cpu, which
# also keeps it from wedging on a multi-host slice a single process cannot init.
check:
	@printf '\n== test suite ==\n'
	$(UV_RUN) python -m pytest tests/ -q
	@printf '\n== static sanity checks ==\n'
	$(UV_RUN) ruff check .
	@printf '\n== the CLI is importable and its surface is intact ==\n'
	$(UV_RUN) rig --help >/dev/null && printf 'rig --help ok\n'
	$(UV_RUN) rig settings >/dev/null && printf 'rig settings ok\n'
	@printf '\n== every recipe resolves a deterministic dev plan ==\n'
	@for entry in $(CURDIR)/recipes/*/train.py; do \
	  name=$$(basename $$(dirname $$entry)); \
	  config=$$(dirname $$entry)/config.yaml; \
	  JAX_PLATFORMS=cpu $(UV_RUN) python $$entry \
	    --config $$config --profile dev --print-plan \
	    | $(UV_RUN) python -m json.tool >/dev/null \
	    && printf 'recipes/%s ok\n' "$$name" || exit 1; \
	done
	@printf '\n== the report builds from the recorded runs ==\n'
	$(UV_RUN) rig report --runs "$(RUNS_PATH)" --output "$(CURDIR)/.check-report.html"
	@rm -f "$(CURDIR)/.check-report.html"
	@printf '\nall checks passed\n\n'

# This is intentionally interactive: the wizard owns personal paths and defaults.
prepare:
	$(UV_BASE) sync --frozen --group dev
	$(UV_RUN) rig prepare --dataset "$(DATASET)" $(if $(TRAIN_SHARDS),--train-shards "$(TRAIN_SHARDS)",)

require-prepare:
	@if [[ ! -f "$(CURDIR)/.rig.toml" ]] || \
	   ! grep -Eq '^[[:space:]]*default_profile[[:space:]]*=[[:space:]]*"(smoke|dev|official)"[[:space:]]*$$' "$(CURDIR)/.rig.toml"; then \
	  printf '%s\n' 'No saved default profile found. Run `make prepare` first.' >&2; \
	  exit 2; \
	fi

validate-target:
	@if [[ ! "$(TARGET)" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$$ ]]; then \
	  printf '%s\n' 'TARGET must be a folder name using letters, digits, dot, underscore, or hyphen.' >&2; \
	  exit 2; \
	fi
	@if [[ ! -f "$(TARGET_ENTRY)" || -L "$(TARGET_ENTRY)" ]]; then \
	  printf '%s\n' 'Missing regular target entry: recipes/$(TARGET)/train.py' >&2; \
	  exit 2; \
	fi
	@if [[ ! -f "$(TARGET_DIR)/config.yaml" || -L "$(TARGET_DIR)/config.yaml" ]]; then \
	  printf '%s\n' 'Missing regular target config: recipes/$(TARGET)/config.yaml' >&2; \
	  exit 2; \
	fi
	@if ! grep -Eq '^schema_version:[[:space:]]*4[[:space:]]*$$' "$(TARGET_DIR)/config.yaml"; then \
	  printf '%s\n' 'TARGET is a legacy fixed-model recipe. Clone the current reference to create a tiered family.' >&2; \
	  exit 2; \
	fi

# Full SHA-256 data checks plus a small BF16 matmul and topology-wide collective.
preflight: require-prepare
	$(UV_RUN) rig doctor \
	  --require-tpu \
	  --color always

# The saved profile/data/checkpoint defaults come from make prepare. The
# target's complete experiment definition lives in its sibling config.yaml.
run: validate-target preflight
	$(UV_RUN) rig run "$(TARGET)" --tier "$(TIER)" \
	  $(if $(CONTEXT),--context "$(CONTEXT)",) \
	  $(if $(TPP),--tokens-per-parameter "$(TPP)",) \
	  $(if $(BASE_LR),--base-learning-rate "$(BASE_LR)",) \
	  $(if $(BATCH_SIZE),--batch-size "$(BATCH_SIZE)",) \
	  $(if $(STOP_AFTER_STEP),--stop-after-step "$(STOP_AFTER_STEP)",) \
	  $(if $(NAME),--name "$(NAME)",) --color always

# Preserve the original user-facing name while `run TARGET=name` is the more
# general upstream interface.
baseline: run

# Diagnostic, not a leaderboard attempt. Compilation precedes tracing; steps 11-20
# are captured by default so the trace stays small while the run still exercises 100
# faithful prefix steps of the selected schedule. Override XPROF_START_STEP/STEPS.
profile: validate-target preflight
	mkdir -p "$(PROFILE_OUTPUT)"
	$(UV_RUN) rig profile "$(TARGET)" \
	  --tier "$(TIER)" \
	  $(if $(CONTEXT),--context "$(CONTEXT)",) \
	  $(if $(TPP),--tokens-per-parameter "$(TPP)",) \
	  $(if $(BASE_LR),--base-learning-rate "$(BASE_LR)",) \
	  $(if $(BATCH_SIZE),--batch-size "$(BATCH_SIZE)",) \
	  --output-dir "$(PROFILE_OUTPUT)" \
	  --stop-after-step 100 \
	  --xprof-start-step $(XPROF_START_STEP) \
	  --xprof-steps $(XPROF_STEPS) \
	  --color always
	@printf '\nProfile ready at %s\nOpen http://localhost:%s (Ctrl-C stops the viewer).\n\n' \
	  "$(PROFILE_OUTPUT)/xprof" "$(XPROF_PORT)"
	$(UV_BASE) tool run --from "xprof==$(XPROF_VERSION)" --with "setuptools<70" xprof \
	  --port="$(XPROF_PORT)" "$(PROFILE_OUTPUT)/xprof"

report:
	$(UV_RUN) rig report --runs "$(RUNS_PATH)" --output "$(REPORT)" --layer-snapshots $(LAYER_SNAPSHOTS)

# Lay a study out the way the dataset repository expects it, ready to upload.
# STUDY names the folder; TARGET is where it lands; SELECT narrows the runs.
#   make study-export STUDY=lr-sweep-8k-60M TARGET=hf-dataset SELECT=reference
study-export:
	@test -n "$(STUDY)" || { echo "set STUDY=<folder name>"; exit 2; }
	$(UV_RUN) rig report --runs "$(RUNS_PATH)" \
	  --study-export-target "$(or $(TARGET),hf-dataset)" --study-name "$(STUDY)" \
	  $(if $(SELECT),--select "$(SELECT)",)
