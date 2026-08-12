SHELL := /bin/bash

UV ?= uv
UV_CACHE_DIR ?= /tmp/uv-cache
DATA_PATH ?= shm
RUNS_PATH ?= runs
SUBMISSION ?= reference
REPORT ?= report.html

XPROF_VERSION ?= 2.22.3
XPROF_PORT ?= 8791
XPROF_START_STEP ?= 11
XPROF_STEPS ?= 10
PROFILE_ID ?= $(shell date -u +%Y%m%dT%H%M%SZ)
PROFILE_ID := $(PROFILE_ID)
PROFILE_OUTPUT ?= $(CURDIR)/profiles/$(PROFILE_ID)-$(SUBMISSION)
PROFILE_OUTPUT := $(PROFILE_OUTPUT)

BASELINE_TRAIN_TOKENS := 624984064
UV_BASE = $(UV) --cache-dir "$(UV_CACHE_DIR)"
UV_RUN = $(UV_BASE) run --frozen --no-sync

.PHONY: help prepare preflight baseline profile report

help:
	@printf '%s\n' \
	  'GPT TPU speedrun workflows:' \
	  '  make prepare   sync the frozen environment and open the setup wizard' \
	  '  make baseline  validate the node/data and run the exact official baseline' \
	  '  make profile   run a 100-step XProf diagnostic, then serve it on :8791' \
	  '  make report    rebuild report.html from integrity-checked run logs' \
	  '' \
	  'Useful overrides: DATA_PATH=shm SUBMISSION=reference REPORT=report.html'

# This is intentionally interactive: the wizard owns personal paths and defaults.
prepare:
	$(UV_BASE) sync --frozen
	$(UV_RUN) speedrun prepare

# Full SHA-256 data checks plus a small BF16 matmul and four-chip collective.
preflight:
	$(UV_RUN) speedrun doctor \
	  --path "$(DATA_PATH)" \
	  --profile official \
	  --require-tpu \
	  --color always

# Exact open-track reference. The fixed token budget preserves the calibrated
# 19,073 optimizer steps at global batch 32 x sequence length 1,024.
baseline: preflight
	$(UV_RUN) speedrun run "$(SUBMISSION)" \
	  --track open \
	  --profile official \
	  --data-path "$(DATA_PATH)" \
	  --seed 1337 \
	  --target-loss 3.28 \
	  --timeout 21600 \
	  --checkpoints qualifying \
	  --color always \
	  -- \
	  --train-tokens $(BASELINE_TRAIN_TOKENS) \
	  --batch-size 32 \
	  --seq-len 1024 \
	  --layers 12 \
	  --heads 12 \
	  --d-model 768 \
	  --mlp-mult 4 \
	  --dtype bfloat16 \
	  --attention-backend dense \
	  --loss-backend dense \
	  --semantic-vocab-size 50304 \
	  --learning-rate 0.0003 \
	  --min-lr-ratio 0.1 \
	  --warmup-steps 715 \
	  --weight-decay 0.1 \
	  --beta1 0.9 \
	  --beta2 0.95 \
	  --grad-clip 1.0 \
	  --eval-batches 320 \
	  --val-every 250 \
	  --val-probe-batches 8 \
	  --diagnostics-every 250 \
	  --log-every 953 \
	  --peak-tflops 1100

# Diagnostic, not a leaderboard attempt. Compilation precedes tracing; steps 11-20
# are captured by default so the trace stays small while the run still exercises 100
# faithful prefix steps of the baseline schedule. Override XPROF_START_STEP/STEPS.
profile: preflight
	mkdir -p "$(PROFILE_OUTPUT)"
	$(UV_RUN) submissions/$(SUBMISSION)/train.py \
	  --output-dir "$(PROFILE_OUTPUT)" \
	  --seed 1337 \
	  --track open \
	  --profile official \
	  --steps 100 \
	  --batch-size 32 \
	  --seq-len 1024 \
	  --layers 12 \
	  --heads 12 \
	  --d-model 768 \
	  --mlp-mult 4 \
	  --dtype bfloat16 \
	  --attention-backend dense \
	  --loss-backend dense \
	  --semantic-vocab-size 50304 \
	  --learning-rate 0.0003 \
	  --min-lr-ratio 0.1 \
	  --warmup-steps 715 \
	  --weight-decay 0.1 \
	  --beta1 0.9 \
	  --beta2 0.95 \
	  --grad-clip 1.0 \
	  --eval-batches 320 \
	  --val-every 0 \
	  --val-probe-batches 8 \
	  --diagnostics-every 0 \
	  --log-every 100 \
	  --peak-tflops 1100 \
	  --data-format llmc \
	  --dataset-id fineweb10b-gpt2 \
	  --tokenizer-id gpt2 \
	  --train-data "$(DATA_PATH)/fineweb_train_000001.bin" \
	  --train-data "$(DATA_PATH)/fineweb_train_000002.bin" \
	  --train-data "$(DATA_PATH)/fineweb_train_000003.bin" \
	  --train-data "$(DATA_PATH)/fineweb_train_000004.bin" \
	  --train-data "$(DATA_PATH)/fineweb_train_000005.bin" \
	  --train-data "$(DATA_PATH)/fineweb_train_000006.bin" \
	  --train-data "$(DATA_PATH)/fineweb_train_000007.bin" \
	  --train-data "$(DATA_PATH)/fineweb_train_000008.bin" \
	  --train-data "$(DATA_PATH)/fineweb_train_000009.bin" \
	  --val-data "$(DATA_PATH)/fineweb_val_000000.bin" \
	  --xprof-dir "$(PROFILE_OUTPUT)/xprof" \
	  --xprof-start-step $(XPROF_START_STEP) \
	  --xprof-steps $(XPROF_STEPS) \
	  --no-final-validation \
	  --no-checkpoint \
	  --color always
	@printf '\nProfile ready at %s\nOpen http://localhost:%s (Ctrl-C stops the viewer).\n\n' \
	  "$(PROFILE_OUTPUT)/xprof" "$(XPROF_PORT)"
	$(UV_BASE) tool run --from "xprof==$(XPROF_VERSION)" --with "setuptools<70" xprof \
	  --port="$(XPROF_PORT)" "$(PROFILE_OUTPUT)/xprof"

report:
	$(UV_RUN) speedrun report --runs "$(RUNS_PATH)" --output "$(REPORT)"
