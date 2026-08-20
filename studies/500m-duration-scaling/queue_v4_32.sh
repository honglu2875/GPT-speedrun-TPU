#!/usr/bin/env bash
set -euo pipefail

cd /home/cubic27/GPT-speedrun-TPU

queue_lock=/tmp/500m-duration-v4-32.lock
queue_pid=/tmp/500m-duration-v4-32.pid
exec 9>"$queue_lock"
if ! flock -n 9; then
  printf 'another 500M duration queue already holds %s\n' "$queue_lock" >&2
  exit 1
fi
printf '%s\n' "$$" >"$queue_pid"

export UV_CACHE_DIR=/tmp/gpt-speedrun-uv-cache

on_exit() {
  status=$?
  printf '=== %s queue exit status %s ===\n' "$(date -u +%FT%TZ)" "$status"
  trap - EXIT
  exit "$status"
}
trap on_exit EXIT

learning_rate() {
  case "$1" in
    6) printf '%s\n' 0.015625 ;;
    7) printf '%s\n' 0.0078125 ;;
    8) printf '%s\n' 0.00390625 ;;
    9) printf '%s\n' 0.001953125 ;;
    *) printf 'unsupported LR exponent: %s\n' "$1" >&2; return 1 ;;
  esac
}

run_point() {
  recipe=$1
  arm=$2
  tpp=$3
  batch=$4
  exponent=$5
  seed=$6
  name="500m-${arm}-bs${batch}-lr2e-${exponent}-s${seed}"

  lr=$(learning_rate "$exponent")
  if [[ "${RIG_QUEUE_PLAN_ONLY:-0}" == 1 ]]; then
    .venv/bin/python -c '
from pathlib import Path
import sys

from rig.plan import resolve_recipe_plan

recipe = sys.argv[1]
resolve_recipe_plan(
    python_executable=Path.cwd() / ".venv/bin/python",
    trainer=(Path("recipes") / recipe / "train.py").resolve(),
    arguments=sys.argv[2:],
    cwd=Path.cwd(),
)
' "$recipe" \
      --profile dev --context 1k --tier 500m \
      --tokens-per-parameter "$tpp" --batch-size "$batch" \
      --base-learning-rate "$lr" --seed "$seed"
    printf 'validated %s\n' "$name"
    return
  fi

  if compgen -G "runs/*-${name}-*/result.json" >/dev/null; then
    printf '=== %s skip %s ===\n' "$(date -u +%FT%TZ)" "$name"
    return
  fi

  if [[ "$tpp" == 5 ]]; then
    run_timeout=9000
  else
    run_timeout=28800
  fi

  printf '=== %s start %s ===\n' "$(date -u +%FT%TZ)" "$name"
  uv run --frozen --no-sync rig run "$recipe" \
    --cluster v4-32 --profile dev --context 1k --tier 500m \
    --tokens-per-parameter "$tpp" --batch-size "$batch" \
    --base-learning-rate "$lr" --seed "$seed" \
    --checkpoint-policy none --timeout "$run_timeout" --name "$name"
  printf '=== %s done %s ===\n' "$(date -u +%FT%TZ)" "$name"
}

printf '=== %s 500M duration study start ===\n' "$(date -u +%FT%TZ)"

# Phase 1: decisive bridge cells. Interleave treatments within each seed so
# infrastructure drift cannot masquerade as a parameterization effect.
for seed in 1337 1338 1339; do
  run_point reference r20 20 128 8 "$seed"
  run_point reference_duration d20 20 128 8 "$seed"
  run_point reference_duration d20 20 128 7 "$seed"
  run_point reference_duration d20 20 512 8 "$seed"
done

# Phase 2: complete the missing 5-TPP LR and batch brackets. Existing batch
# 128/256, LR 2^-8 points are deliberately reused rather than recomputed.
for seed in 1337 1338 1339; do
  run_point reference r5 5 64 8 "$seed"
  run_point reference r5 5 512 8 "$seed"
  run_point reference r5 5 128 7 "$seed"
  run_point reference r5 5 128 9 "$seed"
done

# Phase 3: finish both 20-TPP LR brackets.
for seed in 1337 1338 1339; do
  run_point reference r20 20 128 7 "$seed"
  run_point reference r20 20 128 9 "$seed"
  run_point reference_duration d20 20 128 6 "$seed"
  run_point reference_duration d20 20 128 9 "$seed"
done

# Phase 4: finish the 20-TPP batch shoulders. d20/batch512 was front-loaded.
for seed in 1337 1338 1339; do
  run_point reference r20 20 64 8 "$seed"
  run_point reference r20 20 256 8 "$seed"
  run_point reference r20 20 512 8 "$seed"
  run_point reference_duration d20 20 256 8 "$seed"
done

printf '=== %s 500M DURATION STUDY COMPLETE ===\n' "$(date -u +%FT%TZ)"
