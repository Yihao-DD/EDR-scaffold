#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-20260704}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/repro_rep2/reruns/seed${SEED}"
mkdir -p "$OUT"

python3 "$ROOT/repro_rep2/scripts/lora_phase0.py" \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  train-eval \
  --dataset "$ROOT/repro_rep2/data/distill_main_replay2_capped.jsonl" \
  --output-dir "$OUT/adapter" \
  --eval-output "$OUT/train_eval.json" \
  --rank 16 \
  --lr 5e-5 \
  --epochs 3 \
  --seed "$SEED" \
  --kl-anchor-lambda 2

python3 "$ROOT/repro_rep2/scripts/s07_heldout_eval.py" \
  --root "$ROOT/EDG-EXP3-distill" \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --adapter-dir "$OUT/adapter" \
  --arm main \
  --config rep2 \
  --seed "$SEED" \
  --run-id "repro_rep2_seed${SEED}" \
  --train-dataset "$ROOT/repro_rep2/data/distill_main_replay2_capped.jsonl" \
  --output "$OUT/heldout.json"

python3 "$ROOT/repro_rep2/scripts/s07_sibling_arena_eval.py" \
  --root "$ROOT/EDG-EXP3-distill" \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --adapter-dir "$OUT/adapter" \
  --arm main \
  --config rep2 \
  --seed "$SEED" \
  --run-id "repro_rep2_seed${SEED}" \
  --arena "$ROOT/repro_rep2/data/v17_sibling_arena.json" \
  --output "$OUT/sibling.json"

echo "wrote $OUT"
