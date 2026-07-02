#!/usr/bin/env bash
set -euo pipefail

QUEUE=${1:?usage: run_s07_d2_queue.sh <main_rep1|star_rep1|main_rep2|star_rep2>}
MODEL=${MODEL:-/root/autodl-tmp/hf/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
ROOT=${ROOT:-/root/autodl-tmp/EDG-EXP3-distill}
SEEDS=${SEEDS:-"20260704 20260705 20260706 20260707"}

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
mkdir -p logs results/s07_d2_5seed/main results/s07_d2_5seed/star adapters/s07_d2_5seed/main adapters/s07_d2_5seed/star

lock_dir="logs/s07_d2_${QUEUE}.lock"
if ! mkdir "$lock_dir" 2>/dev/null; then
  echo "[$(date -Is)] lock exists for ${QUEUE}: ${lock_dir}"
  exit 9
fi
trap 'rmdir "$lock_dir" 2>/dev/null || true' EXIT

case "$QUEUE" in
  main_rep1)
    ARM=main
    REP=rep1
    DATA=data/distill_main_replay1.jsonl
    LR=5e-5
    EPOCHS=3
    KL=0
    NAME_PREFIX=main_d2_rep1_r16_lr5e-5_ep3_replay1
    ;;
  star_rep1)
    ARM=star
    REP=rep1
    DATA=data/distill_star_replay1.jsonl
    LR=1e-4
    EPOCHS=3
    KL=0
    NAME_PREFIX=star_d2_rep1_r16_lr1e-4_ep3_replay1
    ;;
  main_rep2)
    ARM=main
    REP=rep2
    DATA=data/distill_main_replay2_capped.jsonl
    LR=5e-5
    EPOCHS=3
    KL=2
    NAME_PREFIX=main_d2_rep2_r16_lr5e-5_ep3_replay2_capped_lam2
    ;;
  star_rep2)
    ARM=star
    REP=rep2
    DATA=data/distill_star_replay2_capped.jsonl
    LR=1e-4
    EPOCHS=3
    KL=2
    NAME_PREFIX=star_d2_rep2_r16_lr1e-4_ep3_replay2_capped_lam2
    ;;
  *)
    echo "unknown queue: ${QUEUE}" >&2
    exit 2
    ;;
esac

echo "[$(date -Is)] start s07 D2 queue=${QUEUE} arm=${ARM} rep=${REP}"
if [ ! -d "$MODEL" ]; then
  echo "[$(date -Is)] missing model: $MODEL" >&2
  exit 3
fi
if [ ! -s "$DATA" ]; then
  echo "[$(date -Is)] missing dataset: $DATA" >&2
  exit 4
fi

for seed in $SEEDS; do
  name="${NAME_PREFIX}_seed${seed}"
  out="results/s07_d2_5seed/${ARM}/${name}.json"
  adapter="adapters/s07_d2_5seed/${ARM}/${name}"
  echo "[$(date -Is)] ${QUEUE} seed=${seed} name=${name}"
  if [ -f "$out" ]; then
    echo "[$(date -Is)] skip existing output ${out}"
    continue
  fi

  if [ -s "${adapter}/train_metadata.json" ]; then
    echo "[$(date -Is)] eval-only recovery for ${name}"
    /root/miniconda3/bin/python scripts/lora_phase0.py \
      --model-id "$MODEL" \
      eval \
      --adapter-dir "$adapter" \
      --output "$out" \
      --progress-every 50
  else
    args=(
      scripts/lora_phase0.py
      --model-id "$MODEL"
      train-eval
      --dataset "$DATA"
      --output-dir "$adapter"
      --rank 16
      --lr "$LR"
      --seed "$seed"
      --epochs "$EPOCHS"
      --batch-size 1
      --grad-accum-steps 8
      --eval-output "$out"
      --progress-every 50
      --log-every 25
    )
    if [ "$KL" != "0" ]; then
      args+=(--kl-anchor-lambda "$KL")
    fi
    /root/miniconda3/bin/python "${args[@]}"
  fi
  if [ -s "${adapter}/train_metadata.json" ]; then
    /root/miniconda3/bin/python - <<PY
import json
from pathlib import Path
p=Path("${adapter}/train_metadata.json")
d=json.loads(p.read_text())
print("[metadata] ${name} skipped_nonfinite_loss=%s skipped_nonfinite_grad=%s kl_anchor_batches=%s train_seconds=%.1f" % (
    d.get("skipped_nonfinite_loss"), d.get("skipped_nonfinite_grad"), d.get("kl_anchor_batches"), float(d.get("train_seconds", 0.0))
), flush=True)
PY
  fi
  echo "[$(date -Is)] done ${name}"
done

echo "[$(date -Is)] complete s07 D2 queue=${QUEUE}"
