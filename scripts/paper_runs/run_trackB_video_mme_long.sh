#!/usr/bin/env bash
# Track B: cascade vs uniform vs random at matched K on event-sparse video.
# Uses the Video-MME *long* bucket, where information is non-uniformly spread and
# uniform sampling is most likely to miss the key moment.
#
# Hypothesis: cascade > uniform at matched K when the video is long / event-sparse.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-python}"
VLM="${VLM:-Qwen/Qwen2.5-VL-7B-Instruct}"
LIMIT="${LIMIT:-200}"          # first N long-bucket items
K="${K:-8}"
FLOOR="${FLOOR:-0.5}"          # uniform-anchor floor for the cascade arm
BUCKET="${BUCKET:-long}"

summarize () {  # out
  local out="$1"
  if [[ -f "$out/qa_summary.json" ]]; then
    $PY -c "import json;d=json.load(open('$out/qa_summary.json'));print('  -> n=%d acc=%.3f'%(d.get('n',0),d.get('accuracy',0)))"
  fi
}

run_uniform () {
  local out="outputs/trackB_${BUCKET}_uniform_k${K}"
  [[ -f "$out/qa_summary.json" ]] && { echo "[SKIP uniform]"; summarize "$out"; return; }
  echo "=== [START uniform k${K}] $(date '+%F %T') ==="
  $PY scripts/run_qa_benchmark.py --dataset video_mme --real --vlm_model "$VLM" \
      --selector uniform --budget "$K" --only_task_type "$BUCKET" --limit "$LIMIT" \
      --output_dir "$out" || echo "  FAILED uniform"
  summarize "$out"
}

run_random () {
  local out="outputs/trackB_${BUCKET}_random_k${K}"
  [[ -f "$out/qa_summary.json" ]] && { echo "[SKIP random]"; summarize "$out"; return; }
  echo "=== [START random k${K}] $(date '+%F %T') ==="
  $PY scripts/run_qa_benchmark.py --dataset video_mme --real --vlm_model "$VLM" \
      --selector random --budget "$K" --only_task_type "$BUCKET" --limit "$LIMIT" \
      --output_dir "$out" || echo "  FAILED random"
  summarize "$out"
}

run_cascade () {
  local tag
  tag=$($PY -c "print(str(${FLOOR}).replace('.','p'))")
  local out="outputs/trackB_${BUCKET}_cascade_k${K}_f${tag}"
  [[ -f "$out/qa_summary.json" ]] && { echo "[SKIP cascade]"; summarize "$out"; return; }
  echo "=== [START cascade k${K} floor${FLOOR}] $(date '+%F %T') ==="
  $PY scripts/run_qa_benchmark.py --dataset video_mme --real --vlm_model "$VLM" \
      --cascade --cascade_planner rule_based --cascade_profile audit \
      --cascade_uniform_floor "$FLOOR" \
      --budget "$K" --only_task_type "$BUCKET" --limit "$LIMIT" \
      --output_dir "$out" || echo "  FAILED cascade"
  summarize "$out"
}

echo "######################################################################"
echo "# Track B start $(date '+%F %T')  bucket=${BUCKET} K=${K} FLOOR=${FLOOR} LIMIT=${LIMIT}"
echo "######################################################################"
run_uniform
run_random
run_cascade
touch outputs/.trackB_done
echo "######################################################################"
echo "# Track B done $(date '+%F %T')"
echo "######################################################################"
