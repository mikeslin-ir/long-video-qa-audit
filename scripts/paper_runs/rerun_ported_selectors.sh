#!/usr/bin/env bash
# Sequential re-run of the three ported selectors after the candidate-pool
# truncation fix (pool covered only the first ~42s of each video).
set -uo pipefail
cd "$(dirname "$0")/../.."
exec 9>/tmp/ported_selectors.lock
flock -n 9 || { echo "another re-run holds the lock; aborting"; exit 1; }
PY="${PY:-python3}"
for S in keyscore aks videotree; do
  echo "=== $(date +%H:%M) starting $S ==="
  "$PY" scripts/run_qa_benchmark.py \
    --dataset video_mme --only_task_type long --limit 900 --budget 8 --tau_sec 2.0 \
    --real --vlm_backbone qwen2_vl --vlm_model Qwen/Qwen2.5-VL-7B-Instruct \
    --item_timeout_sec 600 --resume --selector "$S" \
    --output_dir "outputs/trackB_long_n900_k8_$S"
  echo "=== $(date +%H:%M) finished $S ==="
done
