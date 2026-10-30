#!/usr/bin/env bash
# Single-instance guard: a duplicate run corrupted rows.jsonl on 2026-08-26.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p outputs
exec 9>outputs/.videotree_n900.lock
flock -n 9 || { echo "another videotree n900 run holds the lock; aborting"; exit 1; }
exec python3 scripts/run_qa_benchmark.py \
  --dataset video_mme --only_task_type long --limit 900 --budget 8 --tau_sec 2.0 \
  --real --vlm_backbone qwen2_vl --vlm_model Qwen/Qwen2.5-VL-7B-Instruct \
  --item_timeout_sec 600 --resume --selector videotree \
  --output_dir outputs/trackB_long_n900_k8_videotree
