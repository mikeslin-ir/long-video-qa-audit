#!/usr/bin/env bash
# Extended K-sweep on Video-MME long bucket (cascade's winning regime).
# Produces an accuracy-vs-K curve for cascade vs uniform across budgets.
# To save GPU, random is omitted for the new K values (its floor is established
# at K=4/8); set SELECTORS to override.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-python}"
VLM="${VLM:-Qwen/Qwen2.5-VL-7B-Instruct}"
KS="${KS:-1,2,6}"
FLOOR="${FLOOR:-0.5}"
LIMIT="${LIMIT:-200}"
BUCKET="${BUCKET:-long}"
SELECTORS="${SELECTORS:-uniform,cascade}"

summarize () { local out="$1"
  [[ -f "$out/qa_summary.json" ]] && \
    $PY -c "import json;d=json.load(open('$out/qa_summary.json'));n=d['n'];print('  -> n=%d acc=%.3f sec/clip=%.1f'%(n,d['accuracy'],d['elapsed_sec']/max(n,1)))"
}

run_uniform () { local k="$1"; local out="outputs/trackB_${BUCKET}_uniform_k${k}"
  [[ -f "$out/qa_summary.json" ]] && { echo "[SKIP uniform k$k]"; summarize "$out"; return; }
  echo "=== [START uniform k${k}] $(date '+%F %T') ==="
  $PY scripts/run_qa_benchmark.py --dataset video_mme --real --vlm_model "$VLM" \
      --selector uniform --budget "$k" --only_task_type "$BUCKET" --limit "$LIMIT" \
      --output_dir "$out" || echo "  FAILED uniform k$k"; summarize "$out"; }

run_cascade () { local k="$1"; local tag; tag=$($PY -c "print(str(${FLOOR}).replace('.','p'))")
  local out="outputs/trackB_${BUCKET}_cascade_k${k}_f${tag}"
  [[ -f "$out/qa_summary.json" ]] && { echo "[SKIP cascade k$k]"; summarize "$out"; return; }
  echo "=== [START cascade k${k} floor${FLOOR}] $(date '+%F %T') ==="
  $PY scripts/run_qa_benchmark.py --dataset video_mme --real --vlm_model "$VLM" \
      --cascade --cascade_planner rule_based --cascade_profile audit \
      --cascade_uniform_floor "$FLOOR" \
      --budget "$k" --only_task_type "$BUCKET" --limit "$LIMIT" \
      --output_dir "$out" || echo "  FAILED cascade k$k"; summarize "$out"; }

run_random () { local k="$1"; local out="outputs/trackB_${BUCKET}_random_k${k}"
  [[ -f "$out/qa_summary.json" ]] && { echo "[SKIP random k$k]"; summarize "$out"; return; }
  echo "=== [START random k${k}] $(date '+%F %T') ==="
  $PY scripts/run_qa_benchmark.py --dataset video_mme --real --vlm_model "$VLM" \
      --selector random --budget "$k" --only_task_type "$BUCKET" --limit "$LIMIT" \
      --output_dir "$out" || echo "  FAILED random k$k"; summarize "$out"; }

echo "######################################################################"
echo "# Video-MME long K-sweep start $(date '+%F %T')  KS=${KS} SELECTORS=${SELECTORS} FLOOR=${FLOOR} LIMIT=${LIMIT}"
echo "######################################################################"
IFS=',' read -ra KS_ARR <<< "$KS"
for k in "${KS_ARR[@]}"; do
  [[ ",$SELECTORS," == *",uniform,"* ]] && run_uniform "$k"
  [[ ",$SELECTORS," == *",random,"*  ]] && run_random  "$k"
  [[ ",$SELECTORS," == *",cascade,"* ]] && run_cascade "$k"
done
touch outputs/.trackB_ksweep_done
echo "######################################################################"
echo "# Video-MME long K-sweep done $(date '+%F %T')"
echo "######################################################################"
