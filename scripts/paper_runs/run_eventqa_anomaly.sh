#!/usr/bin/env bash
# Track B extension: binary event-QA on anomaly datasets (event-sparse by design).
# Compares cascade vs uniform vs random at matched K on UCF-Crime and/or
# XD-Violence. The anomaly is a short segment of a long clip, so uniform sampling
# is expected to miss it while content-aware routing catches it.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-python}"
VLM="${VLM:-Qwen/Qwen2.5-VL-7B-Instruct}"
DATASETS="${DATASETS:-ucf_crime,xd_violence}"
K="${K:-8}"
FLOOR="${FLOOR:-0.5}"
LIMIT="${LIMIT:-200}"

summarize () { local out="$1"
  [[ -f "$out/qa_summary.json" ]] && \
    $PY -c "import json;d=json.load(open('$out/qa_summary.json'));n=d['n'];print('  -> n=%d acc=%.3f sec/clip=%.1f'%(n,d['accuracy'],d['elapsed_sec']/max(n,1)))"
}

run_arm () {  # dataset selector_flags tag
  local ds="$1"; local flags="$2"; local tag="$3"
  local out="outputs/eventqa_${ds}_${tag}_k${K}"
  if [[ -f "$out/qa_summary.json" ]]; then echo "[SKIP $ds $tag]"; summarize "$out"; return; fi
  echo "=== [START $ds $tag k${K}] $(date '+%F %T') ==="
  $PY scripts/run_qa_benchmark.py --dataset "$ds" --real --vlm_model "$VLM" \
      --budget "$K" --limit "$LIMIT" $flags --output_dir "$out" \
      || { echo "  FAILED $ds $tag" >&2; return 1; }
  summarize "$out"
}

echo "######################################################################"
echo "# Event-QA (anomaly) start $(date '+%F %T')  DS=${DATASETS} K=${K} FLOOR=${FLOOR} LIMIT=${LIMIT}"
echo "######################################################################"
IFS=',' read -ra DS_ARR <<< "$DATASETS"
for ds in "${DS_ARR[@]}"; do
  run_arm "$ds" "--selector uniform" "uniform"
  run_arm "$ds" "--selector random"  "random"
  run_arm "$ds" "--cascade --cascade_planner rule_based --cascade_uniform_floor $FLOOR" "cascade_f$($PY -c "print(str(${FLOOR}).replace('.','p'))")"
done
touch outputs/.eventqa_done
echo "######################################################################"
echo "# Event-QA (anomaly) done $(date '+%F %T')"
echo "######################################################################"
