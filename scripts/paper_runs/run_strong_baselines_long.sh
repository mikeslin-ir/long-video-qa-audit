#!/usr/bin/env bash
# Strong content-aware baselines on the event-sparse regime (Video-MME long).
# Same backbone (Qwen2.5-VL-7B) and matched K as cascade; only the selector
# changes. Establishes that the cascade beats not just uniform/random but also
# single-skill content-aware selectors (CLIP/SigLIP top-K, motion peaks).
set -euo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-python}"
VLM="${VLM:-Qwen/Qwen2.5-VL-7B-Instruct}"
K="${K:-8}"
LIMIT="${LIMIT:-200}"
BUCKET="${BUCKET:-long}"
SELECTORS="${SELECTORS:-clip,motion}"

summarize () { local out="$1"
  [[ -f "$out/qa_summary.json" ]] && \
    $PY -c "import json;d=json.load(open('$out/qa_summary.json'));n=d['n'];print('  -> n=%d acc=%.3f sec/clip=%.1f'%(n,d['accuracy'],d['elapsed_sec']/max(n,1)))"
}

run_sel () {  # selector
  local s="$1"; local out="outputs/trackB_${BUCKET}_${s}_k${K}"
  if [[ -f "$out/qa_summary.json" ]]; then echo "[SKIP $s]"; summarize "$out"; return; fi
  echo "=== [START $s k${K}] $(date '+%F %T') ==="
  $PY scripts/run_qa_benchmark.py --dataset video_mme --real --vlm_model "$VLM" \
      --selector "$s" --budget "$K" --only_task_type "$BUCKET" --limit "$LIMIT" \
      --output_dir "$out" || echo "  FAILED $s"
  summarize "$out"
}

echo "######################################################################"
echo "# Strong baselines (Video-MME $BUCKET) start $(date '+%F %T')  SEL=${SELECTORS} K=${K} LIMIT=${LIMIT}"
echo "######################################################################"
IFS=',' read -ra SEL_ARR <<< "$SELECTORS"
for s in "${SEL_ARR[@]}"; do run_sel "$s"; done
touch outputs/.strong_baselines_done
echo "######################################################################"
echo "# Strong baselines done $(date '+%F %T')"
echo "######################################################################"
