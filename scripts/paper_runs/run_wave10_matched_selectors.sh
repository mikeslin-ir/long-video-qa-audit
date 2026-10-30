#!/usr/bin/env bash
# Matched-budget selector baselines on Video-MME long (n=900, K=8).
# Same backbone / budget / prompt / grader as every other n=900 run;
# only the selection rule differs.
set -euo pipefail
cd "$(dirname "$0")/../.."

LOGF="logs/wave10_matched_selectors.log"
mkdir -p logs
LOG() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGF"; }

COMMON=(--dataset video_mme --only_task_type long --limit 900
        --budget 8 --tau_sec 2.0 --real
        --vlm_backbone qwen2_vl --vlm_model Qwen/Qwen2.5-VL-7B-Instruct
        --item_timeout_sec 600 --resume)

for SEL in keyscore videotree; do
    OUT="outputs/trackB_long_n900_k8_${SEL}"
    if [ -f "$OUT/qa_summary.json" ]; then
        LOG "$SEL already complete; skipping."
        continue
    fi
    LOG "START $SEL -> $OUT"
    python3 scripts/run_qa_benchmark.py "${COMMON[@]}" \
        --selector "$SEL" --output_dir "$OUT" >>"$LOGF" 2>&1 \
        && LOG "DONE $SEL: $(python3 -c "import json;d=json.load(open('$OUT/qa_summary.json'));print(d['n_correct'],'/',d['n'],'=',round(d['accuracy'],4))")" \
        || { LOG "FAILED $SEL"; exit 1; }
done
LOG "ALL DONE"
