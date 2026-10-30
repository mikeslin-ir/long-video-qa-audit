#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-python}"
VLM="${VLM:-Qwen/Qwen2.5-VL-7B-Instruct}"
LIMIT="${LIMIT:-500}"
KS="${KS:-4,8}"
PLANNER="${PLANNER:-rule_based}"

run () {  # k
  local k="$1"
  local out="outputs/ksweep_cascade_k${k}_egoschema"
  if [[ -f "$out/qa_summary.json" ]]; then
    local n_done acc
    n_done=$($PY -c "import json;d=json.load(open('$out/qa_summary.json'));print(d.get('n',0))" 2>/dev/null || echo 0)
    acc=$($PY -c "import json;d=json.load(open('$out/qa_summary.json'));print(round(d.get('accuracy',0),3))" 2>/dev/null || echo NA)
    echo "=== [SKIP cascade_k${k}] already complete: n=${n_done} acc=${acc} ==="
    return 0
  fi
  echo "=== [START cascade_k${k}] $(date '+%F %T') ==="
  $PY scripts/run_qa_benchmark.py \
      --dataset egoschema --real --vlm_model "$VLM" \
      --cascade --cascade_planner "$PLANNER" \
      --budget "$k" --limit "$LIMIT" \
      --output_dir "$out" \
      || { echo "  FAILED cascade_k${k}" >&2; return 1; }
  echo "=== [DONE  cascade_k${k}] $(date '+%F %T') ==="
}

echo "######################################################################"
echo "# Cascade K-sweep start at $(date '+%F %T')"
echo "# KS=${KS} LIMIT=${LIMIT} PLANNER=${PLANNER}"
echo "######################################################################"

IFS=',' read -ra KS_ARR <<< "$KS"
for k in "${KS_ARR[@]}"; do
  run "$k"
done

touch outputs/.cascade_ksweep_done
echo "######################################################################"
echo "# Cascade K-sweep done at $(date '+%F %T')"
echo "######################################################################"