#!/usr/bin/env bash
# Oracle-gap "killer experiment" runner.
#
# Runs the SAME item set through several fixed selectors at budget K (each a
# separate process, sequential -> no GPU contention), plus a uniform run at 2K
# as the "just add frames" reference, then aggregates with
# scripts/oracle_gap_experiment.py.
#
# The single number that decides everything: ORACLE GAP = oracle - best_fixed.
#   - gap tiny  -> a learned router / dynamic-K / re-think is NOT worth building.
#   - gap large -> head-room exists; proceed to the learned router.
#
# Usage:
#   DATASET=xd_violence LIMIT=100 K=8 bash scripts/paper_runs/run_oracle_gap.sh
#   DATASET=egoschema   LIMIT=150 K=4 bash scripts/paper_runs/run_oracle_gap.sh
#
# Env:
#   DATASET    (default xd_violence)   one of run_qa_benchmark.py --dataset
#   LIMIT      (default 100)           #items (first N, deterministic order)
#   K          (default 8)             frame budget
#   VLM_MODEL  (default Qwen/Qwen2.5-VL-7B-Instruct)
#   OUT_ROOT   (default outputs/oracle_gap)
#   SELECTORS  (default "uniform motion clip aks")  fixed baselines to include
#              (cascade is always added via --cascade)
set -euo pipefail

cd "$(dirname "$0")/../.."

DATASET="${DATASET:-xd_violence}"
LIMIT="${LIMIT:-100}"
K="${K:-8}"
K2="$(( K * 2 ))"
VLM_MODEL="${VLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
OUT_ROOT="${OUT_ROOT:-outputs/oracle_gap}"
SELECTORS="${SELECTORS:-uniform motion clip aks}"

TAG="${DATASET}_k${K}_n${LIMIT}"
BASE="${OUT_ROOT}/${TAG}"
mkdir -p "${BASE}"

echo "=== oracle-gap: dataset=${DATASET} K=${K} limit=${LIMIT} model=${VLM_MODEL} ==="

run_sel () {   # $1 = selector name, $2 = extra args
  local name="$1"; shift
  local dir="${BASE}/${name}"
  if [[ -f "${dir}/qa_summary.json" ]]; then
    echo "[skip] ${name} (already has qa_summary.json)"
    return 0
  fi
  echo "[run ] selector=${name} -> ${dir}"
  python scripts/run_qa_benchmark.py \
    --dataset "${DATASET}" --real --vlm_model "${VLM_MODEL}" \
    --budget "${K}" --limit "${LIMIT}" \
    --output_dir "${dir}" "$@"
}

# 1) fixed single-cue / trivial selectors at K
for s in ${SELECTORS}; do
  run_sel "${s}" --selector "${s}"
done

# 2) cascade (multi-cue, "ours") at K
run_sel "cascade" --cascade --cascade_planner rule_based --cascade_uniform_floor 0.5

# 3) "just add frames" reference: uniform at 2K
DIR2="${BASE}/uniform_k${K2}"
if [[ ! -f "${DIR2}/qa_summary.json" ]]; then
  echo "[run ] more-frames uniform@${K2} -> ${DIR2}"
  python scripts/run_qa_benchmark.py \
    --dataset "${DATASET}" --real --vlm_model "${VLM_MODEL}" \
    --budget "${K2}" --limit "${LIMIT}" \
    --output_dir "${DIR2}" --selector uniform
else
  echo "[skip] uniform@${K2} (already has qa_summary.json)"
fi

# 4) aggregate
RUN_ARGS=()
for s in ${SELECTORS}; do RUN_ARGS+=("${s}=${BASE}/${s}"); done
RUN_ARGS+=("cascade=${BASE}/cascade")

echo "=== aggregating ==="
python scripts/oracle_gap_experiment.py \
  --dataset "${DATASET}" --k "${K}" \
  --runs "${RUN_ARGS[@]}" \
  --more_frames "uniform${K2}=${DIR2}" --more_k "${K2}" \
  --out "${BASE}/oracle_gap_report.json"
