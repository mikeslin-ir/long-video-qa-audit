#!/usr/bin/env bash
# Cross-backbone (#1) generalization: LLaVA-NeXT-Video-7B-hf (Llama LLM + CLIP
# vision == true cross-architecture-family vs Qwen2-VL).
# Runs the F1/F2 core contrasts on Video-MME long, n=200, K=8, matched qids.
set -uo pipefail
cd "$(dirname "$0")/../.."

BK=llava_next_video
DS=video_mme
TT=long
N=200
K=8
TAU=2.0
COMMON="--dataset $DS --only_task_type $TT --limit $N --budget $K --tau_sec $TAU --real --vlm_backbone $BK"

LOGDIR=logs/xbackbone_lnv
mkdir -p "$LOGDIR"

run () {
  local name="$1"; shift
  local out="outputs/xbackbone_lnv_${name}"
  local log="${LOGDIR}/${name}.log"
  echo "=== [$(date +%H:%M:%S)] START $name -> $out ===" | tee -a "$log"
  if [[ -f "$out/qa_summary.json" ]]; then
    echo "    SKIP (already has qa_summary.json)" | tee -a "$log"
    return 0
  fi
  python3 scripts/run_qa_benchmark.py $COMMON "$@" --output_dir "$out" >>"$log" 2>&1
  local rc=$?
  echo "=== [$(date +%H:%M:%S)] END $name rc=$rc ===" | tee -a "$log"
  if [[ -f "$out/qa_summary.json" ]]; then
    python3 -c "import json;s=json.load(open('$out/qa_summary.json'));print('   ',{k:s[k] for k in ('n','n_correct','accuracy','vlm_model','vlm_backbone')})" | tee -a "$log"
  fi
  return $rc
}

# 1) Uniform selection baseline (no cascade, no inject)
run uniform     --selector uniform --inject_evidence_text none

# 2) Cascade selection, no evidence injection
run cascade     --cascade --cascade_planner rule_based --cascade_uniform_floor 0.5 --inject_evidence_text none

# 3) Cascade selection + ASR evidence injection
run cascadeASR  --cascade --cascade_planner rule_based --cascade_uniform_floor 0.5 --inject_evidence_text asr

echo "=== [$(date +%H:%M:%S)] ALL DONE ==="
