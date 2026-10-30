#!/usr/bin/env bash
# Cross-backbone (#1) phase 2: anomaly domains on LLaVA-NeXT-Video-7B-hf.
# Waits for the Video-MME phase-1 driver to finish (avoids 2x 7B GPU contention),
# then runs uniform / cascade / cascade+ASR on XD-Violence and UCF-Crime,
# n=200, K=8, matched qids (first 200 items == cached Qwen runs).
set -uo pipefail
cd "$(dirname "$0")/../.."

BK=llava_next_video
K=8
TAU=2.0
N=200
LOGDIR=logs/xbackbone_lnv
mkdir -p "$LOGDIR"

# --- wait for phase-1 (video_mme) runs to clear the GPU ---
echo "=== [$(date +%H:%M:%S)] phase2 waiting for phase-1 video_mme runs ===" \
  | tee -a "$LOGDIR/anomaly_driver.log"
while pgrep -f "run_qa_benchmark.*xbackbone_lnv_(uniform|cascade|cascadeASR)" >/dev/null; do
  sleep 30
done
echo "=== [$(date +%H:%M:%S)] phase-1 clear; starting anomaly runs ===" \
  | tee -a "$LOGDIR/anomaly_driver.log"

run () {
  local ds="$1"; local name="$2"; shift 2
  local out="outputs/xbackbone_lnv_${ds}_${name}"
  local log="${LOGDIR}/${ds}_${name}.log"
  echo "=== [$(date +%H:%M:%S)] START $ds/$name -> $out ===" | tee -a "$log"
  if [[ -f "$out/qa_summary.json" ]]; then
    echo "    SKIP (already has qa_summary.json)" | tee -a "$log"
    return 0
  fi
  python3 scripts/run_qa_benchmark.py \
      --dataset "$ds" --limit "$N" --budget "$K" --tau_sec "$TAU" \
      --real --vlm_backbone "$BK" "$@" --output_dir "$out" >>"$log" 2>&1
  local rc=$?
  echo "=== [$(date +%H:%M:%S)] END $ds/$name rc=$rc ===" | tee -a "$log"
  if [[ -f "$out/qa_summary.json" ]]; then
    python3 -c "import json;s=json.load(open('$out/qa_summary.json'));print('   ',{k:s[k] for k in ('n','n_correct','accuracy','vlm_model','vlm_backbone')})" | tee -a "$log"
  fi
  return $rc
}

for DS in xd_violence ucf_crime; do
  run "$DS" uniform     --selector uniform --inject_evidence_text none
  run "$DS" cascade     --cascade --cascade_planner rule_based --cascade_uniform_floor 0.5 --inject_evidence_text none
  run "$DS" cascadeASR  --cascade --cascade_planner rule_based --cascade_uniform_floor 0.5 --inject_evidence_text asr
done

echo "=== [$(date +%H:%M:%S)] ANOMALY PHASE DONE ===" | tee -a "$LOGDIR/anomaly_driver.log"
