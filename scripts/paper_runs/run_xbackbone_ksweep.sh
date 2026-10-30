#!/usr/bin/env bash
# Cross-backbone (#1) K-sweep on LLaVA-NeXT-Video-7B-hf.
# Hypothesis: selection (cascade) beats uniform MORE at tight budgets (low K),
# and the gap shrinks as K grows (both saturate). Test Δ(cascade-uniform) vs K.
# Anomaly domains (frame-sensitive) get the full K grid; Video-MME long is a
# control at the bracketing points only. ASR injection is a no-op on anomaly
# (0 discordant at K=8) so we sweep uniform vs cascade only.
set -euo pipefail
cd "$(dirname "$0")/../.."

BK=llava_next_video
TAU=2.0
N=200
LOGDIR=logs/xbackbone_lnv_ksweep
mkdir -p "$LOGDIR"

run () {
  local ds="$1"; local extra_filter="$2"; local k="$3"; local name="$4"; shift 4
  local out="outputs/ksweep_lnv_${ds}_k${k}_${name}"
  local log="${LOGDIR}/${ds}_k${k}_${name}.log"
  echo "=== [$(date +%H:%M:%S)] START ${ds} K=${k} ${name} -> $out ===" | tee -a "$log"
  if [[ -f "$out/qa_summary.json" ]]; then
    echo "    SKIP (already done)" | tee -a "$log"; return 0
  fi
  # shellcheck disable=SC2086
  local rc
  if python3 scripts/run_qa_benchmark.py \
      --dataset "$ds" $extra_filter --limit "$N" --budget "$k" --tau_sec "$TAU" \
      --real --vlm_backbone "$BK" "$@" --output_dir "$out" >>"$log" 2>&1; then
    rc=0
  else
    rc=$?
  fi
  echo "=== [$(date +%H:%M:%S)] END ${ds} K=${k} ${name} rc=$rc ===" | tee -a "$log"
  [[ -f "$out/qa_summary.json" ]] && \
    python3 -c "import json;s=json.load(open('$out/qa_summary.json'));print('   ',s['n_correct'],'/',s['n'],'=',round(s['accuracy']*100,1),'%')" | tee -a "$log"
  return $rc
}

UNIFORM="--selector uniform --inject_evidence_text none"
CASCADE="--cascade --cascade_planner rule_based --cascade_uniform_floor 0.5 --inject_evidence_text none"

# --- anomaly domains: full grid (K=8 already cached elsewhere; we re-derive K!=8) ---
for DS in xd_violence ucf_crime; do
  for K in 2 4 16 32; do
    run "$DS" "" "$K" uniform $UNIFORM
    run "$DS" "" "$K" cascade $CASCADE
  done
done

# --- Video-MME long: control at bracketing points only (slow) ---
for K in 4 16; do
  run video_mme "--only_task_type long" "$K" uniform $UNIFORM
  run video_mme "--only_task_type long" "$K" cascade $CASCADE
done

echo "=== [$(date +%H:%M:%S)] K-SWEEP DONE ===" | tee -a "$LOGDIR/driver.log"
