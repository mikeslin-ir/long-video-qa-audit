#!/usr/bin/env bash
# The cascade, injection and planner arms of the paper's main tables, with the
# flags the published runs used (the paper-run drivers for the selector ports,
# the budget sweeps and MUSIC-AVQA are the other scripts in this directory).
#
#   vmme      Video-MME long, n=900, K=8, Qwen2.5-VL-7B: uniform, cascade, cascade+ASR,
#             cascade+OCR and the ProcMem planner (v3 = inject-only mode).
#   qwen2vl   the same selection and ASR contrast on Qwen2-VL-7B.
#   anomaly   XD-Violence and UCF-Crime, n=200, K=8: uniform, cascade, cascade+ASR,
#             cascade+OCR and the free-plan LLM planner (v1).
#
# Every cascade arm passes --cascade_profile audit, the configuration the published
# runs executed (EARS + uniform frames, motion-difference filter). ASR injection
# needs cached Whisper transcripts, so they are built first; without them an
# --inject_evidence_text asr arm silently injects nothing.
#
# Usage: bash scripts/paper_runs/run_published_cascade_arms.sh [vmme|qwen2vl|anomaly|all]
set -uo pipefail
cd "$(dirname "$0")/../.."

WHICH="${1:-all}"
PY="${PY:-python3}"
VLM="${VLM:-Qwen/Qwen2.5-VL-7B-Instruct}"
VLM_QWEN2="${VLM_QWEN2:-Qwen/Qwen2-VL-7B-Instruct}"
OUT_ROOT="${OUT_ROOT:-outputs}"
MEMORY="configs/procedural_memory_v1.md"

step () {  # [STEP_MODEL=<hf id>] step <output dir> <run_qa_benchmark.py args...>
  local out="$1" model="${STEP_MODEL:-$VLM}"; shift
  if [[ -f "$out/qa_summary.json" ]]; then echo "[skip] $out"; return 0; fi
  echo "=== [$(date '+%F %T')] START $out"
  "$PY" scripts/run_qa_benchmark.py "$@" --real --vlm_backbone qwen2_vl --vlm_model "$model" \
      --item_timeout_sec 600 --resume --output_dir "$out" || echo "=== FAILED $out" >&2
}

CASCADE=(--selector ours --cascade --cascade_planner rule_based --cascade_profile audit --cascade_uniform_floor 0.5)
PLANNER_V3=(--selector ours --cascade --cascade_planner llm --cascade_profile audit --cascade_uniform_floor 0.5
            --inject_evidence_text plan --procedural_memory_file "$MEMORY" --planner_inject_only)
PLANNER_V1=(--selector ours --cascade --cascade_planner llm --cascade_profile audit --cascade_uniform_floor 0.5
            --inject_evidence_text plan --procedural_memory_file "$MEMORY")

vmme () {
  local common=(--dataset video_mme --only_task_type long --limit 900 --budget 8 --tau_sec 2.0)
  "$PY" scripts/prewarm_asr_cache.py --dataset video_mme --task_type long || echo "prewarm failed" >&2
  step "$OUT_ROOT/trackB_long_n900_k8_uniform"                 "${common[@]}" --selector uniform
  step "$OUT_ROOT/trackB_long_n900_k8_cascade_f0p5"            "${common[@]}" "${CASCADE[@]}"
  step "$OUT_ROOT/trackB_long_n900_k8_cascade_f0p5_injectASR"  "${common[@]}" "${CASCADE[@]}" --inject_evidence_text asr
  step "$OUT_ROOT/trackB_long_n900_k8_cascade_f0p5_injectOCR"  "${common[@]}" "${CASCADE[@]}" --inject_evidence_text ocr
  step "$OUT_ROOT/trackB_long_n900_k8_llmPlanner_proceduralV3_injectOnly" "${common[@]}" "${PLANNER_V3[@]}"
}

qwen2vl () {
  local common=(--dataset video_mme --only_task_type long --limit 900 --budget 8 --tau_sec 2.0)
  export STEP_MODEL="$VLM_QWEN2"
  step "$OUT_ROOT/trackB_long_n900_k8_qwen2vl_uniform"                "${common[@]}" --selector uniform
  step "$OUT_ROOT/trackB_long_n900_k8_qwen2vl_cascade_f0p5"           "${common[@]}" "${CASCADE[@]}"
  step "$OUT_ROOT/trackB_long_n900_k8_qwen2vl_cascade_f0p5_injectASR_audit" \
       "${common[@]}" "${CASCADE[@]}" --inject_evidence_text asr
  unset STEP_MODEL
}

anomaly () {
  for ds in xd_violence ucf_crime; do
    local common=(--dataset "$ds" --limit 200 --budget 8 --tau_sec 2.0)
    "$PY" scripts/prewarm_asr_cache.py --dataset "$ds" --limit 200 || echo "prewarm failed" >&2
    step "$OUT_ROOT/trackB_${ds}_uniform_k8"                                 "${common[@]}" --selector uniform
    step "$OUT_ROOT/trackB_${ds}_cascade_k8_f0p5"                            "${common[@]}" "${CASCADE[@]}"
    step "$OUT_ROOT/trackB_${ds}_cascade_k8_f0p5_injectASR_transcribed"      "${common[@]}" "${CASCADE[@]}" --inject_evidence_text asr
    step "$OUT_ROOT/trackB_${ds}_cascade_k8_f0p5_injectOCR"                  "${common[@]}" "${CASCADE[@]}" --inject_evidence_text ocr
    step "$OUT_ROOT/trackB_${ds}_cascade_k8_llmPlanner_planInject_proceduralV1" "${common[@]}" "${PLANNER_V1[@]}"
  done
}

case "$WHICH" in
  vmme) vmme ;;
  qwen2vl) qwen2vl ;;
  anomaly) anomaly ;;
  all) vmme; qwen2vl; anomaly ;;
  *) echo "usage: $0 [vmme|qwen2vl|anomaly|all]" >&2; exit 2 ;;
esac
echo "=== [$(date '+%F %T')] done: $WHICH"
