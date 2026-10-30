#!/usr/bin/env bash
# Extra experiments added in response to review, in five independent lanes so that a
# GPU with enough memory for several 7B models keeps all of them busy:
#
#   LANE=B  ASR injection on XD-Violence / UCF-Crime with transcripts that exist.
#           The original n=200 "cascade + ASR" arms there injected no text at all
#           (no Whisper transcript had been cached for those clips), so they were
#           identical to the plain cascade arm. Transcribe the 400 evaluated clips,
#           then re-run the arm with the configuration of the original runs.
#   LANE=D  The same two arms with the injected block placed BEFORE the question
#           instead of after the options (--evidence_position before). The audited
#           runs append the block after "Answer with the single letter ...", so this
#           checks that what ASR does on the anomaly clips is not an artefact of where
#           the text sits in the prompt.
#   LANE=E  The same placement check on the full Video-MME long bucket (n=900, K=8):
#           cascade + ASR and uniform + ASR with the block before the question.
#   LANE=C  EgoSchema, n=500, K=8, Qwen2.5-VL-7B: uniform / cascade / AKS /
#           KeyScore / VideoTree. A second benchmark, same harness, same grader.
#   LANE=A  (waits for the audio ablation to release the GPU unless WAIT_FOR is
#           set to something that is not running), in priority order:
#           1. Qwen2-VL-7B, Video-MME long, n=900, K=8, cascade + ASR with the same
#              planner profile as its cascade and uniform arms. The original +ASR
#              arm ran after PySceneDetect had been installed (198 proposals per
#              video instead of 132), so it compared a different candidate pool.
#           2. Video-MME long, n=900, K=1: uniform (first frame), uniform_center
#              (middle frame) and cascade. The only budget at which the n=200 sweep
#              found a significant selection effect; the endpoint-inclusive uniform
#              baseline places its single frame at the very start of the video.
#           3. Video-MME long, n=900, K=8, uniform_center: a stronger uniform
#              baseline (segment midpoints instead of endpoint-inclusive spacing).
#           4. Video-MME long, n=900, K=8, cascade and cascade + ASR with the
#              current default planner profile (STACFP + redundancy filter), to
#              check that the audit's conclusions do not depend on the candidate
#              pool of the published runs.
#
# Every step resumes from its rows.jsonl, and a failing step is reported
# without stopping the lane. Point LONG_VIDEO_QA_CASCADE_CACHE at an existing
# cache to avoid recomputing proposals and transcripts. SKIP and ONLY are
# space-separated step labels (e.g. ONLY="qwen2vl_cascade_asr vmme_k1_uniform").
#
#   LANE=F  Second wave, run as four shards each (run_sharded.sh): (1) segment-midpoint
#           uniform at K=8; (2) the evidence-placement control, uniform + ASR on
#           Qwen2.5-VL with the block after the instruction (audited), before the
#           question, between question and options, and between options and
#           instruction; (3) uniform + ASR before/after on Qwen2-VL-7B; (4) K=1 uniform,
#           middle-frame uniform and cascade (all Video-MME long, n=900); (5) the pool-matched
#           Qwen2-VL cascade + ASR (audit profile) and the default STACFP profile with and
#           without ASR (n=900); (6) the AKS / KeyScore / VideoTree ports on EgoSchema (n=500).
#           By-video sharding (SHARD_KEY_SEP=-) keeps a video's questions in one shard.
#
# Usage: LANE=A|B|C|D|E|F bash scripts/paper_runs/run_review_experiments.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

LANE="${LANE:?set LANE=A, B, C, D, E or F}"
PY="${PY:-python3}"
VLM="${VLM:-Qwen/Qwen2.5-VL-7B-Instruct}"
VLM_QWEN2="${VLM_QWEN2:-Qwen/Qwen2-VL-7B-Instruct}"
OUT_ROOT="${OUT_ROOT:-outputs}"
WAIT_FOR="${WAIT_FOR:-run_music_avqa_audio_ablation.sh}"
SKIP="${SKIP:-}"
ONLY="${ONLY:-}"

wanted () {  # wanted <label>
  local label="$1"
  [[ " $SKIP " == *" $label "* ]] && return 1
  [[ -z "$ONLY" || " $ONLY " == *" $label "* ]]
}

step () {  # [STEP_MODEL=<hf id>] step <label> <output dir> <run_qa_benchmark.py args...>
  local label="$1" out="$2" model="${STEP_MODEL:-$VLM}"; shift 2
  if ! wanted "$label"; then echo "[not selected] $label"; return 0; fi
  if [[ -f "$out/qa_summary.json" ]]; then echo "[skip] $label (already complete)"; return 0; fi
  echo "=== [$(date '+%F %T')] START $label"
  if "$PY" scripts/run_qa_benchmark.py "$@" --real --vlm_backbone qwen2_vl \
        --vlm_model "$model" --item_timeout_sec 600 --resume --output_dir "$out"; then
    echo "=== [$(date '+%F %T')] DONE  $label: $("$PY" -c "import json;d=json.load(open('$out/qa_summary.json'));print(d['n_correct'],'/',d['n'],'=',round(100*d['accuracy'],1),'%', d.get('injection') or '')")"
  else
    echo "=== [$(date '+%F %T')] FAILED $label" >&2
  fi
}

AUDIT=(--selector ours --cascade --cascade_planner rule_based --cascade_profile audit --cascade_uniform_floor 0.5)
CURRENT=(--selector ours --cascade --cascade_planner rule_based --cascade_profile stacfp --cascade_uniform_floor 0.5)

lane_b () {
  for ds in xd_violence ucf_crime; do
    wanted "anomaly_${ds}_asr" || { echo "[not selected] anomaly_${ds}_asr"; continue; }
    echo "=== [$(date '+%F %T')] transcribing $ds"
    "$PY" scripts/prewarm_asr_cache.py --dataset "$ds" --limit 200 \
      || echo "=== prewarm FAILED for $ds" >&2
    step "anomaly_${ds}_asr" "$OUT_ROOT/trackB_${ds}_cascade_k8_f0p5_injectASR_transcribed" \
      --dataset "$ds" --limit 200 --budget 8 --tau_sec 2.0 "${AUDIT[@]}" --inject_evidence_text asr
  done
}

lane_d () {  # transcripts must already exist (lane B)
  for ds in xd_violence ucf_crime; do
    step "anomaly_${ds}_asr_before" "$OUT_ROOT/trackB_${ds}_cascade_k8_f0p5_injectASR_transcribed_before" \
      --dataset "$ds" --limit 200 --budget 8 --tau_sec 2.0 "${AUDIT[@]}" --inject_evidence_text asr \
      --evidence_position before
  done
}

lane_e () {  # Video-MME transcripts are cached by the published runs
  local k8=(--dataset video_mme --only_task_type long --limit 900 --tau_sec 2.0 --budget 8)
  step vmme_cascade_asr_before "$OUT_ROOT/trackB_long_n900_k8_cascade_f0p5_injectASR_before" \
    "${k8[@]}" "${AUDIT[@]}" --inject_evidence_text asr --evidence_position before
  step vmme_uniform_asr_before "$OUT_ROOT/trackB_long_n900_k8_uniform_injectASR_before" \
    "${k8[@]}" --selector uniform --inject_evidence_text asr --evidence_position before
}

lane_c () {
  local ego=(--dataset egoschema --limit 500 --budget 8 --tau_sec 2.0)
  step egoschema_uniform "$OUT_ROOT/egoschema_n500_k8_uniform" "${ego[@]}" --selector uniform
  step egoschema_cascade "$OUT_ROOT/egoschema_n500_k8_cascade" "${ego[@]}" "${AUDIT[@]}"
  for sel in aks keyscore videotree; do
    step "egoschema_$sel" "$OUT_ROOT/egoschema_n500_k8_$sel" "${ego[@]}" --selector "$sel"
  done
}

lane_a () {
  while pgrep -f "$WAIT_FOR" >/dev/null 2>&1; do sleep 60; done

  local k8=(--dataset video_mme --only_task_type long --limit 900 --tau_sec 2.0 --budget 8)
  "$PY" scripts/prewarm_asr_cache.py --dataset video_mme --task_type long \
    || echo "=== prewarm FAILED for video_mme" >&2
  STEP_MODEL="$VLM_QWEN2" step qwen2vl_cascade_asr \
    "$OUT_ROOT/trackB_long_n900_k8_qwen2vl_cascade_f0p5_injectASR_audit" \
    "${k8[@]}" "${AUDIT[@]}" --inject_evidence_text asr

  local k1=(--dataset video_mme --only_task_type long --limit 900 --tau_sec 2.0 --budget 1)
  step vmme_k1_uniform "$OUT_ROOT/trackB_long_n900_k1_uniform" "${k1[@]}" --selector uniform
  step vmme_k1_uniform_center "$OUT_ROOT/trackB_long_n900_k1_uniform_center" "${k1[@]}" --selector uniform_center
  step vmme_k1_cascade "$OUT_ROOT/trackB_long_n900_k1_cascade_f0p5" "${k1[@]}" "${AUDIT[@]}"

  step vmme_uniform_center "$OUT_ROOT/trackB_long_n900_k8_uniform_center" "${k8[@]}" --selector uniform_center
  step vmme_stacfp_cascade "$OUT_ROOT/trackB_long_n900_k8_stacfp_cascade_f0p5" "${k8[@]}" "${CURRENT[@]}"
  step vmme_stacfp_cascade_asr "$OUT_ROOT/trackB_long_n900_k8_stacfp_cascade_f0p5_injectASR" \
    "${k8[@]}" "${CURRENT[@]}" --inject_evidence_text asr
}

lane_f () {
  local q="$OUT_ROOT/vmme_long_qids_n900.txt"
  mkdir -p "$OUT_ROOT"
  "$PY" -c "
from router.qa_datasets import load_dataset
items = [it for it in load_dataset('video_mme') if it.task_type == 'long'][:900]
print('\n'.join(str(it.qid) for it in items))" > "$q"
  local q25=(--vlm_backbone qwen2_vl --vlm_model "$VLM") q2=(--vlm_backbone qwen2_vl --vlm_model "$VLM_QWEN2")
  local k8=(--dataset video_mme --only_task_type long --limit 900 --tau_sec 2.0 --budget 8)
  local k1=(--dataset video_mme --only_task_type long --limit 900 --tau_sec 2.0 --budget 1)
  shard () {  # shard <label> <output dir> <args...>
    local label="$1" out="$2"; shift 2
    if ! wanted "$label"; then echo "[not selected] $label"; return 0; fi
    # each shard writes its own qa_summary.json, so completeness is a DONE marker, not that file
    if [[ -f "$out/DONE" ]]; then echo "[skip] $label (already complete)"; return 0; fi
    echo "=== [$(date '+%F %T')] START $label"
    if SHARDS="${SHARDS:-4}" bash scripts/paper_runs/run_sharded.sh "$out" "$q" "$@"; then
      touch "$out/DONE"
    else
      echo "=== [$(date '+%F %T')] FAILED $label" >&2
    fi
  }
  shard vmme_uniform_center "$OUT_ROOT/trackB_long_n900_k8_uniform_center" "${k8[@]}" --selector uniform_center "${q25[@]}"
  local pos suffix
  for pos in after question instruction before; do
    suffix="_$pos"; [[ "$pos" == after ]] && suffix=""
    shard "placement_$pos" "$OUT_ROOT/trackB_long_n900_k8_uniform_injectASR$suffix" "${k8[@]}" \
      --selector uniform --inject_evidence_text asr --evidence_position "$pos" "${q25[@]}"
  done
  for pos in after before; do
    suffix=""; [[ "$pos" == before ]] && suffix="_before"
    shard "qwen2vl_placement_$pos" "$OUT_ROOT/trackB_long_n900_k8_qwen2vl_uniform_injectASR$suffix" "${k8[@]}" \
      --selector uniform --inject_evidence_text asr --evidence_position "$pos" "${q2[@]}"
  done
  shard vmme_k1_uniform "$OUT_ROOT/trackB_long_n900_k1_uniform" "${k1[@]}" --selector uniform "${q25[@]}"
  shard vmme_k1_uniform_center "$OUT_ROOT/trackB_long_n900_k1_uniform_center" "${k1[@]}" --selector uniform_center "${q25[@]}"
  shard vmme_k1_cascade "$OUT_ROOT/trackB_long_n900_k1_cascade_f0p5" "${k1[@]}" "${AUDIT[@]}" "${q25[@]}"

  export SHARD_KEY_SEP=-
  shard qwen2vl_cascade_asr_audit "$OUT_ROOT/trackB_long_n900_k8_qwen2vl_cascade_f0p5_injectASR_audit" \
    "${k8[@]}" "${AUDIT[@]}" --inject_evidence_text asr "${q2[@]}"
  shard vmme_stacfp_cascade "$OUT_ROOT/trackB_long_n900_k8_stacfp_cascade_f0p5" "${k8[@]}" "${CURRENT[@]}" "${q25[@]}"
  shard vmme_stacfp_cascade_asr "$OUT_ROOT/trackB_long_n900_k8_stacfp_cascade_f0p5_injectASR" \
    "${k8[@]}" "${CURRENT[@]}" --inject_evidence_text asr "${q25[@]}"

  local eq="$OUT_ROOT/egoschema_qids_n500.txt"
  "$PY" -c "
from router.qa_datasets import load_dataset
print('\n'.join(str(it.qid) for it in load_dataset('egoschema', limit=500)))" > "$eq"
  local ego=(--dataset egoschema --limit 500 --budget 8 --tau_sec 2.0) sel
  unset SHARD_KEY_SEP
  q="$eq"
  for sel in aks keyscore videotree; do
    shard "egoschema_$sel" "$OUT_ROOT/egoschema_n500_k8_$sel" "${ego[@]}" --selector "$sel" "${q25[@]}"
  done
}

case "$LANE" in
  A) lane_a ;;
  B) lane_b ;;
  C) lane_c ;;
  D) lane_d ;;
  E) lane_e ;;
  F) lane_f ;;
  *) echo "LANE must be A, B, C, D, E or F" >&2; exit 2 ;;
esac
echo "=== [$(date '+%F %T')] lane $LANE done"
