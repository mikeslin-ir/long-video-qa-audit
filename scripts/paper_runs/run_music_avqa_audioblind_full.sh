#!/usr/bin/env bash
# Audio-blind Qwen2-VL-7B FULL MUSIC-AVQA run (n=9129), to pair 1:1 with the
# VideoLLaMA2-AV full run (outputs/wave9av_videollama2av_music_avqa_FULL).
# Mirrors that run EXACTLY except the answering expert:
#   selector=uniform, budget=8, tau_sec=1.0, no evidence injection.
# This fills the audio-blind column of tab:avexpert at full scale so AV-Temporal
# (n=822, was 45) is no longer a small-n cell.
#
# GUARDRAIL: single GPU. We WAIT for the in-flight orchestrator chain to finish
# before claiming the GPU (never run two 7B models concurrently).
#
# Usage:
#   nohup bash scripts/paper_runs/run_music_avqa_audioblind_full.sh [WAIT_PID] \
#       > logs/audioblind_music_avqa_FULL.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
ROOT="$PWD"
LOG() { echo "[ab $(date '+%F %T')] $*"; }

PYENV_PY="$HOME/.pyenv/shims/python"
QWEN2_ID="Qwen/Qwen2-VL-7B-Instruct"
OUT="outputs/wave9base_qwen2vl_music_avqa_FULL_noinject"

# ---- 0) wait for the GPU to free (orchestrator chain) -------------------
WAIT_PID="${1:-$(pgrep -f 'orchestrate_remaining' | head -1)}"
if [ -n "${WAIT_PID:-}" ]; then
    LOG "waiting for orchestrator PID=$WAIT_PID to finish before claiming GPU ..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
    LOG "orchestrator PID=$WAIT_PID exited."
fi
# extra safety: wait until no other run_qa_benchmark is holding the GPU
while pgrep -f 'run_qa_benchmark' >/dev/null 2>&1; do
    LOG "another run_qa_benchmark still active; waiting ..."
    sleep 120
done
LOG "GPU clear; starting audio-blind inference."

# ---- 1) audio-blind inference (uniform K=8, no inject) ------------------
if [ -f "$OUT/qa_summary.json" ]; then
    LOG "STEP 1 skip: $OUT/qa_summary.json already exists."
else
    "$PYENV_PY" scripts/run_qa_benchmark.py \
        --dataset music_avqa --limit 9129 \
        --selector uniform --budget 8 --tau_sec 1.0 \
        --real --item_timeout_sec 600 \
        --vlm_backbone qwen2_vl --vlm_model "${QWEN2_ID}" \
        --inject_evidence_text none \
        --output_dir "$OUT" \
        && LOG "STEP 1 done: $OUT" \
        || { LOG "STEP 1 FAILED."; exit 1; }
fi

# ---- 2) judge (same judge as AV full: Qwen2.5-VL-32B) -------------------
if [ -f "$OUT/judge_summary.json" ]; then
    LOG "STEP 2 skip: $OUT/judge_summary.json already exists."
else
    "$PYENV_PY" scripts/llm_judge.py --run-dir "$OUT" --dataset music_avqa \
        && LOG "STEP 2 done: $OUT/judge_summary.json" \
        || { LOG "STEP 2 FAILED."; exit 1; }
fi

LOG "ALL DONE. Audio-blind full results in $OUT/judge_summary.json"
