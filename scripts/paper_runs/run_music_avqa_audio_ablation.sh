#!/usr/bin/env bash
# Same-model audio ablation on MUSIC-AVQA.
#
# The headline expert contrast swaps the answering model (Qwen2-VL-7B ->
# VideoLLaMA2.1-7B-AV), so it cannot say how much of the gain is audio. Here the
# SAME audio-visual model is run with its soundtrack replaced by digital silence;
# frames, prompt, decoding and the LLM judge are identical to the audio-on run.
#
#   phase "stratified200": 200 questions per question type (about 6 h on one GB10)
#   phase "rest"         : every remaining question        (about 23 h; not needed
#                          for the paper, which reports the stratified phase)
#
# Each phase is judged by the same Qwen2.5-VL-32B judge as the audio-on and
# audio-blind runs. The judge sees only question, gold and answer, so when the
# audio-on / audio-blind runs are judged in $OUT_ROOT their verdicts are reused for
# identical (question, gold, answer) triples and 30 reused verdicts are re-judged as
# a check. Phases are idempotent: re-running resumes where it stopped.
#
# After the first phase:
#   python scripts/audio_ablation_analysis.py --outputs "$OUT_ROOT" --write-arms
#
# Usage:   bash scripts/paper_runs/run_music_avqa_audio_ablation.sh
#          PHASES="stratified200" MODES="silent mismatched" bash scripts/paper_runs/run_music_avqa_audio_ablation.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

VENV_AV="${VENV_AV:-$HOME/.venvs/vllama2av}"   # isolated env: scripts/paper_runs/setup_videollama2av_env.sh
PY_MAIN="${PY_MAIN:-python3}"                  # main env (hosts the judge)
OUT_ROOT="${OUT_ROOT:-outputs}"
SPLIT_DIR="${SPLIT_DIR:-results/audio_ablation}"
PER_TYPE="${PER_TYPE:-200}"
PHASES="${PHASES:-stratified${PER_TYPE} rest}"

if [[ ! -f "$SPLIT_DIR/qids_stratified${PER_TYPE}.txt" ]]; then
  "$PY_MAIN" scripts/make_audio_ablation_subset.py --per-type "$PER_TYPE" --out-dir "$SPLIT_DIR"
fi

MODES="${MODES:-silent}"   # silent, mismatched (each video receives another video's soundtrack), or both

for mode in $MODES; do
for phase in $PHASES; do
  if [[ "$phase" == "rest" ]]; then qids="$SPLIT_DIR/qids_rest.txt"
  else qids="$SPLIT_DIR/qids_${phase}.txt"; fi
  out="$OUT_ROOT/audio_ablation_${mode}_${phase}"

  if [[ ! -f "$out/qa_summary.json" ]]; then
    echo "=== [$(date '+%F %T')] AV $mode: $phase -> $out"
    "$VENV_AV/bin/python" scripts/run_qa_benchmark.py \
        --dataset music_avqa --only_qids "$qids" \
        --selector uniform --budget 8 --tau_sec 1.0 \
        --real --vlm_backbone videollama2_av \
        --vlm_model DAMO-NLP-SG/VideoLLaMA2.1-7B-AV \
        --av_audio_mode "$mode" --item_timeout_sec 600 \
        --output_dir "$out"
  fi
  if [[ ! -f "$out/judge_summary.json" ]]; then
    echo "=== [$(date '+%F %T')] judge: $phase"
    reuse=()
    for arm in wave9av_videollama2av_music_avqa_FULL wave9base_qwen2vl_music_avqa_FULL_noinject; do
      [[ -f "$OUT_ROOT/$arm/judged_rows.jsonl" ]] && reuse+=(--reuse-from "$OUT_ROOT/$arm")
    done
    for other in "$OUT_ROOT"/audio_ablation_*_"$phase"; do
      [[ "$other" != "$out" && -f "$other/judged_rows.jsonl" ]] && reuse+=(--reuse-from "$other")
    done
    "$PY_MAIN" scripts/llm_judge.py --run-dir "$out" --dataset music_avqa \
        ${reuse[@]+"${reuse[@]}"} --audit-reuse 30
  fi
done
done
echo "=== [$(date '+%F %T')] audio ablation done"
