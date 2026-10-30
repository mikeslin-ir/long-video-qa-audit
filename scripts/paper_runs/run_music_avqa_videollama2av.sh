#!/usr/bin/env bash
# Run MUSIC-AVQA with the VideoLLaMA2.1-7B-AV audio-visual backbone.
#
# Must be run inside the isolated env created by setup_videollama2av_env.sh:
#   source ~/.venvs/vllama2av/bin/activate
#
# This swaps the whole expert slot to the AV model (never mixed with the
# audio-blind Qwen2-VL in the same ablation column). Audio is loaded from the
# raw video by the model's own preprocessor (processor['video'](path, va=True)).
set -euo pipefail

cd "$(dirname "$0")/../.."

LIMIT="${LIMIT:-1200}"
OUT="${OUT:-outputs/wave9av_videollama2av_music_avqa_uniform}"

python scripts/run_qa_benchmark.py \
    --dataset music_avqa \
    --limit "$LIMIT" \
    --selector uniform \
    --budget 8 \
    --tau_sec 1.0 \
    --real \
    --vlm_backbone videollama2_av \
    --vlm_model DAMO-NLP-SG/VideoLLaMA2.1-7B-AV \
    --item_timeout_sec 600 \
    --output_dir "$OUT"

echo "[av] done -> $OUT"
echo "[av] re-score with the open judge (open-ended, paper-standard prompt):"
echo "     python scripts/llm_judge.py --run-dir $OUT --dataset music_avqa"
