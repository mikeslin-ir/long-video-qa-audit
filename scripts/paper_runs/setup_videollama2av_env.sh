#!/usr/bin/env bash
# Set up an ISOLATED environment for VideoLLaMA2.1-7B-AV (audio-visual).
#
# Why isolated: the videollama2 package (audio_visual branch) pins
# transformers==4.40.0 / tokenizers==0.19.1, which conflict with the main
# transformers 5.3.0 used by the main environment. We therefore install it into a
# dedicated venv. On the Blackwell GB10 (sm_100) the upstream-pinned
# torch==2.2.0+cu118 will NOT run, so we layer videollama2 on top of a
# Blackwell-capable torch (the same cu13 build the main env uses) and disable
# flash-attn (use sdpa/eager attention instead).
#
# Usage:
#   bash scripts/paper_runs/setup_videollama2av_env.sh
#   # then:  source ~/.venvs/vllama2av/bin/activate
#   #        bash scripts/paper_runs/run_music_avqa_videollama2av.sh
set -euo pipefail

VENV="${VLLAMA2AV_VENV:-$HOME/.venvs/vllama2av}"
REPO="${VLLAMA2AV_REPO:-$HOME/project/code/VideoLLaMA2}"
PYBIN="${PYBIN:-python3}"

echo "[setup] repo=$REPO  venv=$VENV"
test -d "$REPO/videollama2" || { echo "ERROR: VideoLLaMA2 repo not found at $REPO (and on audio_visual branch)"; exit 1; }

# 1) fresh venv that can see the system CUDA torch (do NOT reinstall torch)
"$PYBIN" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -U pip wheel setuptools

# 2) install the Blackwell-capable torch FIRST so videollama2's pin can't
#    downgrade it. Reuse the exact build the main env already runs.
python -m pip install --index-url https://download.pytorch.org/whl/cu130 \
    torch torchaudio || \
    echo "[setup] WARN: torch cu130 install failed; ensure a Blackwell torch is present"

# 3) install videollama2 WITHOUT its torch / flash-attn pins.
#    Strip the conflicting lines from requirements, then editable-install.
cd "$REPO"
if [ -f requirements.txt ]; then
  grep -viE '^(torch|torchaudio|torchvision|flash-attn|flash_attn)\b' requirements.txt \
      > /tmp/vllama2av_req.txt || true
  python -m pip install -r /tmp/vllama2av_req.txt || \
      echo "[setup] WARN: some requirements failed; continuing (transformers/tokenizers are the critical ones)"
fi
# pin the transformers/tokenizers the model code expects (isolated, so safe)
python -m pip install "transformers==4.40.0" "tokenizers==0.19.1" || true
python -m pip install -e . --no-deps

# 4) audio deps used by the AV preprocessor
python -m pip install soundfile librosa "opencv-python==4.5.5.64" einops || true

echo
echo "[setup] DONE. Next:"
echo "  source $VENV/bin/activate"
echo "  bash scripts/paper_runs/run_music_avqa_videollama2av.sh"
echo
echo "[setup] NOTE: if generation errors with a flash-attn requirement, the"
echo "        model must be loaded with attn_implementation='sdpa'. The"
echo "        VideoLLaMA2AVExpert already avoids flash-attn-only paths."
