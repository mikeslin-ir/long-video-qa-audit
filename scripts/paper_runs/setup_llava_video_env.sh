#!/usr/bin/env bash
# Isolated env for LLaVA-Video-7B-Qwen2 (lmms-lab `llava` package).
#
# WHY a separate venv: the lmms-lab LLaVA-NeXT pyproject pins torch==2.1.2 and
# an old transformers git commit. On the GB10 Blackwell GPU torch 2.1.2 has no
# sm_100 kernels, so we layer the `llava` package on top of the SAME Blackwell
# torch cu130 build we use elsewhere and install it WITHOUT its torch / flash-
# attn pins (mirrors scripts/paper_runs/setup_videollama2av_env.sh).
#
# IMPORTANT: this MUST be its own venv. Do NOT install into ~/.venvs/vllama2av
# while the full MUSIC-AVQA AV run is in flight -- it would upgrade torch /
# transformers and break that job.
set -u

VENV="${VENV:-$HOME/.venvs/llava}"
LLAVA_SRC="${LLAVA_SRC:-$HOME/LLaVA-NeXT}"   # git clone https://github.com/LLaVA-VL/LLaVA-NeXT
AV_VENV="$HOME/.venvs/vllama2av"   # source of the working decord PyAV shim

echo "[setup] venv = $VENV"
echo "[setup] llava src = $LLAVA_SRC"

# 1) fresh venv that can see the system CUDA torch
python -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -U pip wheel setuptools

# 2) Blackwell-capable torch FIRST so llava's torch==2.1.2 pin can't clobber it.
python -m pip install --index-url https://download.pytorch.org/whl/cu130 \
    torch torchvision torchaudio || \
    echo "[setup] WARN: torch cu130 install failed; ensure a Blackwell torch is present"

# 3) install the llava package WITHOUT its torch / flash-attn / transformers pins
python -m pip install -e "$LLAVA_SRC" --no-deps

# 4) pin a Qwen2-compatible transformers (same combo that works for the AV model)
python -m pip install "transformers==4.42.3" "tokenizers==0.19.1" || true

# 5) the runtime deps llava actually imports (minus torch/flash-attn)
python -m pip install \
    accelerate sentencepiece protobuf einops einops-exts timm \
    "opencv-python==4.5.5.64" pillow numpy scipy ftfy regex \
    open_clip_torch || true

# 6) decord: no aarch64 wheel -> reuse the working PyAV shim from the AV venv
PYV=$(python -c "import sys;print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
DST="$VENV/lib/$PYV/site-packages/decord"
SRC="$AV_VENV/lib/$PYV/site-packages/decord"
if [ -d "$SRC" ] && [ ! -d "$DST" ]; then
    cp -r "$SRC" "$DST"
    echo "[setup] copied decord PyAV shim from AV venv -> $DST"
else
    python -m pip install av || true
    echo "[setup] NOTE: ensure a decord shim or real decord is importable."
fi

echo
echo "[setup] DONE. Smoke test:"
echo "  source $VENV/bin/activate"
echo "  python -c \"from llava.model.builder import load_pretrained_model; print('llava import ok')\""
