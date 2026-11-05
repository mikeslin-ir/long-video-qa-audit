# Setup

Environment, optional environments, model weights, planner API keys and datasets.

[← Back to README](../README.md)


Reference environment: **Python 3.10** (3.10.14 produced every run added during
release preparation), single **NVIDIA GB10** GPU, CUDA build matching the pinned
`torch`. Dependency versions were frozen 2026-07-06. The CPU test suite and the
offline statistics replay also pass on Python 3.12 (CI runs both).

```bash
git clone https://github.com/mikeslin-ir/long-video-qa-audit.git
cd long-video-qa-audit

python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

**Adapt the `torch` pins if your CUDA differs from the reference.** The pinned
`torch==2.12.0` / `torchvision==0.27.0` match our driver; if yours differs,
adjust those two pins in a local copy of the requirements using
<https://pytorch.org>. Installing different wheels first does not override
the pins in the requirements file. The full GPU environment is a reference
record, not a clean-install compatibility guarantee for other machines.

```bash
pip install -r requirements.txt
```

Verify the install:

```bash
python -m pytest tests/ -q
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

For a **model-free CPU check**, use a separate virtual environment:

```bash
python -m pip install -r requirements-test.txt
python -m pytest tests/ -q
```

This checks the selection stack, synthetic-video cascade, data-root wiring,
and citation consistency without downloading models or touching a GPU.

### Optional environments

Two components deliberately live outside the main environment because they pin
incompatible `torch`/`transformers` versions:

```bash
bash scripts/paper_runs/setup_videollama2av_env.sh   # audio-visual expert (F3 expert row)
bash scripts/paper_runs/setup_llava_video_env.sh     # LLaVA-Video-7B cross-backbone check
```

The reference runs used one 7B model on the GPU at a time. On the unified-memory
GB10 (119 GB) a 7B VLM run takes about 22 GB and the 32B judge about 65 GB, so
at most four 7B jobs fit together (a fifth, or a judge next to more than one
VLM job, is killed by the kernel's out-of-memory handler while loading
weights), and concurrent jobs share throughput: Video-MME items took 13 s alone
and 30-40 s with four jobs running. The run drivers resume from `rows.jsonl`, so
a killed job can simply be restarted. Check before launching:

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

`scripts/llm_judge.py` loads its 65 GB checkpoint one shard at a time. With
`transformers` 5.3 the stock `from_pretrained` keeps every shard it has read
mapped until the end, which on a unified-memory GPU counts as a second copy of
the weights: a judge started alone on the 119 GB machine was killed at 63%
of the load. `JUDGE_STANDARD_LOAD=1` restores `from_pretrained` for machines
where that is not a problem. The judge is text-only (question, gold answer,
predicted answer) and decodes greedily, so `--reuse-from RUN_DIR` copies the
verdict of any already-judged row with the same question, gold and answer
instead of calling the model (a silenced-audio run repeats most of the audio-on
run's answers), and `--audit-reuse N` re-judges N reused rows to check that
the model reproduces them. A partially finished judge run resumes from
`judged_rows.partial.jsonl`.

### Model weights

All pulled from Hugging Face on first use:

| Role | Model |
|---|---|
| Visual expert (default) | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Cross-backbone checks | `LLaVA-Video-7B-Qwen2`, `LLaVA-NeXT-Video-7B` |
| Audio-visual expert | `VideoLLaMA2.1-7B-AV` (isolated env) |
| LLM planner | `Qwen2.5-3B-Instruct` |
| Alignment encoder | SigLIP base/16 |
| Audio encoder | CLAP-2B |
| ASR | `whisper-tiny.en` (auto-download) |
| OCR | EasyOCR (auto-download) |

### Optional: LLM planner API keys

The LLM planner can call a hosted model instead of a local one. Keys are read
from the environment and are never stored in the repo:

```bash
export ANTHROPIC_API_KEY=...      # or
export OPENAI_API_KEY=...
export LLM_PLANNER_MODEL=...      # optional override
```

### Datasets

Datasets are **not** redistributed here. Download them from their official
sources and point the loaders at your local copies:

| Dataset | Use |
|---|---|
| Video-MME (long bucket) | Main F1/F2 evaluation, `n=900` |
| EgoSchema | Cross-benchmark check, `n=500` |
| UCF-Crime, XD-Violence | Anomaly-domain binary event-QA, `n=200` each |
| MUSIC-AVQA | Expert-routing axis |
| MLVU, MSVD-QA | Supplementary |

Set the dataset root **before starting Python**:

```bash
export LONG_VIDEO_QA_DATA_ROOT=/absolute/path/to/datasets
```

The QA loaders, full-split runner, and default CLAP feature lookup share this
setting. Without it, the default is `~/project/datasets`, preserving the
reference-machine layout. An empty value is an error.

Keep the expected subdirectories (`videomme/`, `EgoSchema/`, `UCF-Crime/`,
`XD_violence/test/`, `MUSIC-AVQA/`, etc.). CLAP features belong under
`agentic_routing/features/<dataset>/clap/`.
