# System Overview

The harness behind the audit is a training-free, modular pipeline. Each routing
decision the paper studies is an ablatable stage, so one decision can be varied
while everything else is held fixed. The pipeline figure is
[`figures/figure2_pipeline.png`](../figures/figure2_pipeline.png); the cascade
internals are in [`cascade/README.md`](../cascade/README.md).

## Three routing axes

| Axis | Decision | Where it lives |
|---|---|---|
| Visual | Which `K` frames the VLM sees | `cascade/` (propose → filter → align → select) and the ported selectors in `baselines/` |
| Textual | Which non-visual evidence (Whisper transcript, on-frame OCR) is appended to the prompt | `cascade/evidence_text.py`, `--inject_evidence_text` |
| Model | Which model answers | `router/backbones.py`, `router/experts.py`, `skills/` expert policy |

## Cascade stages

1. **Propose** (`cascade/proposers/`): cheap candidate moments.
   `uniform_stride` is always available; `ears_event` proposes motion/audio
   peaks (external EARS code, first 4,096 frames of a video); `stacfp` clusters
   frame histograms; `shot_boundary` needs PySceneDetect;
   `keyframe_cache` reads an offline keyframe index.
2. **Filter** (`cascade/filters/`): `motion_diff` keeps salient candidates,
   `redundancy` drops near-duplicates.
3. **Align** (`cascade/aligners/`, `evidence_skills/`): per-candidate relevance
   to the question from SigLIP (semantic/object), OCR, Whisper ASR, CLAP or
   audio energy, fused with planner weights.
4. **Select**: lazy-greedy facility location (`submodular.py`) with a uniform
   anchor floor (`--cascade_uniform_floor`; the audited runs reserve
   `round(K/2)` anchors).
5. **Inject / answer**: the selected frames go to the VLM; optional ASR/OCR
   text is *appended after the question and options* with fixed character
   budgets (see `cascade/evidence_text.py`).

The planner chooses which proposers, filters and aligners to run.
`RuleBasedCascadePlanner` uses keyword rules and two profiles (`audit`, the
configuration the paper's runs executed, and `stacfp`, the later default);
`LLMCascadePlanner` asks a small instruction-tuned LLM, optionally reading the
markdown memory in `configs/procedural_memory_v1.md`
([`PROCEDURAL_MEMORY.md`](PROCEDURAL_MEMORY.md)).

## Single-stage router

`skill_router.py` is the earlier single-stage design (uniform candidates →
per-skill scores → `fusion.py` → `temporal_nms.py` or `submodular.py`) kept for
the single-video demo and the skill-ablation scripts. The audit's selection
results use the cascade and the ported baselines.

## Experts

- `vlm`: vision-language model (Qwen2.5-VL-7B for the Video-MME experiments,
  Qwen2-VL-7B for the MUSIC-AVQA base).
- `avlm`: audio-visual model (VideoLLaMA2.1-7B-AV); `--av_audio_mode silent`
  replaces its soundtrack with digital silence for the same-model ablation.
- `alm`, `caplm`: audio-language and text-only experts used by the dispatcher.

The paper's expert contrast is a *model swap*; the silent-audio run bounds how
much of it is audio.

## Run artifacts and caches

Each run directory under `outputs/` holds `rows.jsonl` (per-item prediction and
exact-match correctness), `failures.jsonl`, `qa_summary.json` (accuracy, timing,
arguments, and for injection runs how many items actually received text) and,
for judged open-ended runs, `judged_rows.jsonl`. Heavy intermediate products are
cached and shared across runs:

- `results/cascade_cache/` (`LONG_VIDEO_QA_CASCADE_CACHE`): proposer outputs,
  Whisper transcripts and OCR. An ASR-injection run silently injects nothing for
  a video without a cached transcript, so build them first with
  `scripts/prewarm_asr_cache.py`.
- `results/skill_router_cache/`: SigLIP candidate-pool embeddings used by the
  ported selectors.

`outputs/` and the caches are gitignored. The per-item outcomes needed to
recompute every statistic in the paper are shipped separately in
[`results/predictions/`](../results/predictions).
