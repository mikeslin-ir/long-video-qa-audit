# Agentic Skill Router: High-Level Idea and Usage

## 1. What this project does

This repository implements a training-free, question-conditioned frame selection pipeline for long videos.

Instead of scoring frames with one fixed signal, it dynamically routes to a subset of evidence skills based on the question and dataset context, then fuses the selected skill scores and applies temporal NMS to pick the final frame budget.

## 2. High-level idea

Given:
- video V
- question q
- frame budget K

We compute:
1. Candidate frames sampled from V
2. Planner output: selected skills + fusion weights + prompts
3. Per-skill frame scores
4. Fused score over frames
5. Temporal NMS selection of K frames

Mathematically:

- For each selected skill i, produce score curve s_i(t)
- Normalize each curve
- Fuse by weighted sum:

  F(t) = sum_i w_i * s_i(t) + lambda_uniform * U(t)

- Select K timestamps by temporal NMS with suppression radius tau

## 3. Pipeline diagram

```mermaid
flowchart LR
    A[Video + Question] --> B[Candidate Frame Sampler]
    A --> C[Rule-Based Planner]
    C --> D[Selected Skills + Weights + Prompts]
    B --> E[Skill Scoring]
    D --> E
    E --> F[Fusion]
    F --> G[Temporal NMS]
    G --> H[Selected Frames + Artifacts]
```

## 4. Implemented skills

Current implemented skills:
- motion
- audio
- semantic_siglip
- object_siglip
- asr_whisper
- ocr_text

Compatibility aliases kept:
- object_stub (backed by object_siglip)
- asr_stub (backed by asr_whisper)

Compatibility aliases kept for backward naming:
- ocr_stub (backed by ocr_text)

## 5. Planner behavior summary

The rule-based planner maps question keywords to skill sets:
- violence-related: motion + audio + semantic_siglip
- theft/interaction-related: semantic_siglip + object_siglip + motion
- speech-related: asr_whisper + audio + semantic_siglip
- text-related: ocr_text + semantic_siglip (or ocr_stub if stub mode enabled)

`use_stub_skills: false` in config disables stub-only routing by default.

## 6. Main files

- skill_router.py: orchestrates the full pipeline
- router/planner.py: question-conditioned skill planning
- evidence_skills/: per-skill scoring modules
- fusion.py: normalization + weighted fusion
- temporal_nms.py: top-K with temporal suppression
- run_ablation_skills.py: one-video ablation runner

## 7. Quick usage

Run one question/video:

```bash
python scripts/run_skill_router.py \
  --video_path /path/to/video.mp4 \
  --question "Is someone stealing or robbing in this video?" \
  --dataset ucf_crime \
  --budget 6 \
  --planner rule_based \
  --tau_sec 1.0 \
  --config configs/skill_router.yaml \
  --output_dir outputs/demo_router
```

Run ablations:

```bash
python run_ablation_skills.py \
  --video_path /path/to/video.mp4 \
  --question "Is someone stealing or robbing in this video?" \
  --dataset ucf_crime \
  --budget 6 \
  --tau_sec 1.0 \
  --config configs/skill_router.yaml \
  --output_dir outputs/ablation_demo
```

Run a multi-video subset benchmark (for aggregate runtime/skill usage):

```bash
python scripts/run_subset_benchmark.py \
  --dataset_root "$LONG_VIDEO_QA_DATA_ROOT/XD_violence/test" \
  --dataset xd_violence \
  --max_videos 50 \
  --budget 6 \
  --tau_sec 1.0 \
  --config configs/skill_router.yaml \
  --output_dir outputs/benchmark_xd50
```

## 8. Output artifacts

Each run writes:
- result.json: selected skills, weights, selected frames, runtime
- fused_scores.npy: fused frame-level score curve
- skill_scores.npz: per-skill raw score curves
- selected_frames/: extracted frame images

Ablation runner also writes:
- ablation_summary.json

Subset benchmark runner writes:
- benchmark_rows.json
- benchmark_failures.json
- benchmark_aggregate.json

## 9. Configuration keys (important)

In configs/skill_router.yaml:
- planner
- candidate_stride
- normalization
- lambda_uniform
- use_stub_skills
- semantic_siglip_model
- object_siglip_model
- asr_whisper_model

## 10. Notes on runtime

- SigLIP and Whisper models may download on first run.
- Set HF_TOKEN for more stable/faster Hugging Face access.
- CPU-only execution works but is slower.

## 11. Recommended next extensions

1. Add planner confidence and uncertainty-aware fallback policy.
2. Add OCR language packs and multilingual routing.
3. Extend benchmark summary with task-specific quality metrics.
