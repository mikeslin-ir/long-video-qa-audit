# Runbook — Run the VideoRouteAudit pipeline on any (video, question)

This runbook shows how to run the **VideoRouteAudit** pipeline (the audited five-stage routing cascade)
on a single video + free-form question and read the result. It uses
[`scripts/run_one.py`](../scripts/run_one.py), which reuses the exact same code
paths as the benchmark (`cascade.CascadeRouter` + `router.backbones` experts), so
a single-item result matches what a full benchmark run would produce.

---

## 1. What happens end to end

```
(video, question, K)
      │
      ▼
  PLAN  ── planner picks skills per layer ────────────────┐
      │     (rule keyword planner, or a small LLM planner) │
      ▼                                                    │
  L1 PROPOSE   uniform_stride (+ stacfp / ears_event on long video)
      ▼
  L2 FILTER    redundancy  (drop near-duplicate frames)
      ▼
  L3 ALIGN     semantic_siglip (+ object/ocr/asr/clap/audio_energy)
      │         → per-frame relevance scores, weighted-fused
      ▼
  SELECT       submodular facility-location → K diverse, relevant frames
      ▼
  (optional) INJECT   ASR/OCR text near the selected frames into the prompt
      ▼
  ROUTE        pick the expert (vlm / avlm) ──────────────┘
      ▼
  ANSWER       expert.infer(frames [+audio] [+injected text]) → answer
```

Stages 1–6 are CPU-bound or a single encoder pass; cost is dominated by the
final VLM/AVLM forward.

---

## 2. Prerequisites

- Run from the release repo root: `cd long-video-qa-audit`.
- The visual expert (`Qwen/Qwen2.5-VL-7B-Instruct`) loads in the **main**
  environment.
- The audio-visual expert (`VideoLLaMA2.1-7B-AV`) needs the **isolated**
  `videollama2` env (pins an older torch/transformers; on Blackwell it must live
  in its own venv). If the package is missing, `--expert avlm` fails with a clear
  error.
- **Single-GPU constraint:** only one 7B model is loaded at a time. Do not launch
  `--expert avlm` while another 7B job is using the GPU. Check first:

  ```bash
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
  ```

---

## 3. Basic usage

Visual question, rule-based planner, Qwen2.5-VL-7B expert (defaults):

```bash
python scripts/run_one.py \
    --video /path/to/clip.mp4 \
    --question "What is the man in the red shirt doing?" \
    --budget 8
```

The script prints the chosen PLAN, the selected-frame timestamps, and the final
answer, e.g.:

```
[run_one] PLAN: proposers=['uniform_stride'] filters=['redundancy']
          aligners=['semantic_siglip'] weights={'semantic_siglip': 1.0} expert=vlm
[run_one] selected 8 frames at [2.1, 9.4, 15.0, ...] s
============================================================
Q: What is the man in the red shirt doing?
A: He is carrying a box.
   expert=vlm  inject=none  K=8  select=3.2s  infer=1.8s
   frames + plan + answer saved to: outputs/run_one/20260616_141233
============================================================
```

---

## 4. Common recipes

### 4a. Narrative long video — inject the Whisper transcript

When the answer is in the spoken narration, inject ASR text near the selected
frames:

```bash
python scripts/run_one.py --video lecture.mp4 \
    --question "What does the narrator say about the volcano?" \
    --inject asr --budget 8
```

### 4b. On-screen text — inject OCR

```bash
python scripts/run_one.py --video news.mp4 \
    --question "What is written on the headline banner?" \
    --inject ocr --budget 8
```

### 4c. Audio-visual question — route to the AV expert

**Not yet validated as a standalone demo:** this entry point currently loads
the visual model before the AV model. On a single GPU, use the isolated
benchmark driver described in docs/USAGE.md instead. Do not assume
that the demo enforces the single-model constraint described above.

For questions that need the soundtrack (which instrument, louder/softer,
counting sounds), route to VideoLLaMA2-AV, which ingests the raw audio:

```bash
python scripts/run_one.py --video concert.mp4 \
    --question "Is the violin on the left louder than the cello?" \
    --expert avlm --budget 8
```

### 4d. Let a small LLM planner decide (with procedural memory)

The LLM planner reads a markdown **procedural memory** of priors and decides the
per-question `inject_text`:

```bash
python scripts/run_one.py --video clip.mp4 \
    --question "Why does the crowd start running?" \
    --planner llm \
    --procedural_memory_file configs/procedural_memory_v4.md
```

> `--planner llm` needs a caption LLM as the planner's generate function. See
> [PROCEDURAL_MEMORY.md](PROCEDURAL_MEMORY.md) for how the memory file changes
> the plan. The single-video entry point currently passes no caption model,
> so this demo configuration exits with an error. Use the benchmark's
> `--cascade_planner llm` path instead.

---

## 5. Key flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--video` | (required) | path to the video file |
| `--question` | (required) | free-form question |
| `--budget` | `8` | K, number of final frames |
| `--tau_sec` | `2.0` | temporal-diversity window for the submodular select |
| `--planner` | `rule` | `rule` (keyword) · `relaxed` (force multi-cue) · `llm` |
| `--procedural_memory_file` | `""` | markdown priors for `--planner llm` |
| `--expert` | routed | force `vlm` or `avlm`; default = the plan's routed expert |
| `--inject` | `plan` | `none` · `asr` · `ocr` · `asr+ocr` · `plan` (use planner's decision) |
| `--dataset` | `custom` | tag that gates keyword routing rules; use `music_avqa`/`video_mme` to enable AV routing |
| `--evidence_window_sec` | `6.0` | total ASR window (s) centred on each selected frame, i.e. ± half of it (OCR reads the frames themselves); the benchmark runner defaults to `8`, the paper's ±4 s |
| `--vlm_model` | `Qwen/Qwen2.5-VL-7B-Instruct` | HF id of the visual expert |
| `--av_model` | `DAMO-NLP-SG/VideoLLaMA2.1-7B-AV` | HF id of the AV expert |
| `--max_new_tokens` | `64` | expert generation length |
| `--output_dir` | `outputs/run_one/<timestamp>` | where to write results |

> The cascade's default planner profile is `stacfp` (STACFP + EARS + uniform
> proposals, redundancy filter). The paper's audited runs used
> `--cascade_profile audit` of `scripts/run_qa_benchmark.py` (EARS + uniform
> proposals, motion-difference filter; see docs/CAVEATS.md). This demo entry
> point uses the default profile. ASR injection needs cached transcripts:
> `scripts/prewarm_asr_cache.py` builds them, and a video without one is injected
> with nothing.

---

## 6. Outputs

In the output directory:

| File | Contents |
|------|----------|
| `selected_frames/frame_*.jpg` | the K chosen frames |
| `cascade_result.json` | full plan + per-stage aligner scores + per-stage timing |
| `answer.json` | `{question, answer, expert, inject, plan, selected_frames, runtimes}` |

Inspect the plan and answer:

```bash
python -c "import json; d=json.load(open('outputs/run_one/<ts>/answer.json')); \
print(d['answer']); print(d['plan']['aligners'], d['plan']['expected_expert'])"
```

---

## 7. How routing decides things (so you can predict the plan)

- **Skill selection (L1/L2/L3)** — the rule planner keys off the question:
  - text/sign/subtitle words → `ocr_text`
  - speech/dialog words → `asr_whisper` (and AV expert on `video_mme`/`music_avqa`)
  - object/color/position words → `object_siglip`
  - always adds `semantic_siglip`; adds `stacfp`/`ears_event` only when
    `duration > 120 s`.
  - The catalogue and per-skill cost/use-case hints live in
    [`../skills/`](../skills/) (single source of truth, loaded by the planner).
- **Expert routing** — `expected_expert` defaults to `vlm`, becomes `avlm` for
  speech/AV questions on AV datasets, or force it with `--expert`.
- **Injection** — `--inject plan` follows the planner's `inject_text`; otherwise
  force a fixed set.

For a full discussion of the procedural-memory planner and its experimental
results, see [PROCEDURAL_MEMORY.md](PROCEDURAL_MEMORY.md).
