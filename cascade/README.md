# Agentic Cascade (cascade/)

A 3-level, declarative, agent-planned pipeline for video frame selection.
Designed to make heavy aligners (SigLIP, OCR, Whisper) skip the long tail
of irrelevant frames in long videos.

```
        L1: Frame Proposers          → ProposalSet (anchors)
            uniform_stride            (backbone, always on)
            stacfp        DEFAULT — prior work: spatio-temporal adaptive
                                       clustering. Estimates cut density for
                                       free from its histograms and self-adapts:
                                       centroid clustering on long takes,
                                       snap-to-cuts on edited video (subsumes
                                       shot_boundary). Logs stacfp_cut_density
                                       + stacfp_mode to cascade_result.json.
            shot_boundary ablation only — standalone PySceneDetect; STACFP's
                                       snap_cuts regime covers the same case.
                                       Re-enable via planner use_shot_boundary.
            ears_event     (motion+audio peak)
            keyframe_cache            (offline, prefer_cache)

        L2: Keyframe Filters         → narrowed ProposalSet
            redundancy     DEFAULT — drops near-duplicate frames,
                           keeps static-but-distinct ones (histogram,
                           content-adaptive count)
            motion_diff    ablation only — selects high-motion / drops static

        L3: Evidence Aligners        → per-frame relevance scores
            semantic_siglip           (default backbone)
            object_siglip
            ocr_text
            asr_whisper
            clap_alm                  (non-speech audio / music)
            audio_energy              (loud events)
            semantic_siglip_cached    (reuse offline embeds)

        L4: submodular_select        → K final frames
        L5: evidence text injection  → inject_text=[asr|ocr] into VLM prompt
```

A rendered figure of this pipeline is
[`figures/figure2_pipeline.png`](../figures/figure2_pipeline.png).

## Planner profiles

`RuleBasedCascadePlanner(profile=...)` (CLI: `--cascade_profile`) selects the
proposers and filter:

| Profile | Long-video proposers | Filter | Pool |
|---|---|---|---|
| `stacfp` (default) | STACFP, EARS, uniform | redundancy | 3 × 66 = 197–198 |
| `audit` | EARS, uniform | motion_diff (long videos) | 2 × 66 = 132 |

`audit` is what the paper's Video-MME long, XD-Violence and UCF-Crime cascade
runs actually executed. Those plans listed a PySceneDetect `shot_boundary`
proposer too, but PySceneDetect was not installed, so the router skipped it
silently; the profile leaves it out so the runs reproduce whether or not it is
installed. `ears_event` needs the [EARS](https://github.com/mikeslin-ir/EARS)
code (`$EARS_PATH` or a sibling `../EARS` checkout) and covers only the first
4,096 frames of a video. The router now warns whenever a planned tool fails. The
paper's runs pass `--cascade_uniform_floor 0.5`, which reserves `round(K/2)`
frames for evenly spaced anchors that ignore the alignment scores (none at
`K=1`); the library default is `0.0`, a pure cascade.

## What an agentic CascadePlan looks like

```json
{
  "proposers":   ["shot_boundary", "ears_event", "uniform_stride"],
  "filters":     ["motion_diff"],
  "aligners":    ["semantic_siglip", "ocr_text"],
  "weights":     {"semantic_siglip": 0.55, "ocr_text": 0.45},
  "evidence_prompts": ["text sign", "subtitle", "writing"],
  "K_proposal":  200,
  "K_keyframe":  32,
  "K_final":     8,
  "tau_sec":     2.0,
  "expected_expert": "vlm",
  "reasoning":   "text/OCR keywords",
  "planner":     "rule_based"
}
```

## Usage from the QA harness

```bash
# Drop-in replacement for the single-stage SkillRouter
python scripts/run_qa_benchmark.py \
    --dataset video_mme --real --vlm_model Qwen/Qwen2.5-VL-7B-Instruct \
    --cascade --cascade_planner rule_based \
    --stratify_per_bucket 300 \
    --output_dir outputs/main_video_mme_cascade
```

`--cascade_planner llm` uses the caption-LLM (Qwen2.5-3B) to emit the plan in
JSON, with `RuleBasedCascadePlanner` as a structured fallback.

## Adding a new tool (e.g. a depth aligner)

1. Implement a class that satisfies `EvidenceAligner` protocol:
   ```python
   @dataclass
   class DepthAligner:
       name = "depth"
       modality = "vision"
       cost_per_frame = 0.04
       def align(self, keyframes, frames_rgb, video_path, question, meta) -> np.ndarray:
           ...
   ```
2. Register it in `cascade/registry.py::default_registry()`:
   ```python
   reg.register_aligner("depth", DepthAligner())
   ```
3. Optionally teach the planner about it in `cascade/planner.py`.

No other code touches required — `CascadeRouter` discovers tools through the
registry and the `CascadePlan` names them by string.

## Why this is "Agentic"

* The planner emits a tool-DAG conditioned on the question + video metadata.
* Each cascade level is independently swappable through a Protocol.
* Long videos automatically enable expensive proposers (shot/EARS) while
  short clips stay on cheap uniform stride.
* Question keywords trigger modality-specific aligners (OCR/ASR/object).
* All decisions logged in `cascade_result.json` for reproducibility.
