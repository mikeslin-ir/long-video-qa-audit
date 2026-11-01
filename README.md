# Long-Video QA Audit

**Where do reported long-video QA gains come from?**
A controlled audit of *frame selection*, *evidence injection*, and *expert routing*.

**Shih-Yao Lin · Independent Researcher · WACV 2027**

This repository is the code release for **VideoRouteAudit**, a controlled audit
protocol and harness, **not** a leaderboard method. Everything here exists to make one kind of comparison
possible: hold the backbone, frame budget `K`, prompt scaffold, decoding
configuration and grader fixed, then vary exactly one routing decision at a
time and measure what actually changes.

---

## Headline findings

| | Finding |
|---|---|
| **F1** | At `K=8` on the **full** Video-MME long bucket (`n=900`), no tested selector significantly improves over uniform sampling. Every selection-only rule is statistically *equivalent* to uniform within a ±5 pt margin (none within ±3 pt), so a literature-scale gain is ruled out and a small one is not. |
| **F2** | Three of four matched contrasts are **1.6–3.8× larger** on the commonly used `n=200` subsets and the fourth (planner vs. always-inject ASR) reverses sign. Resampling 200-item subsets from the full-bucket predictions flips the sign of the full-bucket effect in **35.8%** of draws. The minimum detectable effect at `n=200` is **8.3–10.4 pt** — larger than the 1–5 pt gains this literature reports. |
| **F3** | Routing gains are domain-, protocol- and model-dependent. The pooled cascade+ASR gain over uniform (+4.7 pt on the three `n=200` development subsets) shrinks to +1.6 pt outside the development prefix, and the published anomaly-domain "+ASR" arms injected nothing (see [Caveats](#caveats)). With real transcripts, ASR appended after the question *lowers* XD-Violence accuracy by 10.0 pt (91.0 → 81.0) while the identical text placed before the question does not (91.0); on UCF-Crime Whisper hallucinates on silent footage and costs 13.0 pt either way. On Video-MME, ASR placed before the question adds about 4 pt to uniform sampling (+4.4, *p* = 0.002) and to the cascade (+4.0, *p* = 0.024) alike (post hoc, unadjusted; appended it gives +1.7, n.s.), so that gain follows the text and its position, not the frames (text between the question and the options, or after the options, adds only +1 to +2 pt); on Qwen2-VL-7B uniform + ASR with the text first gains +7.2 pt (appended +1.8, n.s.). The largest measured contrast comes from swapping the answering model on MUSIC-AVQA (+15.1 pt micro-average, +33.3 pt on AV-Temporal), but the same audio-visual expert with its soundtrack silenced keeps 92% of that gain (audio share 8%, CI [0, 16]) and with another video's soundtrack still 69%: the expert does read the audio (the wrong soundtrack is 3.4 pt below silence) but the contrast is mostly a model swap. |

### Main result — Video-MME long, `K=8`, `n=900`

| Method | Acc. (%) | Δ | 95% CI | *p* | Wall s/item |
|---|---:|---:|:---:|---:|---:|
| Uniform-`K` | 46.2 | — | — | — | 10.0 |
| Midpoint uniform-`K` (stronger baseline) | 46.2 | 0.0 | [−2.8, +2.9] | 1.00 | — |
| AKS (port) | 48.1 | +1.9 | [−1.2, +5.0] | 0.273 | 22.3 |
| KeyScore (port) | 47.1 | +0.9 | [−2.2, +4.0] | 0.624 | 22.4 |
| VideoTree (port) | 47.8 | +1.6 | [−1.6, +4.8] | 0.360 | 22.3 |
| Cascade (selection only) | 47.4 | +1.2 | [−1.7, +4.1] | 0.452 | 13.3 |
| Cascade + OCR | 47.2 | +1.0 | [−1.9, +4.0] | 0.552 | 21.6 |
| Cascade + ASR | 47.9 | +1.7 | [−1.6, +4.8] | 0.337 | 12.8 |
| Cascade + ASR, text before the question † | 50.2 | +4.0 | [+0.6, +7.3] | 0.024 | — |
| Uniform + ASR, text before the question † | 50.7 | +4.4 | [+1.8, +7.2] | 0.002 | — |
| ProcMem planner | **48.4** | +2.2 | [−0.7, +5.1] | 0.154 | 15.7 |

Backbone: `Qwen2.5-VL-7B-Instruct`. Paired McNemar tests, unadjusted; CIs are paired
bootstrap intervals. † A post hoc placement variant of ASR injection (the evidence
block ahead of the question instead of after the options, `--evidence_position
before`; `question` and `instruction` are the two intermediate positions), added after the anomaly-domain results; they are the only rows with
*p* < 0.05 (the cascade's gain over its no-text version is +2.8 pt, CI [+0.1, +5.6],
*p* = 0.055). No tested *selection rule* significantly improves on uniform.
Every number in this table is recomputed offline from
the released per-item predictions by `python scripts/replay_paper_stats.py`
(see [Verify the numbers without a GPU](#4-verify-the-numbers-without-a-gpu)).
Ported selectors are reimplementations of the *selection rule* scoring the
same 256-frame candidate pool — they are not the original authors' full
pipelines (at `K=8` the official AKS rule returns no frame, so its port is a
binned top-`K`; `docs/` and `results/port_fidelity.json` give the measured
fidelity). The cascade scores its own 132-candidate pool and keeps half of
its frames as uniform anchors.

Historical experiment records:
[`results/RESULTS_SNAPSHOT.md`](results/RESULTS_SNAPSHOT.md).
That earlier snapshot is not the final matched audit; use the table above
for the paper's main result.

### Figures

| File | Content |
|---|---|
| [`figures/fig1_mainidea.pdf`](figures/fig1_mainidea.pdf) | Three routing axes and the audit framing |
| [`figures/figure2_pipeline.png`](figures/figure2_pipeline.png) | Five-stage pipeline (propose → filter → align → select → answer) |
| [`figures/fig3_scalegap.pdf`](figures/fig3_scalegap.pdf) | The `n=200` vs `n=900` scale gap (F2) |
| [`figures/fig_f2_heatmap.pdf`](figures/fig_f2_heatmap.pdf) | Subset-resampling sign-flip heatmap |
| [`figures/fig_f4_expert.pdf`](figures/fig_f4_expert.pdf) | Expert-routing / model-swap contrast and the same-model silent-audio ablation |
| [`figures/fig_routing_ladder.pdf`](figures/fig_routing_ladder.pdf) | Expert-dispatch head-room (base, question-only routing, always-AV, oracle) |
| [`figures/fig_p2_qualitative_main.pdf`](figures/fig_p2_qualitative_main.pdf) | Qualitative selections |

Figures 3 and 4 are rebuilt by `python figures/builders/build_fig3_scalegap.py`
and `python figures/builders/build_fig4_expert.py` (the latter recomputes its
numbers from the shipped `results/audio_ablation/arms.csv.gz`); the heatmap and ladder by
[`scripts/make_result_figures.py`](scripts/make_result_figures.py). The Figure 1
and 2 builders are kept for reference: they also need video frames that are not
redistributed.

---

## Repository layout

```
audit_predictions.py  Loads per-item outcomes (raw run directories or the shipped package)
audit_stats.py        Paired statistics shared by the analysis scripts (McNemar, bootstrap, TOST)
data_paths.py         Dataset-root resolution (LONG_VIDEO_QA_DATA_ROOT)
cascade/            Core five-stage routing pipeline
  proposers/        L1 candidate proposal: uniform_stride, stacfp, shot_boundary, ears_event
  filters/          L1 filtering: redundancy, motion_diff
  aligners/         L2 alignment encoders: semantic/object SigLIP, OCR, ASR, CLAP, audio energy
  planner.py        Rule-based keyword planner + LLM planner
  router.py         CascadeRouter orchestration
  skill_cards.py    Loader for the self-contained skill catalogue
baselines/          Ported selectors: AKS, KeyScore, VideoTree, motion-only, clip-only, uniform, random
evidence_skills/    Cached skill implementations (EARS, Whisper, EasyOCR, SigLIP, CLAP)
router/             Backbones (Qwen2.5-VL, LLaVA-Video, LLaVA-NeXT-Video, VideoLLaMA2-AV), experts, memory, datasets
skills/             Self-contained skill cards (YAML frontmatter, 13 cards) + expert policy
configs/            Per-dataset YAML configs + procedural-memory versions
submodular.py       Lazy-greedy facility-location selector
fusion.py           Score normalization and weighted fusion
temporal_nms.py     Temporal suppression selector
scripts/            Runnable entry points + analysis
  paper_runs/       Paper experiment drivers (see reproduction notes below)
tests/              CPU unit tests (no GPU, no raw outputs needed)
.github/            CI: tests + offline replay of the paper's statistics
requirements.txt    Pinned GPU environment;  requirements-test.txt: CPU tests only
docs/               System overview, runbook, procedural-memory semantics, release checklist
figures/            Paper figures + their build scripts
results/            Durable result records (raw outputs are not shipped)
CITATION.cff        GitHub "Cite this repository" metadata, preferring the paper
citation.bib        Copyable BibTeX for the paper
```

Publication readiness and outstanding artifacts:
[`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md).

---

## Environment setup

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

---

## Usage

### 1. Single (video, question) — the fastest way to see the pipeline

```bash
python scripts/run_one.py \
    --video /path/to/clip.mp4 \
    --question "What is the man in the red shirt doing?" \
    --budget 8
```

Prints the chosen plan, the selected-frame timestamps, and the answer:

```
[run_one] PLAN: proposers=['uniform_stride'] filters=['redundancy']
          aligners=['semantic_siglip'] weights={'semantic_siglip': 1.0} expert=vlm
[run_one] selected 8 frames at [2.1, 9.4, 15.0, ...] s
Q: What is the man in the red shirt doing?
A: He is carrying a box.
   expert=vlm  inject=none  K=8  select=3.2s  infer=1.8s
```

Inject a transcript instead of relying on frames alone:

```bash
python scripts/run_one.py \
    --video /path/to/lecture.mp4 \
    --question "Why does the reaction stop?" \
    --budget 8 --inject asr
```

### 2. Benchmark a single routing decision

The one runner behind every table. `--selector` is the axis under test;
everything else stays fixed.

```bash
# Baseline: uniform sampling
python scripts/run_qa_benchmark.py \
    --dataset video_mme --selector uniform \
    --only_task_type long \
    --budget 8 --real --limit 900 \
    --output_dir outputs/vmme_uniform_k8

# Contrast: a ported selector, same pool, same budget, same grader
python scripts/run_qa_benchmark.py \
    --dataset video_mme --selector aks \
    --only_task_type long \
    --budget 8 --real --limit 900 \
    --output_dir outputs/vmme_aks_k8
```

Supported values:

- `--dataset` — `video_mme`, `egoschema`, `ucf_crime`, `xd_violence`,
  `music_avqa`, `mlvu`, `msvd_qa`, `egotextvqa`, `tutorialvqa`
- `--selector` — `uniform`, `random`, `motion`, `clip`, `ours`, `aks`,
  `keyscore`, `videotree`, `sevila`, `videoagent`
- `--vlm_backbone` — `qwen2_vl`, `llava_video`, `llava_next_video`,
  `videollama2_av`
- `--inject_evidence_text` — `none`, `asr`, `ocr`, `asr+ocr`, `plan`

Omitting `--real` stubs the answering experts only; selectors can still load
encoders. Use the CPU test suite above for a fully model-free wiring check.

### 3. The cascade and the planner

```bash
# Cascade, selection only (the configuration the paper's runs executed)
python scripts/run_qa_benchmark.py \
    --dataset video_mme --selector ours --cascade \
    --only_task_type long \
    --cascade_planner rule_based --cascade_profile audit --cascade_uniform_floor 0.5 \
    --budget 8 --real --limit 900

# Video-MME planner row: LLM planner, inject-only mode, the one memory file
python scripts/run_qa_benchmark.py \
    --dataset video_mme --selector ours --cascade \
    --only_task_type long \
    --cascade_planner llm --cascade_profile audit --cascade_uniform_floor 0.5 \
    --planner_inject_only --procedural_memory_file configs/procedural_memory_v1.md \
    --inject_evidence_text plan --budget 8 --real --limit 900

# Anomaly-domain LLM planner (free plan) using the same memory
python scripts/run_qa_benchmark.py \
    --dataset xd_violence --selector ours --cascade \
    --cascade_planner llm --cascade_profile audit \
    --cascade_uniform_floor 0.5 \
    --procedural_memory_file configs/procedural_memory_v1.md \
    --inject_evidence_text plan --evidence_window_sec 8 \
    --budget 8 --real --limit 200
```

`--planner_inject_only` isolates the planner's injection decision while keeping
the no-inject branch bitwise identical to the non-agentic baseline. This is the
control that makes the injection delta attributable.
`--evidence_window_sec` is the *total* transcript window centred on each
selected frame, so the benchmark default of `8` is the paper's ±4 s.
Build transcripts before any `--inject_evidence_text asr` run:
`python scripts/prewarm_asr_cache.py --dataset xd_violence --limit 200`.
Caches default to `results/cascade_cache`; set `LONG_VIDEO_QA_CASCADE_CACHE` to
reuse an existing one.

### 4. Verify the numbers without a GPU

Every per-item outcome behind the paper's tables is shipped in
[`results/predictions/`](results/predictions): `predictions.csv.gz`
(`run, qid, correct, pred, task_type, evidence_chars`), `manifest.csv` (one row
per run: backbone, budget, selector, scoring rule, cascade profile, candidate-pool
size, injection mode and how many items were actually injected) and
`planner_decisions.csv.gz`. Videos, frames, transcripts and weights are not
redistributed. Nothing below needs a GPU, the datasets or the raw run outputs.

```bash
pip install -r requirements-test.txt
python scripts/package_predictions.py --verify          # shipped outcomes agree with the manifest
python scripts/replay_paper_stats.py --check            # ~10 s; counts and p-values exact, CIs to 0.15 pt
python scripts/replay_paper_stats.py --check --slow     # adds the subset-resampling analysis
```

`replay_paper_stats.py` recomputes the accuracy, paired bootstrap interval,
McNemar test and discordant counts of every comparison in the paper, the pooled
and stratified tests, the judge-validation agreement and the MUSIC-AVQA
per-type table, and fails if any of them drifts from
`results/paper_reference_values.json`.

### 5. Re-running the experiments (GPU, datasets and weights required)

```bash
bash scripts/paper_runs/run_wave10_matched_selectors.sh   # main n=900 selector table
bash scripts/paper_runs/run_videotree_n900.sh             # VideoTree port at full scale
bash scripts/paper_runs/run_published_cascade_arms.sh all # cascade / +ASR / +OCR / planner arms, three domains
bash scripts/paper_runs/run_cascade_ksweep.sh             # EgoSchema K sweep
bash scripts/paper_runs/run_trackB_ksweep.sh              # Video-MME long budget sweep
bash scripts/paper_runs/run_xbackbone_ksweep.sh           # cross-backbone check
bash scripts/paper_runs/run_eventqa_anomaly.sh            # UCF-Crime / XD-Violence
LIMIT=0 bash scripts/paper_runs/run_music_avqa_videollama2av.sh  # full AV split
MODES="silent mismatched" bash scripts/paper_runs/run_music_avqa_audio_ablation.sh  # same AV model, soundtrack silenced / swapped for another video's
LANE=A|B|C|D|E|F bash scripts/paper_runs/run_review_experiments.sh  # post-review additions (see script header); F = placement control, midpoint uniform, K=1, Qwen2-VL, STACFP, EgoSchema ports, run as shards
```

The drivers keep the experiment settings and resolve their own working
directory. Failed runs stop rather than write a success marker, and every step
resumes from its `rows.jsonl`. Set the dataset-root variable above, use the
appropriate model environment, and read the driver before a costly run. The
AV driver defaults to a 1,200-item diagnostic subset; `LIMIT=0` selects the full
split, as required for the headline expert comparison. Other scripts in
`paper_runs/` are historical records of earlier development waves.

To fold new runs into the shipped record, repackage and replay:

```bash
python scripts/package_predictions.py --outputs outputs
python scripts/package_predictions.py --verify --outputs outputs
```

### 6. Analysis and statistics

```bash
python scripts/aggregate_qa_results.py --runs outputs/          # collect qa_summary.json
python scripts/stats_analysis.py                                # McNemar, paired CIs, TOST (the comparison battery)
python scripts/pooled_sensitivity.py                            # F3(a): pooled test, development prefix vs held-out, equal-domain weights
python scripts/subset_stability.py                              # F2: 200-item subset resampling (sign-flip rate, percentile of our prefix)
python scripts/cluster_robustness.py                            # video-clustered bootstrap and design effects
python scripts/audio_ablation_analysis.py                       # MUSIC-AVQA per-type table and the same-model silenced-audio ablation (judge-graded; --grader inloop for the string matcher)
python scripts/routed_expert_analysis.py                        # question-only expert dispatch (shipped dispatch decisions)
python scripts/judge_validation_report.py                       # LLM judge vs human labels, overall and per dataset
python scripts/port_fidelity.py                                 # selector-port fidelity on cached SigLIP pools
python scripts/cost_summary.py RUN [RUN ...]                    # per-stage runtime of a run (results/cost_summary.json)
```

The first seven scripts after `aggregate_qa_results.py` read only the files shipped
in `results/`. `port_fidelity.py` needs the cached candidate pools and
`cost_summary.py` needs raw run directories.

Development-era analyses that are **not** used for any number in the paper (they
read raw run outputs and are kept for transparency, unmaintained):
`pool_significance.py` (superseded by `pooled_sensitivity.py`),
`analyze_sparsity_law.py`, `oracle_gap_experiment.py` (driver
`scripts/paper_runs/run_oracle_gap.sh`) and `lodo_memory_transfer.py`.

Grader validation (the LLM judge is itself audited against human labels):

```bash
python scripts/build_judge_validation_sheet.py
python scripts/score_judge_validation.py
```

---

## Caveats

Read these before drawing conclusions from anything in this repo.

- **The audited cascade is not the library's default pipeline.** The paper's
  Video-MME long, XD-Violence and UCF-Crime cascade runs executed
  *EARS event peaks + uniform frames → motion-difference filter → SigLIP →
  facility location with 50% uniform anchors*. Their plans also listed a
  PySceneDetect shot-boundary proposer, but PySceneDetect was not installed, so
  the router skipped it silently (a 132-frame pool instead of 198). Pass
  `--cascade_profile audit` to reproduce those runs (the paper-run drivers do);
  the default `stacfp` profile (STACFP + EARS + uniform, redundancy filter) is the
  later refinement. At `n=900` it scores 46.4% against 46.2% for uniform (+0.2 pt, CI
  [−2.7, +3.2], equivalent within ±5 pt) and −1.0 pt against the audited cascade
  (lane F of `scripts/paper_runs/run_review_experiments.sh`). A re-run of the
  audit profile reproduces the published selected frames (24/24 items) and,
  end to end, every prediction on 16 planner items and 12 uniform items. The router now warns
  whenever a planned tool fails.
- **EARS is an external dependency.** The `ears_event` proposer imports the
  [EARS](https://github.com/mikeslin-ir/EARS) feature code from a sibling
  `../EARS` checkout, or from `$EARS_PATH`. Without it the proposer is
  skipped (with a warning) and the cascade differs from the paper. Its motion
  signal covers only a video's first 4,096 frames, so its peaks lie in the
  opening minutes of a long video.
- **"Procedural memory v1/v3" name planner configurations, not files.** Every
  planner run loads `configs/procedural_memory_v1.md`; `v1` is the free plan,
  `v2` freezes the aligners and `v3` is `--planner_inject_only`. The memory text
  quotes outcomes on the `n=200` evaluation prefix, so planner numbers are
  test-set-informed diagnostics. `LLMCascadePlanner` defaults to the prompt the
  audited runs used (`prompt_version="paper"`); per-item planner decisions are in
  `results/predictions/planner_decisions.csv.gz`.
- **ASR injection needs transcripts.** `--inject_evidence_text asr` injects
  nothing for a video that has no cached Whisper transcript (the original
  XD-Violence / UCF-Crime "+ASR" arms were identical to the plain cascade for
  this reason). Run `scripts/prewarm_asr_cache.py` first (`--max_new N` bounds its
  memory on long videos); `qa_summary.json` now reports how many items were
  actually injected.
- **Transcript quality and prompt placement decide what injection does.**
  Whisper `tiny.en` hallucinates on silent audio: 160 of the 200 UCF-Crime
  transcripts contain "thank you for watching" at least three times (XD-Violence:
  9 of 200), and nothing in the pipeline filters them. The audited runs append the
  evidence block *after* the question, options and answer instruction;
  `--evidence_position before` (Qwen backbones) places the identical block ahead
  of the question. On XD-Violence the two placements differ by 10 pt and on
  Video-MME by 2.3 pt (n.s.; the before-question arms are post hoc variants: +4.0 pt
  for the cascade, +4.4 pt for uniform, unadjusted); on UCF-Crime they do not differ
  (the paper's F3). The audited numbers use the appended placement.
- **Ported selectors are selection-rule reimplementations**, matched to a common
  candidate pool and budget. They are not the original authors' pipelines and
  should not be read as reproductions of their published numbers. At `K=8` the
  official AKS routine returns no frame at its default depth, so the AKS port is
  a temporally binned top-`K` (`scripts/port_fidelity.py`).
- **Per-item outcomes are shipped; raw run outputs are not.**
  `results/predictions/` holds one 0/1 outcome (plus the model's short answer and
  task type) for every item of every reported run, and `scripts/replay_paper_stats.py`
  recomputes the paper's statistics from it. Generated outputs and result-cache
  directories are gitignored.
- **Layout defaults are overridable.** No author-machine paths ship (a test
  enforces it), but some defaults assume a layout: datasets under
  `~/project/datasets` (`LONG_VIDEO_QA_DATA_ROOT`), a VideoLLaMA2 checkout at
  `$HOME/project/code/VideoLLaMA2` (`VLLAMA2AV_REPO`) and a sibling `../EARS`
  checkout (`EARS_PATH`).
- **The single-video demo is not the benchmark reproduction entry point.**
  Its LLM-planner path is not wired to a caption model, and its AV loading
  path needs separate single-GPU validation. Use the benchmark drivers for
  those configurations; see the release checklist.
- **Sharded runs.** `scripts/paper_runs/run_sharded.sh` (used by lane F) runs one configuration
  as parallel shards in one output directory. Each shard writes its own `qa_summary.json`, so
  its accuracy covers that shard only; `rows.jsonl` (all items) is what the packager and the
  statistics read, and a `DONE` marker says the whole run finished. `--skip_option_logprobs` is
  added (predictions unchanged; cost fields not comparable).
- Some `scripts/paper_runs/*.sh` are one-shot experiment drivers preserved for
  traceability rather than polished tools.

---

## Tests

```bash
python -m pytest tests/ -q
```

Covers the skill-router core, a model-free cascade smoke path, planner profiles,
submodular selection, memory handling, determinism, dataset-root configuration,
the statistics module (exact McNemar, bootstrap, sign-flip), release hygiene
(every paper driver parses, resolves the repository root and references
existing scripts; no author-machine paths) and citation consistency. GitHub
Actions runs these CPU checks and the offline replay of the paper's statistics
(`scripts/replay_paper_stats.py --check`); it does not certify GPU inference or
re-run any model.

---

## Citation

Please cite the paper when using the code or audit protocol. Download
[`citation.bib`](citation.bib), or use GitHub's **Cite this repository**
button (configured by [`CITATION.cff`](CITATION.cff)).

```bibtex
@inproceedings{lin2027longvideoaudit,
  author    = {Lin, Shih-Yao},
  title     = {Where Do Reported Long-Video {QA} Gains Come From? A Controlled Audit of Frame Selection, Evidence Injection, and Expert Routing},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027},
}
```

Add the final paper URL, pages, and DOI when assigned; they are intentionally
not guessed here.

## License

Released under the MIT License — see [`LICENSE`](LICENSE). Datasets and model
weights are governed by their own licenses.
