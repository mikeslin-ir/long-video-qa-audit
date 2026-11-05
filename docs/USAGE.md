# Usage

Runnable entry points, reproduction of the paper runs, analysis scripts, tests and the figure inventory.

[← Back to README](../README.md)


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
[`results/predictions/`](../results/predictions): `predictions.csv.gz`
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

### Figures

| File | Content |
|---|---|
| [`figures/fig1_mainidea.pdf`](../figures/fig1_mainidea.pdf) | Three routing axes and the audit framing |
| [`figures/figure2_pipeline.png`](../figures/figure2_pipeline.png) | Five-stage pipeline (propose → filter → align → select → answer) |
| [`figures/fig3_scalegap.pdf`](../figures/fig3_scalegap.pdf) | The `n=200` vs `n=900` scale gap (F2) |
| [`figures/fig_f2_heatmap.pdf`](../figures/fig_f2_heatmap.pdf) | Subset-resampling sign-flip heatmap |
| [`figures/fig_f4_expert.pdf`](../figures/fig_f4_expert.pdf) | Expert-routing / model-swap contrast and the same-model silent-audio ablation |
| [`figures/fig_routing_ladder.pdf`](../figures/fig_routing_ladder.pdf) | Expert-dispatch head-room (base, question-only routing, always-AV, oracle) |
| [`figures/fig_p2_qualitative_main.pdf`](../figures/fig_p2_qualitative_main.pdf) | Qualitative selections |

Figures 3 and 4 are rebuilt by `python figures/builders/build_fig3_scalegap.py`
and `python figures/builders/build_fig4_expert.py` (the latter recomputes its
numbers from the shipped `results/audio_ablation/arms.csv.gz`); the heatmap and ladder by
[`scripts/make_result_figures.py`](../scripts/make_result_figures.py). The Figure 1
and 2 builders are kept for reference: they also need video frames that are not
redistributed.
