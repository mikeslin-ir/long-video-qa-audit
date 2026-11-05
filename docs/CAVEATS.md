# Caveats

Read these before drawing conclusions from anything in this repository.

[← Back to README](../README.md)


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
