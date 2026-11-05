# Publication and Reproducibility Checklist

Status of this working tree before the public release. Nothing here has been
committed or pushed by the preparation work; the repository visibility is
unchanged.

## Done

- **Citation.** [`citation.bib`](../citation.bib), [`CITATION.cff`](../CITATION.cff)
  and the [README citation](../README.md#citation) agree on author, title and
  venue (`tests/test_publication_metadata.py`).
- **Per-item predictions.** [`results/predictions/`](../results/predictions)
  holds one 0/1 outcome (plus the model's short answer, task type and the
  number of injected evidence characters) for every item of every reported run,
  a manifest with each run's backbone, budget, selector, cascade profile,
  candidate-pool size and injection accounting, and the planner's per-item
  decisions. No video, frame, transcript or weight is redistributed.
  `python scripts/package_predictions.py --verify --outputs <outputs>` checks
  the shipped table against a working tree, and
  `tests/test_audit_stats.py::test_manifest_agrees_with_the_packaged_outcomes`
  checks it against its own manifest in CI.
- **Offline replay of the statistics.** `python scripts/replay_paper_stats.py
  --check` recomputes every accuracy, paired bootstrap interval, McNemar test,
  discordant count, pooled and stratified test, clustered interval, judge
  agreement and MUSIC-AVQA per-type table in the paper from the shipped
  outcomes (about 10 s; `--slow` adds the subset-resampling analysis) and
  compares them with `results/paper_reference_values.json`. CI runs it.
- **Procedural memory.** There is one memory file,
  `configs/procedural_memory_v1.md`; "v1", "v2" and "v3" in the paper name
  planner *configurations* (free plan, frozen aligners, inject-only), not
  separate files. The earlier worry that a "v3" file was missing is therefore
  closed; see [`PROCEDURAL_MEMORY.md`](PROCEDURAL_MEMORY.md). The memory quotes
  outcomes on the `n=200` evaluation prefix, so planner rows are labelled
  test-set-informed.
- **The audited pipeline is reproducible.** `--cascade_profile audit` selects
  exactly what the published cascade runs executed (EARS + uniform proposals,
  motion-difference filter, no shot-boundary proposer). Re-running it
  reproduced the published selected frames on 24/24 items and, end to end,
  the inject decisions, evidence lengths and predictions of 16/16 planner
  items and 12/12 uniform items. The paper-version planner prompt is the
  default (`prompt_version="paper"`). The router now warns whenever a planned
  proposer, filter or aligner fails, and `qa_summary.json` reports how many
  items were actually injected.
- **Experiment drivers.** All drivers in `scripts/paper_runs/` parse, change
  to the repository root, reference existing scripts and contain no
  author-machine paths (`tests/test_release_hygiene.py`).
  `run_published_cascade_arms.sh` runs every cascade / injection / planner arm
  of the main tables with the audited flags.
- **Tests.** The CPU suite passes in a clean virtual environment installed
  only from `requirements-test.txt` with the raw outputs absent
  (`LONG_VIDEO_QA_OUTPUTS=/nonexistent`), so everything it covers works from
  a fresh clone.

## Experiments added after the reviews

Run on one GB10 (119 GB unified memory) with the drivers in
`scripts/paper_runs/run_review_experiments.sh` (lanes B-E) and
`run_music_avqa_audio_ablation.sh`. Their per-item outcomes are in
`results/predictions/` and every statistic below is replayed by
`scripts/replay_paper_stats.py`.

| Experiment | Result |
|---|---|
| XD-Violence / UCF-Crime: ASR with real Whisper transcripts (200 clips each), block after the question | 91.0 -> 81.0 (XD, -10.0 pt) and 85.0 -> 72.0 (UCF, -13.0 pt) |
| The same, block before the question | XD 91.0 (no harm), UCF 72.0 (harm persists: Whisper hallucinates on silent footage) |
| Video-MME long, `n=900`, cascade + ASR with the block before the question | 47.4 -> 50.2 (+2.8 pt, CI [+0.1, +5.6], `p`=0.055; appended: +0.4); +4.0 pt over uniform (CI [+0.6, +7.3], `p`=0.024). Same 865 of 900 questions receive the same text in both layouts. Post hoc and unadjusted |
| Video-MME long, `n=900`, uniform + ASR with the block before the question | 50.7 (+4.4 pt over uniform, CI [+1.8, +7.2], `p`=0.002); equal to the cascade with the text first (`p`=0.83): the gain follows the text, not the frames. Run as four shards; one was OOM-killed and resumed |
| MUSIC-AVQA, same expert with another video's soundtrack (mismatched audio), 1,800 questions | judge-graded, post-stratified: blind 67.0, mismatched 77.5, silenced 80.9, true audio 82.1. Wrong audio is 3.4 pt below silence (`p`=1e-5) and 4.7 below true audio, yet +10.5 over the blind model (69% of the swap): the expert reads the soundtrack, but the contrast is mostly a model swap. Soundtracks are assigned by a fixed seeded derangement of the 6,399 videos |
| Video-MME long, `n=900`, default STACFP profile | 46.4 vs uniform 46.2 (+0.2 pt, CI [-2.7, +3.2], equivalent within ±5 pt); with ASR 46.9 (+0.7, n.s.); -1.0 pt against the audited cascade |
| Video-MME long, `n=900`, Qwen2-VL-7B cascade + ASR with the same 132-candidate pool | 44.6; -0.2 pt within the cascade (`p`=0.92). The original arm's +3.2 pt (`p`=0.007) was the 198 vs 132 pool difference |
| EgoSchema, `n=500`, selector ports | uniform 67.8; AKS 65.2 (-2.6), KeyScore 63.8 (-4.0, CI [-7.8, -0.2], `p`=0.049, not significant after correcting for four contrasts), VideoTree 67.6 (-0.2); cascade 66.2 (-1.6). No selector helps |
| Video-MME long, `n=900`, segment-midpoint uniform at `K=8` | 46.2, identical to endpoint-inclusive uniform (0.0 pt, CI [-2.8, +2.9]); cascade +1.2 and AKS port +1.9 over it, both equivalent within ±5 pt. F1 does not depend on the uniform spacing |
| Video-MME long, `n=900`, uniform + ASR in four prompt positions (Qwen2.5-VL) | no text 46.2; after the instruction (audited) 47.8; between options and instruction 48.1; between question and options 47.2; before the question 50.7. Only the last differs (+4.4 over no text, `p`=0.002; +3.4 over the question/options position, `p`=0.008). Same evidence on all 900 items in every arm |
| Video-MME long, `n=900`, Qwen2-VL-7B, uniform + ASR before / after the question | 45.4 -> 52.7 before (+7.2 pt, CI [+4.4, +9.9], `p`<1e-6), 47.2 after (+1.8, `p`=0.09); before vs after +5.4 (`p`=0.0002). Replicates the placement effect on a second backbone |
| Video-MME long, `n=900`, `K=1`, Qwen2.5-VL-7B: uniform / middle-frame uniform / cascade | 37.0 / 42.1 / 39.7. Cascade vs uniform +2.7 (CI [-0.7, +6.0], `p`=0.13), so the `n=200` +10.0 shrinks; middle-frame uniform +5.1 over first-frame uniform (`p`=0.003) and 2.4 above the cascade (n.s.) |
| EgoSchema, `n=500`, uniform vs audit-profile cascade | 67.8 vs 66.2 (-1.6 pt, CI [-4.6, +1.4]); the early-pipeline cascade run has the same accuracy but disagrees on 60 questions |
| MUSIC-AVQA, same audio-visual expert with its soundtrack silenced, 200 questions per type (1,799 usable; one video has no audio track) | swap +15.1 pt, audio +1.3 pt (CI [-0.1, +2.6]), about 8% of the swap; the in-loop matcher gives the same share |
| Dispatcher analysis on the full split | 78.6%, +12.5 pt over the visual base, 84.9% of questions sent to the AV expert |

**Not run:** the remaining 7,330 MUSIC-AVQA questions of the silenced-audio run (and of the
mismatched-audio run); the packager lists `audio_ablation_silent_rest` under "not found".

Operational notes from these runs: the judge loads its 65 GB checkpoint shard by
shard (`scripts/llm_judge.py`, see docs/USAGE.md), reuses verdicts of identical
question / gold / answer triples (`--reuse-from`; 1,631 of the 1,799 silenced-audio
verdicts, with 30 of 30 re-judged reused verdicts reproduced), and resumes from a
partial file; on this machine at most four 7B jobs fit next to each other. The
Video-MME before-question run was executed as four to six parallel shards
(`--only_qids` lists, `--resume`, one output directory) with
`--skip_option_logprobs`, which leaves predictions unchanged (identical on 20 of 20
items checked); 24 items appear twice in its `rows.jsonl` with identical
outcomes where shards overlapped, and its `expert_cost_sec` / wall times are not
comparable with the other runs. `scripts/paper_runs/run_sharded.sh` packages this pattern (staggered shard launches, automatic retry of shards an out-of-memory kill left unfinished); the Qwen2-VL and `K=1` runs above used it.

## Known differences between the released defaults and the audited runs

These are documented in docs/CAVEATS.md and the paper; they are listed here
so that nobody has to rediscover them.

- The library default planner profile is `stacfp` (STACFP + EARS + uniform,
  redundancy filter), a later refinement. The audited runs used `audit`.
- Published XD-Violence / UCF-Crime "+ASR" arms injected no text (no Whisper
  transcript had been computed), and the planner arms on those clips never
  chose to inject. The drivers now transcribe first; see the paper's F3(a).
- The Qwen2-VL "+ASR" arm of the cross-backbone check was run with a 198-frame
  proposal pool, the other Qwen2-VL arms with 132.
- The EARS proposer's motion signal covers only a video's first 4,096 frames.
- Per-question planner prompts were edited after the published runs; the
  paper-version prompt is preserved as the default and the later one is
  available as `prompt_version="current"`.
- The AKS port is a temporally binned top-`K` because the official routine
  returns no frame at `K=8` (`results/port_fidelity.json`).
- Every published injection arm appends the evidence block after the question,
  the options and the answer instruction. `--evidence_position before` puts it
  ahead of the question (Qwen backbones only); the manifest's
  `evidence_position` column says which layout a run used.
- `results/audio_ablation/arms.csv.gz` holds the three arms of the silenced-audio
  ablation under both graders; rebuild it with
  `python scripts/audio_ablation_analysis.py --outputs <outputs> --write-arms`.

## Needs the authors' decision

- [ ] **Make the repository public** and verify the URL without
  authentication before the camera-ready PDF cites it. SSH access alone does
  not establish public visibility.
- [ ] **Confirm redistribution terms** for the shipped per-item outcomes
  (question ids and short model answers) for Video-MME, EgoSchema,
  XD-Violence, UCF-Crime and MUSIC-AVQA, and for the figures.
- [ ] **EARS.** The `ears_event` proposer imports the EARS code from a sibling
  checkout or `$EARS_PATH`. Pin the commit the audited runs used, or vendor
  the proposer, and state its licence.
- [ ] **Validate a clean GPU installation** on the target CUDA/Python stack.
  The pinned requirements record the reference environment, not a
  cross-platform guarantee.
- [ ] **Single-video demo.** Its AV path loads a visual model first and the
  LLM path has no caption-model generator. The benchmark scripts are the
  reproduction entry point.

## Before announcing the GitHub release

1. Run `python -m pytest tests/ -q` and `python scripts/replay_paper_stats.py
   --check` in an isolated environment installed from `requirements-test.txt`.
2. Review the staged files and Git history for material not intended for
   release, and check dataset, model and figure redistribution terms. This
   preparation is not a completed security or licensing audit.
3. Deliberately commit and push the reviewed release; no automatic push,
   visibility change or release tagging is part of local preparation.
4. After the proceedings appear, add the official paper URL, pages and DOI to
   the BibTeX, CFF preferred citation and README together.
5. Create an archival release/DOI only after verifying the deposited
   artifacts. Do not invent a version, release date or archive identifier.
