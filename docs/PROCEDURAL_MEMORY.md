# Procedural Memory — How It Works, Analysis & Discussion

This note explains the **procedural memory** component of the VideoRouteAudit planner: what
it is, how it is wired into the system, the experimental evidence for it, and an
honest discussion of what it does and does not buy us. It consolidates the
material in the paper (`paper/sections/method.tex` §`sec:procmem`,
`experiments.tex` F3 + scale check) with the implementation
(`cascade/planner.py`).

---

## 1. What it is

Procedural memory is a **short markdown file (≤ 1 page)** of empirically
validated priors about *which text-injection decisions help on which question
phenotypes*. It is **not** a learned parameter, an embedding store, or a
retrieval index — it is plain text appended verbatim to the planner LLM's
prompt. Two files ship in the repo, and **every reported planner run loads
`procedural_memory_v1.md`**; `v4` was drafted afterwards and no reported run
uses it. (The labels "v1", "v2" and "v3" in the paper and in run names are
planner *configurations* — free plan, frozen aligners, `--planner_inject_only` —
not different memory texts.)

- [`configs/procedural_memory_v1.md`](../configs/procedural_memory_v1.md) —
  *selective*: inject ASR only when the question is narrative/causal; suppress
  on purely-visual; never inject ASR+OCR together.
- [`configs/procedural_memory_v4.md`](../configs/procedural_memory_v4.md) —
  *aggressive*: **always inject ASR unless the question is purely visual.**

The "memory" framing is literal: each file encodes lessons **mined from past
execution logs** (e.g. "on the 125 questions where the v3 configuration chose no-inject,
always-inject was +4.8 pt better"), so later drafts are the system *learning
from its own audits* — but the learning happens offline, by a human reading
logs and editing the markdown, not by gradient updates. `v1` itself quotes
outcomes on the `n=200` Video-MME evaluation prefix, so planner numbers are
test-set-informed diagnostics.

---

## 2. How it is wired in

```
configs/procedural_memory_v1.md
        │  --procedural_memory_file
        ▼
LLMCascadePlanner(procedural_memory="...")          # cascade/planner.py
        │
        ▼  _build_prompt(): inserts a
           "PROCEDURAL MEMORY (priors learned from past execution logs ...)"
           block BEFORE the few-shot examples
        ▼
   planner LLM (Qwen2.5-3B-Instruct) → JSON plan
        │
        ▼  inject_only_mode=True  →  keep ONLY plan.inject_text;
           hard-override every other field with the rule-based cascade
        ▼
   CascadeRouter executes the (mostly fixed) plan; the only memory-influenced
   degree of freedom is per-question inject_text ∈ {[], [asr], [ocr], [asr,ocr]}
```

Two design choices make this clean:

1. **Verbatim prompt injection, no parameter updates.** The file text is dropped
   into the system prompt as-is (`cascade/planner.py::_build_prompt`,
   `procedural_memory` block). Swapping `v1 ↔ v4` changes behaviour with zero
   code change.
2. **Inject-only isolation (`inject_only_mode`).** Without it, a free LLM planner
   silently rewrites *other* fields and contaminates attribution — on Video-MME
   long it swapped `semantic_siglip → object_siglip` on **113/200** plans, one of
   the null multi-cue configs from F1. With isolation, the planner's **no-inject
   branch is bitwise identical to the no-inject cascade baseline**, so any
   accuracy delta is attributable purely to the memory-driven inject decision.

---

## 3. Experimental evidence (in the paper)

### F3 — agent value is domain-dependent (K=8, n=200)

| Method (n=200, K=8)            | Video-MME long | XD-Violence | UCF-Crime |
|--------------------------------|:--------------:|:-----------:|:---------:|
| Uniform-K (no inject)          | 50.0           | 87.0        | 81.5      |
| Cascade (no inject)            | 53.5           | 91.0        | **85.0**  |
| Cascade + always-inject ASR    | **56.5**       | 91.0        | **85.0**  |
| Cascade + always-inject OCR    | 55.5           | 89.5        | 83.0      |
| ProcMem planner                | 53.5           | **92.5**    | 83.0      |
| Δ planner vs always-inject     | −3.0           | **+1.5**    | −2.0      |

All cells re-derived from `outputs/*/qa_summary.json`; the XD-Violence and
UCF-Crime planner runs use the free-plan configuration (v1) of memory file `v1`
(`trackB_{xd_violence,ucf_crime}_cascade_k8_llmPlanner_planInject_proceduralV1`),
Video-MME uses the inject-only configuration (v3) of the same file. There is no
"LLM planner (no memory)" run for XD-Violence or UCF-Crime, so that row has
been removed.

*Reading:* on **narrative** video (Video-MME long) every ASR snippet helps, so
always-inject saturates the gain and the selective planner leaves 3.0 pt on the
table. On **both** anomaly domains the always-inject ASR arm injected no text
at all — no Whisper transcript had been cached for those clips, so it is
identical to the plain cascade (91.0→91.0, 85.0→85.0) — and the planner chose
not to inject on any of the 400 questions, so its differences there come from
the tools it chose. This is *not* evidence about ASR injection on anomaly
video; the re-run with real transcripts is in the paper's supplementary. The
planner itself wins on XD-Violence (+1.5) and loses on
UCF-Crime (−2.0); at n=200 neither delta is significant, so no planner effect
on anomaly QA is claimed in either direction.

### Scale check — full Video-MME long bucket (K=8, n=900)

| Method (n=900, K=8)            | Acc. (%) | Wall s/item | p vs. uniform |
|--------------------------------|:--------:|:-----------:|:-------------:|
| Uniform-K                      | 46.2     | 10.0        | —             |
| Cascade (no inject)            | 47.4     | 13.3        | 0.452         |
| Cascade + ASR                  | 47.9     | 12.8        | 0.337         |
| Cascade + OCR                  | 47.2     | 21.6        | 0.552         |
| Procedural-memory planner      | 48.4     | 15.7        | 0.154         |

Wall time is `elapsed_sec / n` of each run (shared GPU); the one-time Whisper
transcription (~94 s per video) is not included, EasyOCR extraction is.

At full scale the planner row is the numerically highest but not distinguishable
from always-inject ASR (+0.6 pt, 95% CI [−1.2, +2.2], p = 0.61) or from uniform
(+2.2 pt, [−0.7, +5.1], p = 0.15). The memory quotes outcomes on the first 200
items, so the planner's number mixes 200 development items with 700 items the
memory never saw; on those 700 it scores 47.0% against 45.4% for always-inject ASR
(+1.6 pt, p = 0.17). No planner advantage is claimed.

### v1 → v4 audit (why the prior flipped)

`v4`'s header records the reasoning derived from the `v3` audit:
- on the 125 questions where v3 chose no-inject, always-inject was **+4.8 pt**
  better (0.544 vs 0.496);
- v3 captured 91.5 % of the oracle ceiling, but always-inject was +3 pt higher
  because v3 "left +4.8 pt on the table" on those 125 items;
- v3's *selectivity had no discriminative power* on the inject subset (the gain
  was ~+4.8 pt whether or not v3 chose to inject).

So `v4` flips the default to "always inject ASR unless purely visual", aiming to
recover the Video-MME loss while preserving the XD-Violence win (where being
selective on rare visual-only items still pays).

---

## 4. Analysis & discussion

**What the memory actually controls.** A single bit per question:
inject vs. don't (and which text stream). It is deliberately the *only*
non-overridable field. This is a feature, not a limitation — it makes the
contribution falsifiable and the ablation clean. The memory does **not** change
which frames are chosen or which encoders run.

**What the evidence supports.** At full scale (n=900) the planner is
numerically highest (48.4%) but is not distinguishable from always-inject ASR
(+0.6 pt, p = 0.61) or from uniform sampling (+2.2 pt, p = 0.15), and on the 700
items outside the development prefix it leads always-inject ASR by 1.6 pt
(p = 0.17). On the development prefix it is 3.0 pt *below* always-inject ASR. On
the anomaly clips it never chose to inject, so its differences there reflect the
tools it chose, not its injection decisions. There is no "LLM planner without
memory" run on those clips either. The planner is therefore an inspectable,
cheap (one call, ≤ 1 s) injection policy, not an established improvement.

**When it could pay for itself (a hypothesis, not a result).** Per-question
routing can only help when an unhelpful injection costs something. On narrative
video ASR rarely hurts, so "always inject" is hard to beat; on short anomaly
clips a mistimed transcript could mislead, but the audited runs never tested
that, because no transcript was available. The supplementary re-runs the anomaly
arms with real transcripts.

**Limitations / threats to validity (stated plainly).**
- Every full-scale planner delta is within noise (all p > 0.15); the planner
  number is test-set-informed because the memory quotes outcomes on the
  evaluation prefix.
- The memory is **hand-authored from log audits**, i.e. a human-in-the-loop
  offline learner. There is no automatic memory-writing loop (the benchmark
  can emit `(q, plan, reward)` records, but turning those into the next
  markdown version is manual).
- The priors are **domain/dataset-specific** (Video-MME narration, XD-Violence
  anomaly). They are not claimed to transfer unchanged to arbitrary video or to
  other backbones: the only second-backbone planner run (n=200) used a different
  candidate pool from its comparison arms, so it is not a matched test.
- Because of single-pass inject-only isolation, the memory cannot rescue a bad
  *selection* — it only governs text injection. Selection quality is the cascade's
  job (F1/F2).

**Future work (as noted in the paper's discussion).** Close the loop: mine the
`(q, plan, reward)` logs automatically into the next memory version, or use them
as supervision for a *learned* planner whose categorical over inject decisions is
biased by retrieved similar past questions — turning the static markdown prior
into a genuine retrieval-augmented procedural memory.

---

## 5. Try it

```bash
# selective prior (the file every reported planner run used), inject-only configuration
python scripts/run_qa_benchmark.py --dataset video_mme --only_task_type long --limit 20 \
    --selector ours --cascade --cascade_planner llm --cascade_profile audit \
    --cascade_uniform_floor 0.5 --planner_inject_only \
    --procedural_memory_file configs/procedural_memory_v1.md \
    --inject_evidence_text plan --budget 8 --real --output_dir outputs/try_memory_v1

# aggressive prior (always inject ASR unless purely visual; drafted, never used in a reported run)
python scripts/run_qa_benchmark.py --dataset video_mme --only_task_type long --limit 20 \
    --selector ours --cascade --cascade_planner llm --cascade_profile audit \
    --cascade_uniform_floor 0.5 --planner_inject_only \
    --procedural_memory_file configs/procedural_memory_v4.md \
    --inject_evidence_text plan --budget 8 --real --output_dir outputs/try_memory_v4
```

Compare the `planner_output` field of each run's `rows.jsonl` (the per-question
`inject_text` decision) and the accuracies. The decisions behind every reported
planner run are in `results/predictions/planner_decisions.csv.gz`. ASR
injection needs cached transcripts (`scripts/prewarm_asr_cache.py`). See
[RUNBOOK.md](RUNBOOK.md) for the full CLI.
