# Skill catalogue

This directory is the **human-readable, self-contained specification** of every
skill the agentic router can select. It mirrors the code that actually
implements and registers these skills, so the repository documents its own
catalogue with no external dependency:

- registration / instantiation: [`cascade/registry.py`](../cascade/registry.py)
  (`default_registry()`),
- the planner's cost / use-case hints fed to the LLM:
  [`cascade/planner.py`](../cascade/planner.py) (`_build_prompt`, `cost_hint`),
- the expert categorical: [`router/planner.py`](../router/planner.py).

Each card is a Markdown file with YAML front-matter (`name`, `tier`,
`modality`, `cost`, `hint`, `output`) followed by *when to use / when not to use
/ behaviour* notes.

**The cards are the single source of truth for the planner's cost/use-case
hints.** At prompt-build time the LLM planner loads every card's `cost` and
one-line `hint` via [`cascade/skill_cards.py`](../cascade/skill_cards.py)
(`load_skill_hints()`) and feeds them to the model; the inline
`_FALLBACK_COST_HINT` dict in `cascade/planner.py` is only a resilience fallback
for cards missing a `hint:` field. Runtime *behaviour* still comes from the
registered Python tools in `cascade/registry.py` — editing a card changes what
the planner is told, not the tool implementation. Keep `hint:` and the tool in
sync.

`cost` values are normalised budget units per video-minute, measured
empirically on this codebase (GB10, fp16), copied from `cost_hint`.

## L1 — frame proposers (high recall)

| skill | cost | role |
|-------|------|------|
| [`uniform_stride`](uniform_stride.skill.md) | 0.0 | always-on backbone |
| [`stacfp`](stacfp.skill.md) | 0.3 | default long-video backbone (gated by `is_long`) |
| [`shot_boundary`](shot_boundary.skill.md) | 0.1 | ablation only |
| [`ears_event`](ears_event.skill.md) | 0.5 | long audio-visual event proposals |
| [`keyframe_cache`](keyframe_cache.skill.md) | cache | reuse offline keyframes |

## L2 — keyframe filters (drop redundancy)

| skill | cost | role |
|-------|------|------|
| [`redundancy`](redundancy.skill.md) | 0.1 | default L2 |
| [`motion_diff`](motion_diff.skill.md) | 0.1 | ablation only |

## L3 — evidence aligners (question-utility ranking)

| skill | cost | modality | role |
|-------|------|----------|------|
| [`semantic_siglip`](semantic_siglip.skill.md) | 1.0 | vision | default visual-text relevance |
| [`object_siglip`](object_siglip.skill.md) | 1.2 | vision | object / colour / position cues |
| [`ocr_text`](ocr_text.skill.md) | 0.4 | vision | text / signs / subtitles |
| [`asr_whisper`](asr_whisper.skill.md) | 0.3 | audio | what was *said* |
| [`clap_alm`](clap_alm.skill.md) | 0.2 | audio | music / sound / instrument |
| [`audio_energy`](audio_energy.skill.md) | 0.1 | audio | loud-event salience |

## Expert pool (modality-matched reasoners)

| expert | role |
|--------|------|
| [`vlm`](experts.md#vlm) | frames-only vision-language model |
| [`alm`](experts.md#alm) | audio-language model |
| [`avlm`](experts.md#avlm) | audio-visual language model |
| [`caplm`](experts.md#caplm) | caption-conditioned language model |
