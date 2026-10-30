---
name: ears_event
tier: L1_proposer
modality: audio_visual
cost: 0.5
hint: "long videos; audio-visual events"
output: frame_indices
gated_by: is_long  # duration_sec > 120
---

# When to use

Long videos where **audio-visual events** (not just visual cuts) localise the
answer: anomalies, onsets, salient acoustic moments aligned to frames. Proposes
candidate frames around detected events.

# When NOT to use

Short clips (gated off by the planner), and purely visual tasks where the audio
track carries no localising signal — the cost (0.5) is the highest among L1
proposers.

# Behaviour

Wraps the EARS event detector to score audio-visual eventfulness over time and
proposes frames near event peaks. Cached under `results/cascade_cache`.
Registered only if EARS imports; otherwise skipped.

Implemented by `cascade/proposers/ears_event.py::EARSEventProposer`.
