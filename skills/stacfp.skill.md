---
name: stacfp
tier: L1_proposer
modality: visual
cost: 0.3
hint: "DEFAULT long-video backbone; spatio-temporal adaptive clustering, content-adaptive anchor count; self-adapts to cut density -- snaps to cuts on edited video, clusters on long takes; subsumes shot_boundary"
output: frame_indices
gated_by: is_long  # duration_sec > 120
---

# When to use

Default **long-video** backbone. STACFP (spatio-temporal adaptive
content-aware frame proposal) uses a content-adaptive anchor count: it snaps to
cuts on heavily edited footage and clusters on long takes, so the candidate set
follows the actual structure of the video.

Subsumes `shot_boundary` — its `snap_cuts` regime already covers standalone shot
detection.

# When NOT to use

On short clips (`duration_sec <= 120`) the planner gates it off; uniform stride
alone is enough and STACFP's cost is not justified.

# Behaviour

Clusters frames by spatio-temporal content and self-adapts the number of anchors
to cut density. Cached under `results/cascade_cache`.

Implemented by `cascade/proposers/stacfp.py::STACFPProposer`.
