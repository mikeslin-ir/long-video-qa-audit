---
name: shot_boundary
tier: L1_proposer
modality: visual
cost: 0.1
hint: "ablation only; standalone shot detection -- STACFP's snap_cuts regime covers this"
output: frame_indices
status: ablation_only
---

# When to use

**Ablation only.** Standalone shot-boundary detection: proposes one
representative frame per detected shot. Useful to isolate the contribution of
shot structure against `stacfp`, whose `snap_cuts` regime already covers this.

# When NOT to use

In the default plan — `stacfp` subsumes it. Selecting both is redundant.

# Behaviour

Runs shot detection (PySceneDetect when available) and emits boundary frames.
Cached under `results/cascade_cache`. Registered only if PySceneDetect imports;
otherwise the cascade degrades gracefully.

Implemented by `cascade/proposers/shot_boundary.py::ShotBoundaryProposer`.
