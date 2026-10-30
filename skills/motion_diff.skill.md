---
name: motion_diff
tier: L2_filter
modality: visual
cost: 0.1
hint: "ablation only; SELECTS high-motion frames and drops static ones -- can lose static signs/speakers"
output: frame_subset
status: ablation_only
---

# When to use

**Ablation only.** Selects high-motion frames and drops static ones. Useful to
study motion-centric questions in isolation, or as a contrast to the default
`redundancy` filter.

# When NOT to use

In the default plan. Because it actively *removes* static frames, it can discard
static-but-decisive evidence (a sign, a held object, a stationary speaker or
instrument), which is why it is not the default.

# Behaviour

Scores inter-frame motion and keeps the high-motion subset.

Implemented by `cascade/filters/motion_filter.py::MotionDiffFilter`.
