---
name: redundancy
tier: L2_filter
modality: visual
cost: 0.1
hint: "default L2; drops near-duplicate frames, keeps static-but-distinct ones; content-adaptive count"
output: frame_subset
---

# When to use

**Default L2 filter.** Drops near-duplicate frames while keeping static-but-
distinct ones, with a content-adaptive kept count. This is the safe default
because it removes redundancy without discarding informative static frames
(signs, a motionless speaker, a still instrument).

# When NOT to use

There is rarely a reason to drop it; the alternative `motion_diff` is reserved
for ablations because it can lose static evidence.

# Behaviour

Compares candidate frames in embedding space and removes those that are
near-duplicates of an already-kept frame; the number kept adapts to content
diversity.

Implemented by `cascade/filters/redundancy_filter.py::RedundancyFilter`.
