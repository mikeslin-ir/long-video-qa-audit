---
name: keyframe_cache
tier: L1_proposer
modality: visual
cost: cache
hint: "reuses offline keyframe cache; ~free at query time on a cache hit"
output: frame_indices
---

# When to use

When offline keyframes / embeddings have already been extracted for the video.
This proposer reuses that cache instead of recomputing, so query-time cost is
effectively zero.

# When NOT to use

When no cache exists for the video — it then has nothing to propose and the
planner falls back to the other proposers.

# Behaviour

Loads precomputed keyframe indices (and optionally their embeddings, which the
cache-aware aligners can reuse) from disk.

Implemented by `cascade/proposers/keyframe_cache.py::KeyframeCacheProposer`.
