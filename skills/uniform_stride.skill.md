---
name: uniform_stride
tier: L1_proposer
modality: visual
cost: 0.0
hint: "free; always include as a backbone"
output: frame_indices
---

# When to use

**Always.** Uniform stride is the free backbone proposer: it samples frames at a
fixed temporal stride to guarantee baseline temporal coverage. Every plan
includes it so that no part of the video is structurally unreachable.

# When NOT to use

There is no reason to drop it; other L1 proposers are *added* on top of it, not
substituted for it.

# Behaviour

Samples `N` candidate frame indices evenly across the clip. On short clips this
already covers the content; on long clips it is complemented by content-adaptive
proposers (`stacfp`, `ears_event`).

Implemented by `cascade/proposers/uniform.py::UniformStrideProposer`.
