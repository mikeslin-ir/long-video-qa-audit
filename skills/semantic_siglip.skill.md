---
name: semantic_siglip
tier: L3_aligner
modality: vision
cost: 1.0
hint: "default; visual-text relevance"
output: per_frame_score
requires: [text_query]
evidence_class: SemanticEvidence
---

# When to use

**Default aligner.** Use whenever the question contains groundable nouns, verbs
or objects that can be matched to visual content. It is the most general-purpose
visual relevance signal and should be in almost every plan unless the task is
purely audio-driven.

# When NOT to use

- Purely audio tasks (use `clap_alm` / `asr_whisper` / `audio_energy`).
- Footage where every frame looks identical to SigLIP (egocentric kitchen
  scenes), where redundancy collapses the signal.

# Behaviour

Encodes the question (and optional negative concepts) with SigLIP/CLIP and
computes per-frame cosine similarity; an optional contrastive variant scores
`sim(positive) - sim(negative)`. A cache-aware variant
(`semantic_siglip_cached`) reuses offline keyframe embeddings for a single text
forward + dot product at query time.

Registered via `evidence_skills.SemanticSigLIPSkill`, wrapped by
`cascade/aligners/adapter.py::EvidenceSkillAligner`.
