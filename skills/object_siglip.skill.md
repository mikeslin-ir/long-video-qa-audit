---
name: object_siglip
tier: L3_aligner
modality: vision
cost: 1.2
hint: "visual objects; helps for color/holding/wearing/position questions"
output: per_frame_score
requires: [text_query]
evidence_class: SemanticEvidence
---

# When to use

Questions about **specific objects and their attributes**: colour, what someone
is holding or wearing, or spatial position ("on the left", "in the middle").
More targeted than `semantic_siglip` for object-grounded queries.

# When NOT to use

Generic scene/event questions where the cheaper `semantic_siglip` already
suffices — `object_siglip` is the most expensive aligner (1.2).

# Behaviour

Encodes object-centric queries with SigLIP and scores per-frame similarity to
the named objects/attributes.

Registered via `evidence_skills.ObjectSigLIPSkill`, wrapped by
`cascade/aligners/adapter.py::EvidenceSkillAligner`.
