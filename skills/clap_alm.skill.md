---
name: clap_alm
tier: L3_aligner
modality: audio
cost: 0.2
hint: "cached audio-language; helps for MUSIC / sound / instrument / which-sounds questions where the answer is acoustic, not spoken words"
output: per_frame_score
evidence_class: AudioLanguageEvidence
---

# When to use

Audio-dependent questions where the answer is **acoustic but not spoken**:
music, sound, which instrument is playing / louder, which sound occurs. The key
signal for datasets like MUSIC-AVQA where neither ASR text nor pixels expose the
answer.

# When NOT to use

Spoken-content questions (use `asr_whisper`) or purely visual questions.

# Behaviour

Question-conditioned audio cueing from pre-extracted CLAP embeddings: one text
forward + cosine similarity at query time (cheap, cached). Scores audio segments
by relevance to the question and maps them back to frames.

Registered via `evidence_skills.CLAPSkill`, wrapped by
`cascade/aligners/adapter.py::EvidenceSkillAligner`.
