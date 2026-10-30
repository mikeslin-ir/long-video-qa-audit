---
name: asr_whisper
tier: L3_aligner
modality: audio
cost: 0.3
hint: "cached; helps for what was SAID / dialog / narration"
output: per_frame_score
evidence_class: SpeechEvidence
---

# When to use

Questions about **what was said**: dialog, narration, spoken answers. Aligns the
transcript to the timeline and ranks frames near the relevant utterance. On
`video_mme` / `music_avqa` speech-style questions the planner also routes toward
the audio-visual expert.

# When NOT to use

Non-speech audio (music, instruments, ambient sound) — use `clap_alm` or
`audio_energy` instead.

# Behaviour

Transcribes the audio track with Whisper (cached) and scores frames by temporal
proximity / relevance to question-matching transcript segments; transcript text
can be injected into the prompt.

Registered via `evidence_skills.ASRWhisperSkill`, wrapped by
`cascade/aligners/adapter.py::EvidenceSkillAligner`.
