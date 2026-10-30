---
name: audio_energy
tier: L3_aligner
modality: audio
cost: 0.1
hint: "cheap acoustic salience; helps locate loud events: screams, gunshots, crashes, applause in event-sparse video"
output: per_frame_score
evidence_class: AudioEvidence
---

# When to use

Locating **loud, sparse acoustic events**: screams, gunshots, crashes, applause,
music onsets — especially in event-sparse anomaly QA where the salient moment is
acoustic. Cheapest aligner (0.1); complements `clap_alm` and `asr_whisper`.

# When NOT to use

Questions that need *semantic* audio understanding (which instrument, what was
said) rather than raw salience — pair with or defer to `clap_alm` / `asr_whisper`.

# Behaviour

Computes short-time audio energy (RMS envelope) and scores frames by acoustic
salience over time, peaking at loud events.

Registered via `evidence_skills.AudioEnergySkill`, wrapped by
`cascade/aligners/adapter.py::EvidenceSkillAligner`.
