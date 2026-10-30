---
name: ocr_text
tier: L3_aligner
modality: vision
cost: 0.4
hint: "helps for text / signs / subtitles / license plates"
output: per_frame_score
evidence_class: TextEvidence
---

# When to use

Questions whose answer is **written in the frame**: on-screen text, signs,
subtitles, license plates, scoreboards, slides. Ranks frames by the presence and
relevance of recognised text.

# When NOT to use

Footage with no salient text, or when the text is incidental to the question —
it then adds cost (0.4) without signal.

# Behaviour

Runs OCR on candidate frames and scores them by how well the recognised text
matches the question. Can also inject recognised text into the prompt.

Registered via `evidence_skills.OCRTextSkill`, wrapped by
`cascade/aligners/adapter.py::EvidenceSkillAligner`.
