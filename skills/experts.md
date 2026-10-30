# Expert pool

The router's final axis selects a **modality-matched reasoner** from a
categorical over four experts. The categorical and its dataset-level priors are
defined in [`router/planner.py`](../router/planner.py); the cascade planner's
`expected_expert` (`cascade/planner.py`) defaults to `vlm` and switches to
`avlm` for speech-style questions on `video_mme` / `music_avqa`.

> Routing is **dataset-/policy-level**, not a per-question decision made by the
> cascade planner. The skill cards above describe per-question evidence skills;
> the expert is chosen by the routing policy that matches the question's
> dominant modality.

## vlm

Frames-only vision-language model (e.g. Qwen2-VL). The default expert for
visual questions; the "audio-blind" baseline in audio-visual comparisons.

## alm

Audio-language model. Selected when the answer is carried almost entirely by the
audio stream (sound / music with little visual grounding). Example dataset prior:
`{vlm: 0.10, alm: 0.70, avlm: 0.15, caplm: 0.05}`.

## avlm

Audio-visual language model (e.g. VideoLLaMA2-AV). Ingests both the selected
frames and the raw audio track, so it can answer questions that need vision +
audio together (e.g. "is the violin on the left louder than the cello?").
Example dataset prior: `{vlm: 0.10, alm: 0.20, avlm: 0.65, caplm: 0.05}`.

## caplm

Caption-conditioned language model: reasons over generated captions / textual
summaries rather than raw pixels. Example dataset prior:
`{vlm: 0.30, alm: 0.05, avlm: 0.05, caplm: 0.60}`.
