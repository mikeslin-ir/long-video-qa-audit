from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    t = text.lower()
    return any(w in t for w in words)


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    s = sum(max(v, 0.0) for v in weights.values())
    if s <= 0:
        n = len(weights)
        return {k: 1.0 / max(n, 1) for k in weights}
    return {k: max(v, 0.0) / s for k, v in weights.items()}


def _use_stub_skills(config: dict[str, Any] | None) -> bool:
    return bool((config or {}).get("use_stub_skills", False))


def _expert_choice(question: str, dataset: str) -> tuple[str, dict[str, float]]:
    """Rule-based expert categorical, returned as (top_expert, beta_dist).

    The categorical is over: 'vlm', 'alm', 'avlm', 'caplm'.
    """
    q = (question or "").lower()
    ds = (dataset or "").lower()
    audio_words = ("scream", "sound", "music", "song", "voice", "noise",
                   "loud", "audio", "speech", "speak", "say", "talk")
    text_words = ("text", "sign", "subtitle", "caption", "license plate",
                  "writing", "letter")
    av_words = ("when does", "what does the person say while",
                "before/after", "music while", "shouted", "screamed")

    if _contains_any(q, av_words) or ds in {"music_avqa"}:
        beta = {"vlm": 0.10, "alm": 0.20, "avlm": 0.65, "caplm": 0.05}
        return "avlm", beta
    if _contains_any(q, audio_words) and not _contains_any(q, text_words):
        beta = {"vlm": 0.10, "alm": 0.70, "avlm": 0.15, "caplm": 0.05}
        return "alm", beta
    if _contains_any(q, text_words):
        beta = {"vlm": 0.30, "alm": 0.05, "avlm": 0.05, "caplm": 0.60}
        return "caplm", beta
    if ds in {"egoschema", "video_mme", "videomme"}:
        beta = {"vlm": 0.65, "alm": 0.05, "avlm": 0.20, "caplm": 0.10}
        return "vlm", beta
    if ds in {"xd_violence", "ucf_crime"}:
        beta = {"vlm": 0.45, "alm": 0.20, "avlm": 0.30, "caplm": 0.05}
        return "vlm", beta
    # Default: VLM-leaning short-video QA.
    beta = {"vlm": 0.70, "alm": 0.05, "avlm": 0.10, "caplm": 0.15}
    return "vlm", beta


@dataclass
class RuleBasedSkillPlanner:
    """Training-free question-conditioned skill planner."""

    def plan(
        self,
        question: str,
        dataset: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        q = (question or "").lower()
        ds = (dataset or "").lower()

        violence_words = (
            "violence", "violent", "fight", "fighting", "assault", "shoot", "explosion", "accident"
        )
        steal_words = (
            "steal", "stealing", "shoplift", "robbery", "arrest", "handcuff", "interaction"
        )
        speech_words = ("say", "speak", "talk", "dialogue")
        text_words = ("text", "sign", "subtitle", "license plate")

        selected: list[str] = []
        weights: dict[str, float] = {}
        prompts: list[str] = []
        reason = ""
        use_stubs = _use_stub_skills(config)

        if _contains_any(q, violence_words):
            selected = ["motion", "audio", "semantic_siglip"]
            weights = {"motion": 0.33, "audio": 0.33, "semantic_siglip": 0.34}
            prompts = ["a person fighting", "explosion", "accident", "weapon"]
            reason = "violence-related keywords"
        elif _contains_any(q, steal_words):
            selected = ["semantic_siglip", "object_siglip", "motion"]
            weights = {"semantic_siglip": 0.45, "object_siglip": 0.25, "motion": 0.3}
            prompts = ["a person stealing", "shoplifting", "person interaction"]
            reason = "crime interaction keywords"
        elif _contains_any(q, speech_words):
            if use_stubs:
                selected = ["asr_stub", "audio", "semantic_siglip"]
                weights = {"asr_stub": 0.2, "audio": 0.4, "semantic_siglip": 0.4}
            else:
                selected = ["asr_whisper", "audio", "semantic_siglip"]
                weights = {"asr_whisper": 0.3, "audio": 0.3, "semantic_siglip": 0.4}
            prompts = ["speaking person", "dialogue scene"]
            reason = "speech keywords"
        elif _contains_any(q, text_words):
            if use_stubs:
                selected = ["ocr_stub", "semantic_siglip"]
                weights = {"ocr_stub": 0.4, "semantic_siglip": 0.6}
            else:
                selected = ["ocr_text", "semantic_siglip"]
                weights = {"ocr_text": 0.45, "semantic_siglip": 0.55}
            prompts = ["text sign", "subtitle", "license plate"]
            reason = "text/OCR keywords"
        elif ds == "xd_violence":
            selected = ["motion", "audio", "semantic_siglip"]
            weights = {"motion": 0.3, "audio": 0.3, "semantic_siglip": 0.4}
            prompts = ["abnormal criminal event", "violent activity"]
            reason = "dataset default: xd_violence"
        elif ds == "ucf_crime":
            selected = ["motion", "semantic_siglip"]
            weights = {"motion": 0.4, "semantic_siglip": 0.6}
            prompts = ["crime event", "suspicious behavior"]
            reason = "dataset default: ucf_crime"
        elif ds in {"egoschema", "videoqa", "general"}:
            selected = ["semantic_siglip"]
            weights = {"semantic_siglip": 0.7}
            reason = "dataset default: long-video QA"
        elif ds in {"msrvtt", "msrvtt_qa", "msvd", "msvd_qa"}:
            selected = ["semantic_siglip", "motion"]
            weights = {"semantic_siglip": 0.7, "motion": 0.3}
            reason = "dataset default: short-clip captioning/QA"
        elif ds in {"music_avqa", "musicavqa"}:
            # AV question answering over music videos: pre-extracted CLAP
            # gives a question-audio relevance prior; ASR catches lyrics
            # and vocals; semantic_siglip anchors visual instruments.
            if use_stubs:
                selected = ["semantic_siglip", "asr_stub", "audio"]
                weights = {"semantic_siglip": 0.4, "asr_stub": 0.3, "audio": 0.3}
            else:
                selected = ["clap_alm", "asr_whisper", "audio", "semantic_siglip"]
                weights = {"clap_alm": 0.35, "asr_whisper": 0.2, "audio": 0.15, "semantic_siglip": 0.3}
            prompts = ["musical instrument", "person playing music"]
            reason = "dataset default: MUSIC-AVQA audio-visual QA"
        elif ds in {"egotextvqa",}:
            # Egocentric scene-text: OCR drives, semantic anchors what's visible.
            if use_stubs:
                selected = ["ocr_stub", "semantic_siglip"]
                weights = {"ocr_stub": 0.55, "semantic_siglip": 0.45}
            else:
                selected = ["ocr_text", "semantic_siglip"]
                weights = {"ocr_text": 0.6, "semantic_siglip": 0.4}
            prompts = ["sign with text", "price tag", "product label"]
            reason = "dataset default: EgoTextVQA scene-text QA"
        elif ds in {"tutorialvqa",}:
            # Software screencast: ASR is gold (official transcripts
            # prewarmed via scripts/prewarm_tutorialvqa_transcripts.py).
            if use_stubs:
                selected = ["asr_stub", "semantic_siglip"]
                weights = {"asr_stub": 0.6, "semantic_siglip": 0.4}
            else:
                selected = ["asr_whisper", "semantic_siglip"]
                weights = {"asr_whisper": 0.65, "semantic_siglip": 0.35}
            prompts = ["software interface", "menu click", "dialog window"]
            reason = "dataset default: TutorialVQA screencast QA"
        elif ds in {"tvqa",}:
            # Text-only routing: pilot videos are not on disk; the
            # injection path falls back to provided subtitles.
            selected = ["semantic_siglip"]
            weights = {"semantic_siglip": 1.0}
            reason = "dataset default: TVQA (subtitle-only injection)"
        elif ds in {"video_mme", "videomme"}:
            if use_stubs:
                selected = ["semantic_siglip", "asr_stub", "motion"]
                weights = {"semantic_siglip": 0.5, "asr_stub": 0.2, "motion": 0.3}
            else:
                selected = ["semantic_siglip", "asr_whisper", "motion"]
                weights = {"semantic_siglip": 0.5, "asr_whisper": 0.2, "motion": 0.3}
            reason = "dataset default: Video-MME long multimodal QA"
        else:
            selected = ["motion", "semantic_siglip"]
            weights = {"motion": 0.5, "semantic_siglip": 0.5}
            reason = "generic fallback"

        expert, beta = _expert_choice(question, dataset)
        return {
            "selected_skills": selected,
            "weights": _normalize_weights(weights),
            "evidence_prompts": prompts,
            "reason": reason,
            "lambda_uniform": float((config or {}).get("lambda_uniform", 0.0)),
            "expert": expert,
            "expert_beta": beta,
        }


_AVAILABLE_SKILLS_DESC = (
    "motion: frame-difference motion intensity (cheap, no question awareness).\n"
    "audio: short-time audio energy (requires audio track).\n"
    "clap_alm: question-conditioned LAION-CLAP audio-language similarity (clip-level, requires pre-extracted CLAP embed).\n"
    "semantic_clip: CLIP ViT-B/32 frame-question similarity.\n"
    "semantic_siglip: SigLIP base/16 frame-question similarity (default semantic skill).\n"
    "object_siglip: SigLIP open-vocabulary object detection from prompts.\n"
    "asr_whisper: Whisper transcription + question keyword matching (needs audio).\n"
    "ocr_text: easyocr/tesseract text-in-frame + question keyword matching.\n"
)


_LLM_PROMPT_TEMPLATE = """You are a planner that picks evidence skills to find video frames most relevant to a question.

Available skills:
{skills}

Dataset: {dataset}
Question: {question}

Return ONLY a single JSON object on one line, no prose, no code fence, with keys:
- selected_skills: list of skill names from the set above (1-4 entries).
- weights: dict mapping each selected skill to a non-negative float; will be re-normalized.
- evidence_prompts: list of short natural-language phrases for the semantic/object skills (may be empty).
- reason: one short sentence explaining the choice.

Constraints:
- Do not include audio/asr_whisper if the question is clearly visual-only.
- Prefer semantic_siglip over semantic_clip unless the question is very generic.
- Use ocr_text only when the question mentions text, signs, license plates, or subtitles.
"""


def _http_post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    import json as _json
    import urllib.request
    req = urllib.request.Request(
        url,
        data=_json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return _json.loads(resp.read().decode("utf-8"))


def _extract_json_object(text: str) -> dict[str, Any] | None:
    import json as _json
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return _json.loads(s[start : end + 1])
    except Exception:
        return None


@dataclass
class LLMPlanner:
    """Question-conditioned planner backed by an LLM (Anthropic or OpenAI).

    Provider/model/key are read from `config["llm_planner"]` or environment variables.
    Falls back to RuleBasedSkillPlanner on any failure (missing key, network, parse error).
    """

    fallback: RuleBasedSkillPlanner = field(default_factory=RuleBasedSkillPlanner)

    def _resolve(self, config: dict[str, Any] | None) -> tuple[str, str, str | None]:
        import os
        cfg = (config or {}).get("llm_planner", {}) if config else {}
        provider = (cfg.get("provider") or os.environ.get("LLM_PLANNER_PROVIDER") or "anthropic").lower()
        if provider == "anthropic":
            model = cfg.get("model") or os.environ.get("LLM_PLANNER_MODEL") or "claude-3-5-sonnet-latest"
            key = cfg.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        elif provider == "openai":
            model = cfg.get("model") or os.environ.get("LLM_PLANNER_MODEL") or "gpt-4o-mini"
            key = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
        return provider, model, key

    def _call_llm(self, provider: str, model: str, key: str, prompt: str) -> str:
        if provider == "anthropic":
            data = _http_post_json(
                "https://api.anthropic.com/v1/messages",
                {"x-api-key": key, "anthropic-version": "2023-06-01"},
                {
                    "model": model,
                    "max_tokens": 512,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            parts = data.get("content") or []
            for p in parts:
                if p.get("type") == "text":
                    return p.get("text", "")
            return ""
        # openai
        data = _http_post_json(
            "https://api.openai.com/v1/chat/completions",
            {"Authorization": f"Bearer {key}"},
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 512,
            },
        )
        choices = data.get("choices") or []
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "")

    def plan(
        self,
        question: str,
        dataset: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            provider, model, key = self._resolve(config)
        except Exception as e:  # bad provider name
            out = self.fallback.plan(question=question, dataset=dataset, config=config)
            out["reason"] = f"[LLMPlanner config error: {e}] {out['reason']}"
            return out

        if not key:
            out = self.fallback.plan(question=question, dataset=dataset, config=config)
            out["reason"] = f"[LLMPlanner: no API key for {provider}, fallback] {out['reason']}"
            return out

        prompt = _LLM_PROMPT_TEMPLATE.format(
            skills=_AVAILABLE_SKILLS_DESC,
            dataset=dataset or "unknown",
            question=question or "",
        )
        try:
            raw = self._call_llm(provider, model, key, prompt)
            obj = _extract_json_object(raw)
            if not obj or "selected_skills" not in obj:
                raise ValueError("LLM did not return parseable JSON plan")
            selected = [str(s) for s in obj.get("selected_skills") or []]
            if not selected:
                raise ValueError("LLM returned empty selected_skills")
            weights_in = obj.get("weights") or {s: 1.0 for s in selected}
            weights = {s: float(weights_in.get(s, 1.0)) for s in selected}
            prompts = [str(p) for p in (obj.get("evidence_prompts") or [])]
            reason = str(obj.get("reason") or "LLM-selected skills")
            return {
                "selected_skills": selected,
                "weights": _normalize_weights(weights),
                "evidence_prompts": prompts,
                "reason": f"[LLM:{provider}/{model}] {reason}",
                "lambda_uniform": float((config or {}).get("lambda_uniform", 0.0)),
            }
        except Exception as e:
            out = self.fallback.plan(question=question, dataset=dataset, config=config)
            out["reason"] = f"[LLMPlanner error: {e}, fallback] {out['reason']}"
            return out
