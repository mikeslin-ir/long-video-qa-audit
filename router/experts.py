"""Heterogeneous expert pool for Agentic Multimodal Routing.

A multimodal expert is a thin wrapper that takes some subset of
(frames, audio waveform, captions, ASR text, OCR text, question) and
produces an answer distribution over a finite answer set, plus an
estimated cost (GPU-seconds).

Concrete experts implemented here are *stubs* by default: they return
deterministic answers based on cheap heuristics, so the full routing
pipeline runs end-to-end without GPU dependencies.  Real backbones can
be plugged in by overriding `_infer()` in subclasses.

Design notes
------------
- Inference path returns `ExpertOutput` with both an answer string
  and (when available) per-token / per-option log-probabilities so
  that the counterfactual reward
  ``log P_e(a* | ev) - log P_e(a* | ev')``
  is computable without re-running the expert.
- Each expert declares which `modalities` it consumes; the router
  uses this for action masking (e.g. ALM is only valid when audio is
  enabled in the routing action).
- `cost_class` is a coarse cost label ('cheap', 'mid', 'heavy') used
  by the planner's budget filter.

Available experts
-----------------
- `VLMExpert`            (vision+text)
- `ALMExpert`            (audio+text)
- `AudioVisualLMExpert`  (vision+audio+text)
- `CaptionLLMExpert`     (text-only over captions/OCR/ASR)
"""
from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------- IO contract ----------------------------------------------------

@dataclass
class ExpertEvidence:
    """Bundle of evidence routed to a single expert."""
    question: str
    frames: list[Any] | None = None        # list[FrameMeta] or list[np.ndarray]
    audio: np.ndarray | None = None        # waveform, mono float32
    sample_rate: int | None = None
    captions: list[str] | None = None
    asr_text: str | None = None
    ocr_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpertOutput:
    """Expert prediction plus log-probability bookkeeping."""
    answer: str
    log_prob: float | None = None              # log P_e(answer | ev)
    option_logprobs: dict[str, float] | None = None   # for MCQ
    expert_name: str = ""
    cost_sec: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)


# ---------- abstract base --------------------------------------------------

class Expert(ABC):
    """Abstract multimodal expert.

    Subclasses implement `_infer(evidence)`; the public `infer()`
    wraps it with timing and logging.
    """
    name: str = "expert"
    modalities: tuple[str, ...] = ()           # subset of {'video','audio','text'}
    cost_class: str = "heavy"                  # 'cheap'|'mid'|'heavy'

    def can_handle(self, evidence: ExpertEvidence) -> bool:
        if "video" in self.modalities and not evidence.frames:
            return False
        if "audio" in self.modalities and evidence.audio is None:
            return False
        return True

    def infer(self, evidence: ExpertEvidence) -> ExpertOutput:
        t0 = time.time()
        out = self._infer(evidence)
        out.expert_name = self.name
        out.cost_sec = time.time() - t0
        return out

    def logprob(self, evidence: ExpertEvidence, answer: str) -> float:
        """Return log P_e(answer | evidence).

        Default implementation re-runs `_infer` and returns the
        option_logprobs[answer] when present, otherwise a calibrated
        proxy in [-1, 0] derived from match strength. Subclasses with
        true language-model log-prob should override.
        """
        out = self._infer(evidence)
        if out.option_logprobs and answer in out.option_logprobs:
            return float(out.option_logprobs[answer])
        if out.log_prob is not None:
            return float(out.log_prob if out.answer == answer else out.log_prob - 1.0)
        # fallback: rough proxy
        match = 1.0 if (out.answer or "").strip().lower() == answer.strip().lower() else 0.0
        return float(np.log(0.05 + 0.9 * match))

    @abstractmethod
    def _infer(self, evidence: ExpertEvidence) -> ExpertOutput: ...


# ---------- helpers --------------------------------------------------------

def _stable_seed(text: str) -> int:
    return int(hashlib.md5((text or "").encode("utf-8")).hexdigest(), 16) % (2**31 - 1)


def _stub_answer(question: str, modality_hint: str = "") -> str:
    """Deterministic placeholder answer for the no-GPU code path."""
    rng = np.random.default_rng(_stable_seed(question + "|" + modality_hint))
    options = ["yes", "no", "unknown"]
    return options[int(rng.integers(0, len(options)))]


# ---------- concrete stub experts -----------------------------------------

class VLMExpert(Expert):
    """Vision-language reasoner stub. Override `_infer` to plug in
    Qwen2-VL / LLaVA / GPT-4V."""
    name = "qwen2_vl"
    modalities = ("video",)
    cost_class = "heavy"

    def __init__(self, backbone: str = "Qwen2-VL-7B", **kw: Any) -> None:
        self.backbone = backbone
        self.kw = kw

    def _infer(self, evidence: ExpertEvidence) -> ExpertOutput:
        # Stub: pretend to see the frames.
        n = len(evidence.frames or [])
        ans = _stub_answer(evidence.question, modality_hint=f"vlm:{n}")
        return ExpertOutput(answer=ans, log_prob=float(np.log(0.34)),
                            extras={"backbone": self.backbone, "num_frames": n})


class ALMExpert(Expert):
    """Audio-language reasoner stub. Override to plug in Qwen-Audio /
    Pengi."""
    name = "qwen_audio"
    modalities = ("audio",)
    cost_class = "mid"

    def __init__(self, backbone: str = "Qwen-Audio", **kw: Any) -> None:
        self.backbone = backbone
        self.kw = kw

    def _infer(self, evidence: ExpertEvidence) -> ExpertOutput:
        n = 0 if evidence.audio is None else int(evidence.audio.size)
        ans = _stub_answer(evidence.question, modality_hint=f"alm:{n}")
        return ExpertOutput(answer=ans, log_prob=float(np.log(0.34)),
                            extras={"backbone": self.backbone, "audio_samples": n})


class AudioVisualLMExpert(Expert):
    """Audio-visual LLM stub (e.g., VideoLLaMA-2)."""
    name = "videollama2"
    modalities = ("video", "audio")
    cost_class = "heavy"

    def __init__(self, backbone: str = "VideoLLaMA-2", **kw: Any) -> None:
        self.backbone = backbone
        self.kw = kw

    def _infer(self, evidence: ExpertEvidence) -> ExpertOutput:
        nf = len(evidence.frames or [])
        na = 0 if evidence.audio is None else int(evidence.audio.size)
        ans = _stub_answer(evidence.question, modality_hint=f"avlm:{nf}:{na}")
        return ExpertOutput(answer=ans, log_prob=float(np.log(0.34)),
                            extras={"backbone": self.backbone,
                                    "num_frames": nf, "audio_samples": na})


class CaptionLLMExpert(Expert):
    """Text-only LLM over captions / OCR / ASR. Cheapest option."""
    name = "cap_llm"
    modalities = ()                            # no raw modalities
    cost_class = "cheap"

    def __init__(self, backbone: str = "Qwen-LLM", **kw: Any) -> None:
        self.backbone = backbone
        self.kw = kw

    def _infer(self, evidence: ExpertEvidence) -> ExpertOutput:
        text_blob = "\n".join(filter(None, [
            "\n".join(evidence.captions or []),
            evidence.asr_text or "",
            evidence.ocr_text or "",
        ]))
        ans = _stub_answer(evidence.question + "|" + text_blob[:200],
                           modality_hint="caplm")
        return ExpertOutput(answer=ans, log_prob=float(np.log(0.34)),
                            extras={"backbone": self.backbone,
                                    "evidence_len": len(text_blob)})


# ---------- registry & default pool ---------------------------------------

EXPERT_REGISTRY = {
    "vlm":   VLMExpert,
    "alm":   ALMExpert,
    "avlm":  AudioVisualLMExpert,
    "caplm": CaptionLLMExpert,
}


def make_expert(name: str, **kw: Any) -> Expert:
    if name not in EXPERT_REGISTRY:
        raise KeyError(f"unknown expert: {name}; available={list(EXPERT_REGISTRY)}")
    return EXPERT_REGISTRY[name](**kw)


def default_pool() -> dict[str, Expert]:
    """Default heterogeneous expert pool (all stubs by default)."""
    return {name: cls() for name, cls in EXPERT_REGISTRY.items()}
