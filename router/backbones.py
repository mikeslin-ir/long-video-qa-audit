"""Real multimodal expert backbones for Agentic Multimodal Routing.

These subclass the stub `Expert` classes in `router.experts` and override
`_infer()` / `logprob()` with actual model forward passes. Heavy imports
(`torch`, `transformers`) are done lazily inside `load()` so that importing
this module is cheap and the stub pipeline still runs on machines without a
GPU.

Backbones
---------
- `Qwen2VLExpert`        : vision-language (Qwen2-VL / Qwen2.5-VL).
- `CaptionLLMRealExpert` : text-only LLM over captions/ASR/OCR (Qwen2.5).

Both expose:
- `infer(evidence)`               -> ExpertOutput (free-form answer)
- `answer_mcq(evidence, options)` -> (raw, letter)
- `option_logprobs(evidence, options)` -> {letter: logprob}   (forced decode)

The forced-decoding option log-probabilities are what make the
expert-counterfactual reward  log P_e(a*|ev) - log P_e(a*|ev')  computable.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from router.experts import (
    AudioVisualLMExpert,
    CaptionLLMExpert,
    Expert,
    ExpertEvidence,
    ExpertOutput,
    VLMExpert,
)

_LETTERS = "ABCDE"
_LETTER_RE = re.compile(r"\b([A-E])\b")

# Injected ASR/OCR text is appended after the question, options and answer instruction (what
# every audited run used). ``set_evidence_position`` moves the same block elsewhere as a
# placement ablation (implemented for the Qwen VLM backbones). The prompt is
#     Question / Options / A.. / Answer-with-the-single-letter instruction
# and the positions are: "before" (ahead of everything), "question" (after the question,
# before the options), "instruction" (after the options, before the answer instruction) and
# "after" (the audited default).
_EVIDENCE_MARKER = ("\n\nAdditional evidence extracted from the video "
                    "(treat as context, not ground truth):\n")
EVIDENCE_POSITIONS = ("after", "before", "question", "instruction")
_EVIDENCE_POSITION = "after"


def set_evidence_position(position: str) -> None:
    global _EVIDENCE_POSITION
    if position not in EVIDENCE_POSITIONS:
        raise ValueError(f"evidence position must be one of {EVIDENCE_POSITIONS}, got {position!r}")
    _EVIDENCE_POSITION = position


def place_evidence(prompt: str) -> str:
    """Return ``prompt`` unchanged ('after'), or with its evidence block moved to the configured place."""
    if _EVIDENCE_POSITION == "after" or _EVIDENCE_MARKER not in prompt:
        return prompt
    body, _, evidence = prompt.partition(_EVIDENCE_MARKER)
    block = _EVIDENCE_MARKER.strip("\n") + "\n" + evidence.strip("\n")
    if _EVIDENCE_POSITION == "before":
        return block + "\n\n" + body
    if _EVIDENCE_POSITION == "question":
        head, sep, rest = body.partition("\nOptions:")
        return head + "\n" + block + "\n" + sep.lstrip("\n") + rest if sep else block + "\n\n" + body
    head, sep, instruction = body.rpartition("\nAnswer with")
    return head + "\n" + block + "\n" + sep.lstrip("\n") + instruction if sep else prompt


def parse_letter(text: str, n_options: int) -> str:
    """Extract the first A..E choice letter from model output ('' if none)."""
    if not text:
        return ""
    t = text.strip().upper()
    first = t.lstrip("(*[ \t\n").lstrip()
    if first and first[0] in _LETTERS[:n_options]:
        return first[0]
    m = _LETTER_RE.search(t)
    if m and m.group(1) in _LETTERS[:n_options]:
        return m.group(1)
    return ""


def build_mcq_prompt(question: str, options: list[str]) -> str:
    """Render an MCQ prompt with letters A..E. `options` may or may not
    already carry a leading 'A. ' prefix."""
    n = len(options)
    letters = _LETTERS[:n]
    lines = [f"Question: {question}", "Options:"]
    for letter, opt in zip(letters, options):
        clean = re.sub(rf"^\s*{letter}[\.\):]\s*", "", str(opt)).strip()
        lines.append(f"{letter}. {clean}")
    lines.append(
        f"Answer with the single letter ({'/'.join(letters)}) of the best "
        f"option, and nothing else."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
#  Vision-language expert (Qwen2-VL / Qwen2.5-VL)
# --------------------------------------------------------------------------

class Qwen2VLExpert(VLMExpert):
    """Real Qwen2-VL / Qwen2.5-VL vision-language reasoner."""

    name = "qwen2_vl"
    modalities = ("video",)
    cost_class = "heavy"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 32,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
        **kw: Any,
    ) -> None:
        super().__init__(backbone=model_id, **kw)
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.model = None
        self.processor = None
        # The answer letter comes from generation; the extra forward pass for option
        # log-probabilities only feeds expert-utility bookkeeping. Skipping it leaves every
        # prediction unchanged and removes about a third of the per-item time.
        self.compute_option_logprobs = True

    # -- model lifecycle ---------------------------------------------------
    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor

        is25 = "2.5" in self.model_id or "2_5" in self.model_id
        if is25:
            from transformers import Qwen2_5_VLForConditionalGeneration as _VLM
        else:
            from transformers import Qwen2VLForConditionalGeneration as _VLM

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = _VLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else self.device,
            local_files_only=True,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            local_files_only=True,
        )

    # -- helpers -----------------------------------------------------------
    def _build_inputs(self, frames: list[np.ndarray], prompt: str):
        from PIL import Image

        prompt = place_evidence(prompt)
        pil = [Image.fromarray(np.asarray(f).astype(np.uint8)) for f in frames]
        content = [{"type": "image", "image": im} for im in pil]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            )
        except Exception:
            inputs = self.processor(
                text=[text], images=pil, padding=True, return_tensors="pt",
            )
        return {k: v.to(self.model.device) if hasattr(v, "to") else v
                for k, v in inputs.items()}

    def _generate(self, frames: list[np.ndarray], prompt: str) -> str:
        import torch

        inputs = self._build_inputs(frames, prompt)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        in_len = inputs["input_ids"].shape[1]
        gen = out[:, in_len:]
        return self.processor.batch_decode(
            gen, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    # -- public API --------------------------------------------------------
    def _infer(self, evidence: ExpertEvidence) -> ExpertOutput:
        self.load()
        frames = evidence.frames or []
        if not frames:
            return ExpertOutput(answer="", log_prob=None,
                                extras={"backbone": self.model_id, "num_frames": 0})
        # Optional textual grounding (e.g. ASR/OCR transcripts aligned with
        # the selected frames). Empty string => no change to behavior.
        evidence_text = (
            (evidence.asr_text or "").strip()
            + ("\n\n" + (evidence.ocr_text or "").strip()
               if (evidence.ocr_text or "").strip() else "")
        ).strip()
        evidence_block = (
            f"\n\nAdditional evidence extracted from the video (treat as "
            f"context, not ground truth):\n{evidence_text}\n"
            if evidence_text else ""
        )

        options = evidence.metadata.get("options")
        if options:
            prompt = build_mcq_prompt(evidence.question, options)
            if evidence_block:
                prompt = prompt + evidence_block
            raw = self._generate(frames, prompt)
            letter = parse_letter(raw, len(options))
            olp = (self.option_logprobs(evidence, options)
                   if self.compute_option_logprobs else {})
            return ExpertOutput(
                answer=letter or raw,
                log_prob=olp.get(letter) if letter else None,
                option_logprobs=olp,
                extras={"backbone": self.model_id, "num_frames": len(frames),
                        "raw": raw,
                        "evidence_text_chars": len(evidence_text)},
            )
        prompt = (
            f"These are sampled frames from a video.\nQuestion: "
            f"{evidence.question}\nAnswer with a single word or short phrase."
        )
        if evidence_block:
            prompt = prompt + evidence_block
        raw = self._generate(frames, prompt)
        return ExpertOutput(answer=raw, log_prob=None,
                            extras={"backbone": self.model_id,
                                    "num_frames": len(frames),
                                    "evidence_text_chars": len(evidence_text)})

    def option_logprobs(self, evidence: ExpertEvidence,
                        options: list[str]) -> dict[str, float]:
        """Forced-decoding log-prob of each choice letter under the model.

        We score the single answer token 'A'..'E' conditioned on the MCQ
        prompt. This gives a calibrated distribution over choices and is the
        quantity used in the expert-counterfactual reward.
        """
        import torch
        import torch.nn.functional as F

        self.load()
        frames = evidence.frames or []
        n = len(options)
        letters = _LETTERS[:n]
        prompt = build_mcq_prompt(evidence.question, options)
        evidence_text = (
            (evidence.asr_text or "").strip()
            + ("\n\n" + (evidence.ocr_text or "").strip()
               if (evidence.ocr_text or "").strip() else "")
        ).strip()
        if evidence_text:
            prompt = (prompt + "\n\nAdditional evidence extracted from the "
                      "video (treat as context, not ground truth):\n"
                      + evidence_text + "\n")
        inputs = self._build_inputs(frames, prompt)
        with torch.no_grad():
            out = self.model(**inputs)
        next_logits = out.logits[0, -1, :].float()
        logprobs = F.log_softmax(next_logits, dim=-1)
        result: dict[str, float] = {}
        for L in letters:
            ids = self.processor.tokenizer.encode(L, add_special_tokens=False)
            if not ids:
                continue
            result[L] = float(logprobs[ids[0]].item())
        return result

    def answer_mcq(self, evidence: ExpertEvidence,
                   options: list[str]) -> tuple[str, str]:
        self.load()
        prompt = build_mcq_prompt(evidence.question, options)
        evidence_text = (
            (evidence.asr_text or "").strip()
            + ("\n\n" + (evidence.ocr_text or "").strip()
               if (evidence.ocr_text or "").strip() else "")
        ).strip()
        if evidence_text:
            prompt = (prompt + "\n\nAdditional evidence extracted from the "
                      "video (treat as context, not ground truth):\n"
                      + evidence_text + "\n")
        raw = self._generate(evidence.frames or [], prompt)
        return raw, parse_letter(raw, len(options))


# --------------------------------------------------------------------------
#  Vision-language expert (LLaVA-Video-7B-Qwen2)
# --------------------------------------------------------------------------

class LLaVAVideoExpert(VLMExpert):
    """LLaVA-Video-7B-Qwen2 vision-language reasoner (lmms-lab).

    Used as a cross-VLM-backbone sanity check for VideoRouteAudit findings. Requires
    the `llava` package from https://github.com/LLaVA-VL/LLaVA-NeXT
    installed on PYTHONPATH (pip install --no-deps git+https://...).

    The interface mirrors `Qwen2VLExpert` so the harness can swap one
    for the other via `--vlm_backbone`.
    """

    name = "llava_video"
    modalities = ("video",)
    cost_class = "heavy"

    def __init__(
        self,
        model_id: str = "lmms-lab/LLaVA-Video-7B-Qwen2",
        device: str = "cuda",
        max_new_tokens: int = 32,
        max_frames: int = 32,  # LLaVA-Video pads frames to a fixed grid
        **kw: Any,
    ) -> None:
        super().__init__(backbone=model_id, **kw)
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_frames = max_frames
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self._conv_template = "qwen_1_5"

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        try:
            from llava.model.builder import load_pretrained_model
        except ImportError as e:
            raise ImportError(
                "LLaVA-Video requires the lmms-lab `llava` package. "
                "Install: pip install --no-deps "
                "git+https://github.com/LLaVA-VL/LLaVA-NeXT.git"
            ) from e

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            self.model_id, None, "llava_qwen",
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else self.device,
            attn_implementation="sdpa",
        )
        self.model.eval()

    def _prepare_frames(self, frames: list[np.ndarray]) -> Any:
        """Process frames into LLaVA-Video image tensor."""
        import torch
        from PIL import Image

        if len(frames) > self.max_frames:
            idx = np.linspace(0, len(frames) - 1, self.max_frames).astype(int)
            frames = [frames[i] for i in idx]

        pil = [Image.fromarray(np.asarray(f).astype(np.uint8)) for f in frames]
        out = self.image_processor.preprocess(pil, return_tensors="pt")
        return out["pixel_values"].to(self.model.device, dtype=self.model.dtype)

    def _build_text_inputs(self, prompt: str, n_frames: int):
        import torch
        from llava.conversation import conv_templates
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.mm_utils import tokenizer_image_token

        # Repeat the image token once per frame; LLaVA-Video aggregates the
        # frame tokens internally via mean-pool over time.
        image_prefix = (DEFAULT_IMAGE_TOKEN + "\n") * 1
        conv = conv_templates[self._conv_template].copy()
        conv.append_message(conv.roles[0], image_prefix + prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(
            full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.model.device)
        return input_ids

    def _generate(self, frames: list[np.ndarray], prompt: str) -> str:
        import torch

        images = self._prepare_frames(frames)
        input_ids = self._build_text_inputs(prompt, len(frames))
        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                images=[images],
                modalities=["video"],
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        text = self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        return text.strip()

    # -- public API --------------------------------------------------------
    def _infer(self, evidence: ExpertEvidence) -> ExpertOutput:
        self.load()
        frames = evidence.frames or []
        if not frames:
            return ExpertOutput(answer="", log_prob=None,
                                extras={"backbone": self.model_id, "num_frames": 0})

        evidence_text = (
            (evidence.asr_text or "").strip()
            + ("\n\n" + (evidence.ocr_text or "").strip()
               if (evidence.ocr_text or "").strip() else "")
        ).strip()
        evidence_block = (
            f"\n\nAdditional evidence extracted from the video (treat as "
            f"context, not ground truth):\n{evidence_text}\n"
            if evidence_text else ""
        )

        options = evidence.metadata.get("options")
        if options:
            prompt = build_mcq_prompt(evidence.question, options)
            if evidence_block:
                prompt = prompt + evidence_block
            raw = self._generate(frames, prompt)
            letter = parse_letter(raw, len(options))
            return ExpertOutput(
                answer=letter or raw,
                log_prob=None,
                option_logprobs={},  # not implemented for LLaVA-Video yet
                extras={"backbone": self.model_id, "num_frames": len(frames),
                        "raw": raw,
                        "evidence_text_chars": len(evidence_text)},
            )
        prompt = (
            f"These are sampled frames from a video.\nQuestion: "
            f"{evidence.question}\nAnswer with a single word or short phrase."
        )
        if evidence_block:
            prompt = prompt + evidence_block
        raw = self._generate(frames, prompt)
        return ExpertOutput(answer=raw, log_prob=None,
                            extras={"backbone": self.model_id,
                                    "num_frames": len(frames),
                                    "evidence_text_chars": len(evidence_text)})

    def option_logprobs(self, evidence: ExpertEvidence,
                        options: list[str]) -> dict[str, float]:
        # Stub -- harness falls back to generated-letter scoring when
        # option_logprobs is empty.
        return {}

    def answer_mcq(self, evidence: ExpertEvidence,
                   options: list[str]) -> tuple[str, str]:
        self.load()
        prompt = build_mcq_prompt(evidence.question, options)
        evidence_text = (
            (evidence.asr_text or "").strip()
            + ("\n\n" + (evidence.ocr_text or "").strip()
               if (evidence.ocr_text or "").strip() else "")
        ).strip()
        if evidence_text:
            prompt = (prompt + "\n\nAdditional evidence extracted from the "
                      "video (treat as context, not ground truth):\n"
                      + evidence_text + "\n")
        raw = self._generate(evidence.frames or [], prompt)
        return raw, parse_letter(raw, len(options))


# --------------------------------------------------------------------------
#  Vision-language expert (LLaVA-NeXT-Video-7B, HF-native)
# --------------------------------------------------------------------------

class LlavaNextVideoExpert(VLMExpert):
    """LLaVA-NeXT-Video-7B vision-language reasoner (HF-native).

    Cross-*architecture-family* backbone for VideoRouteAudit robustness: unlike the
    Qwen2-VL / Qwen2.5-VL backbones (Qwen2 LLM + custom ViT), this model
    pairs a **Llama-2 7B** language model with a **CLIP** vision encoder,
    loaded through the native ``LlavaNextVideoForConditionalGeneration``
    class in transformers (no third-party ``llava`` package, no version
    pin). The interface mirrors ``Qwen2VLExpert`` so the harness can swap
    one for the other via ``--vlm_backbone llava_next_video``.
    """

    name = "llava_next_video"
    modalities = ("video",)
    cost_class = "heavy"

    def __init__(
        self,
        model_id: str = "llava-hf/LLaVA-NeXT-Video-7B-hf",
        device: str = "cuda",
        max_new_tokens: int = 32,
        max_frames: int = 32,
        **kw: Any,
    ) -> None:
        super().__init__(backbone=model_id, **kw)
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_frames = max_frames
        self.model = None
        self.processor = None

    # -- model lifecycle ---------------------------------------------------
    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import (
            AutoProcessor,
            LlavaNextVideoForConditionalGeneration as _VLM,
        )

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = _VLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else self.device,
            local_files_only=True,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, local_files_only=True,
        )

    # -- helpers -----------------------------------------------------------
    def _subsample(self, frames: list[np.ndarray]) -> np.ndarray:
        """Uniformly subsample to <=max_frames and stack as a video clip."""
        if len(frames) > self.max_frames:
            idx = np.linspace(0, len(frames) - 1, self.max_frames).astype(int)
            frames = [frames[i] for i in idx]
        return np.stack([np.asarray(f).astype(np.uint8) for f in frames])

    def _build_inputs(self, frames: list[np.ndarray], prompt: str):
        clip = self._subsample(frames)
        conversation = [{
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text, videos=clip, padding=True, return_tensors="pt",
        )
        return {k: (v.to(self.model.device) if hasattr(v, "to") else v)
                for k, v in inputs.items()}

    def _generate(self, frames: list[np.ndarray], prompt: str) -> str:
        import torch

        inputs = self._build_inputs(frames, prompt)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        in_len = inputs["input_ids"].shape[1]
        gen = out[:, in_len:]
        return self.processor.batch_decode(
            gen, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def _evidence_text(self, evidence: ExpertEvidence) -> str:
        return (
            (evidence.asr_text or "").strip()
            + ("\n\n" + (evidence.ocr_text or "").strip()
               if (evidence.ocr_text or "").strip() else "")
        ).strip()

    # -- public API --------------------------------------------------------
    def _infer(self, evidence: ExpertEvidence) -> ExpertOutput:
        self.load()
        frames = evidence.frames or []
        if not frames:
            return ExpertOutput(answer="", log_prob=None,
                                extras={"backbone": self.model_id,
                                        "num_frames": 0})

        evidence_text = self._evidence_text(evidence)
        evidence_block = (
            f"\n\nAdditional evidence extracted from the video (treat as "
            f"context, not ground truth):\n{evidence_text}\n"
            if evidence_text else ""
        )

        options = evidence.metadata.get("options")
        if options:
            prompt = build_mcq_prompt(evidence.question, options)
            if evidence_block:
                prompt = prompt + evidence_block
            raw = self._generate(frames, prompt)
            letter = parse_letter(raw, len(options))
            olp = self.option_logprobs(evidence, options)
            return ExpertOutput(
                answer=letter or raw,
                log_prob=olp.get(letter) if letter else None,
                option_logprobs=olp,
                extras={"backbone": self.model_id, "num_frames": len(frames),
                        "raw": raw,
                        "evidence_text_chars": len(evidence_text)},
            )
        prompt = (
            f"These are sampled frames from a video.\nQuestion: "
            f"{evidence.question}\nAnswer with a single word or short phrase."
        )
        if evidence_block:
            prompt = prompt + evidence_block
        raw = self._generate(frames, prompt)
        return ExpertOutput(answer=raw, log_prob=None,
                            extras={"backbone": self.model_id,
                                    "num_frames": len(frames),
                                    "evidence_text_chars": len(evidence_text)})

    def option_logprobs(self, evidence: ExpertEvidence,
                        options: list[str]) -> dict[str, float]:
        """Forced-decoding log-prob of each choice letter (mirrors Qwen2VL)."""
        import torch
        import torch.nn.functional as F

        self.load()
        frames = evidence.frames or []
        n = len(options)
        letters = _LETTERS[:n]
        prompt = build_mcq_prompt(evidence.question, options)
        evidence_text = self._evidence_text(evidence)
        if evidence_text:
            prompt = (prompt + "\n\nAdditional evidence extracted from the "
                      "video (treat as context, not ground truth):\n"
                      + evidence_text + "\n")
        inputs = self._build_inputs(frames, prompt)
        with torch.no_grad():
            out = self.model(**inputs)
        next_logits = out.logits[0, -1, :].float()
        logprobs = F.log_softmax(next_logits, dim=-1)
        result: dict[str, float] = {}
        for L in letters:
            ids = self.processor.tokenizer.encode(L, add_special_tokens=False)
            if not ids:
                continue
            result[L] = float(logprobs[ids[0]].item())
        return result

    def answer_mcq(self, evidence: ExpertEvidence,
                   options: list[str]) -> tuple[str, str]:
        self.load()
        prompt = build_mcq_prompt(evidence.question, options)
        evidence_text = self._evidence_text(evidence)
        if evidence_text:
            prompt = (prompt + "\n\nAdditional evidence extracted from the "
                      "video (treat as context, not ground truth):\n"
                      + evidence_text + "\n")
        raw = self._generate(evidence.frames or [], prompt)
        return raw, parse_letter(raw, len(options))


# --------------------------------------------------------------------------
#  Caption / text-only LLM expert (Qwen2.5)
# --------------------------------------------------------------------------

class CaptionLLMRealExpert(CaptionLLMExpert):
    """Real text-only LLM over captions / ASR / OCR evidence."""

    name = "cap_llm"
    modalities = ()
    cost_class = "cheap"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 32,
        **kw: Any,
    ) -> None:
        super().__init__(backbone=model_id, **kw)
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else self.device,
            local_files_only=True,
        )
        self.model.eval()

    def _evidence_blob(self, evidence: ExpertEvidence) -> str:
        return "\n".join(filter(None, [
            "\n".join(evidence.captions or []),
            f"Transcript: {evidence.asr_text}" if evidence.asr_text else "",
            f"On-screen text: {evidence.ocr_text}" if evidence.ocr_text else "",
        ]))

    def _chat(self, user_text: str) -> str:
        import torch

        messages = [{"role": "user", "content": user_text}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen[0], skip_special_tokens=True).strip()

    def _infer(self, evidence: ExpertEvidence) -> ExpertOutput:
        self.load()
        blob = self._evidence_blob(evidence)
        options = evidence.metadata.get("options")
        if options:
            prompt = (f"Video evidence:\n{blob}\n\n"
                      + build_mcq_prompt(evidence.question, options))
            raw = self._chat(prompt)
            letter = parse_letter(raw, len(options))
            return ExpertOutput(answer=letter or raw,
                                extras={"backbone": self.model_id, "raw": raw})
        prompt = (f"Video evidence:\n{blob}\n\nQuestion: {evidence.question}\n"
                  f"Answer with a single word or short phrase.")
        raw = self._chat(prompt)
        return ExpertOutput(answer=raw, extras={"backbone": self.model_id})


# --------------------------------------------------------------------------
#  Audio-visual expert (VideoLLaMA2.1-7B-AV)
# --------------------------------------------------------------------------
class VideoLLaMA2AVExpert(AudioVisualLMExpert):
    """Real audio-visual reasoner (DAMO VideoLLaMA2.1-7B-AV).

    Unlike the visual-only Qwen2-VL backbone, this model jointly attends to the
    video *and* its soundtrack, which is required for MUSIC-AVQA (questions about
    instruments, counting sounds, audio-visual correspondence). It is loaded via
    the official ``videollama2`` package (``audio_visual`` branch): ``model_init``
    returns ``(model, processor, tokenizer)`` and ``mm_infer`` runs generation.
    ``processor['video'](path, va=True)`` loads the synchronized audio+video
    tensor that the AV branch expects.

    ``audio_mode='silent'`` keeps the video and replaces only the soundtrack by
    digital silence. It reproduces what videollama2's own
    ``process_audio_from_video`` does when a soundtrack cannot be decoded
    (30 s of zeros -> 128-bin Kaldi fbank), so the model sees the same tensor
    layout and token count as in the audio-on run. This is the same-model audio
    ablation: it separates "the model can hear" from "a different model answers".

    ``audio_mode='mismatched'`` keeps the video and replaces the soundtrack by the
    soundtrack of a *different* video, chosen by a fixed derangement of the dataset's
    videos (``set_audio_donors``): the model still receives a plausible, correctly
    shaped soundtrack but not the one that belongs to the frames. Comparing it with the
    audio-on and silent runs tells whether the model uses the audio's content at all.

    The heavy import is lazy so this module stays importable in the main env;
    if the ``videollama2`` package is not installed (it pins an older
    torch/transformers and must live in an isolated env on Blackwell), ``load()``
    raises a clear, actionable error instead of failing at import time.
    """

    name = "videollama2_av"
    modalities = ("video", "audio")
    cost_class = "heavy"

    def __init__(
        self,
        model_id: str = "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV",
        device: str = "cuda",
        max_new_tokens: int = 64,
        audio_mode: str = "on",
        **kw: Any,
    ) -> None:
        # AudioVisualLMExpert.__init__ stores backbone/kw; keep that contract.
        super().__init__(backbone=model_id, **kw)
        if audio_mode not in ("on", "silent", "mismatched"):
            raise ValueError(
                f"audio_mode must be 'on', 'silent' or 'mismatched', got {audio_mode!r}")
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.audio_mode = audio_mode
        self._silent_audio = None
        self._donors: dict[str, str] = {}
        self.model = None
        self.processor = None
        self.tokenizer = None

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            from videollama2 import model_init  # type: ignore
            from videollama2.utils import disable_torch_init  # type: ignore
        except Exception as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "VideoLLaMA2-AV requires the `videollama2` package from the "
                "audio_visual branch of DAMO-NLP-SG/VideoLLaMA2, installed in an "
                "isolated environment (it pins torch 2.2 / transformers 4.40, "
                "which conflict with the main env and need a Blackwell-compatible "
                "rebuild). Weights are cached at "
                "~/.cache/huggingface/hub/models--DAMO-NLP-SG--VideoLLaMA2.1-7B-AV. "
                f"Import failed: {exc}"
            ) from exc
        disable_torch_init()
        # model_init returns (model, processor, tokenizer); processor is a dict
        # keyed by modality ('audio', 'video') in the AV branch.
        self.model, self.processor, self.tokenizer = model_init(self.model_id)

    def _infer(self, evidence: ExpertEvidence) -> ExpertOutput:
        self.load()
        from videollama2 import mm_infer  # type: ignore

        video_path = (evidence.metadata or {}).get("video_path")
        if not video_path:
            return ExpertOutput(
                answer="", log_prob=None,
                extras={"backbone": self.model_id,
                        "error": "no video_path in evidence.metadata"})

        # Load synchronized audio+video tensor (va=True pulls the soundtrack).
        try:
            av_tensor = self.processor["video"](video_path, va=True)
        except Exception:
            av_tensor = self.processor["video"](video_path)
        donor = None
        if self.audio_mode == "silent":
            av_tensor = self._silence_audio(av_tensor)
        elif self.audio_mode == "mismatched":
            av_tensor, donor = self._swap_audio(av_tensor, video_path)

        options = (evidence.metadata or {}).get("options")
        if options:
            prompt = build_mcq_prompt(evidence.question, options)
            raw = mm_infer(av_tensor, prompt, model=self.model,
                           tokenizer=self.tokenizer, modal="video",
                           do_sample=False)
            letter = parse_letter(raw, len(options))
            return ExpertOutput(answer=letter or raw,
                                extras={"backbone": self.model_id, "raw": raw,
                                        "modal": "audio_video",
                                        "audio_mode": self.audio_mode,
                                        "audio_donor": donor})
        prompt = (f"{evidence.question}\nAnswer the question using a single word "
                  f"or short phrase.")
        raw = mm_infer(av_tensor, prompt, model=self.model,
                       tokenizer=self.tokenizer, modal="video",
                       do_sample=False)
        return ExpertOutput(answer=str(raw).strip(),
                            extras={"backbone": self.model_id,
                                    "modal": "audio_video",
                                    "audio_mode": self.audio_mode,
                                    "audio_donor": donor})

    def set_audio_donors(self, video_paths, seed: int = 20261029) -> None:
        """Fix which video lends its soundtrack to which (``audio_mode='mismatched'``).

        The unique videos are shuffled once with ``seed`` and each gives its soundtrack to
        the next one in that order (cyclically), so every video receives the soundtrack of
        exactly one other video and none keeps its own. The map depends only on the set of
        videos, not on which questions are run.
        """
        self._donors = audio_donor_map(video_paths, seed)

    def _swap_audio(self, av_tensor, video_path: str):
        if not isinstance(av_tensor, dict) or "audio" not in av_tensor:
            raise RuntimeError(
                "audio_mode='mismatched' needs the audio+video dict returned by "
                "processor['video'](path, va=True)")
        donor = self._donors.get(str(video_path))
        if donor is None:
            raise RuntimeError(
                f"no soundtrack donor for {video_path}; call set_audio_donors() first")
        from videollama2.mm_utils import process_audio_from_video  # type: ignore
        ref = av_tensor["audio"]
        audio = process_audio_from_video(donor, 0.0)
        if tuple(audio.shape) != tuple(ref.shape):
            raise RuntimeError(
                f"donor soundtrack shape {tuple(audio.shape)} does not match "
                f"{tuple(ref.shape)}")
        return {**av_tensor, "audio": audio.to(device=ref.device, dtype=ref.dtype)}, donor

    def _silence_audio(self, av_tensor):
        """Return ``av_tensor`` with the soundtrack replaced by digital silence."""
        if not isinstance(av_tensor, dict) or "audio" not in av_tensor:
            raise RuntimeError(
                "audio_mode='silent' needs the audio+video dict returned by "
                "processor['video'](path, va=True)")
        ref = av_tensor["audio"]
        if self._silent_audio is None:
            import torch  # noqa: PLC0415
            import torchaudio.compliance.kaldi as ta_kaldi  # noqa: PLC0415
            wav = torch.zeros(1, 30 * 16000) * 2 ** 15
            fbank = ta_kaldi.fbank(wav, num_mel_bins=128, sample_frequency=16000,
                                   frame_length=25, frame_shift=10)
            self._silent_audio = fbank.to(torch.bfloat16).unsqueeze(0)
        if tuple(ref.shape) != tuple(self._silent_audio.shape):
            raise RuntimeError(
                f"silent audio shape {tuple(self._silent_audio.shape)} does not "
                f"match the soundtrack shape {tuple(ref.shape)}")
        return {**av_tensor,
                "audio": self._silent_audio.to(device=ref.device, dtype=ref.dtype)}


def audio_donor_map(video_paths, seed: int = 20261029) -> dict[str, str]:
    """video -> the video whose soundtrack it receives: a derangement of the unique videos."""
    import random
    videos = sorted({str(v) for v in video_paths})
    if len(videos) < 2:
        raise ValueError("a mismatched-audio control needs at least two videos")
    order = videos[:]
    random.Random(seed).shuffle(order)
    return {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}


# --------------------------------------------------------------------------
#  Registry of real backbones (opt-in)
# --------------------------------------------------------------------------

REAL_EXPERT_REGISTRY = {
    "vlm": Qwen2VLExpert,
    "vlm_qwen": Qwen2VLExpert,
    "vlm_llava_video": LLaVAVideoExpert,
    "caplm": CaptionLLMRealExpert,
    "avlm": VideoLLaMA2AVExpert,
    "videollama2_av": VideoLLaMA2AVExpert,
}


def make_real_expert(name: str, **kw: Any) -> Expert:
    if name not in REAL_EXPERT_REGISTRY:
        raise KeyError(
            f"no real backbone for {name}; available={list(REAL_EXPERT_REGISTRY)}"
        )
    return REAL_EXPERT_REGISTRY[name](**kw)
