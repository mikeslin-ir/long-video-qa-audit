"""Agentic planners that emit a CascadePlan from (question, dataset, meta).

Two planners are provided:

* `RuleBasedCascadePlanner` -- keyword rules over the question, no LLM call.
* `LLMCascadePlanner` -- prompts a text LLM (default: caption-LLM Qwen2.5-3B)
  for a JSON plan, with the rule-based planner as a robust fallback.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .plan import CascadePlan
from .registry import CascadeRegistry, default_registry


# ----------------------------------------------------------------------------
# Heuristics shared by both planners
# ----------------------------------------------------------------------------
_TEXT_WORDS = ("text", "sign", "subtitle", "license plate", "letter", "word", "written")
_SPEECH_WORDS = ("say", "said", "speak", "spoken", "talk", "talked", "talking",
                 "dialog", "dialogue", "conversation", "shout", "yell")
_AUDIO_WORDS = ("music", "song", "sing", "instrument", "loud", "noise", "sound",
                "clap", "scream", "laugh")
_ACTION_WORDS = ("doing", "action", "happens", "happened", "pick", "carry",
                 "open", "close", "throw", "catch", "run", "walk", "jump")
_OBJECT_WORDS = ("color", "wearing", "holding", "on the table", "in the room",
                 "near the", "next to", "behind", "in front of")
_VIOLENCE_WORDS = ("fight", "violence", "violent", "assault", "shoot", "explosion",
                   "accident", "weapon", "blood")


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    t = (text or "").lower()
    return any(w in t for w in words)


def _video_is_long(meta: dict) -> bool:
    """Long if duration > 120s; relies on meta['duration_sec'] when present."""
    return float(meta.get("duration_sec", 0.0)) > 120.0


# ----------------------------------------------------------------------------
# Rule-based planner
# ----------------------------------------------------------------------------
@dataclass
class RuleBasedCascadePlanner:
    name: str = "rule_based"
    K_proposal_long: int = 200
    K_proposal_short: int = 64
    K_keyframe: int = 32
    K_final: int = 8
    tau_sec: float = 2.0
    uniform_anchor_frac: float = 0.0
    force_multicue: bool = False
    prefer_cache: bool = False  # use offline shot-keyframe cache (L1) + cached
                                # SigLIP aligner (L3) when available
    use_shot_boundary: bool = False  # ablation: re-enable the standalone
                                     # PySceneDetect proposer. Off by default
                                     # because STACFP self-adapts to cuts and
                                     # subsumes shot detection (see proposers/
                                     # stacfp.py snap_cuts regime).
    # Which L1/L2 tools the rule-based plan enables:
    #   "stacfp": STACFP + EARS + uniform stride, redundancy filter (library default)
    #   "audit":  EARS + uniform stride, motion-difference filter. This is what the
    #             audit's Video-MME long, XD-Violence and UCF-Crime cascade runs
    #             actually executed. Those runs were planned with a PySceneDetect
    #             shot-boundary proposer, but PySceneDetect was not installed in
    #             that environment, so the router skipped it without a message
    #             (132 = 2 x 66 proposals per video instead of 198). The profile
    #             omits it, so it reproduces those runs whether or not PySceneDetect
    #             is installed; the paper-run drivers pass --cascade_profile audit.
    profile: str = "stacfp"

    def __post_init__(self):
        if self.profile not in ("stacfp", "audit"):
            raise ValueError(f"unknown cascade profile {self.profile!r}; "
                             "expected 'stacfp' or 'audit'")

    def plan(self, question: str, dataset: str, meta: dict,
             registry: Optional[CascadeRegistry] = None) -> CascadePlan:
        reg = registry or default_registry()
        avail = reg.available()
        q = (question or "").lower()
        ds = (dataset or "").lower()

        is_long = _video_is_long(meta)
        audit = self.profile == "audit"

        # ---- L1 (proposers) ----
        proposers: list[str] = []
        # Offline shot-keyframe cache: cheapest L1 (no decode). Preferred when
        # enabled and the video has a cache entry; uniform_stride still appended
        # below as a backbone/fallback for cache misses.
        if self.prefer_cache and "keyframe_cache" in avail["proposers"]:
            proposers.append("keyframe_cache")
        # STACFP (spatio-temporal adaptive clustering) is the DEFAULT long-video
        # backbone: content-adaptive anchor count, denser where the scene
        # changes and sparser when static. It estimates cut density for free
        # from its histograms and self-adapts -- snapping anchors to cuts on
        # edited video (shot-like) and using centroid clustering on long takes
        # -- so it subsumes the standalone shot_boundary proposer.
        if not audit and is_long and "stacfp" in avail["proposers"]:
            proposers.append("stacfp")
        # shot_boundary (PySceneDetect) is kept ONLY as an ablation baseline,
        # since STACFP's snap_cuts regime covers the same edited-video case.
        if self.use_shot_boundary and is_long and "shot_boundary" in avail["proposers"]:
            proposers.append("shot_boundary")
        if is_long and "ears_event" in avail["proposers"]:
            proposers.append("ears_event")
        # Always include uniform_stride as backbone so the proposal set is dense
        proposers.append("uniform_stride")

        # ---- L2 (filters) ----
        # Default L2 removes near-duplicate (temporally redundant) frames while
        # preserving static-but-distinct frames (e.g. a static speaker or an
        # on-screen sign). Applied to BOTH long and short videos: uniform /
        # boundary proposals can pick near-identical neighbours regardless of
        # duration, and de-duplication is cheap (low-level histogram, no model).
        # The older motion_diff filter -- which *selects* high-motion frames and
        # drops static ones -- is the audit profile's filter (long videos only)
        # and otherwise an ablation baseline.
        filters: list[str] = []
        if audit:
            if is_long and "motion_diff" in avail["filters"]:
                filters.append("motion_diff")
        elif "redundancy" in avail["filters"]:
            filters.append("redundancy")

        # ---- L3 (aligners) -- question-conditioned ----
        aligners: list[str] = []
        weights: dict[str, float] = {}
        prompts: list[str] = []
        reason_parts: list[str] = []
        expected_expert = "vlm"

        if _contains_any(q, _TEXT_WORDS) and "ocr_text" in avail["aligners"]:
            aligners.append("ocr_text")
            weights["ocr_text"] = 0.5
            prompts += ["text sign", "subtitle", "writing"]
            reason_parts.append("text/OCR keywords")

        if _contains_any(q, _SPEECH_WORDS) and "asr_whisper" in avail["aligners"]:
            aligners.append("asr_whisper")
            weights["asr_whisper"] = 0.4
            prompts += ["person speaking", "dialogue"]
            reason_parts.append("speech keywords")
            expected_expert = "avlm" if dataset in {"video_mme", "music_avqa"} else "vlm"

        if _contains_any(q, _OBJECT_WORDS) and "object_siglip" in avail["aligners"]:
            aligners.append("object_siglip")
            weights["object_siglip"] = 0.4
            reason_parts.append("object/scene keywords")

        # ---- Forced multi-cue policy (ablation: does using >1 complementary
        # tool per question help vs. single SigLIP?). When enabled, generic
        # questions that triggered no keyword rule still receive a genuine
        # multi-modal aligner set rather than collapsing to semantic_siglip.
        if self.force_multicue:
            audio_ds = ds in {"video_mme", "videomme", "music_avqa",
                              "ucf_crime", "xd_violence"}
            if "object_siglip" in avail["aligners"] and "object_siglip" not in aligners:
                aligners.append("object_siglip")
                weights["object_siglip"] = 0.30
                reason_parts.append("forced object grounding")
            if (audio_ds or _contains_any(q, _SPEECH_WORDS) or _contains_any(q, _AUDIO_WORDS)) \
                    and "asr_whisper" in avail["aligners"] and "asr_whisper" not in aligners:
                aligners.append("asr_whisper")
                weights["asr_whisper"] = 0.25
                prompts += ["narration", "spoken explanation"]
                reason_parts.append("forced narration/ASR cue")

        # Always include a semantic SigLIP aligner as the default content
        # alignment. When prefer_cache is on and the cached variant is
        # registered, use it (reuses offline embeddings) instead of re-encoding.
        if self.prefer_cache and "semantic_siglip_cached" in avail["aligners"]:
            aligners.append("semantic_siglip_cached")
            weights["semantic_siglip_cached"] = max(0.3, 1.0 - sum(weights.values()))
        elif "semantic_siglip" in avail["aligners"]:
            aligners.append("semantic_siglip")
            weights["semantic_siglip"] = max(0.3, 1.0 - sum(weights.values()))

        if not aligners:
            # Degenerate: at least keep semantic_clip-like if SigLIP missing
            aligners = list(avail["aligners"])[:1]
            if aligners:
                weights[aligners[0]] = 1.0

        # Normalize weights
        sw = sum(weights.values())
        if sw > 0:
            weights = {k: float(v) / sw for k, v in weights.items()}

        # Reasonable default prompts when none were supplied
        if not prompts and q:
            prompts = [q[:80]]

        # ---- budgets ----
        K_prop = self.K_proposal_long if is_long else self.K_proposal_short
        if audit and is_long:
            # The audit's plans listed three proposers, so each got K_proposal // 3
            # = 66 candidates; with the inactive third one removed, two slots of
            # 66 keep the pool at 132 frames (the router splits K_proposal evenly).
            K_prop = 2 * (self.K_proposal_long // 3)
        K_key = min(self.K_keyframe, K_prop)
        K_final = self.K_final

        plan = CascadePlan(
            proposers=proposers,
            filters=filters,
            aligners=aligners,
            weights=weights,
            evidence_prompts=prompts,
            K_proposal=K_prop,
            K_keyframe=K_key,
            K_final=K_final,
            tau_sec=self.tau_sec,
            uniform_anchor_frac=float(self.uniform_anchor_frac),
            expected_expert=expected_expert,
            reasoning="; ".join(reason_parts) or "dataset default",
            planner=self.name,
        )
        return plan


# ----------------------------------------------------------------------------
# LLM planner (uses a callable: prompt -> json string)
# ----------------------------------------------------------------------------
@dataclass
class LLMCascadePlanner:
    """Calls a text LLM to emit a CascadePlan. Robust to malformed JSON by
    falling back to RuleBasedCascadePlanner.

    `generate_fn` is a `(prompt: str) -> str` callable so we don't depend on
    any specific LLM at import time.

    K_final / tau_sec / uniform_anchor_frac coming from the CLI are FORCED
    onto the LLM's plan (so the LLM cannot blow the compute budget); the LLM
    decides only the cue mix, evidence prompts, and whether to inject cue
    text into the downstream VLM prompt.
    """
    name: str = "llm"
    generate_fn: Optional[Callable[[str], str]] = None
    fallback: RuleBasedCascadePlanner = None  # type: ignore[assignment]
    # Hard caps applied after the LLM responds (None = trust LLM).
    K_final_override: Optional[int] = None
    tau_sec_override: Optional[float] = None
    uniform_anchor_frac_override: Optional[float] = None
    # When set, hard-overrides the aligner cue mix to this fixed list after the
    # LLM responds. Used to isolate the inject_text decision from drift in the
    # cue mix (ProcMem-v2 ablation: freeze aligners, let LLM decide only
    # whether to inject ASR/OCR text into the VLM prompt).
    aligners_override: Optional[tuple[str, ...]] = None
    # Allow text injection only from this whitelist (so the LLM can't fabricate
    # cue names). Empty means "no injection allowed".
    allowed_inject: tuple[str, ...] = ("asr", "ocr")
    # Procedural memory: free-form text appended to the planner prompt as a
    # When True, the LLM is consulted ONLY for `inject_text`; every other
    # plan field is taken from the rule-based fallback. This isolates the
    # LLM's per-question inject decision from any drift in the cue mix /
    # proposers / filters / evidence prompts the LLM might introduce.
    inject_only_mode: bool = False
    # "PROCEDURAL MEMORY" block before the few-shot examples. Used to inject
    # priors learned from past execution logs (e.g. "always inject ASR for
    # long narrative videos"). Empty string disables it.
    procedural_memory: str = ""
    # "paper": the prompt behind the audited planner runs (three few-shot examples,
    # shot-boundary / motion-difference tool descriptions). "current": the later prompt
    # that also describes STACFP, redundancy, CLAP and audio-energy tools.
    prompt_version: str = "paper"

    def __post_init__(self):
        if self.prompt_version not in ("paper", "current"):
            raise ValueError(f"unknown prompt_version {self.prompt_version!r}")
        if self.fallback is None:
            self.fallback = RuleBasedCascadePlanner()

    def plan(self, question: str, dataset: str, meta: dict,
             registry: Optional[CascadeRegistry] = None) -> CascadePlan:
        reg = registry or default_registry()
        if self.generate_fn is None:
            return self._with_overrides(
                self.fallback.plan(question, dataset, meta, registry=reg))

        avail = reg.available()
        build = self._build_prompt if self.prompt_version == "current" else self._build_prompt_paper
        prompt = build(question, dataset, meta, avail,
                       procedural_memory=self.procedural_memory)
        try:
            raw = self.generate_fn(prompt)
            plan_dict = _extract_json(raw)
            if not plan_dict:
                raise ValueError("planner LLM produced no JSON")
            if self.inject_only_mode:
                # Take everything from rule_based; overlay only inject_text.
                plan = self.fallback.plan(question, dataset, meta, registry=reg)
                plan.planner = self.name + "+ruleBase"
                plan.inject_text = [c for c in (plan_dict.get("inject_text") or [])
                                    if c in self.allowed_inject]
                return self._with_overrides(plan)
            plan = CascadePlan.from_json(plan_dict)
            plan.planner = self.name
            # Validate tool names against the registry; drop unknown ones.
            plan.proposers = [p for p in plan.proposers if p in avail["proposers"]] \
                              or self.fallback.plan(question, dataset, meta, reg).proposers
            plan.filters = [f for f in plan.filters if f in avail["filters"]]
            plan.aligners = [a for a in plan.aligners if a in avail["aligners"]] \
                              or self.fallback.plan(question, dataset, meta, reg).aligners
            # Clamp injected-text cue names to the whitelist.
            plan.inject_text = [c for c in (plan.inject_text or [])
                                if c in self.allowed_inject]
            return self._with_overrides(plan)
        except Exception:
            return self._with_overrides(
                self.fallback.plan(question, dataset, meta, registry=reg))

    def _with_overrides(self, plan: CascadePlan) -> CascadePlan:
        if self.K_final_override is not None:
            plan.K_final = int(self.K_final_override)
        if self.tau_sec_override is not None:
            plan.tau_sec = float(self.tau_sec_override)
        if self.uniform_anchor_frac_override is not None:
            plan.uniform_anchor_frac = float(self.uniform_anchor_frac_override)
        if self.aligners_override is not None:
            plan.aligners = list(self.aligners_override)
        return plan

    # Tool names the audited planner runs offered (the registry at that time).
    _PAPER_TOOLS = {
        "proposers": ("ears_event", "shot_boundary", "uniform_stride"),
        "filters": ("motion_diff",),
        "aligners": ("asr_whisper", "object_siglip", "ocr_text", "semantic_siglip"),
    }

    @staticmethod
    def _build_prompt_paper(question: str, dataset: str, meta: dict,
                            avail: dict[str, list[str]],
                            procedural_memory: str = "") -> str:
        """The planner prompt exactly as the audited runs used it."""
        cost_hint = {
            "uniform_stride":  "0.0  (free; always include as a backbone)",
            "shot_boundary":   "0.1  (fast; long videos benefit)",
            "ears_event":      "0.5  (long videos; audio-visual events)",
            "motion_diff":     "0.1  (cheap; useful for long static videos)",
            "semantic_siglip": "1.0  (default; visual-text relevance)",
            "object_siglip":   "1.2  (visual objects; helps for color/holding/wearing/position questions)",
            "asr_whisper":     "0.3  (cached; helps for what was SAID / dialog / narration)",
            "ocr_text":        "0.4  (helps for text / signs / subtitles / license plates)",
        }
        cost_lines = []
        for tier in ("proposers", "filters", "aligners"):
            for t in avail[tier]:
                if t in LLMCascadePlanner._PAPER_TOOLS[tier]:
                    cost_lines.append(f"  {t}: {cost_hint.get(t, '? cost')}")

        duration = meta.get("duration_sec", "unknown")
        try:
            dur_tag = "LONG" if float(duration) > 120 else "SHORT"
        except Exception:
            dur_tag = "UNKNOWN"

        few_shot = (
            'Example A. Question: "What color is the car parked next to the building?"\n'
            '  Plan: {"proposers":["uniform_stride"],"filters":[],'
            '"aligners":["object_siglip","semantic_siglip"],'
            '"weights":{"object_siglip":0.6,"semantic_siglip":0.4},'
            '"evidence_prompts":["car parked next to building","colored car"],'
            '"inject_text":[],"reasoning":"object/color question; vision is enough"}\n\n'
            'Example B. Question: "What does the narrator say about the volcano?"\n'
            '  Plan: {"proposers":["uniform_stride","shot_boundary"],"filters":[],'
            '"aligners":["asr_whisper","semantic_siglip"],'
            '"weights":{"asr_whisper":0.7,"semantic_siglip":0.3},'
            '"evidence_prompts":["narrator explaining volcano","spoken narration"],'
            '"inject_text":["asr"],"reasoning":"answer is in narration; inject ASR text"}\n\n'
            'Example C. Question: "What is written on the sign?"\n'
            '  Plan: {"proposers":["uniform_stride"],"filters":[],'
            '"aligners":["ocr_text","semantic_siglip"],'
            '"weights":{"ocr_text":0.7,"semantic_siglip":0.3},'
            '"evidence_prompts":["sign with text","written sign"],'
            '"inject_text":["ocr"],"reasoning":"answer is literal text; inject OCR"}\n'
        )

        memory_block = ""
        if procedural_memory and procedural_memory.strip():
            memory_block = (
                "PROCEDURAL MEMORY (priors learned from past execution logs; "
                "treat as strong defaults, not absolute rules):\n"
                f"{procedural_memory.strip()}\n\n"
            )

        return (
            "You are an agentic video understanding planner. Pick the MINIMAL "
            "set of tools that answers the question; every extra tool costs "
            "compute. The VLM (vision-language model) will see the selected "
            "frames as pixels. If you set inject_text=[\"asr\"] the cached "
            "speech transcript near the selected frames will ALSO be inserted "
            "into the VLM prompt as text (highly effective when the answer is "
            "spoken). Use inject_text=[\"ocr\"] for on-screen text questions.\n\n"
            f"{memory_block}"
            f"Question: {question}\n"
            f"Dataset: {dataset}\n"
            f"Video duration (s): {duration}  [{dur_tag}]\n\n"
            "Available tools (name: relative compute cost):\n"
            + "\n".join(cost_lines)
            + "\n\nGuidelines:\n"
              "- Always include uniform_stride as a proposer.\n"
              "- Always include semantic_siglip as an aligner (default backbone).\n"
              "- Add object_siglip only when the question is about objects, "
              "colors, attributes or spatial layout.\n"
              "- Add asr_whisper (and inject_text=[\"asr\"]) when the answer "
              "is in spoken narration, dialogue, or audio commentary.\n"
              "- Add ocr_text (and inject_text=[\"ocr\"]) when the answer is "
              "on-screen text, a sign, subtitle, or label.\n"
              "- 1-3 aligners is normal; >3 wastes compute.\n"
              "- weights must sum to ~1.0 across aligners.\n\n"
            f"{few_shot}\n"
            "Now produce the JSON plan for the above question. "
            "Reply with ONLY one JSON object with keys: proposers, filters, "
            "aligners, weights, evidence_prompts, inject_text, reasoning. "
            "Do not include any other text.\n"
            "JSON:"
        )

    @staticmethod
    def _build_prompt(question: str, dataset: str, meta: dict,
                      avail: dict[str, list[str]],
                      procedural_memory: str = "") -> str:
        # Compact cost / use-case hints so a small LLM can reason about
        # tradeoffs. Costs are normalized "per video minute" budget units,
        # measured empirically on this codebase (GB10, fp16).
        cost_hint = {
            # L1 proposers
            "uniform_stride":  "0.0  (free; always include as a backbone)",
            "stacfp":          "0.3  (DEFAULT long-video backbone; spatio-temporal adaptive clustering, content-adaptive anchor count; self-adapts to cut density -- snaps to cuts on edited video, clusters on long takes; subsumes shot_boundary)",
            "shot_boundary":   "0.1  (ablation only; standalone shot detection -- STACFP's snap_cuts regime covers this)",
            "ears_event":      "0.5  (long videos; audio-visual events)",
            # L2 filters
            "redundancy":      "0.1  (default L2; drops near-duplicate frames, keeps static-but-distinct ones; content-adaptive count)",
            "motion_diff":     "0.1  (ablation only; SELECTS high-motion frames and drops static ones -- can lose static signs/speakers)",
            # L3 aligners
            "semantic_siglip": "1.0  (default; visual-text relevance)",
            "object_siglip":   "1.2  (visual objects; helps for color/holding/wearing/position questions)",
            "asr_whisper":     "0.3  (cached; helps for what was SAID / dialog / narration)",
            "ocr_text":        "0.4  (helps for text / signs / subtitles / license plates)",
            "clap_alm":        "0.2  (cached audio-language; helps for MUSIC / sound / instrument / which-sounds questions where the answer is acoustic, not spoken words)",
            "audio_energy":    "0.1  (cheap acoustic salience; helps locate loud events: screams, gunshots, crashes, applause in event-sparse video)",
        }
        cost_lines = []
        for tier in ("proposers", "filters", "aligners"):
            for t in avail[tier]:
                cost_lines.append(f"  {t}: {cost_hint.get(t, '? cost')}")

        duration = meta.get("duration_sec", "unknown")
        try:
            dur_f = float(duration)
            dur_tag = "LONG" if dur_f > 120 else "SHORT"
        except Exception:
            dur_tag = "UNKNOWN"

        few_shot = (
            'Example A. Question: "What color is the car parked next to the building?"\n'
            '  Plan: {"proposers":["uniform_stride"],"filters":[],'
            '"aligners":["object_siglip","semantic_siglip"],'
            '"weights":{"object_siglip":0.6,"semantic_siglip":0.4},'
            '"evidence_prompts":["car parked next to building","colored car"],'
            '"inject_text":[],"reasoning":"object/color question; vision is enough"}\n\n'
            'Example B. Question: "What does the narrator say about the volcano?"\n'
            '  Plan: {"proposers":["uniform_stride","shot_boundary"],"filters":[],'
            '"aligners":["asr_whisper","semantic_siglip"],'
            '"weights":{"asr_whisper":0.7,"semantic_siglip":0.3},'
            '"evidence_prompts":["narrator explaining volcano","spoken narration"],'
            '"inject_text":["asr"],"reasoning":"answer is in narration; inject ASR text"}\n\n'
            'Example C. Question: "What is written on the sign?"\n'
            '  Plan: {"proposers":["uniform_stride"],"filters":[],'
            '"aligners":["ocr_text","semantic_siglip"],'
            '"weights":{"ocr_text":0.7,"semantic_siglip":0.3},'
            '"evidence_prompts":["sign with text","written sign"],'
            '"inject_text":["ocr"],"reasoning":"answer is literal text; inject OCR"}\n\n'
            'Example D. Question: "Which instrument plays the loudest while the man is speaking?"\n'
            '  Plan: {"proposers":["uniform_stride","shot_boundary"],"filters":[],'
            '"aligners":["clap_alm","asr_whisper","semantic_siglip"],'
            '"weights":{"clap_alm":0.45,"asr_whisper":0.3,"semantic_siglip":0.25},'
            '"evidence_prompts":["loud instrument","man speaking over music"],'
            '"inject_text":["asr"],"reasoning":"mixes non-speech audio (instrument) and '
            'speech; use CLAP and ASR together -- complementary audio signals"}\n'
        )

        memory_block = ""
        if procedural_memory and procedural_memory.strip():
            memory_block = (
                "PROCEDURAL MEMORY (priors learned from past execution logs; "
                "treat as strong defaults, not absolute rules):\n"
                f"{procedural_memory.strip()}\n\n"
            )

        return (
            "You are an agentic video understanding planner. Select the set of "
            "COMPLEMENTARY tools that best answers the question; each tool adds "
            "compute, so avoid redundant tools of the same modality, but DO use "
            "multiple modalities (vision + audio + text) when the question spans "
            "them. The VLM (vision-language model) will see the selected "
            "frames as pixels. inject_text inserts cached evidence text near the "
            "selected frames INTO the VLM prompt; you may inject MORE THAN ONE "
            "type. Use inject_text=[\"asr\"] when the answer is spoken, "
            "inject_text=[\"ocr\"] for on-screen text, and "
            "inject_text=[\"asr\",\"ocr\"] when both speech and on-screen text "
            "matter.\n\n"
            f"{memory_block}"
            f"Question: {question}\n"
            f"Dataset: {dataset}\n"
            f"Video duration (s): {duration}  [{dur_tag}]\n\n"
            "Available tools (name: relative compute cost):\n"
            + "\n".join(cost_lines)
            + "\n\nGuidelines:\n"
              "- Always include uniform_stride as a proposer.\n"
              "- Always include semantic_siglip as an aligner (default backbone).\n"
              "- Add object_siglip only when the question is about objects, "
              "colors, attributes or spatial layout.\n"
              "- Add asr_whisper (and inject_text=[\"asr\"]) when the answer "
              "is in spoken narration, dialogue, or audio commentary.\n"
              "- Add ocr_text (and inject_text=[\"ocr\"]) when the answer is "
              "on-screen text, a sign, subtitle, or label.\n"
              "- Add clap_alm when the answer depends on NON-SPEECH audio: "
              "music, instruments, genre, which object makes a sound. ASR "
              "(words) and CLAP (sound) are complementary -- use BOTH when a "
              "question mixes speech and non-speech audio.\n"
              "- Add audio_energy when the answer is a loud/sudden event "
              "(scream, gunshot, crash, applause) that is easier to localize "
              "acoustically than visually.\n"
              "- Prefer the SMALLEST set of COMPLEMENTARY signals that each add "
              "distinct evidence; combine multiple modalities (e.g. vision + "
              "audio + text) when the question genuinely spans them, but avoid "
              "redundant signals of the same modality.\n"
              "- weights must sum to ~1.0 across aligners.\n\n"
            f"{few_shot}\n"
            "Now produce the JSON plan for the above question. "
            "Reply with ONLY one JSON object with keys: proposers, filters, "
            "aligners, weights, evidence_prompts, inject_text, reasoning. "
            "Do not include any other text.\n"
            "JSON:"
        )


def _extract_json(s: str) -> dict[str, Any] | None:
    """Find the first JSON object in `s` and parse it."""
    if not s:
        return None
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(s[start:end + 1])
    except Exception:
        return None
