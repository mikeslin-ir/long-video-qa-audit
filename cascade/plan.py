"""CascadePlan dataclass: the declarative recipe an agent emits."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class CascadePlan:
    """Recipe for one cascade execution.

    Tool names must be registered in the active CascadeRegistry. Weights apply
    only to aligners (L3); proposers/filters are combined by union-then-cap.
    """
    proposers: list[str] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    aligners: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    evidence_prompts: list[str] = field(default_factory=list)

    K_proposal: int = 200
    K_keyframe: int = 30
    K_final: int = 8
    tau_sec: float = 2.0
    uniform_anchor_frac: float = 0.0  # fraction of K_final reserved for evenly
                                      # spaced anchors over the full video
                                      # (0.0 = pure cascade selection)

    # Per-question agentic decision: which cue text streams to inject into the
    # downstream VLM prompt. Empty list (default) = no text injection (same as
    # the pre-agentic behaviour). Currently understood by the harness:
    #   "asr"  -- Whisper transcript snippets near the selected frames
    #   "ocr"  -- OCR text aligned to selected frames (when backend available)
    inject_text: list[str] = field(default_factory=list)

    expected_expert: str = "vlm"      # which expert the VLM stage should use
    reasoning: str = ""               # free-form, for logging
    planner: str = ""                 # name of the planner that produced this

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "CascadePlan":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
