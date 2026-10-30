"""Expert-counterfactual oracle (the N2 reward).

Given a question, a set of selected frames (and optional audio/text
evidence), and a *gold* answer, this module measures how much each unit of
evidence contributes to the expert's confidence in the correct answer, by
the change in the answer log-probability when that unit is removed
(leave-one-out) -- and, for the *expert* axis, how the confidence changes
when the question is routed to a different expert.

Three counterfactual signals
-----------------------------
- Delta^frame_i  = logP_e(a*| F) - logP_e(a*| F \ {i})        (frame necessity)
- Delta^skill_s  = logP_e(a*| ev) - logP_e(a*| ev without skill s)  (signal necessity)
- Delta^expert_e = logP_e(a*| ev) - logP_e'(a*| ev)            (expert necessity)

All quantities reuse the expert's `option_logprobs` (forced decoding) so the
oracle never has to free-form generate; it is a pure scoring pass. The
oracle is used (a) as a diagnostic to validate that the router selects
necessary evidence, and (b) as a teacher signal for REINFORCE-style policy
supervision of the planner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from router.experts import Expert, ExpertEvidence


def _answer_logprob(expert: Expert, evidence: ExpertEvidence,
                    options: list[str], gold_letter: str) -> float:
    """log P_e(gold | evidence) via forced-decoding option log-probs."""
    olp_fn = getattr(expert, "option_logprobs", None)
    if olp_fn is not None and options:
        olp = olp_fn(evidence, options)
        if gold_letter in olp:
            return float(olp[gold_letter])
    # fallback: generic logprob proxy
    return float(expert.logprob(evidence, gold_letter))


@dataclass
class CounterfactualResult:
    base_logprob: float
    frame_deltas: list[float] = field(default_factory=list)   # per selected frame
    skill_deltas: dict[str, float] = field(default_factory=dict)
    expert_deltas: dict[str, float] = field(default_factory=dict)
    best_alt_expert: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "base_logprob": self.base_logprob,
            "frame_deltas": self.frame_deltas,
            "skill_deltas": self.skill_deltas,
            "expert_deltas": self.expert_deltas,
            "best_alt_expert": self.best_alt_expert,
            "meta": self.meta,
        }


class CounterfactualOracle:
    """Leave-one-out + expert-swap oracle over a multimodal expert pool."""

    def __init__(self, experts: dict[str, Expert]):
        self.experts = experts

    # -- frame necessity ---------------------------------------------------
    def frame_deltas(
        self,
        expert: Expert,
        question: str,
        frames: Sequence[np.ndarray],
        options: list[str],
        gold_letter: str,
        base_logprob: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[float, list[float]]:
        meta = dict(metadata or {})
        meta["options"] = options
        full = ExpertEvidence(question=question, frames=list(frames),
                              metadata=meta)
        if base_logprob is None:
            base_logprob = _answer_logprob(expert, full, options, gold_letter)
        deltas: list[float] = []
        for i in range(len(frames)):
            loo = [f for j, f in enumerate(frames) if j != i]
            ev = ExpertEvidence(question=question, frames=loo or None,
                                metadata=meta)
            lp = _answer_logprob(expert, ev, options, gold_letter)
            deltas.append(float(base_logprob - lp))   # >0 => frame was helpful
        return float(base_logprob), deltas

    # -- skill / modality necessity ---------------------------------------
    def skill_deltas(
        self,
        expert: Expert,
        question: str,
        frames: Sequence[np.ndarray],
        options: list[str],
        gold_letter: str,
        ablations: dict[str, ExpertEvidence],
        base_logprob: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        meta = dict(metadata or {})
        meta["options"] = options
        if base_logprob is None:
            full = ExpertEvidence(question=question, frames=list(frames),
                                  metadata=meta)
            base_logprob = _answer_logprob(expert, full, options, gold_letter)
        out: dict[str, float] = {}
        for skill, ev in ablations.items():
            if not ev.metadata.get("options"):
                ev.metadata["options"] = options
            lp = _answer_logprob(expert, ev, options, gold_letter)
            out[skill] = float(base_logprob - lp)
        return out

    # -- expert necessity --------------------------------------------------
    def expert_deltas(
        self,
        chosen_name: str,
        question: str,
        evidence: ExpertEvidence,
        options: list[str],
        gold_letter: str,
        base_logprob: float | None = None,
    ) -> tuple[dict[str, float], str | None]:
        if not evidence.metadata.get("options"):
            evidence.metadata["options"] = options
        chosen = self.experts[chosen_name]
        if base_logprob is None:
            base_logprob = _answer_logprob(chosen, evidence, options, gold_letter)
        deltas: dict[str, float] = {}
        best_alt, best_lp = None, -np.inf
        for name, exp in self.experts.items():
            if name == chosen_name:
                continue
            if not exp.can_handle(evidence):
                continue
            lp = _answer_logprob(exp, evidence, options, gold_letter)
            deltas[name] = float(base_logprob - lp)  # >0 => chosen better
            if lp > best_lp:
                best_lp, best_alt = lp, name
        return deltas, best_alt

    # -- full pass ---------------------------------------------------------
    def evaluate(
        self,
        chosen_name: str,
        question: str,
        frames: Sequence[np.ndarray],
        options: list[str],
        gold_letter: str,
        skill_ablations: dict[str, ExpertEvidence] | None = None,
        metadata: dict[str, Any] | None = None,
        do_frames: bool = True,
        do_experts: bool = True,
    ) -> CounterfactualResult:
        expert = self.experts[chosen_name]
        meta = dict(metadata or {})
        meta["options"] = options
        full = ExpertEvidence(question=question, frames=list(frames),
                              metadata=meta)
        base = _answer_logprob(expert, full, options, gold_letter)

        res = CounterfactualResult(base_logprob=float(base))
        if do_frames and frames:
            _, res.frame_deltas = self.frame_deltas(
                expert, question, frames, options, gold_letter,
                base_logprob=base, metadata=metadata)
        if skill_ablations:
            res.skill_deltas = self.skill_deltas(
                expert, question, frames, options, gold_letter,
                skill_ablations, base_logprob=base, metadata=metadata)
        if do_experts:
            res.expert_deltas, res.best_alt_expert = self.expert_deltas(
                chosen_name, question, full, options, gold_letter,
                base_logprob=base)
        res.meta = {"chosen_expert": chosen_name, "gold": gold_letter,
                    "num_frames": len(frames)}
        return res
