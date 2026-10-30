from __future__ import annotations

import pytest

from cascade.planner import LLMCascadePlanner, RuleBasedCascadePlanner
from cascade.registry import CascadeRegistry


class _Tool:
    def __init__(self, name):
        self.name = name


def _registry(proposers=("uniform_stride", "ears_event", "shot_boundary", "stacfp"),
              filters=("motion_diff", "redundancy"),
              aligners=("semantic_siglip", "object_siglip", "asr_whisper", "ocr_text", "clap_alm")):
    reg = CascadeRegistry()
    for name in proposers:
        reg.register_proposer(name, _Tool(name))
    for name in filters:
        reg.register_filter(name, _Tool(name))
    for name in aligners:
        reg.register_aligner(name, _Tool(name))
    return reg


LONG = {"duration_sec": 1800.0}
SHORT = {"duration_sec": 60.0}


def test_audit_profile_reproduces_the_published_pool_for_long_videos():
    plan = RuleBasedCascadePlanner(profile="audit", uniform_anchor_frac=0.5).plan(
        "What is the video about?", "video_mme", LONG, registry=_registry())
    assert plan.proposers == ["ears_event", "uniform_stride"]
    assert plan.filters == ["motion_diff"]
    assert plan.K_proposal == 132  # 2 x 66: the router splits K_proposal evenly
    assert plan.K_keyframe == 32 and plan.uniform_anchor_frac == 0.5


def test_audit_profile_short_clips_use_uniform_stride_only():
    plan = RuleBasedCascadePlanner(profile="audit").plan("Is there a fight?", "xd_violence", SHORT,
                                                         registry=_registry())
    assert plan.proposers == ["uniform_stride"]
    assert plan.filters == []
    assert plan.K_proposal == 64


def test_audit_profile_never_plans_shot_boundary_even_when_installed():
    plan = RuleBasedCascadePlanner(profile="audit").plan("q", "video_mme", LONG, registry=_registry())
    assert "shot_boundary" not in plan.proposers and "stacfp" not in plan.proposers


def test_default_profile_is_stacfp_with_redundancy_filter():
    plan = RuleBasedCascadePlanner().plan("q", "video_mme", LONG, registry=_registry())
    assert plan.proposers == ["stacfp", "ears_event", "uniform_stride"]
    assert plan.filters == ["redundancy"]
    assert plan.K_proposal == 200


def test_shot_boundary_remains_an_explicit_ablation():
    plan = RuleBasedCascadePlanner(use_shot_boundary=True).plan("q", "video_mme", LONG, registry=_registry())
    assert plan.proposers == ["stacfp", "shot_boundary", "ears_event", "uniform_stride"]


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="unknown cascade profile"):
        RuleBasedCascadePlanner(profile="legacy")


def test_keyword_aligners_are_unchanged_by_profile():
    for profile in ("audit", "stacfp"):
        plan = RuleBasedCascadePlanner(profile=profile).plan(
            "What does the narrator say about the sign?", "video_mme", LONG, registry=_registry())
        assert plan.aligners[:2] == ["ocr_text", "asr_whisper"] and plan.aligners[-1] == "semantic_siglip"
        assert abs(sum(plan.weights.values()) - 1.0) < 1e-9


def test_paper_prompt_lists_only_the_audited_tools_and_the_memory():
    prompt = LLMCascadePlanner._build_prompt_paper(
        "What is written on the sign?", "video_mme", LONG, _registry().available(),
        procedural_memory="MEMORY-TEXT")
    assert "Pick the MINIMAL set of tools" in prompt
    assert "PROCEDURAL MEMORY" in prompt and "MEMORY-TEXT" in prompt
    for listed in ("ears_event", "shot_boundary", "uniform_stride", "motion_diff", "semantic_siglip"):
        assert f"  {listed}:" in prompt
    for absent in ("stacfp", "redundancy", "clap_alm", "keyframe_cache"):
        assert f"  {absent}:" not in prompt
    assert prompt.rstrip().endswith("JSON:")
    assert "Example C." in prompt and "Example D." not in prompt


def test_current_prompt_is_selectable_and_differs():
    avail = _registry().available()
    current = LLMCascadePlanner._build_prompt("q", "video_mme", LONG, avail)
    paper = LLMCascadePlanner._build_prompt_paper("q", "video_mme", LONG, avail)
    assert current != paper and "Select the set of" in current
    with pytest.raises(ValueError):
        LLMCascadePlanner(prompt_version="v9")


def test_inject_only_mode_takes_everything_but_inject_text_from_the_rule_planner():
    replies = iter(['{"proposers":["uniform_stride"],"inject_text":["asr","bogus"],"aligners":["clap_alm"]}'])
    llm = LLMCascadePlanner(generate_fn=lambda prompt: next(replies), inject_only_mode=True,
                            fallback=RuleBasedCascadePlanner(profile="audit"))
    plan = llm.plan("What is the video about?", "video_mme", LONG, registry=_registry())
    assert plan.inject_text == ["asr"]                      # whitelist drops unknown names
    assert plan.proposers == ["ears_event", "uniform_stride"] and plan.filters == ["motion_diff"]
    assert plan.planner == "llm+ruleBase"
