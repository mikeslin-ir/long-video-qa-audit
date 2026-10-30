from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from evidence_skills import (
    ASRSkillStub,
    ASRWhisperSkill,
    AudioEnergySkill,
    CLAPSkill,
    MotionSkill,
    ObjectSigLIPSkill,
    OCRTextSkill,
    OCRSkillStub,
    ObjectSkillStub,
    SemanticCLIPSkill,
    SemanticSigLIPSkill,
    sample_candidate_frames,
)
from fusion import fuse_skill_scores
from router.planner import LLMPlanner, RuleBasedSkillPlanner
from router.memory import MemoryAugmentedPlanner, MemoryStore
from router.experts import Expert, ExpertEvidence, default_pool, make_expert
from temporal_nms import select_frames_with_temporal_nms
from submodular import submodular_select


class SkillRouter:
    """Training-free question-conditioned skill routing + frame selection."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        cache_dir = self.config.get("cache_dir", "results/skill_router_cache")
        self.skills = {
            "motion": MotionSkill(cache_dir=cache_dir),
            "audio": AudioEnergySkill(cache_dir=cache_dir),
            "clap_alm": CLAPSkill(
                cache_dir=cache_dir,
                model_id=self.config.get("clap_model", "laion/larger_clap_music_and_speech"),
                feature_roots=self.config.get("clap_feature_roots"),
                device=self.config.get("clap_device", "cpu"),
            ),
            "semantic_clip": SemanticCLIPSkill(
                cache_dir=cache_dir,
                model_id=self.config.get("semantic_clip_model", "openai/clip-vit-base-patch32"),
            ),
            "semantic_siglip": SemanticSigLIPSkill(
                cache_dir=cache_dir,
                model_id=self.config.get("semantic_siglip_model", "google/siglip-base-patch16-224"),
            ),
            "object_siglip": ObjectSigLIPSkill(
                cache_dir=cache_dir,
                model_id=self.config.get("object_siglip_model", self.config.get("semantic_siglip_model", "google/siglip-base-patch16-224")),
            ),
            "asr_whisper": ASRWhisperSkill(
                cache_dir=cache_dir,
                model_id=self.config.get("asr_whisper_model", "openai/whisper-tiny.en"),
            ),
            "ocr_text": OCRTextSkill(
                cache_dir=cache_dir,
                backend=self.config.get("ocr_backend", "auto"),
                conf_threshold=float(self.config.get("ocr_conf_threshold", 0.35)),
            ),
            "ocr_stub": OCRSkillStub(),
            "asr_stub": ASRSkillStub(),
            "object_stub": ObjectSkillStub(),
        }
        planner_mode = self.config.get("planner", "rule_based")
        self.planner = RuleBasedSkillPlanner() if planner_mode == "rule_based" else LLMPlanner()

        # Heterogeneous expert pool (vlm / alm / avlm / caplm).
        # Override via config["experts"] = {name: {kwargs}} or pass
        # custom Expert objects through `set_experts()` after construction.
        exp_cfg = self.config.get("experts", {}) or {}
        if exp_cfg:
            self.experts: dict[str, Expert] = {
                name: make_expert(name, **(kw or {})) for name, kw in exp_cfg.items()
            }
        else:
            self.experts = default_pool()

        # Optional procedural memory.
        mem_cfg = self.config.get("memory", {}) or {}
        self.memory: MemoryStore | None = None
        if mem_cfg.get("enabled", False):
            self.memory = MemoryStore(
                path=mem_cfg.get("path", "results/skill_router_cache/memory.jsonl")
            )
            self.planner = MemoryAugmentedPlanner(
                self.planner,
                memory=self.memory,
                alpha=float(mem_cfg.get("alpha", 0.4)),
                k=int(mem_cfg.get("k", 5)),
                min_similarity=float(mem_cfg.get("min_similarity", 0.3)),
                min_reward=float(mem_cfg.get("min_reward", 0.0)),
            )

    @staticmethod
    def from_yaml(path: str | Path) -> "SkillRouter":
        cfg = yaml.safe_load(Path(path).read_text())
        return SkillRouter(cfg)

    def run(
        self,
        video_path: str,
        question: str,
        dataset: str,
        budget: int,
        tau_sec: float,
        output_dir: str,
        planner_mode: str | None = None,
        enabled_skills: list[str] | None = None,
        use_submodular: bool = False,
        submodular_kernel: str = "gauss",
        planner: Any | None = None,
        invoke_expert: bool = False,
        expert_override: str | None = None,
    ) -> dict[str, Any]:
        t0 = time.time()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        stride = int(self.config.get("candidate_stride", 5))
        max_candidates = self.config.get("max_candidates")
        normalization = self.config.get("normalization", "minmax")

        frames, meta = sample_candidate_frames(video_path, stride=stride, max_frames=max_candidates)
        if not frames:
            raise RuntimeError("No candidate frames decoded.")

        if planner is None:
            planner = self.planner
            if planner_mode == "llm":
                planner = LLMPlanner()
            elif planner_mode == "rule_based":
                planner = RuleBasedSkillPlanner()

        plan = planner.plan(question=question, dataset=dataset, config=self.config)
        selected = list(plan.get("selected_skills", []))
        use_stub_skills = bool(self.config.get("use_stub_skills", False))
        if not use_stub_skills:
            selected = [s for s in selected if not s.endswith("_stub")]
        if enabled_skills:
            keep = set(enabled_skills)
            selected = [s for s in selected if s in keep]
            if not selected:
                selected = list(enabled_skills)

        weights = dict(plan.get("weights", {}))
        prompts = list(plan.get("evidence_prompts", []))

        skill_scores: dict[str, np.ndarray] = {}
        skill_runtime: dict[str, float] = {}
        for name in selected:
            skill = self.skills.get(name)
            if skill is None:
                continue
            st = time.time()
            s = skill.compute(
                video_path=video_path,
                frames=frames,
                question=question,
                metadata={
                    "dataset": dataset,
                    "evidence_prompts": prompts,
                    **meta,
                },
            )
            skill_runtime[name] = time.time() - st
            if len(s) != len(frames):
                x_old = np.linspace(0.0, 1.0, len(s)) if len(s) > 0 else np.array([0.0])
                x_new = np.linspace(0.0, 1.0, len(frames))
                s = np.interp(x_new, x_old, s) if len(s) > 0 else np.zeros((len(frames),), dtype=np.float32)
            skill_scores[name] = np.asarray(s, dtype=np.float32)

        if not skill_scores:
            raise RuntimeError("No valid skill scores produced.")

        # Keep only weights for actually-run skills and renormalize.
        weights = {k: float(weights.get(k, 0.0)) for k in skill_scores.keys()}
        sw = sum(max(v, 0.0) for v in weights.values())
        if sw <= 0:
            weights = {k: 1.0 / len(weights) for k in weights}
        else:
            weights = {k: max(v, 0.0) / sw for k, v in weights.items()}

        fused, normalized = fuse_skill_scores(
            skill_scores=skill_scores,
            weights=weights,
            normalization=normalization,
            lambda_uniform=float(plan.get("lambda_uniform", self.config.get("lambda_uniform", 0.0))),
        )

        timestamps = np.array([f.timestamp_sec for f in frames], dtype=np.float32)
        if use_submodular:
            picked_local = submodular_select(
                scores=fused,
                timestamps=timestamps,
                K=int(budget),
                tau_sec=float(tau_sec),
                kernel=submodular_kernel,
            )
            selection_method = f"submodular-{submodular_kernel}"
        else:
            picked_local = select_frames_with_temporal_nms(
                scores=fused,
                timestamps=timestamps,
                K=int(budget),
                tau_sec=float(tau_sec),
            )
            selection_method = "temporal_nms"

        selected_frames = []
        selected_dir = out_dir / "selected_frames"
        selected_dir.mkdir(parents=True, exist_ok=True)
        for i in picked_local:
            fr = frames[i]
            img_name = f"frame_{fr.frame_index:08d}.jpg"
            img_path = selected_dir / img_name
            if fr.image is not None:
                cv2.imwrite(str(img_path), cv2.cvtColor(fr.image, cv2.COLOR_RGB2BGR))
            selected_frames.append(
                {
                    "frame_index": int(fr.frame_index),
                    "timestamp_sec": float(fr.timestamp_sec),
                    "fused_score": float(fused[i]),
                    "image_path": str(img_path),
                }
            )

        skill_scores_path = out_dir / "skill_scores.npz"
        fused_scores_path = out_dir / "fused_scores.npy"
        np.savez_compressed(skill_scores_path, **{k: v.astype(np.float32) for k, v in skill_scores.items()})
        np.save(fused_scores_path, fused.astype(np.float32))

        result = {
            "video_id": Path(video_path).stem,
            "question": question,
            "dataset": dataset,
            "method": "skill_router",
            "selected_skills": list(skill_scores.keys()),
            "weights": weights,
            "evidence_prompts": prompts,
            "selected_frames": selected_frames,
            "skill_scores_path": str(skill_scores_path),
            "fused_scores_path": str(fused_scores_path),
            "planner_reason": plan.get("reason", ""),
            "planner_output": plan,
            "normalization": normalization,
            "selection_method": selection_method,
            "tau_sec": float(tau_sec),
            "budget": int(budget),
            "num_candidates": len(frames),
            "runtime_sec": time.time() - t0,
            "skill_runtime_sec": skill_runtime,
            "timestamps_path": str(out_dir / "timestamps.npy"),
            "expert_planned": plan.get("expert"),
            "expert_beta": plan.get("expert_beta"),
        }
        np.save(out_dir / "timestamps.npy", timestamps)

        # Optionally route to a multimodal expert and run inference.
        if invoke_expert:
            expert_name = expert_override or plan.get("expert") or "vlm"
            expert = self.experts.get(expert_name)
            if expert is not None:
                ev = ExpertEvidence(
                    question=question,
                    frames=[
                        frames[i].image for i in picked_local
                        if frames[i].image is not None
                    ] or None,
                    audio=None,
                    captions=None,
                    asr_text=None,
                    ocr_text=None,
                    metadata={"dataset": dataset, "video_id": Path(video_path).stem},
                )
                if expert.can_handle(ev):
                    expert_out = expert.infer(ev)
                    result["expert"] = expert_name
                    result["expert_answer"] = expert_out.answer
                    result["expert_log_prob"] = expert_out.log_prob
                    result["expert_cost_sec"] = expert_out.cost_sec
                    result["expert_extras"] = expert_out.extras
                else:
                    result["expert"] = expert_name
                    result["expert_skipped_reason"] = "missing modality"
            else:
                result["expert_skipped_reason"] = f"unknown expert {expert_name}"

        (out_dir / "result.json").write_text(json.dumps(result, indent=2))
        return result
