"""Tool registry. A `CascadeRegistry` is essentially three named dicts."""
from __future__ import annotations

from dataclasses import dataclass, field

from .protocols import FrameProposer, KeyframeFilter, EvidenceAligner


@dataclass
class CascadeRegistry:
    proposers: dict[str, FrameProposer] = field(default_factory=dict)
    filters: dict[str, KeyframeFilter] = field(default_factory=dict)
    aligners: dict[str, EvidenceAligner] = field(default_factory=dict)

    def register_proposer(self, name: str, tool: FrameProposer) -> None:
        self.proposers[name] = tool

    def register_filter(self, name: str, tool: KeyframeFilter) -> None:
        self.filters[name] = tool

    def register_aligner(self, name: str, tool: EvidenceAligner) -> None:
        self.aligners[name] = tool

    def available(self) -> dict[str, list[str]]:
        return {
            "proposers": sorted(self.proposers.keys()),
            "filters": sorted(self.filters.keys()),
            "aligners": sorted(self.aligners.keys()),
        }


# Lazy default registry: filled by default_registry() the first time it's
# accessed so importing this module is free (no torch/cv2 side effects).
_DEFAULT: CascadeRegistry | None = None


def default_registry(
    cache_dir: str | None = None,
    siglip_model: str = "google/siglip-base-patch16-224",
) -> CascadeRegistry:
    """Return a singleton registry populated with the built-in tools.

    Tools that depend on optional third-party packages (PySceneDetect,
    EARS, Whisper) are registered if importable; otherwise skipped with a
    short warning so the cascade gracefully degrades. ``cache_dir`` defaults
    to ``$LONG_VIDEO_QA_CASCADE_CACHE`` or ``results/cascade_cache``.
    """
    global _DEFAULT
    if _DEFAULT is not None:
        return _DEFAULT
    from data_paths import get_cascade_cache_dir
    cache_dir = cache_dir or get_cascade_cache_dir()

    reg = CascadeRegistry()

    # --- Proposers (L1) ------------------------------------------------------
    from .proposers.uniform import UniformStrideProposer
    reg.register_proposer("uniform_stride", UniformStrideProposer())

    try:
        from .proposers.keyframe_cache import KeyframeCacheProposer
        reg.register_proposer("keyframe_cache", KeyframeCacheProposer())
    except Exception as e:
        import warnings
        warnings.warn(f"[cascade] keyframe_cache unavailable: {e}")

    try:
        from .proposers.shot_boundary import ShotBoundaryProposer
        reg.register_proposer("shot_boundary", ShotBoundaryProposer(cache_dir=cache_dir))
    except Exception as e:
        import warnings
        warnings.warn(f"[cascade] shot_boundary unavailable: {e}")

    try:
        from .proposers.ears_event import EARSEventProposer
        reg.register_proposer("ears_event", EARSEventProposer(cache_dir=cache_dir))
    except Exception as e:
        import warnings
        warnings.warn(f"[cascade] ears_event unavailable: {e}")

    try:
        from .proposers.stacfp import STACFPProposer
        reg.register_proposer("stacfp", STACFPProposer(cache_dir=cache_dir))
    except Exception as e:
        import warnings
        warnings.warn(f"[cascade] stacfp unavailable: {e}")

    # --- Filters (L2) --------------------------------------------------------
    from .filters.motion_filter import MotionDiffFilter
    reg.register_filter("motion_diff", MotionDiffFilter())

    from .filters.redundancy_filter import RedundancyFilter
    reg.register_filter("redundancy", RedundancyFilter())

    # --- Aligners (L3) -------------------------------------------------------
    # Adapter wraps any existing evidence_skills.EvidenceSkill as an EvidenceAligner.
    from .aligners.adapter import EvidenceSkillAligner

    try:
        from evidence_skills import SemanticSigLIPSkill, ObjectSigLIPSkill
        reg.register_aligner(
            "semantic_siglip",
            EvidenceSkillAligner(
                "semantic_siglip", "vision", 0.03,
                SemanticSigLIPSkill(cache_dir=cache_dir, model_id=siglip_model),
            ),
        )
        reg.register_aligner(
            "object_siglip",
            EvidenceSkillAligner(
                "object_siglip", "vision", 0.03,
                ObjectSigLIPSkill(cache_dir=cache_dir, model_id=siglip_model),
            ),
        )
    except Exception as e:
        import warnings
        warnings.warn(f"[cascade] siglip aligners unavailable: {e}")

    try:
        from evidence_skills import OCRTextSkill
        reg.register_aligner(
            "ocr_text",
            EvidenceSkillAligner("ocr_text", "vision", 0.05, OCRTextSkill(cache_dir=cache_dir)),
        )
    except Exception as e:
        import warnings
        warnings.warn(f"[cascade] ocr_text unavailable: {e}")

    try:
        from evidence_skills import ASRWhisperSkill
        reg.register_aligner(
            "asr_whisper",
            EvidenceSkillAligner("asr_whisper", "audio", 0.1, ASRWhisperSkill(cache_dir=cache_dir)),
        )
    except Exception as e:
        import warnings
        warnings.warn(f"[cascade] asr_whisper unavailable: {e}")

    # CLAP audio-language relevance: question-conditioned audio cueing from
    # pre-extracted CLAP embeddings. Cheap at query time (one text forward +
    # cosine) and the key signal for audio-dependent datasets (e.g. MUSIC-AVQA)
    # where neither ASR text nor pixels expose the answer.
    try:
        from evidence_skills import CLAPSkill
        reg.register_aligner(
            "clap_alm",
            EvidenceSkillAligner("clap_alm", "audio", 0.05, CLAPSkill(cache_dir=cache_dir)),
        )
    except Exception as e:
        import warnings
        warnings.warn(f"[cascade] clap_alm unavailable: {e}")

    # Short-time audio energy: cheap acoustic-salience cue (loud events,
    # screams, music onsets). Complements CLAP/ASR for event-sparse anomaly QA.
    try:
        from evidence_skills import AudioEnergySkill
        reg.register_aligner(
            "audio_energy",
            EvidenceSkillAligner("audio_energy", "audio", 0.02, AudioEnergySkill(cache_dir=cache_dir)),
        )
    except Exception as e:
        import warnings
        warnings.warn(f"[cascade] audio_energy unavailable: {e}")

    # Cache-aware SigLIP aligner: reuses offline keyframe embeddings; only a
    # single text forward + dot product at query time.
    try:
        from .aligners.cached_siglip import CachedSigLIPAligner
        reg.register_aligner(
            "semantic_siglip_cached",
            CachedSigLIPAligner(model_id=siglip_model),
        )
    except Exception as e:
        import warnings
        warnings.warn(f"[cascade] semantic_siglip_cached unavailable: {e}")

    _DEFAULT = reg
    return reg
