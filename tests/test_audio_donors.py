from __future__ import annotations

import pytest

from router.backbones import VideoLLaMA2AVExpert, audio_donor_map

VIDEOS = [f"/data/video_{i:03d}.mp4" for i in range(50)]


def test_every_video_gets_the_soundtrack_of_exactly_one_other_video():
    donors = audio_donor_map(VIDEOS)
    assert set(donors) == set(VIDEOS)
    assert all(v != d for v, d in donors.items())          # nobody keeps its own soundtrack
    assert sorted(donors.values()) == sorted(VIDEOS)       # a permutation: each lends exactly once


def test_the_map_depends_on_the_set_of_videos_not_their_order_or_repeats():
    a = audio_donor_map(VIDEOS)
    b = audio_donor_map(list(reversed(VIDEOS)) + VIDEOS[:5])
    assert a == b


def test_different_seeds_give_different_maps():
    assert audio_donor_map(VIDEOS, seed=1) != audio_donor_map(VIDEOS, seed=2)


def test_needs_at_least_two_videos():
    with pytest.raises(ValueError):
        audio_donor_map(["only.mp4"])


def test_expert_accepts_the_mismatched_mode_and_rejects_unknown_modes():
    expert = VideoLLaMA2AVExpert(audio_mode="mismatched")
    expert.set_audio_donors(VIDEOS)
    assert expert._donors[VIDEOS[0]] != VIDEOS[0]
    with pytest.raises(ValueError):
        VideoLLaMA2AVExpert(audio_mode="noise")
