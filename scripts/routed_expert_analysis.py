#!/usr/bin/env python3
"""Question-only expert dispatch on MUSIC-AVQA (no gold-type oracle).

Two complete passes over the same questions share one judge:

* ``base`` -- audio-blind Qwen2-VL-7B,
* ``av``   -- the audio-visual expert VideoLLaMA2.1-7B-AV.

The question the per-type table cannot answer: if a dispatcher had to pick the expert from
the *question text alone* (no gold question-type label), how close to the oracle upper
bound would it get? We apply the planner's keyword rule (mirrors ``cascade/planner.py``): a
question goes to the audio-visual expert iff it mentions an audio or speech cue word. On the
common qids we report

    base          every question -> audio-blind model
    av            every question -> audio-visual expert (always on, expensive)
    routed        keyword dispatcher from the question text only
    oracle        per-item best of the two (unreachable upper bound)

plus the dispatcher's precision and recall against the oracle "the expert helped" set and a
paired comparison of routed against base.

The dispatch decision needs the question text, which is not part of the shipped outcome table.
With raw run directories present the script reads the text from ``judged_rows.jsonl`` and
``--write-dispatch`` stores each decision in ``results/predictions/music_avqa_dispatch.csv.gz``;
otherwise it reads that file together with the shipped outcomes, so the analysis replays
offline.

    python scripts/routed_expert_analysis.py [--base RUN] [--av RUN] [--write-dispatch]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import audit_predictions as ap  # noqa: E402
import audit_stats as st  # noqa: E402

BASE = "wave9base_qwen2vl_music_avqa_FULL_noinject"
AV = "wave9av_videollama2av_music_avqa_FULL"
DISPATCH = ROOT / "results" / "predictions" / "music_avqa_dispatch.csv.gz"

# Mirror of cascade/planner.py keyword lists (audio + speech cues).
_AUDIO_WORDS = ("music", "song", "sing", "instrument", "loud", "noise", "sound",
                "clap", "scream", "laugh", "rhythm", "rhythmic", "play", "playing",
                "tune", "melody", "hear", "audio")
_SPEECH_WORDS = ("say", "said", "speak", "spoken", "talk", "talked", "talking",
                 "dialog", "dialogue", "conversation", "shout", "yell")


def needs_audio(question: str) -> bool:
    text = (question or "").lower()
    return any(word in text for word in _AUDIO_WORDS + _SPEECH_WORDS)


def questions_from_runs(run: str, outputs: Path | None = None) -> dict[str, str] | None:
    """qid -> question text from a working-tree run, or None if the run is absent."""
    path = (outputs or ap.outputs_dir()) / run / "judged_rows.jsonl"
    if not path.exists():
        return None
    return {str(r["qid"]): r.get("question") or "" for r in ap._jsonl(path)}


def read_dispatch(path: Path = DISPATCH) -> dict[str, bool]:
    with gzip.open(path, "rt", newline="") as fh:
        return {row["qid"]: bool(int(row["pick_av"])) for row in csv.DictReader(fh)}


def write_dispatch(decisions: dict[str, bool], path: Path = DISPATCH) -> None:
    with gzip.GzipFile(path, "wb", mtime=0) as gz:
        text = "qid,pick_av\n" + "".join(f"{q},{int(v)}\n" for q, v in sorted(decisions.items()))
        gz.write(text.encode())


def compute(base: dict[str, int], av: dict[str, int], pick_av: dict[str, bool], reps: int = 10_000, seed: int = 0) -> dict:
    qids = sorted(set(base) & set(av) & set(pick_av))
    b = np.array([base[q] for q in qids])
    a = np.array([av[q] for q in qids])
    pick = np.array([pick_av[q] for q in qids], dtype=bool)
    routed = np.where(pick, a, b)
    oracle = np.maximum(a, b)
    helps = a > b
    tp, fp = int((pick & helps).sum()), int((pick & ~helps).sum())
    fn = int((~pick & helps).sum())
    gain, loss = st.discordant(b, routed)
    delta, lo, hi = st.paired_bootstrap_ci(routed - b, reps=reps, seed=seed)
    n = len(qids)
    pct = lambda x: 100.0 * float(np.mean(x))  # noqa: E731
    return {
        "n": n, "routed_to_av": int(pick.sum()), "routed_to_av_pct": pct(pick),
        "base": pct(b), "av": pct(a), "routed": pct(routed), "oracle": pct(oracle),
        "routed_minus_base": delta, "ci95": [lo, hi], "p": st.mcnemar_exact_p(gain, loss),
        "headroom_captured_pct": 100.0 * (routed.sum() - b.sum()) / (oracle.sum() - b.sum()),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "expert_helped": int(helps.sum()), "expert_hurt": int((b > a).sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": int((~pick & ~helps).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--av", default=AV)
    parser.add_argument("--write-dispatch", action="store_true",
                        help="store the dispatch decisions (needs the raw judged_rows.jsonl)")
    args = parser.parse_args()

    questions = questions_from_runs(args.base) or questions_from_runs(args.av)
    if questions is not None:
        pick = {q: needs_audio(text) for q, text in questions.items()}
        if args.write_dispatch:
            write_dispatch(pick)
            print(f"wrote {DISPATCH.relative_to(ROOT)} ({len(pick)} decisions)")
    elif DISPATCH.exists():
        pick = read_dispatch()
    else:
        print("no raw judged_rows.jsonl and no shipped dispatch file", file=sys.stderr)
        return 1

    r = compute(ap.load_correct(args.base), ap.load_correct(args.av), pick)
    print(f"n common qids: {r['n']}")
    print(f"routed to the AV expert: {r['routed_to_av']} ({r['routed_to_av_pct']:.1f}%)")
    print("-" * 56)
    for key, label in (("base", "base   (all audio-blind)"), ("av", "AV     (all audio-visual expert)"),
                       ("routed", "routed (keyword on text only)"), ("oracle", "oracle (per-item best)")):
        print(f"  {label:<34}: {r[key]:.1f}")
    print("-" * 56)
    print(f"  routed - base  : {r['routed_minus_base']:+.1f} pt  (exact McNemar p={r['p']:.2e}, "
          f"95% CI [{r['ci95'][0]:+.1f},{r['ci95'][1]:+.1f}])")
    print(f"  routed - AV    : {r['routed'] - r['av']:+.1f} pt")
    print(f"  routed captures {r['headroom_captured_pct']:.0f}% of the oracle head-room over base")
    print(f"  dispatcher vs 'the expert helped' ({r['expert_helped']} items; the expert hurt {r['expert_hurt']}): "
          f"precision={r['precision']:.2f} recall={r['recall']:.2f} "
          f"(tp={r['tp']} fp={r['fp']} fn={r['fn']} tn={r['tn']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
