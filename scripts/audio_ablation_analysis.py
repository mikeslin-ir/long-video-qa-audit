#!/usr/bin/env python3
"""MUSIC-AVQA expert contrast, and what part of it is audio.

Section A replays the paper's full-split table: audio-blind Qwen2-VL-7B against
the audio-visual expert VideoLLaMA2.1-AV, per question type, on the same questions,
graded by the same judge. That contrast is a *model swap*.

Section B is the ablation that separates audio from the swap. The same expert is
run a second time with its soundtrack replaced by silence (a zero waveform, the
model's own no-audio fallback), on a stratified sample of the split. With the
three arms paired by question:

    total      = AV(audio)   - blind          what the swap buys
    audio      = AV(audio)   - AV(silent)     what hearing the soundtrack buys
    non-audio  = AV(silent)  - blind          what the swap buys without audio

Every contrast uses one grader for all three arms: ``--grader judge`` (the LLM judge
that grades every MUSIC-AVQA number in the paper; the default) or ``--grader inloop``
(the matcher that scores each answer inside the QA runner, kept as a robustness
check). Mixing graders would make an arm look better or worse than it is, because the
judge accepts paraphrases the matcher rejects (it moves the audio-blind arm by +2.5 pt).
Both graders put the audio share of the swap at about 8%.

    python scripts/audio_ablation_analysis.py [--grader judge|inloop]
    python scripts/audio_ablation_analysis.py --outputs <raw outputs dir> --write-arms

Section B reads ``results/audio_ablation/arms.csv.gz`` (per-question outcomes of the
three arms under both graders, written by ``--write-arms`` from the raw run
directories), so it replays without the raw outputs.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import audit_predictions as ap  # noqa: E402
import audit_stats as st  # noqa: E402

BLIND = "wave9base_qwen2vl_music_avqa_FULL_noinject"
AV = "wave9av_videollama2av_music_avqa_FULL"
SILENT = "audio_ablation_silent_stratified200"
ARMS_FILE = ROOT / "results" / "audio_ablation" / "arms.csv.gz"
ARM_KEYS = ("blind", "av", "silent")
MISMATCHED = "audio_ablation_mismatched_stratified200"
ALL_KEYS = (*ARM_KEYS, "mismatched")
REPS = 10_000


def type_name(raw: str) -> str:
    try:
        return "-".join(json.loads(raw))
    except (TypeError, ValueError):
        return raw


def paired_arms(runs: dict[str, str]):
    correct = {k: ap.load_correct(v) for k, v in runs.items()}
    types = ap.load_task_types(next(iter(runs.values())))
    qids = sorted(set.intersection(*[set(c) for c in correct.values()]) & set(types))
    return qids, {k: np.array([c[q] for q in qids]) for k, c in correct.items()}, [type_name(types[q]) for q in qids]


def contrast(label: str, base, variant, seed: int) -> dict:
    gain, loss = st.discordant(base, variant)
    delta, lo, hi = st.paired_bootstrap_ci(np.asarray(variant) - np.asarray(base), reps=REPS, seed=seed)
    return {"label": label, "n": len(base), "base": 100 * float(np.mean(base)), "variant": 100 * float(np.mean(variant)),
            "delta": delta, "lo": lo, "hi": hi, "gain": gain, "loss": loss, "p": st.mcnemar_exact_p(gain, loss)}


ARM_COLS = ["qid", "task_type", *[f"{k}_{g}" for k in ALL_KEYS for g in ("inloop", "judge")]]


def _read_run(run: str, roots: list[Path], judged: bool):
    """qid -> verdict (judge) or (verdict, task type) (in-loop) from the first root that has the run."""
    name = "judged_rows.jsonl" if judged else "rows.jsonl"
    for root in roots:
        path = root / run / name
        if path.exists():
            if judged:
                return {str(r["qid"]): int(bool(r["judge_yes"])) for r in ap._jsonl(path)}
            return {str(r["qid"]): (int(bool(r["correct"])), str(r.get("task_type", ""))) for r in ap._jsonl(path)}
    return None


def collect_arms(silent_run: str, roots: list[Path], mismatched_run: str = MISMATCHED) -> dict[str, dict] | None:
    """qid -> task type and both graders' verdicts for the arms, from raw run directories.

    The table covers the questions of the blind, audio-on and silenced arms; the mismatched-audio
    arm is added when its run exists (its cells stay empty otherwise).
    """
    runs = dict(zip(ARM_KEYS, (BLIND, AV, silent_run)))
    inloop = {k: _read_run(r, roots, judged=False) for k, r in runs.items()}
    if any(v is None for v in inloop.values()):
        return None
    judged = {k: _read_run(r, roots, judged=True) for k, r in runs.items()}
    inloop["mismatched"] = _read_run(mismatched_run, roots, judged=False)
    judged["mismatched"] = _read_run(mismatched_run, roots, judged=True)
    table = {}
    for q in sorted(set.intersection(*[set(inloop[k]) for k in ARM_KEYS])):
        row = {"task_type": inloop["silent"][q][1]}
        for k in ALL_KEYS:
            has = inloop[k] is not None and q in inloop[k]
            row[f"{k}_inloop"] = inloop[k][q][0] if has else ""
            row[f"{k}_judge"] = judged[k][q] if judged[k] and q in judged[k] else ""
        table[q] = row
    return table


def write_arms(table: dict[str, dict]) -> None:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(ARM_COLS)
    for q, row in table.items():
        w.writerow([q, row["task_type"], *[row[c] for c in ARM_COLS[2:]]])
    ARMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ARMS_FILE, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(buf.getvalue().encode())


def read_arms() -> dict[str, dict] | None:
    if not ARMS_FILE.exists():
        return None
    with gzip.open(ARMS_FILE, "rt", newline="") as fh:
        return {r["qid"]: r for r in csv.DictReader(fh)}


def arm_arrays(table: dict[str, dict], grader: str, keys: tuple[str, ...] = ARM_KEYS):
    """Paired 0/1 arrays of the arms ``keys`` under one grader, over the questions that have all of them."""
    cols = {k: f"{k}_{grader}" for k in keys}
    qids = [q for q, r in table.items() if all(str(r.get(c, "")) != "" for c in cols.values())]
    arms = {k: np.array([int(table[q][c]) for q in qids]) for k, c in cols.items()}
    return qids, arms, [type_name(table[q]["task_type"]) for q in qids]


def full_split_table() -> dict:
    qids, arms, types = paired_arms({"blind": BLIND, "av": AV})
    types = np.array(types)
    rows = {}
    print("A. Full MUSIC-AVQA split: audio-blind VLM vs audio-visual expert (model swap)")
    print(f"   {'type':<22}{'n':>6}{'blind':>8}{'AV':>8}{'delta':>8}  {'95% CI':<16}{'exact p':>10}")
    for i, t in enumerate(sorted(set(types))):
        m = types == t
        r = contrast(t, arms["blind"][m], arms["av"][m], seed=100 + i)
        rows[t] = r
        print(f"   {t:<22}{r['n']:>6}{r['base']:>8.1f}{r['variant']:>8.1f}{r['delta']:>+8.1f}  "
              f"[{r['lo']:+5.1f},{r['hi']:+5.1f}]  {r['p']:>10.2g}")
    r = contrast("micro", arms["blind"], arms["av"], seed=1)
    macro_b = np.mean([x["base"] for x in rows.values()])
    macro_v = np.mean([x["variant"] for x in rows.values()])
    print(f"   {'micro (all)':<22}{r['n']:>6}{r['base']:>8.1f}{r['variant']:>8.1f}{r['delta']:>+8.1f}  "
          f"[{r['lo']:+5.1f},{r['hi']:+5.1f}]  {r['p']:>10.2g}")
    print(f"   {'macro (types)':<22}{'':>6}{macro_b:>8.1f}{macro_v:>8.1f}{macro_v - macro_b:>+8.1f}")
    return {"types": rows, "micro": r, "macro": [macro_b, macro_v]}


def ablation(table: dict[str, dict], grader: str) -> dict | None:
    qids, arms, types = arm_arrays(table, grader)
    if len(qids) < len(table):
        print(f"\nB. skipped: {len(table) - len(qids)} of {len(table)} questions have no {grader} verdict for some arm "
              "(run scripts/llm_judge.py on the silenced run, then --write-arms)")
        return None
    types = np.array(types)
    full_counts = defaultdict(int)
    for t in ap.load_task_types(AV).values():
        full_counts[type_name(t)] += 1
    print(f"\nB. Audio ablation on {len(qids)} stratified questions, all arms graded by {grader} "
          f"({', '.join(f'{t}:{int((types == t).sum())}' for t in sorted(set(types)))})")
    print(f"   {'type':<22}{'n':>5}{'blind':>7}{'silent':>8}{'AV':>7}  {'AV-blind':>9}{'AV-silent':>11}{'silent-blind':>14}")
    out: dict = {"types": {}}
    per_type = {}
    for i, t in enumerate(sorted(set(types))):
        m = types == t
        b, s, a = arms["blind"][m], arms["silent"][m], arms["av"][m]
        tot, aud, non = contrast("total", b, a, 200 + i), contrast("audio", s, a, 300 + i), contrast("non-audio", b, s, 400 + i)
        out["types"][t] = {"total": tot, "audio": aud, "non_audio": non, "silent_acc": 100 * float(s.mean())}
        per_type[t] = (b, s, a)
        print(f"   {t:<22}{m.sum():>5}{tot['base']:>7.1f}{100 * s.mean():>8.1f}{tot['variant']:>7.1f}  "
              f"{tot['delta']:>+9.1f}{aud['delta']:>+11.1f}{non['delta']:>+14.1f}")
    # post-stratify to the full split's type mix
    total_n = sum(full_counts.values())
    weights = {t: full_counts[t] / total_n for t in per_type}
    rng = np.random.default_rng(7)
    def weighted(sel):
        return 100 * sum(weights[t] * np.mean(sel(per_type[t])) for t in per_type)
    point = {k: weighted(f) for k, f in (("blind", lambda x: x[0]), ("silent", lambda x: x[1]), ("av", lambda x: x[2]))}
    boot = {k: np.empty(REPS) for k in ("total", "audio", "non_audio")}
    for r in range(REPS):
        res = {}
        for t, (b, s, a) in per_type.items():
            ix = rng.integers(0, len(b), len(b))
            res[t] = (b[ix].mean(), s[ix].mean(), a[ix].mean())
        boot["total"][r] = 100 * sum(weights[t] * (res[t][2] - res[t][0]) for t in res)
        boot["audio"][r] = 100 * sum(weights[t] * (res[t][2] - res[t][1]) for t in res)
        boot["non_audio"][r] = 100 * sum(weights[t] * (res[t][1] - res[t][0]) for t in res)
    out["micro_poststratified"] = {"accuracy": point}
    print(f"\n   post-stratified micro-average (weights = full-split type shares): "
          f"blind {point['blind']:.1f}, silent {point['silent']:.1f}, audio-visual {point['av']:.1f}")
    for k, label in (("total", "AV - blind     (swap)"), ("audio", "AV - silent    (audio)"), ("non_audio", "silent - blind (swap without audio)")):
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        value = {"total": point["av"] - point["blind"], "audio": point["av"] - point["silent"],
                 "non_audio": point["silent"] - point["blind"]}[k]
        out["micro_poststratified"][k] = {"delta": value, "ci95": [float(lo), float(hi)]}
        print(f"   {label:<38}{value:>+7.1f} pt  95% CI [{lo:+.1f},{hi:+.1f}]")
    a_total = out["micro_poststratified"]["total"]["delta"]
    if abs(a_total) > 1e-9:
        share = 100 * out["micro_poststratified"]["audio"]["delta"] / a_total
        with np.errstate(divide="ignore", invalid="ignore"):
            share_boot = 100 * boot["audio"] / boot["total"]
        s_lo, s_hi = np.percentile(share_boot[np.isfinite(share_boot)], [2.5, 97.5])
        out["micro_poststratified"]["audio_share_pct"] = {"point": share, "ci95": [float(s_lo), float(s_hi)]}
        print(f"   share of the swap's gain attributable to audio: {share:.0f}%  (95% CI [{s_lo:.0f},{s_hi:.0f}])")
    return out


def mismatched_control(table: dict[str, dict], grader: str) -> dict | None:
    """Silent vs mismatched vs true soundtrack, same expert, post-stratified to the full split's type mix.

    ``mismatched`` gives each video the soundtrack of another video. If the model ignored the
    audio's content, it would score like the silenced run; scoring below it means the wrong
    soundtrack actively misleads, i.e. the model reads the soundtrack.
    """
    qids, arms, types = arm_arrays(table, grader, ALL_KEYS)
    if not qids or len(qids) < 0.9 * len(table):
        print(f"\nC. mismatched-audio control skipped: {len(qids)} of {len(table)} questions have all four "
              f"{grader} verdicts")
        return None
    types = np.array(types)
    full_counts = defaultdict(int)
    for t in ap.load_task_types(AV).values():
        full_counts[type_name(t)] += 1
    total_n = sum(full_counts.values())
    names = sorted(set(types))
    weights = {t: full_counts[t] / total_n for t in names}
    masks = {t: types == t for t in names}
    acc = {k: 100 * sum(weights[t] * arms[k][masks[t]].mean() for t in names) for k in ALL_KEYS}
    rng = np.random.default_rng(11)
    pairs = (("silent", "mismatched"), ("mismatched", "av"), ("silent", "av"), ("blind", "mismatched"))
    boot = {pr: np.empty(REPS) for pr in pairs}
    for r in range(REPS):
        ix = {t: rng.integers(0, int(masks[t].sum()), int(masks[t].sum())) for t in names}
        res = {k: {t: arms[k][masks[t]][ix[t]].mean() for t in names} for k in ALL_KEYS}
        for a, b in pairs:
            boot[(a, b)][r] = 100 * sum(weights[t] * (res[b][t] - res[a][t]) for t in names)
    out: dict = {"n": len(qids), "accuracy": acc, "contrasts": {}}
    print(f"\nC. Mismatched-audio control on {len(qids)} questions, all arms graded by {grader} "
          f"(post-stratified): " + ", ".join(f"{k} {acc[k]:.1f}" for k in ALL_KEYS))
    for a, b in pairs:
        gain, loss = st.discordant(arms[a], arms[b])
        lo, hi = np.percentile(boot[(a, b)], [2.5, 97.5])
        delta = acc[b] - acc[a]
        out["contrasts"][f"{a}->{b}"] = {"delta": delta, "ci95": [float(lo), float(hi)], "gain": gain, "loss": loss,
                                         "p": st.mcnemar_exact_p(gain, loss)}
        print(f"   {a:>10} -> {b:<10}{delta:>+7.1f} pt  95% CI [{lo:+.1f},{hi:+.1f}]  b/c {gain}/{loss}  "
              f"exact p {st.mcnemar_exact_p(gain, loss):.2g}")
    return out


def latex_rows(out: dict) -> str:
    """Rows of the supplementary per-type table: type, n, blind, silenced, audio-on, deltas."""
    lines = []
    for t, r in out["types"].items():
        tot, aud = r["total"], r["audio"]
        lines.append(f"{t.replace('Audio-Visual-', 'AV-')} & {tot['n']} & {tot['base']:.1f} & {r['silent_acc']:.1f} & "
                     f"{tot['variant']:.1f} & ${tot['delta']:+.1f}$ & ${aud['delta']:+.1f}$ \\\\")
    m = out["micro_poststratified"]
    lines.append("\\midrule")
    lines.append(f"Micro (post-stratified) & {sum(r['total']['n'] for r in out['types'].values())} & "
                 f"{m['accuracy']['blind']:.1f} & {m['accuracy']['silent']:.1f} & {m['accuracy']['av']:.1f} & "
                 f"${m['total']['delta']:+.1f}$ & ${m['audio']['delta']:+.1f}$ \\\\")
    return "\n".join(lines)


def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--silent-run", default=SILENT)
    ap_.add_argument("--grader", choices=["judge", "inloop"], default="judge",
                     help="one grader for all three arms of the ablation (default: the LLM judge)")
    ap_.add_argument("--outputs", action="append", default=[],
                     help="raw outputs directory holding the arms' runs (repeatable; default: the working tree)")
    ap_.add_argument("--write-arms", action="store_true",
                     help="rebuild results/audio_ablation/arms.csv.gz from the raw run directories")
    ap_.add_argument("--json", type=Path, default=None)
    ap_.add_argument("--latex", action="store_true", help="also print the rows of the supplementary table")
    args = ap_.parse_args()
    roots = [Path(p).resolve() for p in args.outputs] + [ap.outputs_dir()]
    if args.write_arms:
        table = collect_arms(args.silent_run, roots)
        if table is None:
            print("cannot write arms.csv.gz: one of the three runs is missing from " + ", ".join(map(str, roots)))
            return 1
        write_arms(table)
        print(f"wrote {ARMS_FILE.relative_to(ROOT)} ({len(table)} questions)")
    table = read_arms()
    result = {"full_split": full_split_table(), "ablation": None, "mismatched": None, "grader": args.grader}
    if table is None:
        print("\nB. skipped: results/audio_ablation/arms.csv.gz is missing (run with --outputs ... --write-arms)")
    else:
        result["ablation"] = ablation(table, args.grader)
        if any(str(r.get("mismatched_" + args.grader, "")) != "" for r in table.values()):
            result["mismatched"] = mismatched_control(table, args.grader)
    if args.latex and result["ablation"]:
        print("\n% per-type rows: type & n & blind & silenced & audio on & AV-blind & AV-silenced")
        print(latex_rows(result["ablation"]))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
