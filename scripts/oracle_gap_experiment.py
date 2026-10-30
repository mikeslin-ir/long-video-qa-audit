#!/usr/bin/env python3
"""Oracle-gap "killer experiment" aggregator.

Decides, BEFORE investing in a learned router / dynamic-K / self-reflection,
whether those mechanisms have enough head-room to be worth building.

It does NOT run any VLM or selector itself. Instead it reads the per-item
`rows.jsonl` files produced by `scripts/run_qa_benchmark.py` for several fixed
selectors run on the SAME item set at the SAME budget K, and computes:

  1. per-selector accuracy + mean cost (selection + VLM seconds, #frames);
  2. best-fixed selector accuracy (the strongest single selector);
  3. ORACLE accuracy = per-item max over selectors (upper bound of a perfect
     content-conditioned router);
  4. ORACLE GAP = oracle - best_fixed  ->  the maximum acc a *perfect* router
     could add over always using the best single selector. This is the ceiling
     for ANY learned router. If it is tiny, a learned router is not worth it;
  5. trivial-gate ceilings: for every content selector s, the 2-way oracle of
     {uniform, s}. If a 2-way gate already captures most of the full gap, a
     learned router over many experts adds little novelty;
  6. keyword-rule trivial gate accuracy (a 5-line if/else), to show how much of
     the gap a training-free rule captures;
  7. "just add frames" comparison: if a uniform@(2K) run is supplied, whether
     the oracle@K actually beats simply doubling the frame budget. This bounds
     the head-room for dynamic-K / re-think: if uniform@2K >= oracle@K, then
     spending compute on more frames dominates spending it on smarter routing.

Everything is computed on the qids COMMON to all supplied runs, so the numbers
are paired and directly comparable.

Usage
-----
  python scripts/oracle_gap_experiment.py \
      --dataset xd_violence --k 8 \
      --runs uniform=outputs/og_xd_uniform_k8 \
             motion=outputs/og_xd_motion_k8 \
             clip=outputs/og_xd_clip_k8 \
             aks=outputs/og_xd_aks_k8 \
             cascade=outputs/og_xd_cascade_k8 \
      --more_frames uniform2k=outputs/og_xd_uniform_k16 --more_k 16 \
      --out outputs/og_xd_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------- io helpers
def _load_rows(run_dir: str) -> dict[str, dict]:
    """qid -> row dict, from <run_dir>/rows.jsonl (last write per qid wins)."""
    p = Path(run_dir) / "rows.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"missing rows.jsonl in {run_dir}")
    out: dict[str, dict] = {}
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        qid = r.get("qid")
        if qid is not None:
            out[str(qid)] = r
    if not out:
        raise ValueError(f"no rows parsed from {p}")
    return out


def _parse_runs(pairs: list[str]) -> dict[str, str]:
    runs: dict[str, str] = {}
    for pr in pairs:
        if "=" not in pr:
            raise SystemExit(f"--runs entry must be name=dir, got {pr!r}")
        name, d = pr.split("=", 1)
        runs[name.strip()] = d.strip()
    return runs


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _cost_of(rows: dict[str, dict], qids: list[str]) -> dict[str, float]:
    sel = [rows[q].get("select_runtime_sec") or 0.0 for q in qids]
    vlm = [rows[q].get("expert_cost_sec") or 0.0 for q in qids]
    frm = [rows[q].get("num_selected") or 0 for q in qids]
    return {
        "sel_s": _mean([float(x) for x in sel]),
        "vlm_s": _mean([float(x) for x in vlm]),
        "total_s": _mean([float(a) + float(b) for a, b in zip(sel, vlm)]),
        "frames": _mean([float(x) for x in frm]),
    }


# ---------------------------------------------------------- trivial keyword gate
_CONTENT_WORDS = (
    "violence", "violent", "fight", "fighting", "assault", "shoot", "shooting",
    "explosion", "accident", "steal", "stealing", "shoplift", "robbery",
    "weapon", "blood", "crash", "abuse", "vandalism", "arrest",
    "person", "people", "object", "color", "colour", "wearing", "holding",
    "what", "which", "who", "where", "how many", "count",
)


def _keyword_gate_choice(question: str, available: list[str]) -> str:
    """5-line training-free gate: content query -> semantic (clip), else uniform.

    Falls back gracefully to whatever selectors are actually available.
    """
    q = (question or "").lower()
    is_content = any(w in q for w in _CONTENT_WORDS)
    if is_content:
        for pref in ("clip", "aks", "cascade"):
            if pref in available:
                return pref
    return "uniform" if "uniform" in available else available[0]


def _load_questions(dataset: str) -> dict[str, str]:
    try:
        from router.qa_datasets import load_dataset
        items = load_dataset(dataset, limit=0)
        return {str(it.qid): it.question for it in items}
    except Exception as e:  # dataset not loadable in this env -> skip gate
        print(f"[warn] could not load questions for keyword gate: {e!r}",
              file=sys.stderr)
        return {}


# --------------------------------------------------------------------- core
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--k", type=int, required=True, help="frame budget K used")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="name=dir pairs, one per fixed selector at budget K")
    ap.add_argument("--more_frames", default="",
                    help="optional name=dir for a uniform run at a LARGER "
                         "budget (the 'just add frames' reference)")
    ap.add_argument("--more_k", type=int, default=0,
                    help="the larger budget used by --more_frames")
    ap.add_argument("--gap_worth_it", type=float, default=0.02,
                    help="oracle-gap threshold below which a learned router is "
                         "judged not worth building (default 0.02 = 2 acc pts)")
    ap.add_argument("--out", default="", help="optional JSON report path")
    args = ap.parse_args()

    runs = _parse_runs(args.runs)
    rows_by_sel = {name: _load_rows(d) for name, d in runs.items()}
    sel_names = list(runs.keys())

    # common qids across ALL runs (paired evaluation)
    common = set.intersection(*[set(r.keys()) for r in rows_by_sel.values()])
    if not common:
        raise SystemExit("no common qids across the supplied runs")
    qids = sorted(common)
    n = len(qids)

    # per-item correctness matrix
    correct = {s: {q: bool(rows_by_sel[s][q].get("correct")) for q in qids}
               for s in sel_names}

    # per-selector accuracy + cost
    per_sel = {}
    for s in sel_names:
        acc = _mean([1.0 if correct[s][q] else 0.0 for q in qids])
        per_sel[s] = {"acc": acc, **_cost_of(rows_by_sel[s], qids)}

    best_fixed_name = max(sel_names, key=lambda s: per_sel[s]["acc"])
    best_fixed_acc = per_sel[best_fixed_name]["acc"]

    # full oracle = per-item max over ALL selectors
    oracle_acc = _mean([1.0 if any(correct[s][q] for s in sel_names) else 0.0
                        for q in qids])
    oracle_gap = oracle_acc - best_fixed_acc

    # 2-way oracle ceilings: {uniform, s} for each non-uniform s
    twoway = {}
    if "uniform" in sel_names:
        for s in sel_names:
            if s == "uniform":
                continue
            acc2 = _mean([1.0 if (correct["uniform"][q] or correct[s][q])
                          else 0.0 for q in qids])
            twoway[f"uniform|{s}"] = acc2

    # keyword-rule trivial gate
    questions = _load_questions(args.dataset)
    gate_acc = None
    gate_cov = {}
    if questions:
        hits = []
        for q in qids:
            choice = _keyword_gate_choice(questions.get(q, ""), sel_names)
            gate_cov[choice] = gate_cov.get(choice, 0) + 1
            hits.append(1.0 if correct[choice][q] else 0.0)
        gate_acc = _mean(hits)

    # "just add frames" reference
    more = None
    if args.more_frames:
        mname, mdir = _parse_runs([args.more_frames]).popitem()
        mrows = _load_rows(mdir)
        mcommon = sorted(set(mrows.keys()) & common)
        if mcommon:
            macc = _mean([1.0 if mrows[q].get("correct") else 0.0
                          for q in mcommon])
            # recompute oracle on the SAME sub-set for a fair comparison
            oracle_on_m = _mean([1.0 if any(correct[s][q] for s in sel_names)
                                 else 0.0 for q in mcommon])
            more = {
                "name": mname, "more_k": args.more_k, "n": len(mcommon),
                "acc": macc, "cost": _cost_of(mrows, mcommon),
                "oracle_at_k_on_subset": oracle_on_m,
                "oracle_beats_more_frames": oracle_on_m > macc,
                "delta_oracle_minus_more": oracle_on_m - macc,
            }

    # ----------------------------------------------------------- verdict
    verdicts = []
    if oracle_gap < args.gap_worth_it:
        verdicts.append(
            f"LOW HEAD-ROOM: oracle gap {oracle_gap*100:.1f}pt < "
            f"{args.gap_worth_it*100:.0f}pt -> a learned router can add at most "
            f"this; likely NOT worth building. Ship the best fixed selector "
            f"({best_fixed_name}).")
    else:
        verdicts.append(
            f"HEAD-ROOM EXISTS: oracle gap {oracle_gap*100:.1f}pt -> a learned "
            f"router has room to beat the best fixed selector "
            f"({best_fixed_name}).")
    # how much of the gap a 2-way gate already captures
    if twoway:
        best2_name = max(twoway, key=twoway.get)
        best2_acc = twoway[best2_name]
        cap = (best2_acc - best_fixed_acc)
        frac = cap / oracle_gap if oracle_gap > 1e-9 else 0.0
        verdicts.append(
            f"2-WAY CEILING: best 2-way gate {best2_name} reaches {best2_acc*100:.1f}% "
            f"(captures {frac*100:.0f}% of the gap). "
            + ("A trivial gate suffices; multi-expert learned routing adds little."
               if frac >= 0.7 else
               "A multi-expert learned router still has distinct head-room."))
    if gate_acc is not None:
        verdicts.append(
            f"KEYWORD GATE: training-free if/else reaches {gate_acc*100:.1f}% "
            f"(best fixed {best_fixed_acc*100:.1f}%, oracle {oracle_acc*100:.1f}%).")
    if more is not None:
        if more["oracle_beats_more_frames"]:
            verdicts.append(
                f"SMART-SELECT WINS: oracle@K={args.k} "
                f"({more['oracle_at_k_on_subset']*100:.1f}%) > uniform@{args.more_k} "
                f"({more['acc']*100:.1f}%) at ~{more['cost']['frames']:.0f} frames "
                f"-> smarter routing beats just adding frames; dynamic-K/re-think "
                f"has head-room.")
        else:
            verdicts.append(
                f"MORE-FRAMES WINS: uniform@{args.more_k} "
                f"({more['acc']*100:.1f}%) >= oracle@K={args.k} "
                f"({more['oracle_at_k_on_subset']*100:.1f}%) -> spending compute on "
                f"MORE FRAMES dominates smarter selection; dynamic-K / self-"
                f"reflection head-room is LOW. Do not build them.")

    # ----------------------------------------------------------- report out
    report = {
        "dataset": args.dataset, "k": args.k, "n_common": n,
        "selectors": sel_names,
        "per_selector": per_sel,
        "best_fixed": {"name": best_fixed_name, "acc": best_fixed_acc},
        "oracle_acc": oracle_acc,
        "oracle_gap": oracle_gap,
        "twoway_oracle": twoway,
        "keyword_gate": {"acc": gate_acc, "routing_counts": gate_cov},
        "more_frames": more,
        "verdicts": verdicts,
    }

    # pretty print
    print(f"\n=== Oracle-Gap Report: {args.dataset}  K={args.k}  "
          f"common n={n} ===")
    print(f"{'selector':<12}{'acc':>8}{'sel_s':>8}{'vlm_s':>8}{'frames':>8}")
    for s in sel_names:
        ps = per_sel[s]
        print(f"{s:<12}{ps['acc']*100:>7.1f}%{ps['sel_s']:>8.2f}"
              f"{ps['vlm_s']:>8.2f}{ps['frames']:>8.1f}")
    print("-" * 44)
    print(f"best_fixed   = {best_fixed_name} @ {best_fixed_acc*100:.1f}%")
    print(f"oracle(all)  = {oracle_acc*100:.1f}%   "
          f"gap = {oracle_gap*100:+.1f}pt  (learned-router ceiling)")
    if twoway:
        for k, v in sorted(twoway.items(), key=lambda kv: -kv[1]):
            print(f"  2way {k:<18} = {v*100:.1f}%")
    if gate_acc is not None:
        print(f"keyword_gate = {gate_acc*100:.1f}%   routing={gate_cov}")
    if more is not None:
        print(f"more-frames  = uniform@{args.more_k} {more['acc']*100:.1f}% "
              f"(~{more['cost']['frames']:.0f} frames, "
              f"{more['cost']['total_s']:.2f}s/item, n={more['n']})")
    print("-" * 44)
    for v in verdicts:
        print("VERDICT: " + v)
    print()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
