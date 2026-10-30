"""Self-contained VideoQA dataset loaders for the routing benchmark.

Each loader yields a list of `QAItem` with a uniform schema so the
evaluation harness can treat MCQ and open-ended datasets identically.
Gold answers are loaded so accuracy and the counterfactual oracle can be
computed.

Supported: egoschema, video_mme, msvd_qa, msrvtt_qa, music_avqa.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_paths import get_data_root

_LETTERS = "ABCDE"
_ARTICLES = {"a", "an", "the"}
_PUNCT_RE = re.compile(r"[^\w\s]")

DATA = get_data_root()


@dataclass
class QAItem:
    qid: str
    video_path: str
    question: str
    options: list[str] = field(default_factory=list)   # [] => open-ended
    answer: str = ""          # letter for MCQ, normalized text for open
    answer_idx: int = -1
    task_type: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_mcq(self) -> bool:
        return bool(self.options)


def normalize_answer(text: str) -> str:
    text = (text or "").lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    toks = [t for t in text.split() if t not in _ARTICLES]
    return " ".join(toks).strip()


# ---------------------------------------------------------------- EgoSchema
def load_egoschema(limit: int = 0) -> list[QAItem]:
    root = DATA / "EgoSchema"
    qs = json.loads((root / "questions.json").read_text())
    ans = json.loads((root / "subset_answers.json").read_text())
    vid_dir = root / "videos" / "good_clips_git"
    items: list[QAItem] = []
    for r in qs:
        quid = str(r.get("q_uid"))
        if quid not in ans:
            continue
        vp = vid_dir / f"{quid}.mp4"
        if not vp.exists():
            continue
        opts = [str(r.get(f"option {i}", "")) for i in range(5)]
        idx = int(ans[quid])
        items.append(QAItem(
            qid=quid, video_path=str(vp), question=str(r.get("question", "")),
            options=opts, answer=_LETTERS[idx], answer_idx=idx,
            task_type="egoschema",
        ))
        if limit and len(items) >= limit:
            break
    return items


# ---------------------------------------------------------------- Video-MME
def load_video_mme(limit: int = 0) -> list[QAItem]:
    import pandas as pd

    root = DATA / "videomme"
    vid_dir = root / "videos" / "data"
    pq = list(root.glob("hf_meta/**/*.parquet"))
    if not pq:
        return []
    df = pd.read_parquet(pq[0])
    items: list[QAItem] = []
    for _, r in df.iterrows():
        vid = str(r["videoID"])
        vp = vid_dir / f"{vid}.mp4"
        if not vp.exists():
            continue
        opts = [str(o) for o in (r["options"] if r["options"] is not None else [])]
        ans = str(r["answer"]).strip().upper()[:1]
        idx = _LETTERS.find(ans) if ans else -1
        items.append(QAItem(
            qid=str(r["question_id"]), video_path=str(vp),
            question=str(r["question"]), options=opts, answer=ans,
            answer_idx=idx, task_type=str(r.get("duration", "")),
        ))
        if limit and len(items) >= limit:
            break
    return items


# ---------------------------------------------------------------- MSVD-QA
def load_msvd_qa(limit: int = 0) -> list[QAItem]:
    root = DATA / "MSVD-QA"
    qa = json.loads((root / "test_qa.json").read_text())
    vid_dir = DATA / "MSVD_videos"
    # map numeric id -> youtube clip name via youtube_mapping if present
    mapping_file = root / "youtube_mapping.txt"
    id2name: dict[str, str] = {}
    if mapping_file.exists():
        for line in mapping_file.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                # format: <ytid_clip> vid<NUM>
                name, vid = parts[0], parts[1].replace("vid", "")
                id2name[vid] = name
    items: list[QAItem] = []
    all_vids = {p.stem: p for p in vid_dir.glob("*.mp4")}
    for r in qa:
        vnum = str(r.get("video_id"))
        name = id2name.get(vnum)
        vp = None
        if name and name in all_vids:
            vp = all_vids[name]
        if vp is None:
            continue
        items.append(QAItem(
            qid=str(r.get("id")), video_path=str(vp),
            question=str(r.get("question", "")), options=[],
            answer=normalize_answer(str(r.get("answer", ""))),
            task_type="msvd_qa",
        ))
        if limit and len(items) >= limit:
            break
    return items


# ---------------------------------------------------------------- MUSIC-AVQA
def load_music_avqa(limit: int = 0) -> list[QAItem]:
    root = DATA / "MUSIC-AVQA"
    ann = root / "annotations" / "json_update" / "avqa-test.json"
    if not ann.exists():
        ann = root / "annotations" / "avqa-test.json"
    if not ann.exists():
        return []
    rows = json.loads(ann.read_text())
    real = root / "videos" / "MUSIC-AVQA-videos-Real"
    synth = root / "videos" / "MUCIS-AVQA-videos-Synthetic"
    items: list[QAItem] = []
    for r in rows:
        vid = str(r.get("video_id"))
        vp = None
        for d in (real, synth):
            cand = d / f"{vid}.mp4"
            if cand.exists():
                vp = cand
                break
        if vp is None:
            continue
        # open-ended; answer stored under 'anser' (sic) in original
        ans = str(r.get("anser") or r.get("answer") or "")
        q = str(r.get("question_content", ""))
        # Fill template placeholders (<Object>, <LR>, ...) with templ_values,
        # otherwise the question reaches the model malformed (e.g. "How many
        # sounding <Object> in the video?"). templ_values is a JSON list filling
        # the placeholders left-to-right. Also normalize the full-width '？'.
        templ_raw = r.get("templ_values")
        templ_vals: list[str] = []
        if isinstance(templ_raw, str) and templ_raw.strip():
            try:
                templ_vals = [str(v) for v in json.loads(templ_raw)]
            except Exception:
                templ_vals = []
        elif isinstance(templ_raw, list):
            templ_vals = [str(v) for v in templ_raw]
        if templ_vals:
            ti = iter(templ_vals)
            q = re.sub(r"<[^>]+>", lambda _m: next(ti, _m.group(0)), q)
        q = q.replace("？", "?").strip()
        items.append(QAItem(
            qid=f"{vid}_{r.get('question_id')}", video_path=str(vp),
            question=q, options=[], answer=normalize_answer(ans),
            task_type=str(r.get("type", "")),
            meta={"needs_audio": True},
        ))
        if limit and len(items) >= limit:
            break
    return items


def _balanced_interleave(pos: list[QAItem], neg: list[QAItem]) -> list[QAItem]:
    """Interleave two label groups so any prefix (e.g. under --limit) stays
    roughly class-balanced."""
    out: list[QAItem] = []
    i = j = 0
    while i < len(pos) or j < len(neg):
        if i < len(pos):
            out.append(pos[i]); i += 1
        if j < len(neg):
            out.append(neg[j]); j += 1
    return out


# ---------------------------------------------------------------- UCF-Crime
def load_ucf_crime(limit: int = 0) -> list[QAItem]:
    """Binary event-QA framing of UCF-Crime: 'Does this video contain unusual or
    criminal activity?' Yes (A) for anomaly classes, No (B) for normal videos.
    Event-sparse by construction: the anomaly is a short segment of a long
    surveillance clip, so uniform sampling is likely to miss it."""
    root = DATA / "UCF-Crime"
    split = root / "Anomaly_Detection_splits" / "Anomaly_Test.txt"
    if not split.exists():
        return []
    parts = [root / f"Anomaly-Videos-Part-{p}" for p in (1, 2, 3, 4)]
    normal_dir = root / "Testing_Normal_Videos_Anomaly"
    q = "Does this video contain any unusual, dangerous, or criminal activity?"
    opts = ["Yes", "No"]
    pos: list[QAItem] = []
    neg: list[QAItem] = []
    for line in split.read_text().splitlines():
        rel = line.strip()
        if not rel:
            continue
        is_normal = rel.lower().startswith("testing_normal")
        name = Path(rel).name
        vp = None
        if is_normal:
            cand = normal_dir / name
            if cand.exists():
                vp = cand
        else:
            cls = rel.split("/")[0]
            for base in parts:
                cand = base / cls / name
                if cand.exists():
                    vp = cand
                    break
        if vp is None:
            continue
        idx = 1 if is_normal else 0          # No=B for normal, Yes=A for anomaly
        item = QAItem(
            qid=name, video_path=str(vp), question=q, options=opts,
            answer=_LETTERS[idx], answer_idx=idx,
            task_type="normal" if is_normal else "anomaly",
        )
        (neg if is_normal else pos).append(item)
    items = _balanced_interleave(pos, neg)
    if limit:
        items = items[:limit]
    return items


# ---------------------------------------------------------------- XD-Violence
def load_xd_violence(limit: int = 0) -> list[QAItem]:
    """Binary event-QA framing of XD-Violence: 'Does this video contain violent
    activity?' Yes (A) for Fighting/Shooting, No (B) for Normal."""
    root = DATA / "XD_violence" / "test"
    if not root.exists():
        return []
    q = "Does this video contain violent activity such as fighting or shooting?"
    opts = ["Yes", "No"]
    pos: list[QAItem] = []
    neg: list[QAItem] = []
    for sub in ("Fighting", "Shooting", "Normal"):
        d = root / sub
        if not d.exists():
            continue
        is_violence = sub != "Normal"
        idx = 0 if is_violence else 1        # Yes=A for violence, No=B for normal
        for p in sorted(d.glob("*.mp4")):
            item = QAItem(
                qid=p.stem, video_path=str(p), question=q, options=opts,
                answer=_LETTERS[idx], answer_idx=idx,
                task_type=sub.lower(),
            )
            (pos if is_violence else neg).append(item)
    items = _balanced_interleave(pos, neg)
    if limit:
        items = items[:limit]
    return items


# ---------------------------------------------------------------- MLVU
def load_mlvu(limit: int = 0) -> list[QAItem]:
    """MLVU multiple-choice long-video QA. Loads the four frame-sensitive,
    locally-available task families (needle / count / order / anomaly_reco),
    which stress keyframe selection: the answer hinges on seeing the right
    moment(s) in a multi-minute video. Each record is
    ``{video, duration, question, candidates, answer, question_type}`` with the
    gold answer given as the literal option text."""
    root = DATA / "MLVU"
    tasks = ("2_needle", "4_count", "5_order", "6_anomaly_reco")
    items: list[QAItem] = []
    for task in tasks:
        jf = root / "json" / f"{task}.json"
        vid_dir = root / "videos" / task
        if not jf.exists() or not vid_dir.is_dir():
            continue
        records = json.loads(jf.read_text())
        for i, r in enumerate(records):
            vname = str(r.get("video", ""))
            vp = vid_dir / vname
            if not vp.exists():
                continue
            opts = [str(o) for o in (r.get("candidates") or [])]
            gold = str(r.get("answer", ""))
            if gold not in opts:
                continue  # 2 malformed records across the corpus
            idx = opts.index(gold)
            items.append(QAItem(
                qid=f"{task}__{vp.stem}__q{i}",
                video_path=str(vp),
                question=str(r.get("question", "")),
                options=opts,
                answer=_LETTERS[idx],
                answer_idx=idx,
                task_type=str(r.get("question_type", task)),
                meta={"duration": r.get("duration"), "mlvu_task": task},
            ))
            if limit and len(items) >= limit:
                return items
    return items


# ---------------------------------------------------------------- Agentic-Routing pilot datasets
# All three share the unified JSONL schema produced by
# agentic_routing/scripts/normalize_datasets.py under the dataset root
_AR_ROOT = DATA / "agentic_routing"


def _load_ar_pilot(
    name: str,
    *,
    require_video: bool = True,
    limit: int = 0,
    meta_keys: tuple[str, ...] = (),
) -> list[QAItem]:
    """Common reader for the agentic_routing/processed/<name>/subset_pilot.jsonl
    files. Set ``require_video=False`` for text-only datasets (TVQA) whose
    video frames need a signed agreement and are not on disk."""
    fp = _AR_ROOT / "processed" / name / "subset_pilot.jsonl"
    if not fp.exists():
        return []
    items: list[QAItem] = []
    for line in fp.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        vp_rel = r.get("video_path")
        vp = None
        if vp_rel:
            cand = _AR_ROOT / vp_rel if not Path(vp_rel).is_absolute() else Path(vp_rel)
            if cand.exists():
                vp = str(cand)
        if require_video and vp is None:
            continue
        opts = list(r.get("answer_choices") or [])
        ans_text = str(r.get("answer") or "")
        if opts:
            try:
                idx = next(i for i, o in enumerate(opts) if str(o).strip() == ans_text.strip())
            except StopIteration:
                norm = [str(o).strip().lower() for o in opts]
                try:
                    idx = norm.index(ans_text.strip().lower())
                except ValueError:
                    continue
            ans_letter = _LETTERS[idx] if 0 <= idx < len(_LETTERS) else ""
            qa_ans = ans_letter
            qa_idx = idx
        else:
            qa_ans = normalize_answer(ans_text)
            qa_idx = -1
        meta: dict[str, Any] = {"routing_label": r.get("routing_label")}
        for k in meta_keys:
            v = r.get(k)
            if v is not None:
                meta[k] = v
        for tkey in ("start_time", "end_time"):
            v = r.get(tkey)
            if v is not None:
                meta[tkey] = v
        # audio side-channels (esp. music_avqa CLAP/AST .npz feature paths)
        for k in ("audio_path", "audio_feature_clap_path", "audio_feature_ast_path"):
            v = r.get(k)
            if v:
                cand = _AR_ROOT / v if not Path(v).is_absolute() else Path(v)
                if cand.exists():
                    meta[k] = str(cand)
        items.append(QAItem(
            qid=str(r.get("sample_id") or r.get("video_id")),
            video_path=vp or "",
            question=str(r.get("question", "")),
            options=opts,
            answer=qa_ans,
            answer_idx=qa_idx,
            task_type=str(r.get("question_type") or r.get("routing_label") or name),
            meta=meta,
        ))
        if limit and len(items) >= limit:
            break
    return items


def load_egotextvqa(limit: int = 0) -> list[QAItem]:
    """EgoTextVQA: egocentric scene-text VQA (indoor + outdoor).
    All pilot items are routing_label=visual_ocr, so OCR is the expected
    primary evidence source. Open-ended answers."""
    return _load_ar_pilot(
        "egotextvqa", require_video=True, limit=limit,
        meta_keys=("ocr_text",),
    )


def load_tutorialvqa(limit: int = 0) -> list[QAItem]:
    """TutorialVQA: software-screencast QA. Answers are transcript spans;
    routing labels are mostly visual_asr. Official transcripts available in
    raw/tutorialvqa/videos.json; can be pre-cached into the Whisper ASR
    cache via scripts/prewarm_tutorialvqa_transcripts.py."""
    return _load_ar_pilot(
        "tutorialvqa", require_video=True, limit=limit,
        meta_keys=("subtitle",),
    )


def load_tvqa(limit: int = 0) -> list[QAItem]:
    """TVQA: 5-way MCQ over TV-show clips. Video frames require a signed
    agreement and are not on disk, but official subtitles are. Loaded with
    ``video_path=""`` and ``meta['text_only']=True``; ``meta['subtitle']``
    carries the clipped .srt span. This is a deliberate F2 ablation surface:
    a pure injection-only path with zero visual evidence."""
    items = _load_ar_pilot(
        "tvqa", require_video=False, limit=limit,
        meta_keys=("subtitle",),
    )
    for it in items:
        it.meta["text_only"] = True
    return items


LOADERS = {
    "egoschema": load_egoschema,
    "video_mme": load_video_mme,
    "msvd_qa": load_msvd_qa,
    "music_avqa": load_music_avqa,
    "ucf_crime": load_ucf_crime,
    "xd_violence": load_xd_violence,
    "mlvu": load_mlvu,
    "egotextvqa": load_egotextvqa,
    "tutorialvqa": load_tutorialvqa,
    "tvqa": load_tvqa,
}


def load_dataset(name: str, limit: int = 0) -> list[QAItem]:
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name}; have {list(LOADERS)}")
    return LOADERS[name](limit=limit)
