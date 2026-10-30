#!/usr/bin/env python
"""Open-source LLM-as-judge for open-ended video QA, re-scoring existing runs.

Implements the *paper-standard* GPT-assisted evaluation prompt used by
VideoLLaMA 2 (arXiv:2406.07476, Appendix A Table 9) and EgoTextVQA
(arXiv:2502.07411, Table 16): the judge sees Question / Correct Answer /
Predicted Answer and returns ``{'pred': 'yes'|'no', 'score': 0..5}``.
Accuracy is the percentage of 'yes'.

The only difference from the papers is the judge backbone: instead of the
closed GPT-4o-mini / GPT-3.5, we use a local open model
(Qwen2.5-32B-class) so the metric is fully reproducible. We validate that
this judge reproduces the paper's GPT-4o-mini numbers on EgoTextVQA
(Qwen2-VL-7B video-only macro-5 ~= 28.2) -- see ``--validate``.

This script does NOT re-run any VLM: it reads the frozen ``rows.jsonl``
(qid -> pred) produced by run_qa_benchmark.py and re-maps qid -> question /
gold / question-type via the same deterministic dataset loaders.

Usage
-----
    python scripts/llm_judge.py --run-dir outputs/wave9base_qwen2vl_egotextvqa_uniform_noinject \
        --dataset egotextvqa
    python scripts/llm_judge.py --run-dir <dir> --dataset music_avqa
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from router.qa_datasets import load_dataset  # noqa: E402

# Paper macro-average question-types for EgoTextVQA (the other task_type
# values in our pilot are scene tags, excluded from the paper macro, exactly
# as in scripts/stats_analysis.py).
_EGOTEXT_PAPER_CATS = ["location", "direction", "description",
                       "intention reasoning", "others"]

# Default judge backbone. We reuse the already-cached Qwen2.5-VL-32B-Instruct
# (bf16) text path: it is the Qwen2.5-32B LLM with a vision adapter, so its
# text reasoning is identical to Qwen2.5-32B-Instruct, needs zero extra
# download, and avoids the autoawq aarch64/cu130 kernel problem. bf16 is
# strictly higher precision than the requested AWQ build.
_DEFAULT_JUDGE = "Qwen/Qwen2.5-VL-32B-Instruct"

_JUDGE_SYSTEM = (
    "You are an intelligent chatbot designed for evaluating the correctness "
    "of generative outputs for question-answer pairs. Your task is to compare "
    "the predicted answer with the correct answer and determine if they match "
    "meaningfully. Here's how you can accomplish the task:"
    "\n------"
    "\n##INSTRUCTIONS:"
    "\n- Focus on the meaningful match between the predicted answer and the "
    "correct answer.\n"
    "- Consider synonyms or paraphrases as valid matches.\n"
    "- Evaluate the correctness of the prediction compared to the answer."
)

_JUDGE_USER_TMPL = (
    "Please evaluate the following video-based question-answer pair:\n\n"
    "Question: \"{question}\"\n"
    "Correct Answer: \"{gold}\"\n"
    "Predicted Answer: \"{pred}\"\n\n"
    "Provide your evaluation only as a yes/no and score where the score is an "
    "integer value between 0 and 5, with 5 indicating the highest meaningful "
    "match. Please generate the response in the form of a Python dictionary "
    "string with keys 'pred' and 'score', where value of 'pred' is a string "
    "of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING. DO NOT "
    "PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python "
    "dictionary string. For example, your response should look like this: "
    "{{'pred': 'yes', 'score': 4}}."
)


# --------------------------------------------------------------------------
#  Judge backbone (local open LLM, text-only path of a Qwen2.5 model)
# --------------------------------------------------------------------------
def _stream_load(model_cls, model_id: str, dtype, device: str):
    """Build the model on ``device`` and copy the checkpoint in one shard at a time.

    transformers 5 keeps every safetensors shard it has read open until the whole
    checkpoint is loaded. On a unified-memory GPU (GB10) the pages it has touched
    count as anonymous memory on top of the GPU copy, so a 65 GB checkpoint needs
    ~130 GB and the kernel kills the process at ~60% (the machine has 119 GB).
    Closing each shard as soon as its tensors are copied bounds the extra memory
    by one shard (~4 GB). Weights are copied bit for bit, so the model is the one
    ``from_pretrained`` would return.
    """
    import os
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig, GenerationConfig

    path = model_id if os.path.isdir(model_id) else snapshot_download(model_id, local_files_only=True)
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    with torch.device(device):
        model = model_cls._from_config(config, dtype=dtype, attn_implementation="sdpa")
    state = model.state_dict()

    def target(key: str) -> str:
        # The original Qwen-VL checkpoints name the vision tower "visual" and the
        # decoder "model"; recent transformers nests both under "model".
        if key in state:
            return key
        if key.startswith("visual."):
            return "model." + key
        if key.startswith("model."):
            return "model.language_model." + key[len("model."):]
        return key

    seen: set[str] = set()
    for shard in sorted(f for f in os.listdir(path) if f.endswith(".safetensors")):
        with safe_open(os.path.join(path, shard), framework="pt", device="cpu") as fh:
            for key in fh.keys():
                name = target(key)
                if name not in state:
                    raise RuntimeError(f"checkpoint tensor {key} has no counterpart in the model")
                state[name].copy_(fh.get_tensor(key))
                seen.add(name)
    missing = sorted(set(state) - seen)
    if missing:
        raise RuntimeError(f"{len(missing)} model tensors were not in the checkpoint, e.g. {missing[:3]}")
    model.generation_config = GenerationConfig.from_pretrained(path, local_files_only=True)
    return model


class LocalJudge:
    def __init__(self, model_id: str = _DEFAULT_JUDGE, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._is_vl = False

    def load(self) -> None:
        if self.model is not None:
            return
        import os
        import torch
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        mid = self.model_id
        is_vl = ("-VL-" in mid) or ("vl" in mid.lower() and "qwen" in mid.lower())
        # On the GB10 unified-memory GPU, device_map="auto" offloads layers to
        # the CPU, which cripples generation. Force the whole model onto the GPU
        # (cpu budget 0) so nothing is offloaded. GPU_MEM_GIB caps the pool.
        gpu_gib = os.environ.get("JUDGE_GPU_MEM_GIB", "110")
        if self.device == "cuda":
            device_map = "auto"
            max_memory = {0: f"{gpu_gib}GiB", "cpu": "0GiB"}
        else:
            device_map = self.device
            max_memory = None
        if is_vl:
            self._is_vl = True
            from transformers import AutoProcessor
            if "2.5" in mid or "2_5" in mid:
                from transformers import Qwen2_5_VLForConditionalGeneration as _M
            else:
                from transformers import Qwen2VLForConditionalGeneration as _M
            if self.device == "cuda" and os.environ.get("JUDGE_STANDARD_LOAD") != "1":
                self.model = _stream_load(_M, mid, dtype, "cuda")
            else:
                self.model = _M.from_pretrained(
                    mid, torch_dtype=dtype, device_map=device_map,
                    max_memory=max_memory, local_files_only=True,
                )
            self.processor = AutoProcessor.from_pretrained(
                mid, local_files_only=True)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                mid, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                mid, torch_dtype=dtype, device_map=device_map,
                max_memory=max_memory, local_files_only=True,
            )
        self.model.eval()

    def _chat(self, messages: list[dict]) -> str:
        import torch
        if self.model is None:
            print(f"[judge] loading {self.model_id} ...", flush=True)
            self.load()
            print("[judge] loaded.", flush=True)
        tok = self.processor if self._is_vl else self.tokenizer
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        if self._is_vl:
            inputs = self.processor(text=[text], padding=True,
                                    return_tensors="pt")
        else:
            inputs = self.tokenizer([text], return_tensors="pt")
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=24,
                                      do_sample=False)
        in_len = inputs["input_ids"].shape[1]
        gen = out[:, in_len:]
        dec = (self.processor if self._is_vl else self.tokenizer).batch_decode(
            gen, skip_special_tokens=True)[0]
        return dec.strip()

    def judge(self, question: str, gold: str, pred: str) -> tuple[bool, int, str]:
        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": _JUDGE_USER_TMPL.format(
                question=question, gold=gold, pred=pred)},
        ]
        raw = self._chat(messages)
        return (*_parse_judgement(raw), raw)


def _parse_judgement(raw: str) -> tuple[bool, int]:
    """Robustly parse {'pred': 'yes'|'no', 'score': N} from model text."""
    low = raw.lower()
    yes = None
    m = re.search(r"'?pred'?\s*[:=]\s*'?\"?\s*(yes|no)", low)
    if m:
        yes = (m.group(1) == "yes")
    if yes is None:  # fallback: bare yes/no
        if re.search(r"\byes\b", low) and not re.search(r"\bno\b", low):
            yes = True
        elif re.search(r"\bno\b", low):
            yes = False
    score = 0
    ms = re.search(r"'?score'?\s*[:=]\s*'?\"?\s*([0-5])", low)
    if ms:
        score = int(ms.group(1))
    elif yes:
        score = 4
    return (bool(yes), score)


# --------------------------------------------------------------------------
#  Re-scoring driver
# --------------------------------------------------------------------------
def _load_rows(run_dir: Path) -> list[dict]:
    fp = run_dir / "rows.jsonl"
    return [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]


def _verdict_cache(run_dirs: list[Path]) -> dict[tuple, tuple[bool, int, str]]:
    """(qid, question, gold, pred) -> (yes, score, source run) from runs that are already judged.

    The judge sees only the question, the gold answer and the predicted answer and
    decodes greedily, so identical inputs give the identical verdict.
    """
    cache: dict[tuple, tuple[bool, int, str]] = {}
    for d in run_dirs:
        for line in (d / "judged_rows.jsonl").read_text().splitlines():
            if line.strip():
                x = json.loads(line)
                cache.setdefault((str(x["qid"]), x["question"], x["gold"], x["pred"]),
                                 (bool(x["judge_yes"]), int(x["judge_score"]), d.name))
    return cache


def rescore(run_dir: Path, dataset: str, judge: LocalJudge,
            limit: int = 0, reuse_dirs: list[Path] | None = None,
            audit_reuse: int = 0) -> dict:
    """Judge every row of ``run_dir``; verdicts persist in ``judged_rows.partial.jsonl`` as they are
    produced, so an interrupted run resumes where it stopped.

    ``reuse_dirs`` are judged runs of the same dataset: a row whose question, gold and predicted
    answer equal a judged row's takes that verdict without calling the model (a silenced-audio
    run mostly repeats the audio-on run's answers). ``audit_reuse`` re-judges that many reused rows
    with the model and records how many verdicts it reproduces.
    """
    rows = _load_rows(run_dir)
    if limit:
        rows = rows[:limit]
    items = load_dataset(dataset)
    by_qid = {it.qid: it for it in items}
    cache = _verdict_cache(reuse_dirs or [])
    partial = run_dir / "judged_rows.partial.jsonl"
    resumed: dict[str, dict] = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            if line.strip():
                x = json.loads(line)
                resumed[str(x["qid"])] = x

    out_rows = []
    reused: list[tuple[int, tuple]] = []
    n_yes = 0
    cat_tot: dict[str, int] = {}
    cat_yes: dict[str, int] = {}
    t0 = time.time()
    n_model = 0
    with open(partial, "a") as sink:
        for i, r in enumerate(rows):
            qid = r.get("qid")
            it = by_qid.get(qid)
            question = it.question if it else r.get("question", "")
            gold = it.answer if it else r.get("gold", "")
            if not gold:
                gold = r.get("gold", "")
            cat = (it.task_type if it else r.get("task_type", "")) or "unknown"
            pred = r.get("pred", "")
            key = (str(qid), question, gold, pred)
            source = None
            if str(qid) in resumed:
                yes, score = resumed[str(qid)]["judge_yes"], resumed[str(qid)]["judge_score"]
                source = resumed[str(qid)].get("verdict_from")
            elif key in cache:
                yes, score, source = cache[key]
                reused.append((i, key))
            else:
                yes, score, _ = judge.judge(question, gold, pred)
                n_model += 1
            n_yes += int(yes)
            cat_tot[cat] = cat_tot.get(cat, 0) + 1
            cat_yes[cat] = cat_yes.get(cat, 0) + int(yes)
            row = {"qid": qid, "question": question, "gold": gold,
                   "pred": pred, "task_type": cat,
                   "judge_yes": yes, "judge_score": score}
            if source:
                row["verdict_from"] = source
            out_rows.append(row)
            if str(qid) not in resumed:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
            if n_model and n_model % 25 == 0 and (time.time() - t0) > 0:
                print(f"[judge] {dataset} {run_dir.name}: {i+1}/{len(rows)} rows, {n_model} judged by the "
                      f"model, {len(reused)} reused ({n_model / (time.time() - t0):.2f}/s)", flush=True)

    audit = None
    if audit_reuse and reused:
        step = max(1, len(reused) // audit_reuse)
        checked = reused[::step][:audit_reuse]
        agree = 0
        for i, key in checked:
            row = out_rows[i]
            yes, _, _ = judge.judge(row["question"], row["gold"], row["pred"])
            agree += int(yes == row["judge_yes"])
        audit = {"checked": len(checked), "agree": agree}
        print(f"[judge] reuse audit: the model reproduced {agree}/{len(checked)} reused verdicts", flush=True)

    n = len(out_rows)
    micro = 100.0 * n_yes / n if n else 0.0
    cat_acc = {k: 100.0 * cat_yes[k] / cat_tot[k] for k in cat_tot}
    macro_all = sum(cat_acc.values()) / len(cat_acc) if cat_acc else 0.0
    paper_cats = [k for k in _EGOTEXT_PAPER_CATS if k in cat_tot]
    macro_paper = (sum(cat_acc[k] for k in paper_cats) / len(paper_cats)
                   if paper_cats else None)

    summary = {
        "run_dir": str(run_dir), "dataset": dataset, "judge_model": judge.model_id,
        "n": n, "micro_acc": round(micro, 2),
        "macro_all_acc": round(macro_all, 2),
        "macro_paper_acc": (round(macro_paper, 2) if macro_paper is not None else None),
        "per_cat": {k: round(v, 2) for k, v in sorted(cat_acc.items())},
        "per_cat_n": cat_tot,
    }
    if reuse_dirs:
        summary["n_reused_verdicts"] = len(reused)
        summary["n_judged_by_model"] = n_model
        summary["reuse_sources"] = [d.name for d in reuse_dirs]
        summary["reuse_audit"] = audit
    # write artefacts
    (run_dir / "judged_rows.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in out_rows))
    (run_dir / "judge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    partial.unlink(missing_ok=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default=_DEFAULT_JUDGE)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reuse-from", action="append", default=[], metavar="RUN_DIR",
                    help="judged run (repeatable) whose verdicts are reused for rows with the same "
                         "question, gold and predicted answer")
    ap.add_argument("--audit-reuse", type=int, default=0, metavar="N",
                    help="re-judge N reused rows with the model and report how many verdicts it reproduces")
    args = ap.parse_args()

    judge = LocalJudge(model_id=args.model)
    summary = rescore(Path(args.run_dir), args.dataset, judge, limit=args.limit,
                      reuse_dirs=[Path(p) for p in args.reuse_from], audit_reuse=args.audit_reuse)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
