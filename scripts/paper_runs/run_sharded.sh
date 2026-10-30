#!/usr/bin/env bash
# Run one run_qa_benchmark.py configuration as parallel shards that share an output
# directory, then retry whatever a shard left undone (an OOM-killed shard simply resumes).
#
#   SHARDS=4 bash scripts/paper_runs/run_sharded.sh <output dir> <ordered qid file> [run_qa_benchmark.py args...]
#
# <ordered qid file> lists the run's qids one per line (for example the qids of an
# existing run's rows.jsonl). Shard i takes every SHARDS-th qid, or, with
# SHARD_KEY_SEP=- , every SHARDS-th *video* (the qid text before the first "-"), so that a
# video's questions share a shard and shards never compute the same cached proposals or
# transcripts at the same time (needed for the first run of a cascade profile).
# Model weights are loaded offline (HF_HUB_OFFLINE=1): a shard that stalled on a network
# request while loading blocked a whole run once. Predictions are
# unchanged by sharding; --skip_option_logprobs is added (it only affects cost bookkeeping),
# so expert_cost_sec and wall times of sharded runs are not comparable with other runs.
# On a 119 GB unified-memory GPU keep SHARDS <= 4 for 7B models.
set -uo pipefail
cd "$(dirname "$0")/../.."

out="${1:?output dir}"; qids="${2:?qid file}"; shift 2
SHARDS="${SHARDS:-4}"
PY="${PY:-python3}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
mkdir -p "$out"

launch () {  # launch <attempt>
  local i
  for ((i = 0; i < SHARDS; i++)); do
    awk -v n="$SHARDS" -v i="$i" -v sep="${SHARD_KEY_SEP:-}" '
      { key = $0; if (sep != "") { split($0, a, sep); key = a[1] }
        if (!(key in id)) id[key] = cnt++
        if (id[key] % n == i) print }' "$qids" > "$tmp/shard$i.txt"
    "$PY" scripts/run_qa_benchmark.py "$@" --real --item_timeout_sec 600 --resume \
        --skip_option_logprobs --only_qids "$tmp/shard$i.txt" --output_dir "$out" \
        >> "$out/shard$i.log" 2>&1 &
    sleep "${STAGGER_SEC:-60}"   # concurrent weight loads of several 7B models can exhaust memory
  done
  wait
}

for attempt in 1 2 3; do
  launch "$@"
  # count unique finished items; skip torn lines (a killed writer can leave NUL bytes)
  done_n=$(python3 -c "
import json
q = set()
for l in open('$out/rows.jsonl', 'rb'):
    try:
        q.add(json.loads(l.strip(b'\\x00').strip())['qid'])
    except Exception:
        pass
print(len(q))" 2>/dev/null || echo 0)
  want=$(grep -c . "$qids")
  echo "[sharded] attempt $attempt: $done_n / $want items in $out"
  [[ "$done_n" -ge "$want" ]] && exit 0
done
exit 1
