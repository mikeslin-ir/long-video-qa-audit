#!/usr/bin/env bash
# Compare EgoSchema QA across selectors and K values under one scoring protocol.
# Methods: ours (submodular), uniform, random; plus optional EARS historical table merge.
set -euo pipefail
cd "$(dirname "$0")/../.."

KS_CSV="${KS:-4,8}"
LIMIT="${LIMIT:-0}"   # 0 = full 500
VLM="${VLM:-Qwen/Qwen2.5-VL-7B-Instruct}"
OUT_ROOT="${OUT_ROOT:-outputs/egoschema_k_sweep}"

mkdir -p "$OUT_ROOT"

echo "[k-sweep] KS=$KS_CSV LIMIT=$LIMIT VLM=$VLM"

run_case () {  # method k extra_args...
  local method="$1"; local k="$2"; shift 2
  local out="$OUT_ROOT/${method}_k${k}"
  echo "=== method=${method} k=${k} ==="
  python scripts/run_qa_benchmark.py \
    --dataset egoschema \
    --selector "$method" \
    --budget "$k" \
    --limit "$LIMIT" \
    --real \
    --vlm_model "$VLM" \
    --output_dir "$out" \
    "$@"
}

IFS=',' read -r -a KS_ARR <<< "$KS_CSV"
for k in "${KS_ARR[@]}"; do
  run_case ours "$k" --use_submodular
  run_case uniform "$k"
  run_case random "$k"
done

python - <<'PY'
import csv, glob, json, os
from pathlib import Path

root = Path('outputs/egoschema_k_sweep')
rows = []
for f in sorted(glob.glob(str(root / '*_k*' / 'qa_summary.json'))):
    d = json.load(open(f))
    run = Path(f).parent.name
    rows.append({
        'source': 'ours-stack',
        'run': run,
        'method': d.get('selector', '').lower(),
        'k': d.get('budget'),
        'n': d.get('n'),
        'accuracy': d.get('accuracy'),
        'elapsed_sec': d.get('elapsed_sec'),
        'vlm_model': d.get('vlm_model'),
    })

ears_csv = Path(os.environ.get('EARS_EGOSCHEMA_CSV', ''))  # optional historical EARS table
if ears_csv.is_file():
    with ears_csv.open() as fp:
        rr = csv.DictReader(fp)
        for r in rr:
            for k in ('4', '8'):
                key = f'K={k}'
                v = r.get(key)
                if v is None or v == '':
                    continue
                rows.append({
                    'source': 'EARS-historical',
                    'run': f"ears_{r.get('Method','').strip()}_k{k}",
                    'method': r.get('Method', '').strip().lower(),
                    'k': int(k),
                    'n': 500,
                    'accuracy': float(v),
                    'elapsed_sec': None,
                    'vlm_model': 'Qwen/Qwen2.5-VL-7B-Instruct',
                })

rows = sorted(rows, key=lambda x: (x['k'] if x['k'] is not None else 999, x['source'], x['method']))
out_csv = root / 'comparison_table.csv'
out_json = root / 'comparison_table.json'

with out_csv.open('w', newline='') as fp:
    w = csv.DictWriter(fp, fieldnames=['source','run','method','k','n','accuracy','elapsed_sec','vlm_model'])
    w.writeheader()
    w.writerows(rows)

out_json.write_text(json.dumps(rows, indent=2))
print(f'wrote {out_csv}')
print(f'wrote {out_json}')
for r in rows:
    a = r['accuracy']
    a = f"{a:.3f}" if isinstance(a, float) else str(a)
    print(f"{r['source']:14s} method={r['method']:<10s} k={str(r['k']):>2s} acc={a}")
PY

echo "done: $OUT_ROOT/comparison_table.csv"
