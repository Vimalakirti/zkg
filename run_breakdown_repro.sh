#!/bin/bash
# Reproduce Table 2 (tab:breakdown): Prover time breakdown by component
# Runs GCN and GAT on Cora and PubMed, parses BREAKDOWN|... lines.
set -u
cd "$(dirname "$0")"

mkdir -p repro_logs

run_breakdown() {
  local model=$1 dataset=$2
  local logfile="repro_logs/breakdown_${model}_${dataset}.log"

  echo ">>> [$(date '+%H:%M:%S')] Breakdown: $model on $dataset"
  ./target/release/${model} config.yaml pyg/weights "$dataset" > "$logfile" 2>&1

  if ! grep -q "^verified: true" "$logfile"; then
    echo "    FAILED"
    return
  fi

  python3 - "$logfile" "$model" "$dataset" <<'PY'
import sys, re
from collections import defaultdict

log = open(sys.argv[1]).read()
model = sys.argv[2]
dataset = sys.argv[3]

# Per-node: BREAKDOWN|node|<type>|<ms>ms
node_ms = defaultdict(float)
for t, ms in re.findall(r"BREAKDOWN\|node\|(\S+?)\|([\d.]+)ms", log):
    node_ms[t] += float(ms)

# Lookup / Opening aggregates
lookup_ms = 0.0
opening_ms = 0.0
for m in re.findall(r"BREAKDOWN\|lookup\|\S+?\|([\d.]+)ms", log):
    lookup_ms += float(m)
for m in re.findall(r"BREAKDOWN\|opening\|\S+?\|([\d.]+)ms", log):
    opening_ms += float(m)

spmm_ms    = node_ms.get("SpMM", 0.0)
matmul_ms  = node_ms.get("MatMul", 0.0)
other_types = ["Add/Sub","SignBit","Scale","RangeCheck","Exp","ElemDiv","Reducer","Other"]
other_ms = sum(node_ms.get(t, 0.0) for t in other_types)

categories = [
    ("SpMM sumcheck",   spmm_ms),
    ("MatMul sumcheck", matmul_ms),
    ("Lookups",         lookup_ms),
    ("Opening (PCS)",   opening_ms),
    ("Other",           other_ms),
]
total_ms = sum(v for _, v in categories)

print(f"\n=== {model.upper()} on {dataset} ===")
print(f"Total prove time: {total_ms/1000:.2f} s")
for name, ms in categories:
    pct = 100.0 * ms / total_ms if total_ms > 0 else 0
    print(f"  {name:18s} {ms/1000:8.2f} s   {pct:5.1f}%")
PY
}

echo "======================================"
echo "  Prover Time Breakdown (Table 2)"
echo "  Start: $(date)"
echo "======================================"

run_breakdown gcn cora
run_breakdown gcn pubmed
run_breakdown gat cora
run_breakdown gat pubmed

echo ""
echo "======================================"
echo "  Done: $(date)"
echo "======================================"
