#!/bin/bash
# Run GAT on PubMed and compute prover-time breakdown percentages.
# Parses BREAKDOWN|... lines emitted by src/dag/mod.rs during proving.
#
# Usage: ./scripts/gat_pubmed_breakdown.sh [output_log]
set -e

CONFIG=config.yaml
DATA_DIR=pyg/weights
DATASET=pubmed
LOG=${1:-gat_pubmed_breakdown.log}

echo "=== Running GAT on $DATASET ==="
./target/release/gat "$CONFIG" "$DATA_DIR" "$DATASET" 2>&1 | tee "$LOG"

echo ""
echo "=== Breakdown percentages (parsed from $LOG) ==="
python3 - "$LOG" <<'PY'
import sys, re
from collections import defaultdict

log = open(sys.argv[1]).read()

# Per-node: BREAKDOWN|node|<type>|<ms>ms
node_ms = defaultdict(float)
for t, ms in re.findall(r"BREAKDOWN\|node\|(\S+?)\|([\d.]+)ms", log):
    node_ms[t] += float(ms)

# Lookup / Opening aggregates
lookup_ms = 0.0
opening_ms = 0.0
for t, ms in re.findall(r"BREAKDOWN\|lookup\|\S+?\|([\d.]+)ms", log):
    lookup_ms += float(ms)
for t, ms in re.findall(r"BREAKDOWN\|opening\|\S+?\|([\d.]+)ms", log):
    opening_ms += float(ms)

# Map node types into the table's categories
spmm_ms    = node_ms.get("SpMM", 0.0)
matmul_ms  = node_ms.get("MatMul", 0.0)
# "Other" = everything else that ran as a node (Add/Sub, Scale, SignBit,
# RangeCheck, Exp, ElemDiv, Reducer, Other)
other_types = ["Add/Sub","SignBit","Scale","RangeCheck","Exp","ElemDiv","Reducer","Other"]
other_ms = sum(node_ms.get(t, 0.0) for t in other_types)

categories = [
    ("SpMM sumcheck",   spmm_ms),
    ("MatMul sumcheck", matmul_ms),
    ("Lookup (range)",  lookup_ms),
    ("Opening (PCS)",   opening_ms),
    ("Other",           other_ms),
]
total_ms = sum(v for _, v in categories)

print(f"Total accounted: {total_ms/1000:.2f} s")
for name, ms in categories:
    pct = 100.0 * ms / total_ms if total_ms > 0 else 0
    print(f"  {name:18s} {ms/1000:8.2f} s   {pct:5.1f}%")

# Also dump the raw per-type node times for transparency
print("\nRaw per-type node times:")
for t, ms in sorted(node_ms.items(), key=lambda kv: -kv[1]):
    print(f"  {t:12s} {ms/1000:8.2f} s")
PY
