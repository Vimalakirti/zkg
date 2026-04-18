#!/bin/bash
# Reproduce Table 4 (tab:ezkl) EZKL column: run ezkl on subgraph-extracted GraphSAGE.
# Requires: pip install ezkl (tested with v23.0.5)
# Run AFTER run_subgraph_repro.sh (which produces the zkGNN column).
set -u
cd "$(dirname "$0")"

echo "======================================"
echo "  Table 4: EZKL Subgraph Comparison"
echo "  Start: $(date)"
echo "======================================"

# Elliptic subgraphs (3, 7, 10, 20, 50, 75, 100 nodes)
echo ""
echo "--- Elliptic ---"
python3 ezkl_graphsage/run_ezkl_subgraph.py \
  --dataset elliptic \
  --sub_ids 188332,196141,192834,170495,145462,160165,185164 \
  --output_csv ezkl_elliptic_results.csv

# DGraphFin subgraphs (3, 10, 20, 45, 93 nodes)
echo ""
echo "--- DGraphFin ---"
python3 ezkl_graphsage/run_ezkl_subgraph.py \
  --dataset dgraphfin \
  --sub_ids 644186,44336,284080,28234,1832174 \
  --output_csv ezkl_dgraphfin_results.csv

echo ""
echo "======================================"
echo "  Done: $(date)"
echo "======================================"
