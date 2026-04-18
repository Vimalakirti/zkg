#!/bin/bash
# Generate all training data and weights from scratch.
# This replaces the pre-trained pyg/weights/raw/ directory.
#
# Prerequisites: Python 3.10+ with torch, torch_geometric, numpy, scikit-learn
# Total time: ~1-2 hours depending on GPU availability
set -eu
cd "$(dirname "$0")/pyg"

mkdir -p weights/raw

echo "======================================"
echo "  zkGNN Data Setup"
echo "  Start: $(date)"
echo "======================================"

# Step 1: Citation networks (Cora, CiteSeer, PubMed)
echo ""
echo "--- Step 1: Train GCN/GraphSAGE/GAT on citation networks ---"
python train_citation.py

echo ""
echo "--- Step 2: Export citation weights to binary ---"
python export_raw.py

# Step 2: TU kernel benchmarks (MUTAG, PROTEINS)
echo ""
echo "--- Step 3: Train GCN/GraphSAGE on TU benchmarks ---"
python train_kernel.py

echo ""
echo "--- Step 4: Export kernel weights to binary ---"
python export_kernel.py

# Step 3: Fraud detection
echo ""
echo "--- Step 5: Train GraphSAGE on Elliptic Bitcoin ---"
python train_elliptic.py

echo ""
echo "--- Step 6: Train GraphSAGE on DGraphFin ---"
python train_dgraphfin.py

# Step 4: Synthetic graphs for ablation and ZK overhead experiments
echo ""
echo "--- Step 7: Generate synthetic Erdős–Rényi graphs ---"
python gen_fake.py

# Step 5: Subgraph extraction for large-graph experiments
echo ""
echo "--- Step 8: Extract subgraphs for Elliptic ---"
python extract_subgraph.py elliptic "test:1000" --khops 2

echo ""
echo "--- Step 9: Extract subgraphs for DGraphFin ---"
python extract_subgraph.py dgraphfin "test:1000" --khops 2

echo ""
echo "======================================"
echo "  Data setup complete: $(date)"
echo "======================================"
echo ""
echo "Generated data in pyg/weights/raw/:"
ls weights/raw/ | wc -l
echo "directories"
