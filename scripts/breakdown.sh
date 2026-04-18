#!/bin/bash
# Prover time breakdown script
# Parses the existing binary output to aggregate proving time by component type.
# Usage: ./scripts/breakdown.sh <binary> <config> <data_dir> <dataset> [extra_args...]
#
# Example: ./scripts/breakdown.sh gcn config.yaml pyg/weights cora

set -e
BINARY=$1; shift
CONFIG=$1; shift
DATA_DIR=$1; shift
DATASET=$1; shift

echo "=== Prover Time Breakdown: $BINARY on $DATASET ==="
echo ""

# Run the binary and capture full output
OUTPUT=$(./target/release/$BINARY $CONFIG $DATA_DIR $DATASET "$@" 2>&1)

# Extract commit, prove, verify times
echo "$OUTPUT" | grep -E "commit time|prove time|verify time|Total proof size"
echo ""

# Parse per-node proving times from timed! output
# The timed! macro outputs lines like: "prove node X | kind Type(...)"
# followed by timing info in the TimingTree
# But we can also measure by parsing the "SpMV adjacency partial eval time" lines

echo "--- SpMM Partial Eval Times ---"
echo "$OUTPUT" | grep "partial eval time" || echo "  (none)"
echo ""

echo "--- Node Count by Type ---"
echo "$OUTPUT" | grep "proving node\|proving reducer" | sed 's/.*kind //' | sed 's/(.*//' | sort | uniq -c | sort -rn
echo ""

echo "--- Proving Phase Breakdown ---"
echo "$OUTPUT" | grep -E "proving opening|prove_range|prove_two_pow|time taken" || true
echo ""

echo "--- Full Output (last 5 lines) ---"
echo "$OUTPUT" | tail -5
