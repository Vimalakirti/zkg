#!/bin/bash
# Table 3: Scalability comparison of three SpMV prover baselines on synthetic graphs.
#   - Dense O(N^2):       CLASSICAL_SPMV=1
#   - Sparse-naive O(MlogN): NAIVE_SPMV=1
#   - Ours O(M):          (default)
# Uses generated datasets (fake_*_d10) with average degree 10.
# Must be run from the project root directory.
set -e
cd "$(dirname "$0")"

RESULTS="table3_results.csv"
echo "nodes,degree,mode,spmv_l1_ms,spmv_l2_ms,prove_s,commit_s,verify_ms,verified" > "$RESULTS"

parse_time_ms() {
  local line="$1"
  if echo "$line" | grep -q "ms \["; then
    echo "$line" | sed -E 's/.*time: ([0-9.]+)ms.*/\1/'
  else
    local secs
    secs=$(echo "$line" | sed -E 's/.*time: ([0-9.]+)s.*/\1/')
    echo "$secs" | awk '{printf "%.3f", $1 * 1000}'
  fi
}

parse_time_s() {
  local line="$1"
  if echo "$line" | grep -qE '[0-9]ms'; then
    local ms
    ms=$(echo "$line" | sed -E 's/.*time: ([0-9.]+)ms.*/\1/')
    echo "$ms" | awk '{printf "%.3f", $1 / 1000}'
  else
    echo "$line" | sed -E 's/.*time: ([0-9.]+)s.*/\1/'
  fi
}

parse_verify_ms() {
  local line="$1"
  if echo "$line" | grep -qE '[0-9]ms'; then
    echo "$line" | sed -E 's/.*time: ([0-9.]+)ms.*/\1/'
  else
    local secs
    secs=$(echo "$line" | sed -E 's/.*time: ([0-9.]+)s.*/\1/')
    echo "$secs" | awk '{printf "%.3f", $1 * 1000}'
  fi
}

run_one() {
  local nodes=$1 degree=$2 mode=$3
  local name="fake_${nodes}_d${degree}"
  local env_prefix=""
  case "$mode" in
    classical) env_prefix="CLASSICAL_SPMV=1 " ;;
    naive)     env_prefix="NAIVE_SPMV=1 " ;;
    proposed)  env_prefix="" ;;
  esac

  echo -n ">>> $name [$mode]... "

  local output
  output=$(eval ${env_prefix}cargo run --release --bin graphsage \
    -- config.yaml pyg/weights "$name" 2>/dev/null) || true

  # Check for failure / OOM
  if ! echo "$output" | grep -q "^verified:"; then
    echo "FAILED/OOM"
    echo "$nodes,$degree,$mode,OOM,OOM,OOM,OOM,OOM,false" >> "$RESULTS"
    return
  fi

  # Parse SpMV times (two lines)
  local spmv_lines
  spmv_lines=$(echo "$output" | grep "adjacency partial eval time:")
  local line1 line2
  line1=$(echo "$spmv_lines" | head -1)
  line2=$(echo "$spmv_lines" | tail -1)
  local spmv1_ms spmv2_ms
  spmv1_ms=$(parse_time_ms "$line1")
  spmv2_ms=$(parse_time_ms "$line2")

  local prove_s commit_s verify_ms verified
  prove_s=$(parse_time_s "$(echo "$output" | grep "prove time:")")
  commit_s=$(parse_time_s "$(echo "$output" | grep "commit time:")")
  verify_ms=$(parse_verify_ms "$(echo "$output" | grep "verify time:")")
  verified=$(echo "$output" | grep "^verified:" | awk '{print $2}')

  echo "SpMV=${spmv1_ms}/${spmv2_ms}ms prove=${prove_s}s commit=${commit_s}s v=$verified"
  echo "$nodes,$degree,$mode,$spmv1_ms,$spmv2_ms,$prove_s,$commit_s,$verify_ms,$verified" >> "$RESULTS"
}

DEGREE=10

echo "========================================="
echo "  Table 3: Three-Baseline Scalability"
echo "  (GraphSAGE, degree=$DEGREE)"
echo "========================================="

# Our proposed prover: 2^12 .. 2^17
echo ""
echo "--- PROPOSED O(M) ---"
for log_n in 12 13 14 15 16 17; do
  nodes=$((1 << log_n))
  run_one $nodes $DEGREE proposed
done

# Sparse-naive O(M log N): 2^12 .. 2^17
echo ""
echo "--- SPARSE-NAIVE O(M log N) ---"
for log_n in 12 13 14 15 16 17; do
  nodes=$((1 << log_n))
  run_one $nodes $DEGREE naive
done

# Classical dense O(N^2): 2^12 .. 2^15 (2^16+ OOMs)
echo ""
echo "--- CLASSICAL DENSE O(N^2) ---"
for log_n in 12 13 14 15; do
  nodes=$((1 << log_n))
  run_one $nodes $DEGREE classical
done
# Mark 2^16 and 2^17 as OOM for classical
echo "65536,$DEGREE,classical,OOM,OOM,OOM,OOM,OOM,false" >> "$RESULTS"
echo "131072,$DEGREE,classical,OOM,OOM,OOM,OOM,OOM,false" >> "$RESULTS"

echo ""
echo "========================================="
echo "  Results saved to $RESULTS"
echo "========================================="
echo ""
column -t -s',' "$RESULTS"
