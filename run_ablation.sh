#!/bin/bash
# SpMV ablation study: proposed vs classical, varying nodes and sparsity.
# Must be run from the project root directory.
set -e
cd "$(dirname "$0")"

RESULTS="ablation_results.csv"
echo "nodes,degree,mode,spmv_l1_ms,spmv_l2_ms,prove_s,commit_s,verify_ms,verified" > "$RESULTS"

parse_time_ms() {
  # Input: a line like "SpMV adjacency partial eval time: 2.75631ms [proposed]"
  #    or: "SpMV adjacency partial eval time: 47.019522715s [classical]"
  # Output: time in ms
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
  # Input: a line like "  prove time: 9.413s" or "  commit time: 945.387ms"
  # Output: time in seconds
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
  [ "$mode" = "classical" ] && env_prefix="CLASSICAL_SPMV=1 "

  echo -n ">>> $name [$mode]... "

  local output
  output=$(eval ${env_prefix}cargo run --release --bin graphsage --features '"arkworks bn254"' \
    -- config.yaml pyg/weights "$name" 2>&1) || true

  # Check for failure
  if ! echo "$output" | grep -q "^verified:"; then
    echo "FAILED"
    echo "$nodes,$degree,$mode,FAIL,FAIL,FAIL,FAIL,FAIL,FAIL" >> "$RESULTS"
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

  echo "SpMV=${spmv1_ms}/${spmv2_ms}ms prove=${prove_s}s v=$verified"
  echo "$nodes,$degree,$mode,$spmv1_ms,$spmv2_ms,$prove_s,$commit_s,$verify_ms,$verified" >> "$RESULTS"
}

DEGREES="5 10 20 50 100"

echo "========================================="
echo "  SpMV Ablation Study"
echo "========================================="

# Proposed mode: 2^12 .. 2^17
echo ""
echo "--- PROPOSED MODE ---"
for log_n in 12 13 14 15 16 17; do
  nodes=$((1 << log_n))
  for deg in $DEGREES; do
    run_one $nodes $deg proposed
  done
done

# Classical mode: 2^12 .. 2^15 (2^16+ OOMs at ~128GB)
echo ""
echo "--- CLASSICAL MODE ---"
for log_n in 12 13 14 15; do
  nodes=$((1 << log_n))
  for deg in $DEGREES; do
    run_one $nodes $deg classical
  done
done

echo ""
echo "========================================="
echo "  Results saved to $RESULTS"
echo "========================================="
echo ""
column -t -s',' "$RESULTS"
