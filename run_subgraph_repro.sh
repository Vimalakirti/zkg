#!/bin/bash
# Reproduce Table 3 (tab:subgraph): Subgraph extraction on Elliptic and DGraphFin
# Runs GraphSAGE on 20 random subgraphs per dataset and reports averages.
# Also reproduces Table 4 (tab:ezkl): per-subgraph zkGNN numbers for EZKL comparison.
set -u
cd "$(dirname "$0")"

mkdir -p repro_logs

parse_time_ms() {
  local line="$1"
  [ -z "$line" ] && { echo "NA"; return; }
  if echo "$line" | grep -qE '[0-9]ms'; then
    echo "$line" | sed -E 's/.*time: ([0-9.]+)ms.*/\1/'
  else
    local secs
    secs=$(echo "$line" | sed -E 's/.*time: ([0-9.]+)s.*/\1/')
    awk -v s="$secs" 'BEGIN{printf "%.3f", s*1000}'
  fi
}

# =====================================================================
# Part 1: Table 3 — average over 20 subgraphs per dataset
# =====================================================================
echo "======================================"
echo "  Table 3: Subgraph Extraction"
echo "  Start: $(date)"
echo "======================================"

for dataset in elliptic dgraphfin; do
  echo ""
  echo "--- $dataset (20 random subgraphs) ---"

  # Pick 20 random subgraph dirs
  subs=$(ls -d pyg/weights/raw/${dataset}_sub_* | shuf -n 20 | sort)

  total_prove=0
  total_verify=0
  total_proof_kb=0
  count=0

  for sub_dir in $subs; do
    sub_name=$(basename "$sub_dir")
    logfile="repro_logs/sub_${sub_name}.log"

    ./target/release/graphsage config.yaml pyg/weights "$sub_name" "$dataset" > "$logfile" 2>&1

    if ! grep -q "^verified: true" "$logfile"; then
      echo "  $sub_name: FAILED"
      continue
    fi

    prove=$(parse_time_ms "$(grep 'prove time:' "$logfile" | tail -1)")
    verify=$(parse_time_ms "$(grep 'verify time:' "$logfile" | tail -1)")
    proof_line=$(grep 'Total proof size:' "$logfile" | tail -1)

    # Parse proof size to KB
    if echo "$proof_line" | grep -q "MB"; then
      proof_kb=$(echo "$proof_line" | sed -E 's/.*: ([0-9.]+) MB/\1/' | awk '{printf "%.2f", $1 * 1024}')
    else
      proof_kb=$(echo "$proof_line" | sed -E 's/.*: ([0-9.]+) KB/\1/')
    fi

    total_prove=$(awk "BEGIN{print $total_prove + $prove}")
    total_verify=$(awk "BEGIN{print $total_verify + $verify}")
    total_proof_kb=$(awk "BEGIN{print $total_proof_kb + $proof_kb}")
    count=$((count + 1))

    nodes=$(python3 -c "import json; print(json.load(open('$sub_dir/meta.json'))['num_nodes'])")
    echo "  [$count] $sub_name: ${nodes} nodes, prove=${prove}ms verify=${verify}ms proof=${proof_kb}KB"
  done

  if [ $count -gt 0 ]; then
    avg_prove=$(awk "BEGIN{printf \"%.1f\", $total_prove / $count}")
    avg_verify=$(awk "BEGIN{printf \"%.1f\", $total_verify / $count}")
    avg_proof=$(awk "BEGIN{printf \"%.1f\", $total_proof_kb / $count}")
    echo ""
    echo "  Average ($count subgraphs): prove=${avg_prove}ms verify=${avg_verify}ms proof=${avg_proof}KB"
  fi
done

# =====================================================================
# Part 2: Table 4 — specific subgraphs matching EZKL comparison
# =====================================================================
echo ""
echo "======================================"
echo "  Table 4: zkGNN on EZKL subgraphs"
echo "======================================"

EZKL_RESULTS="repro_ezkl_zkgnn.csv"
echo "dataset,sub_id,num_nodes,prove_ms,verify_ms,proof_kb" > "$EZKL_RESULTS"

# Elliptic subgraphs used in EZKL comparison
ELLIPTIC_SUBS="188332 196141 192834 170495 145462 160165 185164"
# DGraphFin subgraphs used in EZKL comparison
DGRAPHFIN_SUBS="644186 44336 284080 28234 1832174"

for sub_id in $ELLIPTIC_SUBS; do
  sub_name="elliptic_sub_${sub_id}"
  logfile="repro_logs/ezkl_${sub_name}.log"
  ./target/release/graphsage config.yaml pyg/weights "$sub_name" elliptic > "$logfile" 2>&1

  if ! grep -q "^verified: true" "$logfile"; then
    echo "  $sub_name: FAILED"
    continue
  fi

  prove=$(parse_time_ms "$(grep 'prove time:' "$logfile" | tail -1)")
  verify=$(parse_time_ms "$(grep 'verify time:' "$logfile" | tail -1)")
  proof_line=$(grep 'Total proof size:' "$logfile" | tail -1)
  if echo "$proof_line" | grep -q "MB"; then
    proof_kb=$(echo "$proof_line" | sed -E 's/.*: ([0-9.]+) MB/\1/' | awk '{printf "%.2f", $1 * 1024}')
  else
    proof_kb=$(echo "$proof_line" | sed -E 's/.*: ([0-9.]+) KB/\1/')
  fi
  nodes=$(python3 -c "import json; print(json.load(open('pyg/weights/raw/$sub_name/meta.json'))['num_nodes'])")

  echo "  Elliptic N=$nodes: prove=${prove}ms verify=${verify}ms proof=${proof_kb}KB"
  echo "elliptic,$sub_id,$nodes,$prove,$verify,$proof_kb" >> "$EZKL_RESULTS"
done

for sub_id in $DGRAPHFIN_SUBS; do
  sub_name="dgraphfin_sub_${sub_id}"
  logfile="repro_logs/ezkl_${sub_name}.log"
  ./target/release/graphsage config.yaml pyg/weights "$sub_name" dgraphfin > "$logfile" 2>&1

  if ! grep -q "^verified: true" "$logfile"; then
    echo "  $sub_name: FAILED"
    continue
  fi

  prove=$(parse_time_ms "$(grep 'prove time:' "$logfile" | tail -1)")
  verify=$(parse_time_ms "$(grep 'verify time:' "$logfile" | tail -1)")
  proof_line=$(grep 'Total proof size:' "$logfile" | tail -1)
  if echo "$proof_line" | grep -q "MB"; then
    proof_kb=$(echo "$proof_line" | sed -E 's/.*: ([0-9.]+) MB/\1/' | awk '{printf "%.2f", $1 * 1024}')
  else
    proof_kb=$(echo "$proof_line" | sed -E 's/.*: ([0-9.]+) KB/\1/')
  fi
  nodes=$(python3 -c "import json; print(json.load(open('pyg/weights/raw/$sub_name/meta.json'))['num_nodes'])")

  echo "  DGraphFin N=$nodes: prove=${prove}ms verify=${verify}ms proof=${proof_kb}KB"
  echo "dgraphfin,$sub_id,$nodes,$prove,$verify,$proof_kb" >> "$EZKL_RESULTS"
done

echo ""
echo "Results saved to $EZKL_RESULTS"
column -t -s',' "$EZKL_RESULTS"

echo ""
echo "======================================"
echo "  Done: $(date)"
echo "======================================"
