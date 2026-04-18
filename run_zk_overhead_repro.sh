#!/bin/bash
# Reproduce ZK overhead Table 4 (GCN/GraphSAGE/GAT × N=2^12..2^16 × ZK off/on)
# Runs all experiments sequentially. Logs are written to repro_logs/<model>_<N>_<mode>.log
# Aggregated summary goes to repro_summary.csv

set -u
cd "$(dirname "$0")"

mkdir -p repro_logs
RESULTS="repro_summary.csv"
echo "model,N,mode,prove_s,commit_s,verify_ms,proof_total,zk_extra_kb,verified" > "$RESULTS"

parse_prove_s() {
  local line="$1"
  [ -z "$line" ] && { echo "NA"; return; }
  if echo "$line" | grep -qE '[0-9]ms'; then
    local ms
    ms=$(echo "$line" | sed -E 's/.*time: ([0-9.]+)ms.*/\1/')
    awk -v ms="$ms" 'BEGIN{printf "%.3f", ms/1000}'
  else
    echo "$line" | sed -E 's/.*time: ([0-9.]+)s.*/\1/'
  fi
}

parse_verify_ms() {
  local line="$1"
  [ -z "$line" ] && { echo "NA"; return; }
  if echo "$line" | grep -qE 'time: [0-9.]+ms'; then
    echo "$line" | sed -E 's/.*time: ([0-9.]+)ms.*/\1/'
  else
    local secs
    secs=$(echo "$line" | sed -E 's/.*time: ([0-9.]+)s.*/\1/')
    awk -v s="$secs" 'BEGIN{printf "%.3f", s*1000}'
  fi
}

run_one() {
  local model=$1 logN=$2 mode=$3
  local N=$((1 << logN))
  local dataset_suffix="d10"
  [ "$model" = "gat" ] && dataset_suffix="d10_gat"
  local dataset="fake_${N}_${dataset_suffix}"
  local zk_flag=""
  [ "$mode" = "on" ] && zk_flag="--zk"
  local logfile="repro_logs/${model}_${logN}_${mode}.log"

  echo ">>> [$(date '+%H:%M:%S')] $model N=2^$logN mode=$mode dataset=$dataset"

  ./target/release/${model} config.yaml pyg/weights "$dataset" $zk_flag > "$logfile" 2>&1

  if ! grep -q "^verified: true" "$logfile"; then
    echo "    FAILED (no 'verified: true')"
    echo "${model},${logN},${mode},FAIL,FAIL,FAIL,FAIL,FAIL,false" >> "$RESULTS"
    return
  fi

  local prove commit verify total_proof zk_extra
  prove=$(parse_prove_s "$(grep 'prove time:' "$logfile" | tail -1)")
  commit=$(parse_prove_s "$(grep 'commit time:' "$logfile" | tail -1)")
  verify=$(parse_verify_ms "$(grep 'verify time:' "$logfile" | tail -1)")
  total_proof=$(grep 'Total proof size:' "$logfile" | tail -1 | sed -E 's/Total proof size: //')
  zk_extra=$(grep 'ZK extra proof size:' "$logfile" | tail -1 | sed -E 's/.*: ([0-9.]+) KB/\1/')
  [ -z "$zk_extra" ] && zk_extra="0"

  echo "    prove=${prove}s commit=${commit}s verify=${verify}ms proof=${total_proof} zk_extra=${zk_extra}KB"
  echo "${model},${logN},${mode},${prove},${commit},${verify},${total_proof},${zk_extra},true" >> "$RESULTS"
}

echo "======================================"
echo "  ZK Overhead Table 4 Reproduction"
echo "  Start: $(date)"
echo "======================================"

# GCN first (fastest)
for logN in 12 13 14 15 16; do
  for mode in off on; do
    run_one gcn $logN $mode
  done
done

# GraphSAGE
for logN in 12 13 14 15 16; do
  for mode in off on; do
    run_one graphsage $logN $mode
  done
done

# GAT (longest)
for logN in 12 13 14 15; do
  for mode in off on; do
    run_one gat $logN $mode
  done
done

echo ""
echo "======================================"
echo "  Done: $(date)"
echo "======================================"
echo ""
column -t -s',' "$RESULTS"
