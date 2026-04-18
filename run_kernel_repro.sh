#!/bin/bash
# Reproduce Table 1 (tab:proving) TU kernel benchmarks: GCN/GraphSAGE on MUTAG/PROTEINS
# Reports per-graph average commit, prove, verify times and proof sizes.
set -u
cd "$(dirname "$0")"

mkdir -p repro_logs
RESULTS="repro_kernel.csv"
echo "model,dataset,commit_ms,prove_ms,verify_ms,proof_size,verified" > "$RESULTS"

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

run_kernel() {
  local model=$1 dataset=$2
  local logfile="repro_logs/kernel_${model}_${dataset}.log"

  echo ">>> [$(date '+%H:%M:%S')] Kernel: $model on $dataset"

  ./target/release/kernel config.yaml pyg/weights "$dataset" "$model" > "$logfile" 2>&1

  if ! grep -q "^verified: true" "$logfile"; then
    echo "    FAILED"
    echo "${model},${dataset},FAIL,FAIL,FAIL,FAIL,false" >> "$RESULTS"
    return
  fi

  local commit prove verify proof
  commit=$(parse_time_ms "$(grep 'commit time:' "$logfile" | tail -1)")
  prove=$(parse_time_ms "$(grep 'prove time:' "$logfile" | tail -1)")
  verify=$(parse_time_ms "$(grep 'verify time:' "$logfile" | tail -1)")
  proof=$(grep 'Total proof size:' "$logfile" | tail -1 | sed -E 's/Total proof size: //')

  echo "    commit=${commit}ms prove=${prove}ms verify=${verify}ms proof=${proof}"
  echo "${model},${dataset},${commit},${prove},${verify},${proof},true" >> "$RESULTS"
}

echo "======================================"
echo "  Kernel Benchmarks (TU datasets)"
echo "  Start: $(date)"
echo "======================================"

for model in gcn graphsage; do
  for dataset in mutag proteins; do
    run_kernel $model $dataset
  done
done

echo ""
echo "======================================"
echo "  Done: $(date)"
echo "======================================"
echo ""
column -t -s',' "$RESULTS"
