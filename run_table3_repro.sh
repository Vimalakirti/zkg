#!/bin/bash
# Reproduce Table 3: End-to-end proving on citation networks (Cora, CiteSeer, PubMed)
# Also reruns GAT N=2^13 ZK overhead (off + on) for Table 4
set -u
cd "$(dirname "$0")"

mkdir -p repro_logs
RESULTS="repro_table3.csv"
echo "model,dataset,commit_s,prove_s,verify_ms,proof_size,verified" > "$RESULTS"

parse_time_s() {
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

run_table3() {
  local model=$1 dataset=$2
  local logfile="repro_logs/table3_${model}_${dataset}.log"

  echo ">>> [$(date '+%H:%M:%S')] Table3: $model on $dataset"

  ./target/release/${model} config.yaml pyg/weights "$dataset" > "$logfile" 2>&1

  if ! grep -q "^verified: true" "$logfile"; then
    echo "    FAILED"
    echo "${model},${dataset},FAIL,FAIL,FAIL,FAIL,false" >> "$RESULTS"
    return
  fi

  local commit prove verify proof
  commit=$(parse_time_s "$(grep 'commit time:' "$logfile" | tail -1)")
  prove=$(parse_time_s "$(grep 'prove time:' "$logfile" | tail -1)")
  verify=$(parse_verify_ms "$(grep 'verify time:' "$logfile" | tail -1)")
  proof=$(grep 'Total proof size:' "$logfile" | tail -1 | sed -E 's/Total proof size: //')

  echo "    commit=${commit}s prove=${prove}s verify=${verify}ms proof=${proof}"
  echo "${model},${dataset},${commit},${prove},${verify},${proof},true" >> "$RESULTS"
}

run_gat_zk() {
  local logN=$1 mode=$2
  local N=$((1 << logN))
  local dataset="fake_${N}_d10_gat"
  local zk_flag=""
  [ "$mode" = "on" ] && zk_flag="--zk"
  local logfile="repro_logs/gat_${logN}_${mode}_v2.log"

  echo ">>> [$(date '+%H:%M:%S')] GAT ZK: N=2^$logN mode=$mode"

  ./target/release/gat config.yaml pyg/weights "$dataset" $zk_flag > "$logfile" 2>&1

  if ! grep -q "^verified: true" "$logfile"; then
    echo "    FAILED"
    return
  fi

  local prove verify proof zk_extra
  prove=$(parse_time_s "$(grep 'prove time:' "$logfile" | tail -1)")
  verify=$(parse_verify_ms "$(grep 'verify time:' "$logfile" | tail -1)")
  proof=$(grep 'Total proof size:' "$logfile" | tail -1 | sed -E 's/Total proof size: //')
  zk_extra=$(grep 'ZK extra proof size:' "$logfile" | tail -1 | sed -E 's/.*: ([0-9.]+) KB/\1/')
  [ -z "$zk_extra" ] && zk_extra="0"

  echo "    prove=${prove}s verify=${verify}ms proof=${proof} zk_extra=${zk_extra}KB"
}

echo "======================================"
echo "  Table 3 + GAT 2^13 ZK Reproduction"
echo "  Start: $(date)"
echo "======================================"

# GAT 2^13 rerun first (so it runs on a clean machine)
echo ""
echo "--- GAT N=2^13 ZK overhead rerun ---"
run_gat_zk 13 off
run_gat_zk 13 on

# Table 3: Citation networks
echo ""
echo "--- Table 3: Citation Networks ---"
for model in gcn graphsage gat; do
  for dataset in cora citeseer pubmed; do
    run_table3 $model $dataset
  done
done

echo ""
echo "======================================"
echo "  Done: $(date)"
echo "======================================"
echo ""
echo "--- Table 3 Results ---"
column -t -s',' "$RESULTS"
