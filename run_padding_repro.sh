#!/bin/bash
# Measure the data-independent public-capacity mode used to hide the exact
# additive-decomposition size. Runs the same ZK binary with and without
# capacity padding so the only changed setting is the number of factor terms.
set -u
cd "$(dirname "$0")"

mkdir -p repro_logs
RESULTS="repro_padding.csv"
echo "model,dataset,mode,capacity,commit_s,prove_s,verify_ms,proof_size,peak_rss_kb,verified" > "$RESULTS"

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

run_one() {
  local model=$1 dataset=$2 capacity=$3 mode=$4
  local logfile="repro_logs/padding_${model}_${dataset}_${mode}.log"
  local capacity_args=()
  if [ "$mode" = "padded" ]; then
    capacity_args=(--factor-capacity "$capacity")
  fi

  echo ">>> $model $dataset $mode (capacity=$capacity)"
  /usr/bin/time -v "./target/release/${model}" config.yaml pyg/weights "$dataset" \
    --zk "${capacity_args[@]}" > "$logfile" 2>&1

  if ! grep -q "^verified: true" "$logfile"; then
    echo "    FAILED; see $logfile"
    echo "$model,$dataset,$mode,$capacity,FAIL,FAIL,FAIL,FAIL,FAIL,false" >> "$RESULTS"
    return
  fi

  local commit prove verify proof memory
  commit=$(parse_time_s "$(grep 'commit time:' "$logfile" | tail -1)")
  prove=$(parse_time_s "$(grep 'prove time:' "$logfile" | tail -1)")
  verify=$(parse_verify_ms "$(grep 'verify time:' "$logfile" | tail -1)")
  proof=$(grep 'Total proof size:' "$logfile" | tail -1 | sed -E 's/Total proof size: //')
  memory=$(grep 'Maximum resident set size' "$logfile" | tail -1 | sed -E 's/.*: ([0-9]+)$/\1/')
  echo "$model,$dataset,$mode,$capacity,$commit,$prove,$verify,$proof,$memory,true" >> "$RESULTS"
}

for model in gcn graphsage; do
  for entry in "cora 32" "citeseer 32" "pubmed 128"; do
    read -r dataset capacity <<< "$entry"
    run_one "$model" "$dataset" "$capacity" unpadded
    run_one "$model" "$dataset" "$capacity" padded
  done
done

echo "Results written to $RESULTS"
