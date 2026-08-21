#!/bin/bash
# Collect only the GAT measurements invalidated by the corrected division proof:
#   - tab:proving: Cora, CiteSeer, and PubMed end-to-end rows
#   - tab:breakdown: Cora and PubMed GAT columns (from the same runs)
#   - tab:zk-overhead: GAT, N=2^12..2^15, ZK off/on
#
# Private-M padding is a separate experiment; use run_padding_repro.sh for it.
set -u
cd "$(dirname "$0")"

# Override these variables to rerun only a subset without overwriting a prior
# complete result set.  For example, after the first run reported a PubMed
# verification failure and ran out of SRS storage at 2^14:
#   GAT_RESULT_TAG=failed_only \
#   GAT_CITATION_DATASETS="pubmed" \
#   GAT_ZK_LOG_NS="14 15" \
#   ./run_required_gat_reruns.sh
GAT_RESULT_TAG="${GAT_RESULT_TAG:-required_gat}"
GAT_CITATION_DATASETS="${GAT_CITATION_DATASETS:-cora citeseer pubmed}"
GAT_ZK_LOG_NS="${GAT_ZK_LOG_NS:-12 13 14 15}"
GAT_ZK_MODES="${GAT_ZK_MODES:-off on}"

LOG_DIR="repro_logs/${GAT_RESULT_TAG}"
PROVING_RESULTS="repro_${GAT_RESULT_TAG}_proving.csv"
BREAKDOWN_RESULTS="repro_${GAT_RESULT_TAG}_breakdown.csv"
ZK_RESULTS="repro_${GAT_RESULT_TAG}_zk.csv"

mkdir -p "$LOG_DIR"
echo "dataset,commit_s,prove_s,verify_ms,proof_size,verified" > "$PROVING_RESULTS"
echo "dataset,total_s,spmm_pct,matmul_pct,lookups_pct,opening_pct,other_pct,verified" > "$BREAKDOWN_RESULTS"
echo "N,mode,commit_s,prove_s,verify_ms,proof_size,zk_extra_kb,verified" > "$ZK_RESULTS"

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
    local seconds
    seconds=$(echo "$line" | sed -E 's/.*time: ([0-9.]+)s.*/\1/')
    awk -v seconds="$seconds" 'BEGIN{printf "%.3f", seconds*1000}'
  fi
}

append_breakdown() {
  local logfile="$1" dataset="$2"
  python3 - "$logfile" "$dataset" "$BREAKDOWN_RESULTS" <<'PY'
import re
import sys
from collections import defaultdict

log_path, dataset, output_path = sys.argv[1:]
log = open(log_path).read()

node_ms = defaultdict(float)
for block_type, elapsed in re.findall(
        r"BREAKDOWN\|node\|(\S+?)\|([\d.]+)ms", log):
    node_ms[block_type] += float(elapsed)

lookup_ms = sum(float(value) for value in re.findall(
    r"BREAKDOWN\|lookup\|\S+?\|([\d.]+)ms", log))
opening_ms = sum(float(value) for value in re.findall(
    r"BREAKDOWN\|opening\|\S+?\|([\d.]+)ms", log))
other_types = [
    "Add/Sub", "SignBit", "Scale", "RangeCheck", "Exp", "ElemDiv",
    "Reducer", "Other",
]
categories = [
    node_ms.get("SpMM", 0.0),
    node_ms.get("MatMul", 0.0),
    lookup_ms,
    opening_ms,
    sum(node_ms.get(block_type, 0.0) for block_type in other_types),
]
total_ms = sum(categories)

with open(output_path, "a") as output:
    if total_ms == 0:
        output.write(f"{dataset},FAIL,FAIL,FAIL,FAIL,FAIL,FAIL,false\n")
    else:
        percentages = [100.0 * value / total_ms for value in categories]
        output.write(
            f"{dataset},{total_ms / 1000:.3f},"
            + ",".join(f"{value:.2f}" for value in percentages)
            + ",true\n"
        )
PY
}

run_citation() {
  local dataset="$1"
  local logfile="$LOG_DIR/citation_${dataset}.log"

  echo ">>> [$(date '+%H:%M:%S')] GAT end-to-end: $dataset"
  ./target/release/gat config.yaml pyg/weights "$dataset" > "$logfile" 2>&1
  local status=$?

  if [ "$status" -ne 0 ]; then
    echo "    COMMAND FAILED (exit ${status}); see $logfile"
    echo "${dataset},FAIL,FAIL,FAIL,FAIL,false" >> "$PROVING_RESULTS"
    if [ "$dataset" = "cora" ] || [ "$dataset" = "pubmed" ]; then
      echo "${dataset},FAIL,FAIL,FAIL,FAIL,FAIL,FAIL,false" >> "$BREAKDOWN_RESULTS"
    fi
    return
  fi

  if ! grep -q "^verified: true" "$logfile"; then
    if grep -q "^verified: false" "$logfile"; then
      echo "    VERIFICATION FAILED; see $logfile"
    else
      echo "    INCOMPLETE OUTPUT (no verification result); see $logfile"
    fi
    echo "${dataset},FAIL,FAIL,FAIL,FAIL,false" >> "$PROVING_RESULTS"
    if [ "$dataset" = "cora" ] || [ "$dataset" = "pubmed" ]; then
      echo "${dataset},FAIL,FAIL,FAIL,FAIL,FAIL,FAIL,false" >> "$BREAKDOWN_RESULTS"
    fi
    return
  fi

  local commit prove verify proof
  commit=$(parse_time_s "$(grep 'commit time:' "$logfile" | tail -1)")
  prove=$(parse_time_s "$(grep 'prove time:' "$logfile" | tail -1)")
  verify=$(parse_verify_ms "$(grep 'verify time:' "$logfile" | tail -1)")
  proof=$(grep 'Total proof size:' "$logfile" | tail -1 | sed -E 's/Total proof size: //')
  echo "${dataset},${commit},${prove},${verify},${proof},true" >> "$PROVING_RESULTS"
  echo "    commit=${commit}s prove=${prove}s verify=${verify}ms proof=${proof}"

  # The same proof log supplies the breakdown columns; no duplicate run needed.
  if [ "$dataset" = "cora" ] || [ "$dataset" = "pubmed" ]; then
    append_breakdown "$logfile" "$dataset"
  fi
}

run_synthetic() {
  local log_n="$1" mode="$2"
  local node_capacity=$((1 << log_n))
  local dataset="fake_${node_capacity}_d10_gat"
  local logfile="$LOG_DIR/zk_${log_n}_${mode}.log"
  local command=(./target/release/gat config.yaml pyg/weights "$dataset")
  [ "$mode" = "on" ] && command+=(--zk)

  echo ">>> [$(date '+%H:%M:%S')] GAT ZK overhead: N=2^${log_n} mode=${mode}"
  "${command[@]}" > "$logfile" 2>&1
  local status=$?

  if [ "$status" -ne 0 ]; then
    echo "    COMMAND FAILED (exit ${status}); see $logfile"
    echo "2^${log_n},${mode},FAIL,FAIL,FAIL,FAIL,FAIL,false" >> "$ZK_RESULTS"
    return
  fi

  if ! grep -q "^verified: true" "$logfile"; then
    if grep -q "^verified: false" "$logfile"; then
      echo "    VERIFICATION FAILED; see $logfile"
    else
      echo "    INCOMPLETE OUTPUT (no verification result); see $logfile"
    fi
    echo "2^${log_n},${mode},FAIL,FAIL,FAIL,FAIL,FAIL,false" >> "$ZK_RESULTS"
    return
  fi

  local commit prove verify proof zk_extra
  commit=$(parse_time_s "$(grep 'commit time:' "$logfile" | tail -1)")
  prove=$(parse_time_s "$(grep 'prove time:' "$logfile" | tail -1)")
  verify=$(parse_verify_ms "$(grep 'verify time:' "$logfile" | tail -1)")
  proof=$(grep 'Total proof size:' "$logfile" | tail -1 | sed -E 's/Total proof size: //')
  zk_extra=$(grep 'ZK extra proof size:' "$logfile" | tail -1 | sed -E 's/.*: ([0-9.]+) KB/\1/')
  [ -z "$zk_extra" ] && zk_extra="0"

  echo "2^${log_n},${mode},${commit},${prove},${verify},${proof},${zk_extra},true" >> "$ZK_RESULTS"
  echo "    commit=${commit}s prove=${prove}s verify=${verify}ms proof=${proof}"
}

show_results() {
  local title="$1" path="$2"
  echo ""
  echo "--- $title ---"
  if command -v column >/dev/null 2>&1; then
    column -t -s',' "$path"
  else
    sed -n '1,200p' "$path"
  fi
}

if [ ! -x ./target/release/gat ]; then
  echo "Missing ./target/release/gat"
  echo "Build it with: cargo build --release --bin gat"
  exit 1
fi

missing_data=0
for dataset in $GAT_CITATION_DATASETS; do
  if [ ! -f "pyg/weights/raw/${dataset}/meta.json" ]; then
    echo "Missing citation data: pyg/weights/raw/${dataset}/meta.json"
    missing_data=1
  fi
done
for log_n in $GAT_ZK_LOG_NS; do
  node_capacity=$((1 << log_n))
  dataset="fake_${node_capacity}_d10_gat"
  if [ ! -f "pyg/weights/raw/${dataset}/meta.json" ]; then
    echo "Missing GAT data: pyg/weights/raw/${dataset}/meta.json"
    missing_data=1
  fi
done
if [ "$missing_data" -ne 0 ]; then
  echo "Generate the missing synthetic inputs with:"
  echo "  (cd pyg && python gen_fake.py --gat-only)"
  exit 1
fi

echo "======================================"
echo "  Required corrected-GAT reruns"
echo "  Start: $(date)"
echo "======================================"

echo ""
echo "--- tab:proving and tab:breakdown ---"
for dataset in $GAT_CITATION_DATASETS; do
  run_citation "$dataset"
done

echo ""
echo "--- tab:zk-overhead (GAT rows only) ---"
for log_n in $GAT_ZK_LOG_NS; do
  for mode in $GAT_ZK_MODES; do
    run_synthetic "$log_n" "$mode"
  done
done

echo ""
echo "======================================"
echo "  Done: $(date)"
echo "======================================"

show_results "tab:proving GAT rows" "$PROVING_RESULTS"
show_results "tab:breakdown GAT columns" "$BREAKDOWN_RESULTS"
show_results "tab:zk-overhead GAT rows" "$ZK_RESULTS"

echo ""
echo "Raw logs: $LOG_DIR/"
