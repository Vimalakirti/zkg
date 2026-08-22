#!/bin/bash
# Repeat the six representative zkGNN/ezkl subgraph comparisons from the main
# paper table. System order alternates across repetitions. The ezkl command
# includes circuit setup but the reported `prove_s` remains its prove phase,
# matching the paper's comparison metric.
set -u
set -o pipefail
cd "$(dirname "$0")"

REPETITIONS="${PAIRED_REPETITIONS:-3}"
POINTS="${PAIRED_POINTS:-elliptic:188332 elliptic:170495 elliptic:185164 dgraphfin:644186 dgraphfin:284080 dgraphfin:1832174}"
RESULT_TAG="${PAIRED_RESULT_TAG:-ezkl_paired}"
LOG_DIR="repro_logs/${RESULT_TAG}"
RAW_RESULTS="repro_${RESULT_TAG}_raw.csv"
SUMMARY_RESULTS="repro_${RESULT_TAG}.csv"

if ! [[ "$REPETITIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "PAIRED_REPETITIONS must be a positive integer" >&2
  exit 2
fi
if [ ! -x ./target/release/graphsage ]; then
  echo "Missing ./target/release/graphsage; build it first." >&2
  exit 2
fi
python3 -c 'import ezkl' >/dev/null 2>&1 || {
  echo "The ezkl Python package is required (tested with ezkl==23.0.5)." >&2
  exit 2
}

mkdir -p "$LOG_DIR"
echo "system,dataset,sub_id,num_nodes,rep,prove_s,verify_ms,proof_kb,peak_rss_kb,verified" > "$RAW_RESULTS"

parse_time_s() {
  local line="$1"
  [ -z "$line" ] && { echo "NA"; return; }
  if echo "$line" | grep -qE 'time: [0-9.]+ms'; then
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

parse_proof_kb() {
  local line="$1" value unit
  read -r value unit <<< "$(echo "$line" | sed -E 's/.*Total proof size: ([0-9.]+) ([KMG]?B).*/\1 \2/')"
  case "$unit" in
    KB) awk -v v="$value" 'BEGIN{printf "%.3f", v}' ;;
    MB) awk -v v="$value" 'BEGIN{printf "%.3f", v*1024}' ;;
    B)  awk -v v="$value" 'BEGIN{printf "%.3f", v/1024}' ;;
    *) echo "NA" ;;
  esac
}

run_zkgnn() {
  local dataset="$1" sub_id="$2" rep="$3" num_nodes="$4"
  local sub_name="${dataset}_sub_${sub_id}"
  local logfile="$LOG_DIR/zkgnn_${dataset}_${sub_id}_rep${rep}.log"
  echo ">>> [$(date '+%H:%M:%S')] zkGNN ${dataset}/${sub_id} rep=${rep}"
  if ! /usr/bin/time -v ./target/release/graphsage config.yaml pyg/weights "$sub_name" "$dataset" > "$logfile" 2>&1; then
    echo "zkgnn,$dataset,$sub_id,$num_nodes,$rep,NA,NA,NA,NA,false" >> "$RAW_RESULTS"
    return 1
  fi
  if ! grep -q '^verified: true' "$logfile"; then
    echo "zkgnn,$dataset,$sub_id,$num_nodes,$rep,NA,NA,NA,NA,false" >> "$RAW_RESULTS"
    return 1
  fi
  local prove verify proof memory
  prove=$(parse_time_s "$(grep 'prove time:' "$logfile" | tail -1)")
  verify=$(parse_verify_ms "$(grep 'verify time:' "$logfile" | tail -1)")
  proof=$(parse_proof_kb "$(grep 'Total proof size:' "$logfile" | tail -1)")
  memory=$(grep 'Maximum resident set size' "$logfile" | tail -1 | sed -E 's/.*: ([0-9]+)$/\1/')
  echo "zkgnn,$dataset,$sub_id,$num_nodes,$rep,$prove,$verify,$proof,$memory,true" >> "$RAW_RESULTS"
}

run_ezkl() {
  local dataset="$1" sub_id="$2" rep="$3" num_nodes="$4"
  local logfile="$LOG_DIR/ezkl_${dataset}_${sub_id}_rep${rep}.log"
  local csvfile="$LOG_DIR/ezkl_${dataset}_${sub_id}_rep${rep}.csv"
  echo ">>> [$(date '+%H:%M:%S')] ezkl ${dataset}/${sub_id} rep=${rep}"
  if ! /usr/bin/time -v python3 ezkl_graphsage/run_ezkl_subgraph.py \
      --dataset "$dataset" --sub_ids "$sub_id" --output_csv "$csvfile" > "$logfile" 2>&1; then
    echo "ezkl,$dataset,$sub_id,$num_nodes,$rep,NA,NA,NA,NA,false" >> "$RAW_RESULTS"
    return 1
  fi
  if ! python3 - "$csvfile" "$rep" "$logfile" "$RAW_RESULTS" <<'PY'
import csv
import re
import sys

csv_path, rep, log_path, output_path = sys.argv[1:]
with open(csv_path, newline="") as source:
    rows = list(csv.DictReader(source))
if len(rows) != 1 or str(rows[0].get("verified", "")).lower() != "true":
    raise SystemExit(1)
row = rows[0]
log = open(log_path).read()
match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", log)
peak = match.group(1) if match else "NA"
with open(output_path, "a") as output:
    output.write(
        f"ezkl,{row['dataset']},{row['sub_id']},{row['num_nodes']},{rep},"
        f"{row['prove_s']},{float(row['verify_s']) * 1000:.3f},"
        f"{row['proof_size_kb']},{peak},true\n"
    )
PY
  then
    echo "ezkl,$dataset,$sub_id,$num_nodes,$rep,NA,NA,NA,NA,false" >> "$RAW_RESULTS"
    return 1
  fi
}

failed=0
for point in $POINTS; do
  dataset=${point%%:*}
  sub_id=${point##*:}
  meta="pyg/weights/raw/${dataset}_sub_${sub_id}/meta.json"
  if [ ! -f "$meta" ]; then
    echo "Missing subgraph metadata: $meta" >&2
    exit 2
  fi
  num_nodes=$(python3 -c "import json; print(json.load(open('$meta'))['num_nodes'])")
  for rep in $(seq 1 "$REPETITIONS"); do
    if [ $((rep % 2)) -eq 1 ]; then
      run_zkgnn "$dataset" "$sub_id" "$rep" "$num_nodes" || failed=1
      run_ezkl "$dataset" "$sub_id" "$rep" "$num_nodes" || failed=1
    else
      run_ezkl "$dataset" "$sub_id" "$rep" "$num_nodes" || failed=1
      run_zkgnn "$dataset" "$sub_id" "$rep" "$num_nodes" || failed=1
    fi
  done
done

echo "system,dataset,sub_id,num_nodes,successful_reps,prove_mean_s,prove_std_s,verify_mean_ms,verify_std_ms,proof_mean_kb,peak_rss_max_kb,all_verified" > "$SUMMARY_RESULTS"
awk -F, '
  NR == 1 { next }
  {
    key=$1 SUBSEP $2 SUBSEP $3 SUBSEP $4
    expected[key]++; system[key]=$1; dataset[key]=$2; subid[key]=$3; nodes[key]=$4
    if ($10 != "true") next
    n[key]++; p[key]+=$6; p2[key]+=$6*$6; v[key]+=$7; v2[key]+=$7*$7; sz[key]+=$8
    if ($9+0 > peak[key]+0) peak[key]=$9
  }
  END {
    for (key in expected) {
      count=n[key]+0
      if (count == 0) {
        printf "%s,%s,%s,%s,0,NA,NA,NA,NA,NA,NA,false\n", system[key],dataset[key],subid[key],nodes[key]
        continue
      }
      pm=p[key]/count; vm=v[key]/count
      pv=p2[key]/count-pm*pm; if (pv < 0) pv=0
      vv=v2[key]/count-vm*vm; if (vv < 0) vv=0
      all=(count == expected[key] ? "true" : "false")
      printf "%s,%s,%s,%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%s\n", \
        system[key],dataset[key],subid[key],nodes[key],count,pm,sqrt(pv),vm,sqrt(vv),sz[key]/count,peak[key],all
    }
  }
' "$RAW_RESULTS" | sort -t, -k2,2 -k4,4n -k1,1 >> "$SUMMARY_RESULTS"

echo "Raw results: $RAW_RESULTS"
echo "Summary:     $SUMMARY_RESULTS"
column -t -s',' "$SUMMARY_RESULTS" 2>/dev/null || true
exit "$failed"
