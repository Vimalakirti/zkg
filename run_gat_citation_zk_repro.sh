#!/bin/bash
# Measure zero-knowledge overhead for GAT on the three real citation graphs.
# Off/on order alternates across repetitions to reduce systematic machine-load
# and cache-order bias. Raw measurements and mean/std summaries are separate.
set -u
set -o pipefail
cd "$(dirname "$0")"

REPETITIONS="${GAT_CITATION_ZK_REPETITIONS:-3}"
DATASETS="${GAT_CITATION_ZK_DATASETS:-cora citeseer pubmed}"
RESULT_TAG="${GAT_CITATION_ZK_RESULT_TAG:-gat_citation_zk}"
LOG_DIR="repro_logs/${RESULT_TAG}"
RAW_RESULTS="repro_${RESULT_TAG}_raw.csv"
SUMMARY_RESULTS="repro_${RESULT_TAG}.csv"

if ! [[ "$REPETITIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "GAT_CITATION_ZK_REPETITIONS must be a positive integer" >&2
  exit 2
fi
if [ ! -x ./target/release/gat ]; then
  echo "Missing ./target/release/gat; build it with:" >&2
  echo "  cargo build --release --bin gat" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
echo "dataset,mode,rep,commit_s,prove_s,verify_ms,proof_kb,zk_extra_kb,peak_rss_kb,verified" > "$RAW_RESULTS"

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
  [ -z "$line" ] && { echo "NA"; return; }
  read -r value unit <<< "$(echo "$line" | sed -E 's/.*Total proof size: ([0-9.]+) ([KMG]?B).*/\1 \2/')"
  case "$unit" in
    B)  awk -v v="$value" 'BEGIN{printf "%.3f", v/1024}' ;;
    KB) awk -v v="$value" 'BEGIN{printf "%.3f", v}' ;;
    MB) awk -v v="$value" 'BEGIN{printf "%.3f", v*1024}' ;;
    GB) awk -v v="$value" 'BEGIN{printf "%.3f", v*1024*1024}' ;;
    *) echo "NA" ;;
  esac
}

run_one() {
  local dataset="$1" mode="$2" rep="$3"
  local logfile="${LOG_DIR}/${dataset}_${mode}_rep${rep}.log"
  local command=(./target/release/gat config.yaml pyg/weights "$dataset")
  [ "$mode" = "on" ] && command+=(--zk)

  echo ">>> [$(date '+%H:%M:%S')] GAT ${dataset} mode=${mode} rep=${rep}"
  if ! /usr/bin/time -v "${command[@]}" > "$logfile" 2>&1; then
    echo "    FAILED (nonzero exit); see $logfile"
    echo "$dataset,$mode,$rep,NA,NA,NA,NA,NA,NA,false" >> "$RAW_RESULTS"
    return 1
  fi
  if ! grep -q '^verified: true' "$logfile"; then
    echo "    FAILED (no 'verified: true'); see $logfile"
    echo "$dataset,$mode,$rep,NA,NA,NA,NA,NA,NA,false" >> "$RAW_RESULTS"
    return 1
  fi

  local commit prove verify proof zk_extra memory
  commit=$(parse_time_s "$(grep 'commit time:' "$logfile" | tail -1)")
  prove=$(parse_time_s "$(grep 'prove time:' "$logfile" | tail -1)")
  verify=$(parse_verify_ms "$(grep 'verify time:' "$logfile" | tail -1)")
  proof=$(parse_proof_kb "$(grep 'Total proof size:' "$logfile" | tail -1)")
  zk_extra=$(grep 'ZK extra proof size:' "$logfile" | tail -1 | sed -E 's/.*: ([0-9.]+) KB/\1/')
  [ -z "$zk_extra" ] && zk_extra="0"
  memory=$(grep 'Maximum resident set size' "$logfile" | tail -1 | sed -E 's/.*: ([0-9]+)$/\1/')
  [ -z "$memory" ] && memory="NA"

  echo "    commit=${commit}s prove=${prove}s verify=${verify}ms proof=${proof}KB peak=${memory}KB"
  echo "$dataset,$mode,$rep,$commit,$prove,$verify,$proof,$zk_extra,$memory,true" >> "$RAW_RESULTS"
  return 0
}

failed=0
for dataset in $DATASETS; do
  if [ ! -f "pyg/weights/raw/${dataset}/meta.json" ]; then
    echo "Missing citation data: pyg/weights/raw/${dataset}/meta.json" >&2
    exit 2
  fi
  for rep in $(seq 1 "$REPETITIONS"); do
    if [ $((rep % 2)) -eq 1 ]; then
      run_one "$dataset" off "$rep" || failed=1
      run_one "$dataset" on "$rep" || failed=1
    else
      run_one "$dataset" on "$rep" || failed=1
      run_one "$dataset" off "$rep" || failed=1
    fi
  done
done

echo "dataset,mode,successful_reps,commit_mean_s,commit_std_s,prove_mean_s,prove_std_s,verify_mean_ms,verify_std_ms,proof_mean_kb,zk_extra_mean_kb,peak_rss_max_kb,all_verified" > "$SUMMARY_RESULTS"
awk -F, '
  NR == 1 { next }
  {
    key=$1 SUBSEP $2
    expected[key]++
    dataset[key]=$1; mode[key]=$2
    if ($10 != "true") next
    n[key]++
    c[key]+=$4; c2[key]+=$4*$4
    p[key]+=$5; p2[key]+=$5*$5
    v[key]+=$6; v2[key]+=$6*$6
    sz[key]+=$7; zksz[key]+=$8
    if ($9+0 > peak[key]+0) peak[key]=$9
  }
  END {
    for (key in expected) {
      count=n[key]+0
      if (count == 0) {
        printf "%s,%s,0,NA,NA,NA,NA,NA,NA,NA,NA,NA,false\n", dataset[key],mode[key]
        continue
      }
      cm=c[key]/count; pm=p[key]/count; vm=v[key]/count
      cv=c2[key]/count-cm*cm; if (cv < 0) cv=0
      pv=p2[key]/count-pm*pm; if (pv < 0) pv=0
      vv=v2[key]/count-vm*vm; if (vv < 0) vv=0
      all=(count == expected[key] ? "true" : "false")
      printf "%s,%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%s\n", \
        dataset[key],mode[key],count,cm,sqrt(cv),pm,sqrt(pv),vm,sqrt(vv), \
        sz[key]/count,zksz[key]/count,peak[key],all
    }
  }
' "$RAW_RESULTS" | sort -t, -k1,1 -k2,2 >> "$SUMMARY_RESULTS"

echo "Raw results: $RAW_RESULTS"
echo "Summary:     $SUMMARY_RESULTS"
column -t -s',' "$SUMMARY_RESULTS" 2>/dev/null || true
exit "$failed"
