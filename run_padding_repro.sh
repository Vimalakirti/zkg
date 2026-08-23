#!/bin/bash
# Measure the cost of hiding the exact factor count in data-independent mode.
# Each repetition runs the same ZK workload with the true factor count and
# with a predeclared public capacity. The order alternates across repetitions
# to limit systematic bias from warm caches or machine drift. Raw and mean/std
# summaries are emitted separately.
set -u
set -o pipefail
cd "$(dirname "$0")"

REPETITIONS="${PADDING_REPETITIONS:-3}"
MODELS="${PADDING_MODELS:-gcn graphsage}"
DATASETS="${PADDING_DATASETS:-cora:32 citeseer:32 pubmed:128}"
RESULT_TAG="${PADDING_RESULT_TAG:-padding}"
LOG_DIR="repro_logs/${RESULT_TAG}"
RAW_RESULTS="repro_${RESULT_TAG}_raw.csv"
SUMMARY_RESULTS="repro_${RESULT_TAG}.csv"

if ! [[ "$REPETITIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "PADDING_REPETITIONS must be a positive integer" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
echo "model,dataset,mode,capacity,rep,commit_s,prove_s,verify_ms,proof_kb,peak_rss_kb,verified" > "$RAW_RESULTS"

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
    local secs
    secs=$(echo "$line" | sed -E 's/.*time: ([0-9.]+)s.*/\1/')
    awk -v s="$secs" 'BEGIN{printf "%.3f", s*1000}'
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
  local model=$1 dataset=$2 capacity=$3 mode=$4 rep=$5
  local logfile="${LOG_DIR}/${model}_${dataset}_${mode}_rep${rep}.log"
  local capacity_args=()
  if [ "$mode" = "padded" ]; then
    capacity_args=(--factor-capacity "$capacity")
  fi

  echo ">>> [$(date '+%H:%M:%S')] $model $dataset $mode rep=$rep (capacity=$capacity)"
  if ! /usr/bin/time -v "./target/release/${model}" config.yaml pyg/weights "$dataset" \
      --zk "${capacity_args[@]}" > "$logfile" 2>&1; then
    echo "    FAILED (nonzero exit); see $logfile"
    echo "$model,$dataset,$mode,$capacity,$rep,NA,NA,NA,NA,NA,false" >> "$RAW_RESULTS"
    return 1
  fi

  if ! grep -q "^verified: true" "$logfile"; then
    echo "    FAILED (no 'verified: true'); see $logfile"
    echo "$model,$dataset,$mode,$capacity,$rep,NA,NA,NA,NA,NA,false" >> "$RAW_RESULTS"
    return 1
  fi

  local commit prove verify proof memory
  commit=$(parse_time_s "$(grep 'commit time:' "$logfile" | tail -1)")
  prove=$(parse_time_s "$(grep 'prove time:' "$logfile" | tail -1)")
  verify=$(parse_verify_ms "$(grep 'verify time:' "$logfile" | tail -1)")
  proof=$(parse_proof_kb "$(grep 'Total proof size:' "$logfile" | tail -1)")
  memory=$(grep 'Maximum resident set size' "$logfile" | tail -1 | sed -E 's/.*: ([0-9]+)$/\1/')
  [ -z "$memory" ] && memory="NA"

  echo "    commit=${commit}s prove=${prove}s verify=${verify}ms proof=${proof}KB peak=${memory}KB"
  echo "$model,$dataset,$mode,$capacity,$rep,$commit,$prove,$verify,$proof,$memory,true" >> "$RAW_RESULTS"
  return 0
}

failed=0
for model in $MODELS; do
  if [ ! -x "./target/release/${model}" ]; then
    echo "Missing ./target/release/${model}; build the paper binaries first." >&2
    exit 2
  fi
  for entry in $DATASETS; do
    dataset=${entry%%:*}
    capacity=${entry##*:}
    if [ "$dataset" = "$capacity" ] || ! [[ "$capacity" =~ ^[1-9][0-9]*$ ]]; then
      echo "Invalid PADDING_DATASETS entry '$entry'; expected dataset:capacity" >&2
      exit 2
    fi
    for rep in $(seq 1 "$REPETITIONS"); do
      if [ $((rep % 2)) -eq 1 ]; then
        run_one "$model" "$dataset" "$capacity" unpadded "$rep" || failed=1
        run_one "$model" "$dataset" "$capacity" padded "$rep" || failed=1
      else
        run_one "$model" "$dataset" "$capacity" padded "$rep" || failed=1
        run_one "$model" "$dataset" "$capacity" unpadded "$rep" || failed=1
      fi
    done
  done
done

echo "model,dataset,mode,capacity,successful_reps,commit_mean_s,commit_std_s,prove_mean_s,prove_std_s,verify_mean_ms,verify_std_ms,proof_mean_kb,peak_rss_max_kb,all_verified" > "$SUMMARY_RESULTS"
awk -F, '
  NR == 1 { next }
  {
    key=$1 SUBSEP $2 SUBSEP $3 SUBSEP $4
    expected[key]++
    model[key]=$1; dataset[key]=$2; mode[key]=$3; capacity[key]=$4
    if ($11 != "true") next
    n[key]++
    c[key]+=$6; c2[key]+=$6*$6
    p[key]+=$7; p2[key]+=$7*$7
    v[key]+=$8; v2[key]+=$8*$8
    sz[key]+=$9
    if ($10+0 > peak[key]+0) peak[key]=$10
  }
  END {
    for (key in expected) {
      count=n[key]+0
      if (count == 0) {
        printf "%s,%s,%s,%s,0,NA,NA,NA,NA,NA,NA,NA,NA,false\n", \
          model[key],dataset[key],mode[key],capacity[key]
        continue
      }
      cmean=c[key]/count; pmean=p[key]/count; vmean=v[key]/count
      cvar=(c2[key]/count)-cmean*cmean; if (cvar < 0) cvar=0
      pvar=(p2[key]/count)-pmean*pmean; if (pvar < 0) pvar=0
      vvar=(v2[key]/count)-vmean*vmean; if (vvar < 0) vvar=0
      cstd=sqrt(cvar); pstd=sqrt(pvar); vstd=sqrt(vvar)
      all=(count == expected[key] ? "true" : "false")
      printf "%s,%s,%s,%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%s\n", \
        model[key],dataset[key],mode[key],capacity[key],count, \
        cmean,cstd,pmean,pstd,vmean,vstd,sz[key]/count,peak[key],all
    }
  }
' "$RAW_RESULTS" | sort -t, -k1,1 -k2,2 -k3,3 >> "$SUMMARY_RESULTS"

echo "Raw results: $RAW_RESULTS"
echo "Summary:     $SUMMARY_RESULTS"
column -t -s',' "$SUMMARY_RESULTS" 2>/dev/null || true
exit "$failed"
