#!/bin/bash
# Measure generation, loading, serialized size, and peak RSS for the reusable
# KZH3 SRS used by the balanced adjacency decomposition. For N=2^16 and 2^17,
# the largest factor polynomials have 24 and 26 variables, respectively.
set -u
set -o pipefail
cd "$(dirname "$0")"

SRS_DEGREES="${SRS_SETUP_DEGREES:-24 26}"
RESULT_TAG="${SRS_SETUP_RESULT_TAG:-srs_setup}"
RESULTS="repro_${RESULT_TAG}.csv"
LOG_DIR="$(pwd)/repro_logs/${RESULT_TAG}"
SETUP_BIN="$(pwd)/target/release/setup"

if [ ! -x "$SETUP_BIN" ]; then
  echo "Missing $SETUP_BIN; build the ICICLE setup binary with:" >&2
  echo "  cargo build --release --no-default-features --features icicle --bin setup" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/zkg-srs-setup.XXXXXX")
cleanup() {
  case "$WORK_DIR" in
    "${TMPDIR:-/tmp}"/zkg-srs-setup.*) rm -rf -- "$WORK_DIR" ;;
  esac
}
if [ "${SRS_SETUP_KEEP_FILES:-0}" != "1" ]; then
  trap cleanup EXIT
else
  echo "SRS files will be retained in $WORK_DIR"
fi

echo "factor_poly_vars,node_capacity,generation_wall_s,generation_peak_rss_kb,load_wall_s,load_peak_rss_kb,file_bytes,loaded" > "$RESULTS"

for degree in $SRS_DEGREES; do
  if ! [[ "$degree" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid SRS_SETUP_DEGREES entry '$degree'" >&2
    exit 2
  fi
  case "$degree" in
    24) node_capacity="2^16" ;;
    26) node_capacity="2^17" ;;
    *) node_capacity="NA" ;;
  esac

  generate_log="$LOG_DIR/generate_${degree}.log"
  generate_metrics="$LOG_DIR/generate_${degree}.time"
  load_log="$LOG_DIR/load_${degree}.log"
  load_metrics="$LOG_DIR/load_${degree}.time"

  echo ">>> [$(date '+%H:%M:%S')] generate KZH3 SRS for ${degree}-variable factors"
  if ! (cd "$WORK_DIR" && /usr/bin/time -f '%e,%M' -o "$generate_metrics" "$SETUP_BIN" generate "$degree" > "$generate_log" 2>&1); then
    echo "    FAILED during generation; see $generate_log"
    echo "$degree,$node_capacity,NA,NA,NA,NA,NA,false" >> "$RESULTS"
    continue
  fi
  IFS=, read -r generation_wall generation_peak < "$generate_metrics"
  file_path="$WORK_DIR/${degree}.srs"
  if [ ! -f "$file_path" ]; then
    echo "    FAILED: setup did not create $file_path"
    echo "$degree,$node_capacity,$generation_wall,$generation_peak,NA,NA,NA,false" >> "$RESULTS"
    continue
  fi
  file_bytes=$(stat -c '%s' "$file_path")

  echo ">>> [$(date '+%H:%M:%S')] load KZH3 SRS for ${degree}-variable factors"
  if ! (cd "$WORK_DIR" && /usr/bin/time -f '%e,%M' -o "$load_metrics" "$SETUP_BIN" load "$degree" > "$load_log" 2>&1); then
    echo "    FAILED during loading; see $load_log"
    echo "$degree,$node_capacity,$generation_wall,$generation_peak,NA,NA,$file_bytes,false" >> "$RESULTS"
    continue
  fi
  IFS=, read -r load_wall load_peak < "$load_metrics"
  echo "$degree,$node_capacity,$generation_wall,$generation_peak,$load_wall,$load_peak,$file_bytes,true" >> "$RESULTS"
done

echo "Results: $RESULTS"
column -t -s',' "$RESULTS" 2>/dev/null || true
