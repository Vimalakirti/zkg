#!/bin/bash
# Rerun only the measurements that did not complete in the first corrected-GAT
# run: PubMed end-to-end/breakdown and synthetic 2^14--2^15 with ZK off/on.
set -u
cd "$(dirname "$0")"

export GAT_RESULT_TAG="failed_only"
export GAT_CITATION_DATASETS="pubmed"
export GAT_ZK_LOG_NS="14 15"
export GAT_ZK_MODES="off on"

exec ./run_required_gat_reruns.sh
