#!/usr/bin/env python3
"""Run zk-torch-3 on subgraphs and compute AUC from output logits.

Runs the graphsage binary on each extracted subgraph, parses LOGIT lines,
and computes ROC-AUC over all target nodes.
"""

import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score, classification_report

ZKGNN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.path.join(ZKGNN_DIR, "target", "release", "graphsage")


def run_subgraph(sub_name, weights_name="elliptic", data_dir="weights", prove=False):
    """Run zk-torch-3 on a subgraph, return logits for test nodes."""
    cmd = [BINARY, "config.yaml", os.path.join("pyg", data_dir), sub_name, weights_name]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ZKGNN_DIR,
                            timeout=300)

    logits = []
    prove_ms = None
    verify_ms = None
    verified = None

    for line in result.stdout.split("\n"):
        if line.startswith("LOGIT|"):
            parts = line.split("|")
            node_idx = int(parts[1])
            label = int(parts[2])
            logit_0 = int(parts[3])
            logit_1 = int(parts[4])
            logits.append((node_idx, label, logit_0, logit_1))
        elif "prove time:" in line:
            import re
            m = re.search(r'([\d.]+)ms', line)
            if m:
                prove_ms = float(m.group(1))
            else:
                m = re.search(r'([\d.]+)s', line)
                if m:
                    prove_ms = float(m.group(1)) * 1000
        elif "verify time:" in line:
            import re
            m = re.search(r'([\d.]+)ms', line)
            if m:
                verify_ms = float(m.group(1))
        elif "verified:" in line:
            verified = "true" in line

    return logits, prove_ms, verify_ms, verified


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="weights")
    parser.add_argument("--weights_name", default="elliptic")
    parser.add_argument("--dataset", default="elliptic")
    args = parser.parse_args()

    data_dir = args.data_dir
    raw_dir = os.path.join(data_dir, "raw")

    # Find all subgraph directories
    sub_dirs = sorted([
        d for d in os.listdir(raw_dir)
        if d.startswith(f"{args.dataset}_sub_") and not d.startswith(f"{args.dataset}_sub_batch")
    ])

    if not sub_dirs:
        print(f"No subgraphs found in {raw_dir} matching {args.dataset}_sub_*")
        return

    print(f"Found {len(sub_dirs)} subgraphs")

    all_labels = []
    all_scores = []  # logit_1 - logit_0 as score for AUC
    all_preds = []
    total_prove_ms = 0
    total_verify_ms = 0
    proved = 0

    for i, sub_name in enumerate(sub_dirs):
        # Load mapping to get original node info
        mapping_path = os.path.join(raw_dir, sub_name, "mapping.json")
        mapping = json.load(open(mapping_path))

        logits, prove_ms, verify_ms, verified = run_subgraph(
            sub_name, args.weights_name, data_dir)

        for node_idx, label, logit_0, logit_1 in logits:
            all_labels.append(label)
            all_scores.append(logit_1 - logit_0)  # higher = more likely illicit
            all_preds.append(1 if logit_1 > logit_0 else 0)

        if prove_ms is not None:
            total_prove_ms += prove_ms
            total_verify_ms += verify_ms
            proved += 1

        orig_target = mapping["target_nodes_orig"][0]
        meta = json.load(open(os.path.join(raw_dir, sub_name, "meta.json")))
        status = f"verified={verified}" if verified is not None else "no-prove"

        if (i + 1) % 20 == 0 or i < 5:
            print(f"  [{i+1}/{len(sub_dirs)}] {sub_name}: "
                  f"{meta['num_nodes']} nodes, {status}")

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores, dtype=float)
    all_preds = np.array(all_preds)

    # Compute metrics
    auc = roc_auc_score(all_labels, all_scores)
    acc = (all_preds == all_labels).mean()

    print(f"\n=== Results: GraphSAGE on {args.dataset} (subgraph proving) ===")
    print(f"  Nodes evaluated: {len(all_labels)}")
    print(f"  Label distribution: licit={int((all_labels == 0).sum())}, "
          f"illicit={int((all_labels == 1).sum())}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC:      {auc:.4f}")
    if proved > 0:
        print(f"  Avg prove time:  {total_prove_ms / proved:.1f}ms")
        print(f"  Avg verify time: {total_verify_ms / proved:.1f}ms")

    # Per-class metrics
    print(f"\n{classification_report(all_labels, all_preds, target_names=['licit', 'illicit'])}")


if __name__ == "__main__":
    main()
