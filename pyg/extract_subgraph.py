#!/usr/bin/env python3
"""Extract k-hop subgraphs around target nodes for efficient ZK proving.

For a k-layer GNN, a node's output only depends on its k-hop neighborhood.
This script extracts minimal subgraphs so the prover only works on O(|neighborhood|)
nodes instead of the full graph.

Usage:
  python extract_subgraph.py <dataset> <target_nodes> [--khops K] [--output_dir DIR]

  target_nodes: comma-separated node indices, "test" for all test nodes,
                or "test:N" for N random test nodes

Output: one subgraph per target node (or batched) in binary format:
  subgraph_<id>/
    x.bin, edge_src.bin, edge_dst.bin, y.bin, meta.json
    mapping.json  (subgraph_idx -> original_idx, target_node info)
"""

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np


def load_dataset(data_dir, dataset_name):
    ds_dir = os.path.join(data_dir, "raw", dataset_name)
    meta = json.load(open(os.path.join(ds_dir, "meta.json")))

    x = np.fromfile(os.path.join(ds_dir, "x.bin"), dtype=np.float32)
    x = x.reshape(meta["num_nodes"], meta["num_features"])

    y = np.fromfile(os.path.join(ds_dir, "y.bin"), dtype=np.int32)
    edge_src = np.fromfile(os.path.join(ds_dir, "edge_src.bin"), dtype=np.int32)
    edge_dst = np.fromfile(os.path.join(ds_dir, "edge_dst.bin"), dtype=np.int32)

    train_mask = np.fromfile(os.path.join(ds_dir, "train_mask.bin"), dtype=np.uint8)
    test_mask = np.fromfile(os.path.join(ds_dir, "test_mask.bin"), dtype=np.uint8)

    # Build undirected adjacency for neighborhood discovery
    adj = defaultdict(set)
    for s, d in zip(edge_src.tolist(), edge_dst.tolist()):
        adj[s].add(d)
        adj[d].add(s)

    return {
        "x": x, "y": y, "edge_src": edge_src, "edge_dst": edge_dst,
        "train_mask": train_mask, "test_mask": test_mask,
        "adj": adj, "meta": meta,
    }


def khop_neighborhood(adj, target_nodes, k):
    """Find all nodes within k hops of any target node."""
    visited = set(target_nodes)
    frontier = set(target_nodes)
    for _ in range(k):
        next_frontier = set()
        for n in frontier:
            next_frontier.update(adj[n])
        next_frontier -= visited
        visited.update(next_frontier)
        frontier = next_frontier
    return visited


def extract_subgraph(dataset, target_nodes, k):
    """Extract k-hop subgraph around target_nodes.

    Returns a dict with subgraph data and mapping info.
    """
    # Find k-hop neighborhood
    sub_nodes = khop_neighborhood(dataset["adj"], target_nodes, k)
    sub_nodes = sorted(sub_nodes)

    # Create mapping: original_idx -> subgraph_idx
    orig_to_sub = {orig: sub for sub, orig in enumerate(sub_nodes)}

    # Extract node features and labels
    sub_x = dataset["x"][sub_nodes]
    sub_y = dataset["y"][sub_nodes]

    # Extract edges within subgraph (directed, from original edge list)
    edge_src = dataset["edge_src"]
    edge_dst = dataset["edge_dst"]
    sub_node_set = set(sub_nodes)

    sub_edges_src = []
    sub_edges_dst = []
    for s, d in zip(edge_src.tolist(), edge_dst.tolist()):
        if s in sub_node_set and d in sub_node_set:
            sub_edges_src.append(orig_to_sub[s])
            sub_edges_dst.append(orig_to_sub[d])

    sub_edge_src = np.array(sub_edges_src, dtype=np.int32)
    sub_edge_dst = np.array(sub_edges_dst, dtype=np.int32)

    # Map target nodes to subgraph indices
    target_sub = [orig_to_sub[t] for t in target_nodes]

    # Create masks: target nodes get test_mask=1
    sub_train_mask = np.zeros(len(sub_nodes), dtype=np.uint8)
    sub_test_mask = np.zeros(len(sub_nodes), dtype=np.uint8)
    for t in target_sub:
        sub_test_mask[t] = 1

    return {
        "x": sub_x.astype(np.float32),
        "y": sub_y.astype(np.int32),
        "edge_src": sub_edge_src,
        "edge_dst": sub_edge_dst,
        "train_mask": sub_train_mask,
        "test_mask": sub_test_mask,
        "num_nodes": len(sub_nodes),
        "num_features": sub_x.shape[1],
        "num_edges": len(sub_edge_src),
        "num_classes": dataset["meta"]["num_classes"],
        # Mapping info
        "sub_to_orig": sub_nodes,
        "target_nodes_orig": list(target_nodes),
        "target_nodes_sub": target_sub,
    }


def save_subgraph(subgraph, output_dir, name):
    """Save subgraph in binary format compatible with zk-torch-3."""
    out_dir = os.path.join(output_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    subgraph["x"].tofile(os.path.join(out_dir, "x.bin"))
    subgraph["y"].tofile(os.path.join(out_dir, "y.bin"))
    subgraph["edge_src"].tofile(os.path.join(out_dir, "edge_src.bin"))
    subgraph["edge_dst"].tofile(os.path.join(out_dir, "edge_dst.bin"))
    subgraph["train_mask"].tofile(os.path.join(out_dir, "train_mask.bin"))
    subgraph["test_mask"].tofile(os.path.join(out_dir, "test_mask.bin"))

    # Also write a dummy val_mask
    np.zeros(subgraph["num_nodes"], dtype=np.uint8).tofile(
        os.path.join(out_dir, "val_mask.bin"))

    meta = {
        "num_nodes": subgraph["num_nodes"],
        "num_features": subgraph["num_features"],
        "num_edges": subgraph["num_edges"],
        "num_classes": subgraph["num_classes"],
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    mapping = {
        "sub_to_orig": subgraph["sub_to_orig"],
        "target_nodes_orig": subgraph["target_nodes_orig"],
        "target_nodes_sub": subgraph["target_nodes_sub"],
    }
    with open(os.path.join(out_dir, "mapping.json"), "w") as f:
        json.dump(mapping, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Extract k-hop subgraphs for ZK proving")
    parser.add_argument("dataset", type=str, help="Dataset name (e.g., elliptic)")
    parser.add_argument("targets", type=str,
                        help="Target nodes: comma-separated indices, 'test', or 'test:N'")
    parser.add_argument("--khops", type=int, default=2,
                        help="Number of hops (should match GNN layers, default: 2)")
    parser.add_argument("--data_dir", type=str, default="weights",
                        help="Data directory containing raw/<dataset>/")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: weights/raw/)")
    parser.add_argument("--batch", action="store_true",
                        help="Batch all target nodes into one subgraph")
    parser.add_argument("--max_batch_size", type=int, default=100,
                        help="Max target nodes per batch (default: 100)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, "raw")

    print(f"Loading {args.dataset}...")
    dataset = load_dataset(args.data_dir, args.dataset)
    print(f"  Nodes: {dataset['meta']['num_nodes']}, "
          f"Edges: {dataset['meta']['num_edges']}")

    # Parse target nodes
    if args.targets == "test":
        target_list = np.where(dataset["test_mask"] > 0)[0].tolist()
    elif args.targets.startswith("test:"):
        n = int(args.targets.split(":")[1])
        all_test = np.where(dataset["test_mask"] > 0)[0].tolist()
        random.seed(42)
        target_list = random.sample(all_test, min(n, len(all_test)))
    else:
        target_list = [int(x) for x in args.targets.split(",")]

    print(f"  Target nodes: {len(target_list)}")

    model_tag = f"graphsage_{args.dataset}"

    if args.batch:
        # Batch all targets into one subgraph
        for i in range(0, len(target_list), args.max_batch_size):
            batch = target_list[i:i + args.max_batch_size]
            sub = extract_subgraph(dataset, batch, args.khops)
            name = f"{args.dataset}_sub_batch{i // args.max_batch_size}"
            save_subgraph(sub, args.output_dir, name)
            print(f"  {name}: {sub['num_nodes']} nodes, {sub['num_edges']} edges, "
                  f"{len(batch)} targets")
    else:
        # One subgraph per target node
        sizes = []
        for idx, target in enumerate(target_list):
            sub = extract_subgraph(dataset, [target], args.khops)
            name = f"{args.dataset}_sub_{target}"
            save_subgraph(sub, args.output_dir, name)
            sizes.append(sub["num_nodes"])
            if idx < 10 or (idx + 1) % 100 == 0:
                print(f"  Node {target}: {sub['num_nodes']} nodes, "
                      f"{sub['num_edges']} edges")

        sizes = np.array(sizes)
        print(f"\nSubgraph stats ({len(sizes)} subgraphs):")
        print(f"  Nodes: min={sizes.min()}, max={sizes.max()}, "
              f"mean={sizes.mean():.1f}, median={np.median(sizes):.0f}")


if __name__ == "__main__":
    main()
