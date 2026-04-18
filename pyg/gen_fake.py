"""Generate fake GraphSage datasets for ablation study.

Usage:
    # Generate a grid of (nodes, avg_degree) combinations:
    python gen_fake.py --nodes 12 13 14 15 --degrees 5 10 20 50

    # Single dataset:
    python gen_fake.py --nodes 14 --degrees 10

    # Defaults: nodes 2^12..2^18, degree 10
    python gen_fake.py

Dataset naming: fake_{num_nodes}_d{avg_degree}
  e.g., fake_4096_d10, fake_16384_d50

Each dataset has:
  - 16 input features, 16 hidden, 7 classes
  - Random Erdos-Renyi-like edges (configurable avg degree)
  - Random node features, labels, masks
  - Random GraphSage weights (2-layer, with bias)
"""

import argparse
import json
import os

import numpy as np


def gen_dataset(num_nodes, avg_degree=10, num_features=16, hidden=16,
                num_classes=7, out_dir="weights"):
    name = f"fake_{num_nodes}_d{avg_degree}"
    ds_dir = os.path.join(out_dir, "raw", name)
    w_dir = os.path.join(out_dir, "raw", f"graphsage_{name}")
    os.makedirs(ds_dir, exist_ok=True)
    os.makedirs(w_dir, exist_ok=True)

    rng = np.random.default_rng(42)

    # --- Node features: (num_nodes, num_features), float32 ---
    x = rng.standard_normal((num_nodes, num_features)).astype(np.float32) * 0.1
    x.tofile(os.path.join(ds_dir, "x.bin"))

    # --- Edges: random sparse graph, ~avg_degree edges per node ---
    num_edges = num_nodes * avg_degree
    src = rng.integers(0, num_nodes, size=num_edges).astype(np.int32)
    dst = rng.integers(0, num_nodes, size=num_edges).astype(np.int32)
    # Remove self-loops
    mask = src != dst
    src, dst = src[mask], dst[mask]
    src.tofile(os.path.join(ds_dir, "edge_src.bin"))
    dst.tofile(os.path.join(ds_dir, "edge_dst.bin"))

    # --- Labels: random ---
    y = rng.integers(0, num_classes, size=num_nodes).astype(np.int32)
    y.tofile(os.path.join(ds_dir, "y.bin"))

    # --- Masks ---
    train_mask = np.zeros(num_nodes, dtype=np.uint8)
    val_mask = np.zeros(num_nodes, dtype=np.uint8)
    test_mask = np.zeros(num_nodes, dtype=np.uint8)
    train_mask[:num_nodes // 2] = 1
    val_mask[num_nodes // 2: num_nodes * 3 // 4] = 1
    test_mask[num_nodes * 3 // 4:] = 1
    train_mask.tofile(os.path.join(ds_dir, "train_mask.bin"))
    val_mask.tofile(os.path.join(ds_dir, "val_mask.bin"))
    test_mask.tofile(os.path.join(ds_dir, "test_mask.bin"))

    # --- meta.json ---
    meta = {
        "num_nodes": num_nodes,
        "num_features": num_features,
        "num_classes": num_classes,
        "num_edges": int(len(src)),
        "avg_degree": avg_degree,
    }
    with open(os.path.join(ds_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    # --- GraphSage weights (2-layer, random) ---
    # Layer 1: W_neighbor (hidden, num_features), W_self (hidden, num_features), bias (hidden,)
    # Layer 2: W_neighbor (num_classes, hidden), W_self (num_classes, hidden), bias (num_classes,)
    # PyTorch layout: (out_features, in_features), row-major float32
    scale = 0.1
    w1_l = (rng.standard_normal((hidden, num_features)) * scale).astype(np.float32)
    w1_r = (rng.standard_normal((hidden, num_features)) * scale).astype(np.float32)
    b1 = (rng.standard_normal(hidden) * scale).astype(np.float32)
    w2_l = (rng.standard_normal((num_classes, hidden)) * scale).astype(np.float32)
    w2_r = (rng.standard_normal((num_classes, hidden)) * scale).astype(np.float32)
    b2 = (rng.standard_normal(num_classes) * scale).astype(np.float32)

    w1_l.tofile(os.path.join(w_dir, "conv1.lin_l.weight.bin"))
    w1_r.tofile(os.path.join(w_dir, "conv1.lin_r.weight.bin"))
    b1.tofile(os.path.join(w_dir, "conv1.lin_l.bias.bin"))
    w2_l.tofile(os.path.join(w_dir, "conv2.lin_l.weight.bin"))
    w2_r.tofile(os.path.join(w_dir, "conv2.lin_r.weight.bin"))
    b2.tofile(os.path.join(w_dir, "conv2.lin_l.bias.bin"))

    w_meta = {"hidden": hidden, "num_features": num_features, "num_classes": num_classes}
    with open(os.path.join(w_dir, "meta.json"), "w") as f:
        json.dump(w_meta, f)

    # --- GCN weights (2-layer, random) ---
    gcn_dir = os.path.join(out_dir, "raw", f"gcn_{name}")
    os.makedirs(gcn_dir, exist_ok=True)
    gcn_w1 = (rng.standard_normal((hidden, num_features)) * scale).astype(np.float32)
    gcn_w2 = (rng.standard_normal((num_classes, hidden)) * scale).astype(np.float32)
    gcn_w1.tofile(os.path.join(gcn_dir, "conv1.lin.weight.bin"))
    gcn_w2.tofile(os.path.join(gcn_dir, "conv2.lin.weight.bin"))
    gcn_meta = {"weights": [
        {"name": "conv1.lin.weight", "shape": [hidden, num_features], "dtype": "float32"},
        {"name": "conv2.lin.weight", "shape": [num_classes, hidden], "dtype": "float32"},
    ]}
    with open(os.path.join(gcn_dir, "meta.json"), "w") as f:
        json.dump(gcn_meta, f)

    # --- GAT weights (2-layer, 4 heads -> 1 head, random) ---
    num_heads_1 = 4
    head_dim_1 = hidden // num_heads_1  # 4 heads, each hidden/4 dim
    num_heads_2 = 1
    gat_dir = os.path.join(out_dir, "raw", f"gat_{name}")
    os.makedirs(gat_dir, exist_ok=True)
    gat_w1 = (rng.standard_normal((num_heads_1 * head_dim_1, num_features)) * scale).astype(np.float32)
    gat_att1_src = (rng.standard_normal((1, num_heads_1, head_dim_1)) * scale).astype(np.float32)
    gat_att1_dst = (rng.standard_normal((1, num_heads_1, head_dim_1)) * scale).astype(np.float32)
    gat_w2 = (rng.standard_normal((num_classes, hidden)) * scale).astype(np.float32)
    gat_att2_src = (rng.standard_normal((1, num_heads_2, num_classes)) * scale).astype(np.float32)
    gat_att2_dst = (rng.standard_normal((1, num_heads_2, num_classes)) * scale).astype(np.float32)
    gat_w1.tofile(os.path.join(gat_dir, "conv1.lin.weight.bin"))
    gat_att1_src.tofile(os.path.join(gat_dir, "conv1.att_src.bin"))
    gat_att1_dst.tofile(os.path.join(gat_dir, "conv1.att_dst.bin"))
    gat_w2.tofile(os.path.join(gat_dir, "conv2.lin.weight.bin"))
    gat_att2_src.tofile(os.path.join(gat_dir, "conv2.att_src.bin"))
    gat_att2_dst.tofile(os.path.join(gat_dir, "conv2.att_dst.bin"))
    gat_meta = {"weights": [
        {"name": "conv1.att_src", "shape": [1, num_heads_1, head_dim_1], "dtype": "float32"},
        {"name": "conv1.att_dst", "shape": [1, num_heads_1, head_dim_1], "dtype": "float32"},
        {"name": "conv1.lin.weight", "shape": [num_heads_1 * head_dim_1, num_features], "dtype": "float32"},
        {"name": "conv2.att_src", "shape": [1, num_heads_2, num_classes], "dtype": "float32"},
        {"name": "conv2.att_dst", "shape": [1, num_heads_2, num_classes], "dtype": "float32"},
        {"name": "conv2.lin.weight", "shape": [num_classes, hidden], "dtype": "float32"},
    ]}
    with open(os.path.join(gat_dir, "meta.json"), "w") as f:
        json.dump(gat_meta, f)

    print(f"  {name}: {num_nodes} nodes, {len(src)} edges (avg_degree={avg_degree}), "
          f"{num_features} features, {num_classes} classes")


def main():
    parser = argparse.ArgumentParser(
        description="Generate fake GraphSage datasets for ablation study")
    parser.add_argument("--nodes", type=int, nargs="+", default=list(range(12, 19)),
                        help="log2 of node counts (default: 12 13 14 15 16 17 18)")
    parser.add_argument("--degrees", type=int, nargs="+", default=[10],
                        help="Average degrees to generate (default: 10)")
    parser.add_argument("--out_dir", default="weights",
                        help="Output directory (default: weights)")
    args = parser.parse_args()

    print("Generating fake datasets for ablation study...")
    for log_n in args.nodes:
        num_nodes = 1 << log_n
        for deg in args.degrees:
            gen_dataset(num_nodes, avg_degree=deg, out_dir=args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
