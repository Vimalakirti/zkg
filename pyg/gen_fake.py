"""Generate synthetic datasets for the ablation and ZK-overhead studies.

Usage:
    # Generate a grid of (node capacity, average degree) combinations:
    python gen_fake.py --nodes 12 13 14 15 --degrees 5 10 20 50

    # Regenerate only the GAT-specific datasets:
    python gen_fake.py --gat-only

    # Defaults: capacities 2^12..2^18, degree 10
    python gen_fake.py

The GCN/GraphSAGE dataset fake_{N}_d{degree} contains N real nodes. GAT uses
fake_{N}_d{degree}_gat, which contains N-1 real nodes and reserves the final
slot of its N-node padded space for padded edges. This keeps the advertised
power-of-two experiment size without doubling the GAT proof domain.
"""

import argparse
import json
import os

import numpy as np


def gen_dataset(num_nodes, avg_degree=10, num_features=16, hidden=16,
                num_classes=7, out_dir="weights", name=None,
                models=("graphsage", "gcn"), node_capacity=None):
    if name is None:
        name = f"fake_{num_nodes}_d{avg_degree}"

    ds_dir = os.path.join(out_dir, "raw", name)
    os.makedirs(ds_dir, exist_ok=True)

    rng = np.random.default_rng(42)

    # --- Node features: (num_nodes, num_features), float32 ---
    x = rng.standard_normal((num_nodes, num_features)).astype(np.float32) * 0.1
    x.tofile(os.path.join(ds_dir, "x.bin"))

    # --- Edges: random sparse graph, approximately avg_degree per node ---
    num_edges = num_nodes * avg_degree
    src = rng.integers(0, num_nodes, size=num_edges).astype(np.int32)
    dst = rng.integers(0, num_nodes, size=num_edges).astype(np.int32)
    mask = src != dst
    src, dst = src[mask], dst[mask]
    src.tofile(os.path.join(ds_dir, "edge_src.bin"))
    dst.tofile(os.path.join(ds_dir, "edge_dst.bin"))

    # --- Labels and masks ---
    y = rng.integers(0, num_classes, size=num_nodes).astype(np.int32)
    y.tofile(os.path.join(ds_dir, "y.bin"))

    train_mask = np.zeros(num_nodes, dtype=np.uint8)
    val_mask = np.zeros(num_nodes, dtype=np.uint8)
    test_mask = np.zeros(num_nodes, dtype=np.uint8)
    train_mask[:num_nodes // 2] = 1
    val_mask[num_nodes // 2:num_nodes * 3 // 4] = 1
    test_mask[num_nodes * 3 // 4:] = 1
    train_mask.tofile(os.path.join(ds_dir, "train_mask.bin"))
    val_mask.tofile(os.path.join(ds_dir, "val_mask.bin"))
    test_mask.tofile(os.path.join(ds_dir, "test_mask.bin"))

    meta = {
        "num_nodes": num_nodes,
        "num_features": num_features,
        "num_classes": num_classes,
        "num_edges": int(len(src)),
        "avg_degree": avg_degree,
    }
    if node_capacity is not None:
        meta["node_capacity"] = node_capacity
    with open(os.path.join(ds_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    scale = 0.1

    if "graphsage" in models:
        # Two GraphSAGE layers, each with neighbor/self weights and a bias.
        w_dir = os.path.join(out_dir, "raw", f"graphsage_{name}")
        os.makedirs(w_dir, exist_ok=True)
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

        w_meta = {
            "hidden": hidden,
            "num_features": num_features,
            "num_classes": num_classes,
        }
        with open(os.path.join(w_dir, "meta.json"), "w") as f:
            json.dump(w_meta, f)

    if "gcn" in models:
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

    if "gat" in models:
        # Match src/bin/gat.rs: four 8-dimensional heads, then one output head.
        num_heads_1 = 4
        head_dim_1 = 8
        gat_hidden = num_heads_1 * head_dim_1
        num_heads_2 = 1
        gat_dir = os.path.join(out_dir, "raw", f"gat_{name}")
        os.makedirs(gat_dir, exist_ok=True)

        gat_w1 = (rng.standard_normal((gat_hidden, num_features)) * scale).astype(np.float32)
        gat_att1_src = (rng.standard_normal((1, num_heads_1, head_dim_1)) * scale).astype(np.float32)
        gat_att1_dst = (rng.standard_normal((1, num_heads_1, head_dim_1)) * scale).astype(np.float32)
        gat_w2 = (rng.standard_normal((num_classes, gat_hidden)) * scale).astype(np.float32)
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
            {"name": "conv1.lin.weight", "shape": [gat_hidden, num_features], "dtype": "float32"},
            {"name": "conv2.att_src", "shape": [1, num_heads_2, num_classes], "dtype": "float32"},
            {"name": "conv2.att_dst", "shape": [1, num_heads_2, num_classes], "dtype": "float32"},
            {"name": "conv2.lin.weight", "shape": [num_classes, gat_hidden], "dtype": "float32"},
        ]}
        with open(os.path.join(gat_dir, "meta.json"), "w") as f:
            json.dump(gat_meta, f)

    capacity_text = ""
    if node_capacity is not None:
        capacity_text = f", padded capacity={node_capacity}"
    print(f"  {name}: {num_nodes} real nodes, {len(src)} edges "
          f"(avg_degree={avg_degree}{capacity_text}), {num_features} features, "
          f"{num_classes} classes")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic GNN datasets for paper experiments")
    parser.add_argument("--nodes", type=int, nargs="+", default=list(range(12, 19)),
                        help="log2 of node capacities (default: 12 13 14 15 16 17 18)")
    parser.add_argument("--degrees", type=int, nargs="+", default=[10],
                        help="Average degrees to generate (default: 10)")
    parser.add_argument("--out_dir", default="weights",
                        help="Output directory (default: weights)")
    parser.add_argument("--gat-only", action="store_true",
                        help="Generate only GAT-specific datasets and weights")
    args = parser.parse_args()

    print("Generating synthetic datasets for paper experiments...")
    for log_n in args.nodes:
        node_capacity = 1 << log_n
        for deg in args.degrees:
            if not args.gat_only:
                gen_dataset(node_capacity, avg_degree=deg, out_dir=args.out_dir)

            gat_name = f"fake_{node_capacity}_d{deg}_gat"
            gen_dataset(
                node_capacity - 1,
                avg_degree=deg,
                out_dir=args.out_dir,
                name=gat_name,
                models=("gat",),
                node_capacity=node_capacity,
            )
    print("Done.")


if __name__ == "__main__":
    main()
