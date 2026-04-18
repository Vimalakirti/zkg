#!/usr/bin/env python3
"""Export kernel benchmark weights and graphs to raw binary format for Rust.

For GIN: folds BatchNorm into adjacent Linear layers so the ZK circuit
only sees Linear → ReLU → Linear → ReLU (no BN).

Output directory structure:
  kernel_raw/<dataset>/graph_<i>/
    meta.json, x.bin, edge_src.bin, edge_dst.bin, y.bin (scalar label)
  kernel_raw/<model>_<dataset>/
    meta.json + weight .bin files
"""

import json
import os
import sys
import numpy as np
import torch

KERNEL_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel_weights")
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel_raw")

DATASETS = ["mutag", "proteins", "imdb-binary", "reddit-binary"]
MODELS = ["gcn", "graphsage", "gin", "gat"]


def load_npy(path):
    return np.load(path)


def save_bin(arr, path):
    arr.tofile(path)


# ---------------------------------------------------------------------------
# Export dataset graphs
# ---------------------------------------------------------------------------

def export_graphs(dataset_name, graph_indices=None):
    """Export selected graphs from a dataset."""
    ds_dir = os.path.join(KERNEL_WEIGHTS, dataset_name)
    meta_path = os.path.join(ds_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    num_graphs = meta["num_graphs"]
    if graph_indices is None:
        # Export first 5 graphs + a few larger ones
        graph_indices = list(range(min(5, num_graphs)))

    out_ds = os.path.join(RAW_DIR, dataset_name)
    os.makedirs(out_ds, exist_ok=True)

    # Save dataset-level meta
    ds_meta = {
        "num_graphs": num_graphs,
        "num_features": meta["num_features"],
        "num_classes": meta["num_classes"],
        "exported_graphs": graph_indices,
        "graph_sizes": [meta["graph_sizes"][i] for i in graph_indices],
    }
    with open(os.path.join(out_ds, "meta.json"), "w") as f:
        json.dump(ds_meta, f, indent=2)

    # Save graph labels for all graphs
    labels = load_npy(os.path.join(ds_dir, "graph_labels.npy"))

    for gi in graph_indices:
        g_src = os.path.join(ds_dir, f"graph_{gi}")
        g_dst = os.path.join(out_ds, f"graph_{gi}")
        os.makedirs(g_dst, exist_ok=True)

        x = load_npy(os.path.join(g_src, "x.npy")).astype(np.float32)
        edge_src = load_npy(os.path.join(g_src, "edge_src.npy")).astype(np.int32)
        edge_dst = load_npy(os.path.join(g_src, "edge_dst.npy")).astype(np.int32)

        save_bin(x, os.path.join(g_dst, "x.bin"))
        save_bin(edge_src, os.path.join(g_dst, "edge_src.bin"))
        save_bin(edge_dst, os.path.join(g_dst, "edge_dst.bin"))

        g_meta = {
            "num_nodes": int(x.shape[0]),
            "num_features": int(x.shape[1]),
            "num_edges": int(len(edge_src)),
            "label": int(labels[gi]),
        }
        with open(os.path.join(g_dst, "meta.json"), "w") as f:
            json.dump(g_meta, f, indent=2)

    print(f"  {dataset_name}: exported {len(graph_indices)} graphs")


# ---------------------------------------------------------------------------
# GCN / GraphSAGE export (straightforward)
# ---------------------------------------------------------------------------

def export_gcn(dataset_name):
    """Export GCN weights (conv weights, biases, lin1, lin2)."""
    tag = f"gcn_{dataset_name}"
    src_dir = os.path.join(KERNEL_WEIGHTS, tag)
    dst_dir = os.path.join(RAW_DIR, tag)
    os.makedirs(dst_dir, exist_ok=True)

    # Load meta to get shapes
    with open(os.path.join(src_dir, "meta.json")) as f:
        meta = json.load(f)

    weights_info = []
    for key, info in meta.items():
        fname = info["file"]
        arr = load_npy(os.path.join(src_dir, fname)).astype(np.float32)
        out_fname = key + ".bin"
        save_bin(arr, os.path.join(dst_dir, out_fname))
        weights_info.append({"name": key, "shape": list(arr.shape)})

    with open(os.path.join(dst_dir, "meta.json"), "w") as f:
        json.dump({"weights": weights_info}, f, indent=2)

    print(f"  {tag}: exported {len(weights_info)} weight tensors")


def export_graphsage(dataset_name):
    """Export GraphSAGE weights."""
    tag = f"graphsage_{dataset_name}"
    src_dir = os.path.join(KERNEL_WEIGHTS, tag)
    dst_dir = os.path.join(RAW_DIR, tag)
    os.makedirs(dst_dir, exist_ok=True)

    with open(os.path.join(src_dir, "meta.json")) as f:
        meta = json.load(f)

    weights_info = []
    for key, info in meta.items():
        fname = info["file"]
        arr = load_npy(os.path.join(src_dir, fname)).astype(np.float32)
        out_fname = key + ".bin"
        save_bin(arr, os.path.join(dst_dir, out_fname))
        weights_info.append({"name": key, "shape": list(arr.shape)})

    with open(os.path.join(dst_dir, "meta.json"), "w") as f:
        json.dump({"weights": weights_info}, f, indent=2)

    print(f"  {tag}: exported {len(weights_info)} weight tensors")


# ---------------------------------------------------------------------------
# GIN export with BatchNorm folding
# ---------------------------------------------------------------------------

def fold_bn_into_linear(W, b, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """Fold a BatchNorm that follows ReLU into the NEXT Linear layer.

    If BN(x) = gamma * (x - mean) / std + beta  (where std = sqrt(var + eps))
    and the next layer is Linear: y = W @ x + b
    then y = W @ BN(x) + b
           = W @ (diag(gamma/std) @ x + (beta - gamma*mean/std)) + b
           = (W @ diag(gamma/std)) @ x + (W @ offset + b)

    Returns (W_new, b_new).
    """
    std = np.sqrt(bn_var + eps)
    scale = bn_weight / std          # (C,)
    offset = bn_bias - bn_weight * bn_mean / std  # (C,)

    # W: (out, in), scale: (in,) → W_new = W * scale[None, :]
    W_new = W * scale[None, :]
    b_new = W @ offset + b
    return W_new.astype(np.float32), b_new.astype(np.float32)


def export_gin(dataset_name):
    """Export GIN weights with BatchNorm folded into Linear layers.

    Original GINConv MLP: Linear1 → ReLU → BN1 → Linear2 → ReLU → BN2
    After folding BN1 into Linear2:  Linear1 → ReLU → Linear2' → ReLU → BN2
    After folding BN2 into next layer's Linear1 (or lin1 for last conv):
      Conv becomes: Linear1 → ReLU → Linear2' → ReLU
      (with modified weights/biases)

    Exported weights per conv layer:
      conv{k}.w1.bin, conv{k}.b1.bin   — MLP first linear (potentially with BN from prev layer folded in)
      conv{k}.w2.bin, conv{k}.b2.bin   — MLP second linear (with BN1 folded in)
    Plus:
      lin1.weight.bin, lin1.bias.bin   — MLP head layer 1 (with last BN2 folded in)
      lin2.weight.bin, lin2.bias.bin   — MLP head layer 2
    """
    tag = f"gin_{dataset_name}"
    src_dir = os.path.join(KERNEL_WEIGHTS, tag)
    dst_dir = os.path.join(RAW_DIR, tag)
    os.makedirs(dst_dir, exist_ok=True)

    # Load all parameters
    def load_param(name):
        fname = name.replace('.', '__') + ".npy"
        return load_npy(os.path.join(src_dir, fname)).astype(np.float64)

    # Identify conv layers: conv1 + convs.0, convs.1, ...
    with open(os.path.join(src_dir, "meta.json")) as f:
        meta = json.load(f)

    # Determine number of conv layers
    conv_names = ["conv1"]
    i = 0
    while f"convs.{i}.nn.0.weight" in meta:
        conv_names.append(f"convs.{i}")
        i += 1
    num_convs = len(conv_names)

    # Load per-conv parameters
    convs = []
    for cname in conv_names:
        prefix = cname + ".nn"
        c = {
            "w1": load_param(f"{prefix}.0.weight"),     # Linear1 weight
            "b1": load_param(f"{prefix}.0.bias"),        # Linear1 bias
            "bn1_w": load_param(f"{prefix}.2.weight"),   # BN1 gamma
            "bn1_b": load_param(f"{prefix}.2.bias"),     # BN1 beta
            "bn1_mean": load_param(f"{prefix}.2.running_mean"),
            "bn1_var": load_param(f"{prefix}.2.running_var"),
            "w2": load_param(f"{prefix}.3.weight"),      # Linear2 weight
            "b2": load_param(f"{prefix}.3.bias"),        # Linear2 bias
            "bn2_w": load_param(f"{prefix}.5.weight"),   # BN2 gamma
            "bn2_b": load_param(f"{prefix}.5.bias"),     # BN2 beta
            "bn2_mean": load_param(f"{prefix}.5.running_mean"),
            "bn2_var": load_param(f"{prefix}.5.running_var"),
        }
        convs.append(c)

    # Load MLP head
    lin1_w = load_param("lin1.weight")
    lin1_b = load_param("lin1.bias")
    lin2_w = load_param("lin2.weight")
    lin2_b = load_param("lin2.bias")

    # Step 1: For each conv, fold BN1 into Linear2
    for c in convs:
        c["w2"], c["b2"] = fold_bn_into_linear(
            c["w2"], c["b2"],
            c["bn1_w"], c["bn1_b"], c["bn1_mean"], c["bn1_var"])

    # Step 2: Fold each conv's BN2 into the NEXT conv's Linear1 (or lin1 for last)
    for i in range(num_convs - 1):
        convs[i + 1]["w1"], convs[i + 1]["b1"] = fold_bn_into_linear(
            convs[i + 1]["w1"], convs[i + 1]["b1"],
            convs[i]["bn2_w"], convs[i]["bn2_b"],
            convs[i]["bn2_mean"], convs[i]["bn2_var"])

    # Fold last conv's BN2 into lin1
    lin1_w, lin1_b = fold_bn_into_linear(
        lin1_w, lin1_b,
        convs[-1]["bn2_w"], convs[-1]["bn2_b"],
        convs[-1]["bn2_mean"], convs[-1]["bn2_var"])

    # Export folded weights
    weights_info = []

    for i, c in enumerate(convs):
        for suffix, key in [("w1", "w1"), ("b1", "b1"), ("w2", "w2"), ("b2", "b2")]:
            fname = f"conv{i}.{suffix}.bin"
            arr = c[key].astype(np.float32)
            save_bin(arr, os.path.join(dst_dir, fname))
            weights_info.append({"name": fname, "shape": list(arr.shape)})

    for name, arr in [("lin1.weight.bin", lin1_w), ("lin1.bias.bin", lin1_b),
                       ("lin2.weight.bin", lin2_w), ("lin2.bias.bin", lin2_b)]:
        arr = arr.astype(np.float32)
        save_bin(arr, os.path.join(dst_dir, name))
        weights_info.append({"name": name, "shape": list(arr.shape)})

    with open(os.path.join(dst_dir, "meta.json"), "w") as f:
        json.dump({
            "weights": weights_info,
            "num_conv_layers": num_convs,
            "note": "BatchNorm folded into Linear layers",
        }, f, indent=2)

    print(f"  {tag}: exported {len(weights_info)} weight tensors "
          f"({num_convs} conv layers, BN folded)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    print("Exporting graphs...")
    for ds in DATASETS:
        # Export all graphs (they're small)
        ds_dir = os.path.join(KERNEL_WEIGHTS, ds)
        with open(os.path.join(ds_dir, "meta.json")) as f:
            meta = json.load(f)
        all_indices = list(range(meta["num_graphs"]))
        export_graphs(ds, all_indices)

    print("\nExporting model weights...")
    for ds in DATASETS:
        export_gcn(ds)
        export_graphsage(ds)
        export_gin(ds)
        # GAT uses same format as GCN (no BN folding needed)
        gat_tag = f"gat_{ds}"
        gat_src = os.path.join(KERNEL_WEIGHTS, gat_tag)
        if os.path.exists(gat_src):
            export_gcn.__code__ and None  # reuse GCN export logic
            tag = gat_tag
            src_dir = gat_src
            dst_dir = os.path.join(RAW_DIR, tag)
            os.makedirs(dst_dir, exist_ok=True)
            with open(os.path.join(src_dir, "meta.json")) as f:
                meta = json.load(f)
            weights_info = []
            for key, info in meta.items():
                fname = info["file"]
                arr = load_npy(os.path.join(src_dir, fname)).astype(np.float32)
                out_fname = key + ".bin"
                save_bin(arr, os.path.join(dst_dir, out_fname))
                weights_info.append({"name": key, "shape": list(arr.shape)})
            with open(os.path.join(dst_dir, "meta.json"), "w") as f:
                json.dump({"weights": weights_info}, f, indent=2)
            print(f"  {tag}: exported {len(weights_info)} weight tensors")

    print("\nDone.")


if __name__ == "__main__":
    main()
