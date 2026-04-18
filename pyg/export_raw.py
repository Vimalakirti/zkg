#!/usr/bin/env python3
"""Export npz weight files to raw binary format for Rust consumption."""

import json
import os
import numpy as np

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
RAW_DIR = os.path.join(WEIGHTS_DIR, "raw")

DATASETS = ["cora", "citeseer", "pubmed"]
MODELS = ["gcn", "graphsage", "gat"]


def export_dataset(name):
    """Export a dataset npz to raw binary files."""
    src = os.path.join(WEIGHTS_DIR, f"{name}.npz")
    dst = os.path.join(RAW_DIR, name)
    os.makedirs(dst, exist_ok=True)

    d = np.load(src)

    # x: float32 row-major
    x = d["x"].astype(np.float32)
    x.tofile(os.path.join(dst, "x.bin"))

    # y: int32
    y = d["y"].astype(np.int32)
    y.tofile(os.path.join(dst, "y.bin"))

    # edge_index: split into src/dst, int32
    edge_index = d["edge_index"].astype(np.int32)
    edge_index[0].tofile(os.path.join(dst, "edge_src.bin"))
    edge_index[1].tofile(os.path.join(dst, "edge_dst.bin"))

    # masks: uint8
    for mask_name in ["train_mask", "val_mask", "test_mask"]:
        mask = d[mask_name].astype(np.uint8)
        mask.tofile(os.path.join(dst, f"{mask_name}.bin"))

    num_classes = int(d["num_classes"])
    num_nodes, num_features = x.shape
    num_edges = edge_index.shape[1]

    meta = {
        "num_nodes": num_nodes,
        "num_features": num_features,
        "num_edges": num_edges,
        "num_classes": num_classes,
    }
    with open(os.path.join(dst, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  {name}: nodes={num_nodes}, features={num_features}, edges={num_edges}, classes={num_classes}")


def export_model(model, dataset):
    """Export a model weight npz to raw binary files."""
    tag = f"{model}_{dataset}"
    src = os.path.join(WEIGHTS_DIR, f"{tag}.npz")
    dst = os.path.join(RAW_DIR, tag)
    os.makedirs(dst, exist_ok=True)

    d = np.load(src)

    weights_meta = []
    for key in d.files:
        arr = d[key].astype(np.float32)
        # Use dots-to-slashes for subdirectory or just replace dots with dots in filename
        fname = key.replace(".", "/") + ".bin"  # e.g. conv1/lin/weight.bin
        # Actually, keep it flat with dots: conv1.lin.weight.bin
        fname = key + ".bin"
        arr.tofile(os.path.join(dst, fname))
        weights_meta.append({"name": key, "shape": list(arr.shape), "dtype": "float32"})
        print(f"    {fname}: shape={list(arr.shape)}")

    meta = {"weights": weights_meta}
    with open(os.path.join(dst, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main():
    print("Exporting datasets...")
    for ds in DATASETS:
        export_dataset(ds)

    print("\nExporting model weights...")
    for model in MODELS:
        for ds in DATASETS:
            tag = f"{model}_{ds}"
            print(f"  {tag}:")
            export_model(model, ds)

    print("\nDone.")


if __name__ == "__main__":
    main()
