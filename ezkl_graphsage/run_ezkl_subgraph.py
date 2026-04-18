#!/usr/bin/env python3
"""
Run EZKL on subgraph-extracted GraphSAGE for Elliptic and DGraphFin.
Exports each subgraph as a fresh ONNX model (dense adjacency baked in),
then runs the full EZKL pipeline: gen_settings -> calibrate -> compile ->
get_srs -> setup -> gen_witness -> prove -> verify.

Usage:
  python run_ezkl_subgraph.py --dataset elliptic --num_subgraphs 20
  python run_ezkl_subgraph.py --dataset dgraphfin --num_subgraphs 20
"""

import argparse
import asyncio
import csv
import json
import os
import shutil
import struct
import time

import ezkl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DATA_DIR = "/scratch/bjchen4_icgpu/zkgnn/pyg/weights/raw"
WORK_DIR = "/taiga/illinois/eng/cs/ddkang/bjchen4/ezkl_subgraph_work"


def read_f32_bin(path):
    data = open(path, "rb").read()
    n = len(data) // 4
    return np.array(struct.unpack(f"{n}f", data), dtype=np.float32)


def read_i32_bin(path):
    data = open(path, "rb").read()
    n = len(data) // 4
    return np.array(struct.unpack(f"{n}i", data), dtype=np.int32)


class GraphSAGESubgraph(nn.Module):
    """GraphSAGE with dense adjacency baked in."""

    def __init__(self, A, W_neighbor1, W_self1, b1, W_neighbor2, W_self2, b2):
        super().__init__()
        self.register_buffer("A", torch.tensor(A))
        self.W_neighbor1 = nn.Parameter(torch.tensor(W_neighbor1))
        self.W_self1 = nn.Parameter(torch.tensor(W_self1))
        self.b1 = nn.Parameter(torch.tensor(b1))
        self.W_neighbor2 = nn.Parameter(torch.tensor(W_neighbor2))
        self.W_self2 = nn.Parameter(torch.tensor(W_self2))
        self.b2 = nn.Parameter(torch.tensor(b2))

    def forward(self, x):
        y = torch.matmul(x, self.W_neighbor1.t())
        z_neighbor = torch.matmul(self.A, y)
        z_self = torch.matmul(x, self.W_self1.t())
        h = F.relu(z_self + z_neighbor + self.b1)

        y = torch.matmul(h, self.W_neighbor2.t())
        z_neighbor = torch.matmul(self.A, y)
        z_self = torch.matmul(h, self.W_self2.t())
        out = z_self + z_neighbor + self.b2
        return out


def load_weights(dataset_name):
    """Load GraphSAGE weights for a dataset."""
    weight_dir = os.path.join(DATA_DIR, f"graphsage_{dataset_name}")
    meta = json.load(open(os.path.join(weight_dir, "meta.json")))

    hidden = meta["hidden"]
    num_features = meta["num_features"]
    num_classes = meta["num_classes"]

    W_neighbor1 = read_f32_bin(os.path.join(weight_dir, "conv1.lin_l.weight.bin")).reshape(hidden, num_features)
    W_self1 = read_f32_bin(os.path.join(weight_dir, "conv1.lin_r.weight.bin")).reshape(hidden, num_features)
    b1 = read_f32_bin(os.path.join(weight_dir, "conv1.lin_l.bias.bin"))
    W_neighbor2 = read_f32_bin(os.path.join(weight_dir, "conv2.lin_l.weight.bin")).reshape(num_classes, hidden)
    W_self2 = read_f32_bin(os.path.join(weight_dir, "conv2.lin_r.weight.bin")).reshape(num_classes, hidden)
    b2 = read_f32_bin(os.path.join(weight_dir, "conv2.lin_l.bias.bin"))

    return {
        "W_neighbor1": W_neighbor1, "W_self1": W_self1, "b1": b1,
        "W_neighbor2": W_neighbor2, "W_self2": W_self2, "b2": b2,
        "num_features": num_features, "num_classes": num_classes, "hidden": hidden,
    }


def load_subgraph(dataset_name, sub_id):
    """Load a subgraph's data."""
    sub_dir = os.path.join(DATA_DIR, f"{dataset_name}_sub_{sub_id}")
    meta = json.load(open(os.path.join(sub_dir, "meta.json")))

    x = read_f32_bin(os.path.join(sub_dir, "x.bin")).reshape(meta["num_nodes"], meta["num_features"])
    edge_src = read_i32_bin(os.path.join(sub_dir, "edge_src.bin"))
    edge_dst = read_i32_bin(os.path.join(sub_dir, "edge_dst.bin"))

    # Build mean-normalized adjacency (matches GraphSAGE convention)
    N = meta["num_nodes"]
    A = np.zeros((N, N), dtype=np.float32)
    in_degree = np.zeros(N, dtype=np.int32)
    for d in edge_dst:
        in_degree[d] += 1
    for s, d in zip(edge_src, edge_dst):
        A[d, s] = 1.0 / max(in_degree[d], 1)

    return x, A, meta


def export_onnx(model, x, work_dir):
    """Export model + input to ONNX and JSON."""
    onnx_path = os.path.join(work_dir, "model.onnx")
    input_path = os.path.join(work_dir, "input.json")

    x_tensor = torch.tensor(x)
    torch.onnx.export(
        model, x_tensor, onnx_path,
        input_names=["input"], output_names=["output"],
        opset_version=13, do_constant_folding=True,
    )

    input_json = {"input_data": [x.flatten().tolist()]}
    with open(input_path, "w") as f:
        json.dump(input_json, f)

    return onnx_path, input_path


async def run_ezkl_pipeline(work_dir):
    """Run full EZKL pipeline. Returns dict of timings + proof size, or None on failure."""
    model_path = os.path.join(work_dir, "model.onnx")
    input_path = os.path.join(work_dir, "input.json")
    settings_path = os.path.join(work_dir, "settings.json")
    compiled_path = os.path.join(work_dir, "model.compiled")
    srs_path = os.path.join(work_dir, "kzg.srs")
    vk_path = os.path.join(work_dir, "vk.key")
    pk_path = os.path.join(work_dir, "pk.key")
    witness_path = os.path.join(work_dir, "witness.json")
    proof_path = os.path.join(work_dir, "proof.json")

    timings = {}

    try:
        # gen_settings
        py_run_args = ezkl.PyRunArgs()
        py_run_args.input_visibility = "public"
        py_run_args.output_visibility = "public"
        py_run_args.param_visibility = "fixed"

        t0 = time.time()
        ezkl.gen_settings(model_path, settings_path, py_run_args=py_run_args)
        timings["gen_settings"] = time.time() - t0

        with open(settings_path) as f:
            settings = json.load(f)
        num_rows = settings["num_rows"]
        logrows = settings["run_args"]["logrows"]

        # calibrate
        t0 = time.time()
        ezkl.calibrate_settings(input_path, model_path, settings_path, "resources")
        timings["calibrate"] = time.time() - t0

        with open(settings_path) as f:
            settings = json.load(f)
        logrows = settings["run_args"]["logrows"]

        # compile
        t0 = time.time()
        ezkl.compile_circuit(model_path, compiled_path, settings_path)
        timings["compile"] = time.time() - t0

        # get_srs (async in ezkl v23)
        t0 = time.time()
        await ezkl.get_srs(settings_path=settings_path, srs_path=srs_path)
        timings["get_srs"] = time.time() - t0

        # setup
        t0 = time.time()
        ezkl.setup(model=compiled_path, vk_path=vk_path, pk_path=pk_path, srs_path=srs_path)
        timings["setup"] = time.time() - t0

        # gen_witness
        t0 = time.time()
        ezkl.gen_witness(data=input_path, model=compiled_path, output=witness_path)
        timings["gen_witness"] = time.time() - t0

        # prove
        t0 = time.time()
        ezkl.prove(witness=witness_path, model=compiled_path, pk_path=pk_path,
                   proof_path=proof_path, srs_path=srs_path)
        timings["prove"] = time.time() - t0

        # verify
        t0 = time.time()
        result = ezkl.verify(proof_path=proof_path, settings_path=settings_path,
                             vk_path=vk_path, srs_path=srs_path)
        timings["verify"] = time.time() - t0

        proof_size = os.path.getsize(proof_path)

        return {
            "verified": result,
            "num_rows": num_rows,
            "logrows": logrows,
            "proof_size_bytes": proof_size,
            **timings,
        }

    except Exception as e:
        print(f"  EZKL pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def find_subgraphs_by_size(dataset_name, target_sizes, tolerance=2):
    """Find subgraph IDs close to each target size."""
    prefix = f"{dataset_name}_sub_"
    all_subs = []
    for entry in os.listdir(DATA_DIR):
        if entry.startswith(prefix):
            sub_id = entry[len(prefix):]
            meta_path = os.path.join(DATA_DIR, entry, "meta.json")
            if os.path.exists(meta_path):
                meta = json.load(open(meta_path))
                all_subs.append((sub_id, meta["num_nodes"], meta["num_edges"]))

    results = []
    used_ids = set()
    for target in target_sizes:
        best = None
        best_diff = float("inf")
        for sub_id, n, e in all_subs:
            if sub_id in used_ids:
                continue
            diff = abs(n - target)
            if diff < best_diff:
                best = (sub_id, n, e)
                best_diff = diff
        if best:
            results.append(best)
            used_ids.add(best[0])

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["elliptic", "dgraphfin"])
    parser.add_argument("--num_subgraphs", type=int, default=20,
                        help="Number of subgraphs to test (picks a range of sizes)")
    parser.add_argument("--sub_ids", type=str, default=None,
                        help="Comma-separated subgraph IDs to run (overrides --num_subgraphs)")
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = f"ezkl_{args.dataset}_results.csv"

    os.makedirs(WORK_DIR, exist_ok=True)

    # Load shared weights
    print(f"Loading GraphSAGE weights for {args.dataset}...")
    weights = load_weights(args.dataset)
    print(f"  Features: {weights['num_features']}, Hidden: {weights['hidden']}, Classes: {weights['num_classes']}")

    # Find subgraphs
    if args.sub_ids:
        # Use explicit subgraph IDs
        subgraphs = []
        for sub_id in args.sub_ids.split(","):
            sub_id = sub_id.strip()
            meta_path = os.path.join(DATA_DIR, f"{args.dataset}_sub_{sub_id}", "meta.json")
            meta = json.load(open(meta_path))
            subgraphs.append((sub_id, meta["num_nodes"], meta["num_edges"]))
        print(f"  Using {len(subgraphs)} specified subgraphs: {[(n, e) for _, n, e in subgraphs]}")
    else:
        # Auto-select from size range
        prefix = f"{args.dataset}_sub_"
        all_sizes = []
        for entry in os.listdir(DATA_DIR):
            if entry.startswith(prefix):
                meta_path = os.path.join(DATA_DIR, entry, "meta.json")
                if os.path.exists(meta_path):
                    meta = json.load(open(meta_path))
                    all_sizes.append(meta["num_nodes"])

        all_sizes.sort()
        print(f"  Found {len(all_sizes)} subgraphs, sizes {min(all_sizes)}-{max(all_sizes)}, median {all_sizes[len(all_sizes)//2]}")

        if args.num_subgraphs <= 1:
            target_sizes = [all_sizes[len(all_sizes) // 2]]
        elif args.num_subgraphs >= len(set(all_sizes)):
            target_sizes = sorted(set(all_sizes))
        else:
            step = (max(all_sizes) - min(all_sizes)) / (args.num_subgraphs - 1)
            target_sizes = [int(min(all_sizes) + i * step) for i in range(args.num_subgraphs)]

        subgraphs = find_subgraphs_by_size(args.dataset, target_sizes)
        print(f"  Selected {len(subgraphs)} subgraphs: {[n for _, n, _ in subgraphs]}")

    # Run EZKL on each subgraph
    results = []
    for idx, (sub_id, num_nodes, num_edges) in enumerate(subgraphs):
        print(f"\n[{idx+1}/{len(subgraphs)}] Subgraph {sub_id}: {num_nodes} nodes, {num_edges} edges")

        # Load subgraph data
        x, A, meta = load_subgraph(args.dataset, sub_id)

        # Build model with this subgraph's adjacency
        model = GraphSAGESubgraph(
            A, weights["W_neighbor1"], weights["W_self1"], weights["b1"],
            weights["W_neighbor2"], weights["W_self2"], weights["b2"],
        )
        model.eval()

        # Export
        sub_work = os.path.join(WORK_DIR, f"sub_{sub_id}")
        os.makedirs(sub_work, exist_ok=True)
        export_onnx(model, x, sub_work)

        # Run pipeline
        result = asyncio.run(run_ezkl_pipeline(sub_work))

        if result:
            row = {
                "dataset": args.dataset,
                "sub_id": sub_id,
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "num_rows": result["num_rows"],
                "logrows": result["logrows"],
                "prove_s": f"{result['prove']:.2f}",
                "verify_s": f"{result['verify']:.4f}",
                "proof_size_kb": f"{result['proof_size_bytes'] / 1024:.2f}",
                "setup_s": f"{result['setup']:.2f}",
                "gen_settings_s": f"{result['gen_settings']:.2f}",
                "calibrate_s": f"{result['calibrate']:.2f}",
                "verified": result["verified"],
            }
            results.append(row)
            print(f"  Prove: {result['prove']:.2f}s, Verify: {result['verify']:.4f}s, "
                  f"Proof: {result['proof_size_bytes']/1024:.2f} KB, "
                  f"Rows: {result['num_rows']}, LogRows: {result['logrows']}")
        else:
            results.append({
                "dataset": args.dataset, "sub_id": sub_id,
                "num_nodes": num_nodes, "num_edges": num_edges,
                "error": "pipeline_failed",
            })

        # Clean up intermediate files to save disk
        shutil.rmtree(sub_work, ignore_errors=True)

    # Write CSV
    csv_path = os.path.join("/scratch/bjchen4_icgpu/zkgnn/ezkl_graphsage", args.output_csv)
    if results:
        fieldnames = list(results[0].keys())
        # Union all keys
        for r in results:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {csv_path}")

    # Summary
    print("\n" + "=" * 70)
    print(f"  EZKL GraphSAGE Subgraph Results — {args.dataset}")
    print("=" * 70)
    print(f"{'Nodes':>6} {'Edges':>6} {'Rows':>10} {'LR':>3} {'Prove(s)':>9} {'Verify(s)':>10} {'Proof(KB)':>10}")
    for r in results:
        if "error" in r:
            print(f"{r['num_nodes']:>6} {r['num_edges']:>6} {'FAILED':>10}")
        else:
            print(f"{r['num_nodes']:>6} {r['num_edges']:>6} {r['num_rows']:>10} {r['logrows']:>3} "
                  f"{r['prove_s']:>9} {r['verify_s']:>10} {r['proof_size_kb']:>10}")
    print("=" * 70)


if __name__ == "__main__":
    main()
