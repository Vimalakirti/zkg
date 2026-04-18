"""
Export GraphSAGE on Cora as ONNX for ezkl proving.
Architecture matches zkGNN exactly:
  Layer 1: h = ReLU(X @ W_self1 + A @ (X @ W_neighbor1) + b1)   -> (N, 16)
  Layer 2: out = H @ W_self2 + A @ (H @ W_neighbor2) + b2       -> (N, 7)
Adjacency A is baked in as a model parameter (dense weight matrix).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import struct
import os

DATA_DIR = "/scratch/bjchen4_icgpu/zkgnn/pyg/weights/raw"
OUT_DIR = "/scratch/bjchen4_icgpu/zkgnn/ezkl_graphsage"


def read_f32_bin(path):
    data = open(path, "rb").read()
    n = len(data) // 4
    return np.array(struct.unpack(f"{n}f", data), dtype=np.float32)


def read_i32_bin(path):
    data = open(path, "rb").read()
    n = len(data) // 4
    return np.array(struct.unpack(f"{n}i", data), dtype=np.int32)


# --- Load Cora dataset ---
meta = json.load(open(f"{DATA_DIR}/cora/meta.json"))
num_nodes = meta["num_nodes"]   # 2708
num_features = meta["num_features"]  # 1433
num_classes = meta["num_classes"]  # 7
num_edges = meta["num_edges"]   # 10556

print(f"Cora: {num_nodes} nodes, {num_features} features, {num_classes} classes, {num_edges} edges")

# Load node features
x_flat = read_f32_bin(f"{DATA_DIR}/cora/x.bin")
X = x_flat.reshape(num_nodes, num_features)
print(f"X shape: {X.shape}")

# Load edges
edge_src = read_i32_bin(f"{DATA_DIR}/cora/edge_src.bin")
edge_dst = read_i32_bin(f"{DATA_DIR}/cora/edge_dst.bin")
print(f"Edges: {len(edge_src)} directed edges")

# Build mean-normalized adjacency (matching zkGNN graphsage.rs)
# A[dst, src] = 1 / in_degree[dst]  (no self-loops for GraphSAGE)
A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
in_degree = np.zeros(num_nodes, dtype=np.int32)
for s, d in zip(edge_src, edge_dst):
    in_degree[d] += 1

for s, d in zip(edge_src, edge_dst):
    A[d, s] = 1.0 / max(in_degree[d], 1)

print(f"A shape: {A.shape}, nnz: {np.count_nonzero(A)}")

# --- Load trained weights ---
wmeta = json.load(open(f"{DATA_DIR}/graphsage_cora/meta.json"))
print("Weight shapes:", {w["name"]: w["shape"] for w in wmeta["weights"]})

W_neighbor1 = read_f32_bin(f"{DATA_DIR}/graphsage_cora/conv1.lin_l.weight.bin").reshape(16, 1433)
W_self1 = read_f32_bin(f"{DATA_DIR}/graphsage_cora/conv1.lin_r.weight.bin").reshape(16, 1433)
b1 = read_f32_bin(f"{DATA_DIR}/graphsage_cora/conv1.lin_l.bias.bin")
W_neighbor2 = read_f32_bin(f"{DATA_DIR}/graphsage_cora/conv2.lin_l.weight.bin").reshape(7, 16)
W_self2 = read_f32_bin(f"{DATA_DIR}/graphsage_cora/conv2.lin_r.weight.bin").reshape(7, 16)
b2 = read_f32_bin(f"{DATA_DIR}/graphsage_cora/conv2.lin_l.bias.bin")

print(f"W_neighbor1: {W_neighbor1.shape}, W_self1: {W_self1.shape}, b1: {b1.shape}")
print(f"W_neighbor2: {W_neighbor2.shape}, W_self2: {W_self2.shape}, b2: {b2.shape}")


# --- Define PyTorch model ---
class GraphSAGECora(nn.Module):
    """GraphSAGE with adjacency baked in as a parameter."""

    def __init__(self, A, W_neighbor1, W_self1, b1, W_neighbor2, W_self2, b2):
        super().__init__()
        # Register adjacency as buffer (constant, not trained)
        self.register_buffer("A", torch.tensor(A))
        # Layer 1 weights (PyTorch convention: (out_features, in_features))
        self.W_neighbor1 = nn.Parameter(torch.tensor(W_neighbor1))
        self.W_self1 = nn.Parameter(torch.tensor(W_self1))
        self.b1 = nn.Parameter(torch.tensor(b1))
        # Layer 2
        self.W_neighbor2 = nn.Parameter(torch.tensor(W_neighbor2))
        self.W_self2 = nn.Parameter(torch.tensor(W_self2))
        self.b2 = nn.Parameter(torch.tensor(b2))

    def forward(self, x):
        # Layer 1: mean aggregation + self transform + ReLU
        y = torch.matmul(x, self.W_neighbor1.t())        # (N, 16)
        z_neighbor = torch.matmul(self.A, y)              # (N, 16)
        z_self = torch.matmul(x, self.W_self1.t())        # (N, 16)
        h = F.relu(z_self + z_neighbor + self.b1)         # (N, 16)

        # Layer 2: mean aggregation + self transform (no ReLU)
        y = torch.matmul(h, self.W_neighbor2.t())         # (N, 7)
        z_neighbor = torch.matmul(self.A, y)              # (N, 7)
        z_self = torch.matmul(h, self.W_self2.t())        # (N, 7)
        out = z_self + z_neighbor + self.b2               # (N, 7)
        return out


model = GraphSAGECora(A, W_neighbor1, W_self1, b1, W_neighbor2, W_self2, b2)
model.eval()

# --- Verify output ---
x_tensor = torch.tensor(X)
with torch.no_grad():
    out = model(x_tensor)
    preds = out.argmax(dim=1).numpy()

labels = read_i32_bin(f"{DATA_DIR}/cora/y.bin")
test_mask_raw = open(f"{DATA_DIR}/cora/test_mask.bin", "rb").read()
test_mask = np.array([b for b in test_mask_raw], dtype=np.bool_)
acc = (preds[test_mask] == labels[test_mask]).mean()
print(f"Float32 accuracy on test set: {acc:.4f}")

# --- Export to ONNX ---
onnx_path = os.path.join(OUT_DIR, "graphsage_cora.onnx")
dummy_input = x_tensor
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
    do_constant_folding=True,
)
print(f"ONNX model saved to {onnx_path}")

# --- Save input data for ezkl ---
input_json = {"input_data": [X.flatten().tolist()]}
input_path = os.path.join(OUT_DIR, "input.json")
with open(input_path, "w") as f:
    json.dump(input_json, f)
print(f"Input data saved to {input_path} ({X.size} values)")
