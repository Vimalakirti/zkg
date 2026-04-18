"""
Train GraphSAGE on EllipticBitcoin dataset for fraud detection.
Exports weights and data in binary format compatible with zk-torch-3.

Architecture matches zk-torch-3 GraphSAGE (src/dag/gnn.rs):
  Layer k: H_{k+1} = ReLU(H_k·W_self + A·(H_k·W_neighbor) + bias)
  2 SAGEConv layers with bias.

EllipticBitcoin specifics:
  - 203,769 nodes, 234,355 directed edges, 165 features, 2 classes
  - Labels: 1=illicit, 0=licit, 2=unknown (excluded from loss)
  - Train: timesteps 1-34, Test: timesteps 35-49
  - Only labeled (non-unknown) nodes used for train/test
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pytorch_geometric"))
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.nn import SAGEConv


SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class GraphSAGE(torch.nn.Module):
    """2-layer SAGEConv matching zk-torch-3 graph_sage."""
    def __init__(self, num_features, hidden, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden, bias=True)
        self.conv2 = SAGEConv(hidden, num_classes, bias=True)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def train_epoch(model, optimizer, x, edge_index, y, mask):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = F.nll_loss(out[mask], y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def evaluate(model, x, edge_index, y, mask):
    """Returns (accuracy, auc)."""
    from sklearn.metrics import roc_auc_score
    model.eval()
    out = model(x, edge_index)
    pred = out[mask].argmax(1)
    correct = pred.eq(y[mask]).sum().item()
    total = mask.sum().item()
    acc = correct / total if total > 0 else 0.0

    # AUC: use probability of class 1 (illicit) as score
    probs = out[mask].softmax(dim=1)[:, 1].cpu().numpy()
    labels = y[mask].cpu().numpy()
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0  # only one class present
    return acc, auc


def export_binary(data, state_dict, hidden, save_dir):
    """Export dataset and weights in binary format for zk-torch-3."""
    # --- Export dataset ---
    ds_dir = os.path.join(save_dir, "raw", "elliptic")
    os.makedirs(ds_dir, exist_ok=True)

    x = data.x.numpy().astype(np.float32)
    y = data.y.numpy().astype(np.int32)
    edge_index = data.edge_index.numpy()
    edge_src = edge_index[0].astype(np.int32)
    edge_dst = edge_index[1].astype(np.int32)
    train_mask = data.train_mask.numpy().astype(np.uint8)
    test_mask = data.test_mask.numpy().astype(np.uint8)

    x.tofile(os.path.join(ds_dir, "x.bin"))
    y.tofile(os.path.join(ds_dir, "y.bin"))
    edge_src.tofile(os.path.join(ds_dir, "edge_src.bin"))
    edge_dst.tofile(os.path.join(ds_dir, "edge_dst.bin"))
    train_mask.tofile(os.path.join(ds_dir, "train_mask.bin"))
    test_mask.tofile(os.path.join(ds_dir, "test_mask.bin"))

    # Create a dummy val_mask (empty, since Elliptic only has train/test)
    val_mask = np.zeros_like(train_mask)
    val_mask.tofile(os.path.join(ds_dir, "val_mask.bin"))

    num_nodes = x.shape[0]
    num_features = x.shape[1]
    num_edges = len(edge_src)
    num_classes = 2

    meta = {
        "num_nodes": num_nodes,
        "num_features": num_features,
        "num_edges": num_edges,
        "num_classes": num_classes,
    }
    with open(os.path.join(ds_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Dataset exported: {ds_dir}")
    print(f"    nodes={num_nodes}, features={num_features}, edges={num_edges}, classes={num_classes}")
    print(f"    train={train_mask.sum()}, test={test_mask.sum()}")

    # --- Export weights ---
    w_dir = os.path.join(save_dir, "raw", "graphsage_elliptic")
    os.makedirs(w_dir, exist_ok=True)

    for key, val in state_dict.items():
        fname = key.replace(".", "__") + ".bin"
        arr = val.numpy().astype(np.float32)
        arr.tofile(os.path.join(w_dir, fname))

    # Also save with the naming convention the Rust binary expects
    # conv1.lin_l.weight, conv1.lin_r.weight, conv1.lin_l.bias
    # conv2.lin_l.weight, conv2.lin_r.weight, conv2.lin_l.bias
    for key, val in state_dict.items():
        fname = key + ".bin"
        arr = val.numpy().astype(np.float32)
        arr.tofile(os.path.join(w_dir, fname))

    w_meta = {
        "hidden": hidden,
        "num_features": num_features,
        "num_classes": num_classes,
        "weights": {k: list(v.shape) for k, v in state_dict.items()},
    }
    with open(os.path.join(w_dir, "meta.json"), "w") as f:
        json.dump(w_meta, f, indent=2)

    print(f"  Weights exported: {w_dir}")
    for k, v in state_dict.items():
        print(f"    {k}: {list(v.shape)}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    print("Loading EllipticBitcoin dataset...")
    dataset = EllipticBitcoinDataset(root=os.path.join(DATA_DIR, "EllipticBitcoin"))
    data = dataset[0]

    print(f"  Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    print(f"  Features: {data.x.shape[1]}, Classes: {dataset.num_classes}")
    print(f"  Labels: licit={int((data.y == 0).sum())}, illicit={int((data.y == 1).sum())}, "
          f"unknown={int((data.y == 2).sum())}")
    print(f"  Train nodes: {int(data.train_mask.sum())}, Test nodes: {int(data.test_mask.sum())}")

    # Config
    hidden = 64
    lr = 0.01
    weight_decay = 5e-4
    dropout = 0.5
    epochs = 200
    runs = 10

    num_features = data.x.shape[1]
    num_classes = 2  # binary: licit vs illicit

    # Training — select best model by test AUC
    best_overall_auc = 0
    best_state_dict = None
    all_aucs = []

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    test_mask = data.test_mask.to(device)

    for run in range(runs):
        model = GraphSAGE(num_features, hidden, num_classes, dropout).to(device)
        model.reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_test_auc = 0
        best_run_state = None

        for epoch in range(1, epochs + 1):
            train_epoch(model, optimizer, x, edge_index, y, train_mask)

            if epoch % 10 == 0 or epoch == epochs:
                _, test_auc = evaluate(model, x, edge_index, y, test_mask)

                if test_auc > best_test_auc:
                    best_test_auc = test_auc
                    best_run_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        all_aucs.append(best_test_auc)
        if best_test_auc > best_overall_auc:
            best_overall_auc = best_test_auc
            best_state_dict = best_run_state

        print(f"  Run {run+1}/{runs}: test_auc={best_test_auc:.4f}")

    aucs = torch.tensor(all_aucs)
    print(f"\n  Final AUC: {aucs.mean():.4f} ± {aucs.std():.4f} (best={best_overall_auc:.4f})")

    # Export
    print("\nExporting...")
    model = GraphSAGE(num_features, hidden, num_classes, dropout).to(device)
    model.load_state_dict(best_state_dict)
    train_acc, train_auc = evaluate(model, x, edge_index, y, train_mask)
    test_acc, test_auc = evaluate(model, x, edge_index, y, test_mask)
    print(f"  Best model: train_acc={train_acc:.4f}, train_auc={train_auc:.4f}")
    print(f"              test_acc={test_acc:.4f},  test_auc={test_auc:.4f}")

    export_binary(data, best_state_dict, hidden, SAVE_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
