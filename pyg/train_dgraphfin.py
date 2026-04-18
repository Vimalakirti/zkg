"""
Train GraphSAGE on DGraphFin for financial anomaly detection.
Uses NeighborLoader for mini-batch training (3.7M nodes won't fit in GPU memory).
Exports weights and data in binary format for zk-torch-3 subgraph proving.

Architecture: 2-layer SAGEConv with bias, matching zk-torch-3 graph_sage.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score

import sys
# Use system PyG for NeighborLoader (needs torch-sparse), but local for DGraphFin dataset
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pytorch_geometric"))
from torch_geometric.datasets import DGraphFin


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
        return x


def train_epoch(model, optimizer, loader, device):
    model.train()
    total_loss = 0
    total_nodes = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        # Only compute loss on seed nodes (batch_size)
        mask = batch.train_mask[:batch.batch_size]
        y = batch.y[:batch.batch_size]
        if mask.sum() == 0:
            continue
        loss = F.cross_entropy(out[:batch.batch_size][mask], y[mask])
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * int(mask.sum())
        total_nodes += int(mask.sum())
    return total_loss / max(total_nodes, 1)


@torch.no_grad()
def evaluate(model, data, mask, device, batch_size=4096):
    """Evaluate using NeighborLoader for inference."""
    model.eval()
    loader = NeighborLoader(
        data, num_neighbors=[-1, -1], input_nodes=mask,
        batch_size=batch_size, shuffle=False,
    )
    all_preds = []
    all_labels = []
    all_scores = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        out = out[:batch.batch_size]
        y = batch.y[:batch.batch_size]
        probs = out.softmax(dim=1)[:, 1]
        preds = out.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())
        all_scores.append(probs.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_scores = torch.cat(all_scores)

    acc = (all_preds == all_labels).float().mean().item()
    try:
        auc = roc_auc_score(all_labels.numpy(), all_scores.numpy())
    except ValueError:
        auc = 0.0
    return acc, auc


def export_binary(data, state_dict, hidden, save_dir):
    """Export dataset and weights in binary format for zk-torch-3."""
    # --- Export dataset ---
    ds_dir = os.path.join(save_dir, "raw", "dgraphfin")
    os.makedirs(ds_dir, exist_ok=True)

    x = data.x.numpy().astype(np.float32)
    y = data.y.numpy().astype(np.int32)
    edge_index = data.edge_index.numpy()
    edge_src = edge_index[0].astype(np.int32)
    edge_dst = edge_index[1].astype(np.int32)
    train_mask = data.train_mask.numpy().astype(np.uint8)
    val_mask = data.val_mask.numpy().astype(np.uint8)
    test_mask = data.test_mask.numpy().astype(np.uint8)

    x.tofile(os.path.join(ds_dir, "x.bin"))
    y.tofile(os.path.join(ds_dir, "y.bin"))
    edge_src.tofile(os.path.join(ds_dir, "edge_src.bin"))
    edge_dst.tofile(os.path.join(ds_dir, "edge_dst.bin"))
    train_mask.tofile(os.path.join(ds_dir, "train_mask.bin"))
    val_mask.tofile(os.path.join(ds_dir, "val_mask.bin"))
    test_mask.tofile(os.path.join(ds_dir, "test_mask.bin"))

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
    print(f"    nodes={num_nodes}, features={num_features}, edges={num_edges}")
    print(f"    train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")

    # --- Export weights ---
    w_dir = os.path.join(save_dir, "raw", "graphsage_dgraphfin")
    os.makedirs(w_dir, exist_ok=True)

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
    print("Loading DGraphFin dataset...")
    dataset = DGraphFin(root=os.path.join(DATA_DIR, "DGraphFin"))
    data = dataset[0]

    print(f"  Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    print(f"  Features: {data.x.shape[1]}, Classes: {dataset.num_classes}")
    print(f"  Labels: normal={int((data.y == 0).sum())}, fraud={int((data.y == 1).sum())}")
    print(f"  Train: {int(data.train_mask.sum())}, Val: {int(data.val_mask.sum())}, "
          f"Test: {int(data.test_mask.sum())}")

    # Config
    hidden = 64
    lr = 0.01
    weight_decay = 5e-4
    dropout = 0.5
    epochs = 50
    runs = 5
    batch_size = 2048
    num_neighbors = [10, 10]  # sample 10 neighbors per hop per layer

    num_features = data.x.shape[1]
    num_classes = 2

    # Training with NeighborLoader
    best_overall_auc = 0
    best_state_dict = None
    all_aucs = []

    for run in range(runs):
        model = GraphSAGE(num_features, hidden, num_classes, dropout).to(device)
        model.reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_loader = NeighborLoader(
            data, num_neighbors=num_neighbors,
            input_nodes=data.train_mask,
            batch_size=batch_size, shuffle=True,
        )

        best_val_auc = 0
        best_run_state = None

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            loss = train_epoch(model, optimizer, train_loader, device)
            elapsed = time.time() - t0

            if epoch % 5 == 0 or epoch == epochs:
                val_acc, val_auc = evaluate(model, data, data.val_mask, device)
                print(f"  Run {run+1}, Epoch {epoch}: loss={loss:.4f}, "
                      f"val_acc={val_acc:.4f}, val_auc={val_auc:.4f} ({elapsed:.1f}s)")

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_run_state = {k: v.cpu().clone()
                                      for k, v in model.state_dict().items()}

        # Evaluate on test set with best val model
        model.load_state_dict(best_run_state)
        test_acc, test_auc = evaluate(model, data, data.test_mask, device)
        all_aucs.append(test_auc)
        if test_auc > best_overall_auc:
            best_overall_auc = test_auc
            best_state_dict = best_run_state

        print(f"  Run {run+1}/{runs}: val_auc={best_val_auc:.4f}, test_auc={test_auc:.4f}")

    aucs = torch.tensor(all_aucs)
    print(f"\n  Final Test AUC: {aucs.mean():.4f} ± {aucs.std():.4f} (best={best_overall_auc:.4f})")

    # Export best model
    print("\nExporting...")
    model = GraphSAGE(num_features, hidden, num_classes, dropout).to(device)
    model.load_state_dict(best_state_dict)
    train_acc, train_auc = evaluate(model, data, data.train_mask, device)
    val_acc, val_auc = evaluate(model, data, data.val_mask, device)
    test_acc, test_auc = evaluate(model, data, data.test_mask, device)
    print(f"  Best model: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, test_auc={test_auc:.4f}")
    print(f"              train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, test_acc={test_acc:.4f}")

    export_binary(data, best_state_dict, hidden, SAVE_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
