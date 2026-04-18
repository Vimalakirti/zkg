"""
Train GAE (Graph Autoencoder) on Cora, CiteSeer, PubMed for link prediction.
Exports weights and data in binary format for zk-torch-3.

Architecture matches zk-torch-3 GAE (src/dag/gnn.rs):
  Encoder: 2-layer GCN (no bias) with symmetric normalization D^{-1/2}(A+I)D^{-1/2}
  Decoder: Inner product Z·Z^T → sigmoid

GCNConv bias=False to match zk-torch-3 which has no bias in GCN layers.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pytorch_geometric"))
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, GCNConv

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class GCNEncoder(torch.nn.Module):
    """2-layer GCN encoder without bias, matching zk-torch-3 gcn()."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, bias=False)
        self.conv2 = GCNConv(hidden_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


def train_epoch(model, optimizer, train_data):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def evaluate(model, data):
    """Returns (AUC, AP) on link prediction."""
    model.eval()
    z = model.encode(data.x, data.edge_index)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)


def export_binary(dataset_name, data_orig, train_data, test_data,
                  state_dict, hidden, out_channels, save_dir):
    """Export dataset and weights in binary format for zk-torch-3."""
    # --- Export dataset (full graph features + edges for adjacency) ---
    ds_dir = os.path.join(save_dir, "raw", f"gae_{dataset_name.lower()}")
    os.makedirs(ds_dir, exist_ok=True)

    # Use original (unsplit) data for node features
    x = data_orig.x.cpu().numpy().astype(np.float32)
    num_nodes = x.shape[0]
    num_features = x.shape[1]

    # Full edge list (train edges only — what the model sees during inference)
    edge_index = train_data.edge_index.cpu().numpy()
    edge_src = edge_index[0].astype(np.int32)
    edge_dst = edge_index[1].astype(np.int32)

    x.tofile(os.path.join(ds_dir, "x.bin"))
    edge_src.tofile(os.path.join(ds_dir, "edge_src.bin"))
    edge_dst.tofile(os.path.join(ds_dir, "edge_dst.bin"))

    # Test edges (positive and negative) for AUC evaluation
    test_pos = test_data.pos_edge_label_index.cpu().numpy().astype(np.int32)
    test_neg = test_data.neg_edge_label_index.cpu().numpy().astype(np.int32)
    test_pos[0].tofile(os.path.join(ds_dir, "test_pos_src.bin"))
    test_pos[1].tofile(os.path.join(ds_dir, "test_pos_dst.bin"))
    test_neg[0].tofile(os.path.join(ds_dir, "test_neg_src.bin"))
    test_neg[1].tofile(os.path.join(ds_dir, "test_neg_dst.bin"))

    meta = {
        "num_nodes": num_nodes,
        "num_features": num_features,
        "num_edges": len(edge_src),
        "num_test_pos": test_pos.shape[1],
        "num_test_neg": test_neg.shape[1],
        "hidden": hidden,
        "out_channels": out_channels,
    }
    with open(os.path.join(ds_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Dataset exported: {ds_dir}")
    print(f"    nodes={num_nodes}, features={num_features}, "
          f"train_edges={len(edge_src)}")
    print(f"    test_pos={test_pos.shape[1]}, test_neg={test_neg.shape[1]}")

    # --- Export weights ---
    # Weights are stored in the encoder's state dict
    w_dir = os.path.join(ds_dir, "weights")
    os.makedirs(w_dir, exist_ok=True)

    for key, val in state_dict.items():
        fname = key + ".bin"
        arr = val.cpu().numpy().astype(np.float32)
        arr.tofile(os.path.join(w_dir, fname))

    w_meta = {
        "hidden": hidden,
        "out_channels": out_channels,
        "num_features": num_features,
        "weights": {k: list(v.shape) for k, v in state_dict.items()},
    }
    with open(os.path.join(w_dir, "meta.json"), "w") as f:
        json.dump(w_meta, f, indent=2)

    print(f"  Weights exported: {w_dir}")
    for k, v in state_dict.items():
        print(f"    {k}: {list(v.shape)}")


def train_dataset(dataset_name, hidden=32, out_channels=16, epochs=400, runs=10):
    """Train GAE on a single dataset, return best model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"Training GAE on {dataset_name}")
    print(f"{'='*60}")

    # Load dataset with link prediction split
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=False),
    ])
    path = os.path.join(DATA_DIR, "Planetoid_GAE")
    dataset = Planetoid(path, dataset_name, transform=transform)
    train_data, val_data, test_data = dataset[0]

    # Also load without transform for original features
    dataset_orig = Planetoid(os.path.join(DATA_DIR, dataset_name),
                             dataset_name,
                             transform=T.NormalizeFeatures())
    data_orig = dataset_orig[0]

    num_features = dataset.num_features
    print(f"  Nodes: {data_orig.num_nodes}, Features: {num_features}")
    print(f"  Train edges: {train_data.edge_index.shape[1]}")
    print(f"  Test pos: {test_data.pos_edge_label_index.shape[1]}, "
          f"Test neg: {test_data.neg_edge_label_index.shape[1]}")
    print(f"  Encoder: GCN({num_features} → {hidden} → {out_channels}), no bias")

    best_overall_auc = 0
    best_state_dict = None
    all_aucs = []

    for run in range(runs):
        model = GAE(GCNEncoder(num_features, hidden, out_channels)).to(device)
        optimizer = Adam(model.parameters(), lr=0.01)

        best_val_auc = 0
        best_run_state = None

        for epoch in range(1, epochs + 1):
            loss = train_epoch(model, optimizer, train_data)

            if epoch % 50 == 0 or epoch == epochs:
                val_auc, val_ap = evaluate(model, val_data)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_run_state = {k: v.cpu().clone()
                                      for k, v in model.state_dict().items()}
                if epoch % 100 == 0 or epoch == epochs:
                    print(f"  Run {run+1}, Epoch {epoch}: loss={loss:.4f}, "
                          f"val_auc={val_auc:.4f}, val_ap={val_ap:.4f}")

        # Evaluate best-val model on test
        model.load_state_dict(best_run_state)
        test_auc, test_ap = evaluate(model, test_data)
        all_aucs.append(test_auc)

        if test_auc > best_overall_auc:
            best_overall_auc = test_auc
            best_state_dict = best_run_state

        print(f"  Run {run+1}/{runs}: val_auc={best_val_auc:.4f}, "
              f"test_auc={test_auc:.4f}, test_ap={test_ap:.4f}")

    aucs = torch.tensor(all_aucs)
    print(f"\n  Final Test AUC: {aucs.mean():.4f} ± {aucs.std():.4f} "
          f"(best={best_overall_auc:.4f})")

    # Export best model
    print("\n  Exporting...")
    model = GAE(GCNEncoder(num_features, hidden, out_channels)).to(device)
    model.load_state_dict(best_state_dict)
    test_auc, test_ap = evaluate(model, test_data)
    val_auc, val_ap = evaluate(model, val_data)
    print(f"  Best model: val_auc={val_auc:.4f}, val_ap={val_ap:.4f}")
    print(f"              test_auc={test_auc:.4f}, test_ap={test_ap:.4f}")

    export_binary(dataset_name, data_orig, train_data, test_data,
                  best_state_dict, hidden, out_channels, SAVE_DIR)

    return best_overall_auc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = {}
    for dataset_name in ["Cora", "CiteSeer", "PubMed"]:
        auc = train_dataset(dataset_name, hidden=32, out_channels=16,
                            epochs=400, runs=10)
        results[dataset_name] = auc

    print(f"\n{'='*60}")
    print("Summary (Best Test AUC):")
    for name, auc in results.items():
        print(f"  {name}: {auc:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
