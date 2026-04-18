"""
Train GCN, GraphSage, GAT on citation networks (Cora, CiteSeer, PubMed).
Model architectures match zk-torch-3 definitions in src/dag/gnn.rs.

Key architectural choices to match zk-torch-3:
  - GCN: ReLU(A · (H · W)) per layer, no bias
  - GraphSage: ReLU(H·W_self + A·(H·W_neighbor)) per layer, no bias
  - GAT: single-head attention, ReLU (not LeakyReLU) in attention, no bias
  - All models: 2 conv layers (first with ReLU, second feeds into log_softmax)
"""

import os
import argparse
import json
import time

import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


# ---------------------------------------------------------------------------
# Models matching zk-torch-3 architecture
# ---------------------------------------------------------------------------

class GCN(torch.nn.Module):
    """
    Matches zk-torch-3 gcn (src/dag/gnn.rs):
      Layer k: H_{k+1} = ReLU(A · (H_k · W_k))
    Two layers: input→hidden (ReLU), hidden→num_classes (no ReLU, log_softmax).
    No bias to match zk-torch-3 which only has weight matrices.
    """
    def __init__(self, num_features, hidden, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden, bias=False)
        self.conv2 = GCNConv(hidden, num_classes, bias=False)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GraphSage(torch.nn.Module):
    """
    Matches zk-torch-3 graph_sage (src/dag/gnn.rs):
      Layer k: H_{k+1} = ReLU(H_k·W_self + A·(H_k·W_neighbor) + bias)
    SAGEConv with mean aggregation and bias enabled.
    """
    def __init__(self, num_features, hidden, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden, bias=True)
        self.conv2 = SAGEConv(hidden, num_classes, bias=True)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    """
    Matches zk-torch-3 gat (src/dag/gnn.rs):
      - Multi-head attention (configurable heads)
      - ReLU in attention (negative_slope=0), not LeakyReLU
      - Intermediate layers: concat heads → output dim = heads * hidden
      - Last layer: single head → output dim = num_classes
    No bias to match zk-torch-3.
    """
    def __init__(self, num_features, hidden, num_classes, heads=4, dropout=0.5):
        super().__init__()
        self.heads = heads
        # Layer 1: multi-head, concat → output dim = heads * hidden
        self.conv1 = GATConv(num_features, hidden, heads=heads, concat=True,
                             negative_slope=0.0, bias=False, dropout=dropout)
        # Layer 2: single head → output dim = num_classes
        self.conv2 = GATConv(hidden * heads, num_classes, heads=1, concat=False,
                             negative_slope=0.0, bias=False, dropout=dropout)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_epoch(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    out = model(data)
    results = {}
    for split in ['train', 'val', 'test']:
        mask = data[f'{split}_mask']
        loss = float(F.nll_loss(out[mask], data.y[mask]))
        pred = out[mask].argmax(1)
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        results[f'{split}_loss'] = loss
        results[f'{split}_acc'] = acc
    return results


def train_model(model, dataset, lr, weight_decay, epochs, early_stopping,
                runs, device):
    """Train model for multiple runs, return best model state dict and stats."""
    best_overall_acc = 0
    best_state_dict = None
    all_accs = []

    for run in range(runs):
        data = dataset[0].to(device)
        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss = float('inf')
        test_acc = 0
        val_loss_history = []
        best_run_state = None

        for epoch in range(1, epochs + 1):
            train_epoch(model, optimizer, data)
            info = evaluate(model, data)

            if info['val_loss'] < best_val_loss:
                best_val_loss = info['val_loss']
                test_acc = info['test_acc']
                best_run_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            val_loss_history.append(info['val_loss'])
            if early_stopping > 0 and epoch > epochs // 2:
                recent = torch.tensor(val_loss_history[-(early_stopping + 1):-1])
                if info['val_loss'] > recent.mean().item():
                    break

        all_accs.append(test_acc)
        if test_acc > best_overall_acc:
            best_overall_acc = test_acc
            best_state_dict = best_run_state

        if (run + 1) % 10 == 0 or run == 0:
            print(f"  Run {run+1}/{runs}: test_acc={test_acc:.4f}")

    accs_t = torch.tensor(all_accs)
    print(f"  Final: {accs_t.mean():.4f} ± {accs_t.std():.4f} "
          f"(best={best_overall_acc:.4f})")
    return best_state_dict, float(accs_t.mean()), float(accs_t.std())


def save_dataset(data, dataset, save_dir, dataset_name):
    """Save graph data as .npz for loading in zk-torch-3.

    Contents:
      x:          (num_nodes, num_features)  float32  — node features
      y:          (num_nodes,)               int64    — class labels
      edge_index: (2, num_edges)             int64    — COO edge list [src; dst]
      train_mask: (num_nodes,)               bool
      val_mask:   (num_nodes,)               bool
      test_mask:  (num_nodes,)               bool
      num_classes: scalar
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{dataset_name}.npz")
    np.savez(
        path,
        x=data.x.numpy(),
        y=data.y.numpy(),
        edge_index=data.edge_index.numpy(),
        train_mask=data.train_mask.numpy(),
        val_mask=data.val_mask.numpy(),
        test_mask=data.test_mask.numpy(),
        num_classes=np.array(dataset.num_classes),
    )
    print(f"  Dataset saved: {path}")
    print(f"    x: {list(data.x.shape)}, edge_index: {list(data.edge_index.shape)}, "
          f"y: {list(data.y.shape)}, classes: {dataset.num_classes}")


def save_weights(state_dict, save_dir, model_name, dataset_name):
    """Save model weights as both .pt (PyTorch) and .npz (numpy) files."""
    os.makedirs(save_dir, exist_ok=True)

    # Save PyTorch checkpoint
    pt_path = os.path.join(save_dir, f"{model_name}_{dataset_name}.pt")
    torch.save(state_dict, pt_path)

    # Save as numpy arrays for easier loading in Rust/other languages
    np_dict = {k: v.numpy() for k, v in state_dict.items()}
    npz_path = os.path.join(save_dir, f"{model_name}_{dataset_name}.npz")
    np.savez(npz_path, **np_dict)

    print(f"  Saved: {pt_path}")
    print(f"  Saved: {npz_path}")

    # Print weight shapes
    for k, v in state_dict.items():
        print(f"    {k}: {list(v.shape)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    'gcn': {
        'hidden': 16,
        'lr': 0.01,
        'weight_decay': 5e-4,
        'dropout': 0.5,
        'epochs': 200,
        'early_stopping': 10,
    },
    'graphsage': {
        'hidden': 16,
        'lr': 0.01,
        'weight_decay': 5e-4,
        'dropout': 0.5,
        'epochs': 200,
        'early_stopping': 10,
    },
    'gat': {
        'hidden': 8,
        'heads': 4,
        'lr': 0.005,
        'weight_decay': 5e-4,
        'dropout': 0.6,
        'epochs': 1000,
        'early_stopping': 100,
    },
}

DATASETS = ['Cora', 'CiteSeer', 'PubMed']


def build_model(model_name, num_features, num_classes, cfg):
    if model_name == 'gcn':
        return GCN(num_features, cfg['hidden'], num_classes, cfg['dropout'])
    elif model_name == 'graphsage':
        return GraphSage(num_features, cfg['hidden'], num_classes, cfg['dropout'])
    elif model_name == 'gat':
        return GAT(num_features, cfg['hidden'], num_classes,
                   heads=cfg.get('heads', 4), dropout=cfg['dropout'])
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['gcn', 'graphsage', 'gat'],
                        choices=['gcn', 'graphsage', 'gat'])
    parser.add_argument('--datasets', nargs='+', default=DATASETS,
                        choices=DATASETS)
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of training runs per experiment')
    parser.add_argument('--save_dir', type=str, default='weights',
                        help='Directory to save model weights')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory for downloaded datasets')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    results = {}

    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        dataset = Planetoid(root=os.path.join(args.data_dir, dataset_name),
                            name=dataset_name,
                            transform=NormalizeFeatures())
        data = dataset[0]
        print(f"  Nodes: {data.num_nodes}, Edges: {data.num_edges}, "
              f"Features: {dataset.num_features}, Classes: {dataset.num_classes}")

        save_dataset(data, dataset, args.save_dir, dataset_name.lower())

        for model_name in args.models:
            cfg = MODEL_CONFIGS[model_name]
            print(f"\n--- {model_name.upper()} (hidden={cfg['hidden']}) ---")

            model = build_model(model_name, dataset.num_features,
                                dataset.num_classes, cfg)
            print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")

            t0 = time.time()
            state_dict, mean_acc, std_acc = train_model(
                model, dataset, cfg['lr'], cfg['weight_decay'],
                cfg['epochs'], cfg['early_stopping'], args.runs, device,
            )
            elapsed = time.time() - t0
            print(f"  Training time: {elapsed:.1f}s")

            save_weights(state_dict, args.save_dir, model_name, dataset_name.lower())

            results[f"{model_name}_{dataset_name}"] = {
                'mean_acc': mean_acc,
                'std_acc': std_acc,
            }

    # Save summary
    summary_path = os.path.join(args.save_dir, 'results.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults summary saved to {summary_path}")


if __name__ == '__main__':
    main()
