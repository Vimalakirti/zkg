"""
Train GCN, GraphSage, GIN on TUDataset graph-classification benchmarks
(MUTAG, PROTEINS, IMDB-BINARY, REDDIT-BINARY).

Model architectures follow pyg/pytorch_geometric/benchmark/kernel/.

Key architecture details:
  - GCN: GCNConv layers → global_mean_pool → Linear → ReLU → Linear → log_softmax
  - GraphSAGE: SAGEConv layers → global_add_pool → Linear → ReLU → Linear → log_softmax
  - GIN (GIN0, eps=0): GINConv(MLP w/ BN) layers → global_mean_pool → Linear → ReLU → Linear → log_softmax

Saves trained model weights and dataset as .npy files for loading into zk-torch-3.
"""

import os
import argparse
import json
import time

import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import BatchNorm1d as BN, Linear, ReLU, Sequential
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import degree
import torch_geometric.transforms as T


# ---------------------------------------------------------------------------
# Feature transforms (from benchmark/kernel/datasets.py)
# ---------------------------------------------------------------------------

class NormalizedDegree:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def get_dataset(name, data_dir):
    path = os.path.join(data_dir, name)
    dataset = TUDataset(path, name)
    dataset.data.edge_attr = None

    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    return dataset


# ---------------------------------------------------------------------------
# Models (matching benchmark/kernel/ architectures)
# ---------------------------------------------------------------------------

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers, hidden):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers, hidden):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_add_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


class GIN0(torch.nn.Module):
    """GIN with train_eps=False (eps fixed at 0)."""
    def __init__(self, num_features, num_classes, num_layers, hidden):
        super().__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(num_features, hidden), ReLU(), BN(hidden),
                Linear(hidden, hidden), ReLU(), BN(hidden),
            ), train_eps=False)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(
                Sequential(
                    Linear(hidden, hidden), ReLU(), BN(hidden),
                    Linear(hidden, hidden), ReLU(), BN(hidden),
                ), train_eps=False))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


class GAT(torch.nn.Module):
    """GAT for graph classification: 3 conv layers + mean pool + MLP head."""
    def __init__(self, num_features, num_classes, num_layers, hidden, heads=4):
        super().__init__()
        self.conv1 = GATConv(num_features, hidden // heads, heads=heads, concat=True, bias=False)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden, hidden // heads, heads=heads, concat=True, bias=False))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


# ---------------------------------------------------------------------------
# Training / evaluation (from benchmark/kernel/train_eval.py)
# ---------------------------------------------------------------------------

def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)
    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))
    val_indices = [test_indices[i - 1] for i in range(folds)]
    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))
    return train_indices, test_indices, val_indices


def train_epoch(model, optimizer, loader, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval()
    loss = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)


def cross_validation(dataset, model, device, folds=10, epochs=100,
                     batch_size=128, lr=0.01, lr_decay_factor=0.5,
                     lr_decay_step_size=50, weight_decay=0):
    """10-fold cross-validation. Returns best state_dict, mean acc, std."""
    val_losses_all, accs_all = [], []
    best_overall_acc = 0
    best_state_dict = None

    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, folds))):
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss = float('inf')
        best_test_acc = 0
        best_fold_state = None

        for epoch in range(1, epochs + 1):
            train_epoch(model, optimizer, train_loader, device)
            val_loss = eval_loss(model, val_loader, device)
            test_acc = eval_acc(model, test_loader, device)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_acc = test_acc
                best_fold_state = {k: v.cpu().clone()
                                   for k, v in model.state_dict().items()}

            if epoch % lr_decay_step_size == 0:
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_decay_factor * pg['lr']

        val_losses_all.append(best_val_loss)
        accs_all.append(best_test_acc)

        if best_test_acc > best_overall_acc:
            best_overall_acc = best_test_acc
            best_state_dict = best_fold_state

        print(f"  Fold {fold+1}/{folds}: val_loss={best_val_loss:.4f}, "
              f"test_acc={best_test_acc:.4f}")

    accs_t = torch.tensor(accs_all)
    mean_acc = accs_t.mean().item()
    std_acc = accs_t.std().item()
    print(f"  Result: {mean_acc:.3f} ± {std_acc:.3f} "
          f"(best fold={best_overall_acc:.4f})")
    return best_state_dict, mean_acc, std_acc


# ---------------------------------------------------------------------------
# Save dataset and weights as .npy files
# ---------------------------------------------------------------------------

def save_dataset(dataset, save_dir, dataset_name):
    """Save TUDataset as individual .npy files for each graph + metadata.

    Directory structure:
      <save_dir>/<dataset_name>/
        meta.json         — num_graphs, num_features, num_classes, graph sizes
        graph_labels.npy  — (num_graphs,) int32
        graph_<i>/
          x.npy           — (num_nodes_i, num_features) float32
          edge_src.npy    — (num_edges_i,) int32
          edge_dst.npy    — (num_edges_i,) int32
    """
    out_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    num_features = dataset.num_features
    num_classes = dataset.num_classes
    num_graphs = len(dataset)
    graph_sizes = []
    labels = []

    for i, data in enumerate(dataset):
        g_dir = os.path.join(out_dir, f"graph_{i}")
        os.makedirs(g_dir, exist_ok=True)

        x = data.x.numpy().astype(np.float32)
        np.save(os.path.join(g_dir, "x.npy"), x)

        edge_index = data.edge_index.numpy().astype(np.int32)
        np.save(os.path.join(g_dir, "edge_src.npy"), edge_index[0])
        np.save(os.path.join(g_dir, "edge_dst.npy"), edge_index[1])

        graph_sizes.append(data.num_nodes)
        labels.append(data.y.item())

    labels = np.array(labels, dtype=np.int32)
    np.save(os.path.join(out_dir, "graph_labels.npy"), labels)

    meta = {
        "num_graphs": num_graphs,
        "num_features": num_features,
        "num_classes": num_classes,
        "graph_sizes": graph_sizes,
        "max_nodes": max(graph_sizes),
    }
    with open(os.path.join(out_dir, "meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Dataset saved: {out_dir}")
    print(f"    Graphs: {num_graphs}, Features: {num_features}, "
          f"Classes: {num_classes}, Max nodes: {max(graph_sizes)}")


def save_weights(state_dict, save_dir, model_name, dataset_name):
    """Save model weights as individual .npy files.

    Directory structure:
      <save_dir>/<model_name>_<dataset_name>/
        meta.json           — layer names and shapes
        <param_name>.npy    — one file per parameter
    """
    out_dir = os.path.join(save_dir, f"{model_name}_{dataset_name}")
    os.makedirs(out_dir, exist_ok=True)

    meta = {}
    for k, v in state_dict.items():
        arr = v.numpy()
        # Replace dots with underscores for filesystem compatibility
        fname = k.replace('.', '__') + ".npy"
        np.save(os.path.join(out_dir, fname), arr)
        meta[k] = {"shape": list(arr.shape), "dtype": str(arr.dtype),
                    "file": fname}

    with open(os.path.join(out_dir, "meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Weights saved: {out_dir}")
    for k, v in state_dict.items():
        print(f"    {k}: {list(v.shape)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASETS = ['MUTAG', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY']

# Hyperparameters matching benchmark/kernel/main.py defaults
HPARAMS = {
    'epochs': 100,
    'batch_size': 128,
    'lr': 0.01,
    'lr_decay_factor': 0.5,
    'lr_decay_step_size': 50,
    'weight_decay': 0,
    'folds': 10,
}

MODEL_CONFIGS = {
    'gcn':       {'num_layers': 3, 'hidden': 64},
    'graphsage': {'num_layers': 3, 'hidden': 64},
    'gin':       {'num_layers': 3, 'hidden': 64},
    'gat':       {'num_layers': 3, 'hidden': 64},
}


def build_model(model_name, num_features, num_classes, cfg):
    if model_name == 'gcn':
        return GCN(num_features, num_classes, cfg['num_layers'], cfg['hidden'])
    elif model_name == 'graphsage':
        return GraphSAGE(num_features, num_classes, cfg['num_layers'],
                         cfg['hidden'])
    elif model_name == 'gin':
        return GIN0(num_features, num_classes, cfg['num_layers'],
                    cfg['hidden'])
    elif model_name == 'gat':
        return GAT(num_features, num_classes, cfg['num_layers'],
                   cfg['hidden'])
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+',
                        default=['gcn', 'graphsage', 'gin', 'gat'],
                        choices=['gcn', 'graphsage', 'gin', 'gat'])
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    parser.add_argument('--save_dir', type=str, default='kernel_weights',
                        help='Directory to save model weights and datasets')
    parser.add_argument('--data_dir', type=str, default='kernel_data',
                        help='Directory for downloaded TUDatasets')
    parser.add_argument('--num_layers', type=int, default=None,
                        help='Override number of layers')
    parser.add_argument('--hidden', type=int, default=None,
                        help='Override hidden dimension')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    results = {}

    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        dataset = get_dataset(dataset_name, args.data_dir)
        print(f"  Graphs: {len(dataset)}, Features: {dataset.num_features}, "
              f"Classes: {dataset.num_classes}")

        save_dataset(dataset, args.save_dir, dataset_name.lower())

        for model_name in args.models:
            cfg = MODEL_CONFIGS[model_name].copy()
            if args.num_layers is not None:
                cfg['num_layers'] = args.num_layers
            if args.hidden is not None:
                cfg['hidden'] = args.hidden

            print(f"\n--- {model_name.upper()} "
                  f"(layers={cfg['num_layers']}, hidden={cfg['hidden']}) ---")

            model = build_model(model_name, dataset.num_features,
                                dataset.num_classes, cfg)
            nparams = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {nparams}")

            t0 = time.time()
            state_dict, mean_acc, std_acc = cross_validation(
                dataset, model, device,
                folds=HPARAMS['folds'],
                epochs=HPARAMS['epochs'],
                batch_size=HPARAMS['batch_size'],
                lr=HPARAMS['lr'],
                lr_decay_factor=HPARAMS['lr_decay_factor'],
                lr_decay_step_size=HPARAMS['lr_decay_step_size'],
                weight_decay=HPARAMS['weight_decay'],
            )
            elapsed = time.time() - t0
            print(f"  Training time: {elapsed:.1f}s")

            save_weights(state_dict, args.save_dir, model_name,
                         dataset_name.lower())

            results[f"{model_name}_{dataset_name}"] = {
                'mean_acc': mean_acc,
                'std_acc': std_acc,
                'num_layers': cfg['num_layers'],
                'hidden': cfg['hidden'],
            }

    # Save summary
    summary_path = os.path.join(args.save_dir, 'results.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults summary saved to {summary_path}")


if __name__ == '__main__':
    main()
