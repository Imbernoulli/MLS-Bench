"""Graph Classification Readout/Pooling Benchmark.

Train GNN models on TU graph classification datasets (MUTAG, PROTEINS, NCI1)
to evaluate graph-level readout and pooling mechanisms.

FIXED: GNN backbone (GIN message-passing layers), data pipeline, training loop.
EDITABLE: GraphReadout class (graph-level pooling/readout mechanism).

Usage:
    python custom_graph_cls.py --dataset MUTAG --seed 42 --output-dir ./output
"""

import argparse
import math
import os
import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool
from torch_geometric.utils import degree, to_dense_adj, to_dense_batch

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


# ============================================================================
# Graph-Level Readout / Pooling Mechanism
# ============================================================================

# -- EDITABLE REGION START (lines 41-81) ------------------------------------
class GraphReadout(nn.Module):
    """Custom graph-level readout/pooling mechanism.

    Aggregates node-level representations into a single graph-level
    representation for classification. Receives node embeddings from
    a GIN backbone and must produce a fixed-size graph embedding.

    Args:
        hidden_dim (int): Dimension of node embeddings from the GNN backbone.
        num_layers (int): Number of GNN layers (for JK-style readout).

    Input:
        x (Tensor): Node embeddings [N_total, hidden_dim] (batched).
        edge_index (LongTensor): Edge index [2, E_total] (batched).
        batch (LongTensor): Batch assignment vector [N_total].
        layer_outputs (list[Tensor]): Per-layer node embeddings from GNN,
            each [N_total, hidden_dim]. len == num_layers.

    Output:
        Tensor: Graph-level embeddings [B, output_dim].
            output_dim is accessible via self.output_dim attribute.

    Design considerations:
        - How to aggregate variable-size node sets into fixed-size vectors
        - Whether to use simple permutation-invariant ops (sum/mean/max)
        - Whether to learn attention weights over nodes
        - Whether to exploit multi-scale information from different GNN layers
        - Whether to use hierarchical coarsening (cluster, pool, repeat)
        - Interaction between pooling and downstream classifier
    """

    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Default: simple sum pooling over final-layer node embeddings
        self.output_dim = hidden_dim

    def forward(self, x, edge_index, batch, layer_outputs):
        # Default: global sum pooling on last-layer embeddings
        return global_add_pool(x, batch)
# -- EDITABLE REGION END (lines 41-81) --------------------------------------


# ============================================================================
# GIN Backbone (FIXED)
# ============================================================================

class GINBackbone(nn.Module):
    """Graph Isomorphism Network backbone (Xu et al., 2019).

    Standard 5-layer GIN with batch normalization. Produces per-node
    embeddings at each layer for flexible readout.
    """

    def __init__(self, input_dim, hidden_dim, num_layers=5):
        super().__init__()
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch):
        layer_outputs = []
        h = x
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            h = self.bns[i](h)
            h = F.relu(h)
            layer_outputs.append(h)
        return h, layer_outputs


# ============================================================================
# Full Classifier Model (FIXED)
# ============================================================================

class GraphClassifier(nn.Module):
    """GIN backbone + custom readout + MLP classifier."""

    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=5,
                 dropout=0.5):
        super().__init__()
        self.backbone = GINBackbone(input_dim, hidden_dim, num_layers)
        self.readout = GraphReadout(hidden_dim, num_layers)
        self.dropout = dropout

        # MLP classifier head
        readout_dim = self.readout.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        node_emb, layer_outputs = self.backbone(x, edge_index, batch)
        graph_emb = self.readout(node_emb, edge_index, batch, layer_outputs)
        return self.classifier(graph_emb)


# ============================================================================
# Data Loading (FIXED)
# ============================================================================

def load_dataset(name, data_root='/data/TUDataset'):
    """Load TU dataset with one-hot degree features if no node features."""
    dataset = TUDataset(root=data_root, name=name, use_node_attr=True)

    # If no node features, use one-hot degree encoding
    if dataset.num_node_features == 0:
        max_degree = 0
        for data in dataset:
            d = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))

        for data in dataset:
            d = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
            data.x = F.one_hot(d, num_classes=max_degree + 1).float()

    return dataset


# ============================================================================
# Training & Evaluation (FIXED)
# ============================================================================

def train_epoch(model, loader, optimizer, device):
    """Train one epoch. Returns average loss."""
    model.train()
    total_loss = 0
    total_graphs = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        total_graphs += data.num_graphs
    return total_loss / total_graphs


def evaluate(model, loader, device):
    """Evaluate model. Returns (accuracy%, macro_f1)."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds) * 100.0
    f1 = f1_score(all_labels, all_preds, average='macro') * 100.0
    return acc, f1


def run_fold(dataset, train_idx, test_idx, args, device):
    """Run training on one fold. Returns test (acc, f1) at epoch of best val acc."""
    # Carve out a stratified 10% validation split from the training fold
    # so that epoch selection does not peek at the test set.
    train_idx = np.asarray(train_idx)
    train_labels = np.array([dataset[int(i)].y.item() for i in train_idx])
    val_frac = 0.1
    rng = np.random.RandomState(args.seed)
    unique_labels = np.unique(train_labels)
    val_mask = np.zeros(len(train_idx), dtype=bool)
    for lbl in unique_labels:
        lbl_positions = np.where(train_labels == lbl)[0]
        rng.shuffle(lbl_positions)
        n_val = max(1, int(round(len(lbl_positions) * val_frac)))
        val_mask[lbl_positions[:n_val]] = True
    # Guard: ensure at least one training sample remains
    if val_mask.all():
        val_mask[0] = False

    val_idx = train_idx[val_mask]
    sub_train_idx = train_idx[~val_mask]

    train_dataset = dataset[sub_train_idx.tolist()]
    val_dataset = dataset[val_idx.tolist()]
    test_dataset = dataset[test_idx.tolist()]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    input_dim = dataset.num_node_features
    num_classes = dataset.num_classes

    model = GraphClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    # ── Parameter Budget Check (first fold only) ──
    # Budget = 1.05x total model params with largest baseline readout.
    # GMT readout: seed(H) + MultiheadAttention(4*H*H+4*H) + 2*LayerNorm(4*H)
    #   + FFN(4*H*H+3*H) = 8*H*H + 12*H
    # Set2Set readout: LSTM(H,H) = 8*H*H+8*H, proj Linear(2H,H) = 2*H*H+H
    #   total = 10*H*H + 9*H
    # Set2Set is larger, so we take the max.
    _H = args.hidden_dim
    _n_params = sum(p.numel() for p in model.parameters())
    _readout_params = sum(p.numel() for p in model.readout.parameters())
    _fixed_params = _n_params - _readout_params
    _gmt_readout = 8 * _H * _H + 12 * _H
    _set2set_readout = 10 * _H * _H + 9 * _H  # LSTM(8H^2+8H) + proj(2H^2+H)
    _max_readout = max(_gmt_readout, _set2set_readout)
    _param_budget = int((_fixed_params + _max_readout) * 1.05)
    print(f"Model parameters: {_n_params:,} (budget: {_param_budget:,})", flush=True)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = -1.0
    best_test_acc = 0.0
    best_test_f1 = 0.0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_acc, val_f1 = evaluate(model, val_loader, device)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Only touch the test set when val improves; the reported
            # numbers correspond to the epoch selected by val accuracy.
            test_acc, test_f1 = evaluate(model, test_loader, device)
            best_test_acc = test_acc
            best_test_f1 = test_f1

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"TRAIN_METRICS: fold_epoch epoch={epoch+1} "
                f"train_loss={train_loss:.4f} val_acc={val_acc:.2f} "
                f"val_f1={val_f1:.2f} lr={optimizer.param_groups[0]['lr']:.6f}",
                flush=True,
            )

    return best_test_acc, best_test_f1


def main():
    parser = argparse.ArgumentParser(description="Graph Classification Readout Benchmark")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['MUTAG', 'PROTEINS', 'NCI1'])
    parser.add_argument('--data-root', type=str, default='/data/TUDataset')
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num-folds', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='.')
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    dataset = load_dataset(args.dataset, args.data_root)
    print(f"Dataset: {args.dataset}, Graphs: {len(dataset)}, "
          f"Features: {dataset.num_node_features}, Classes: {dataset.num_classes}",
          flush=True)

    # 10-fold stratified cross-validation
    labels = np.array([d.y.item() for d in dataset])
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True,
                          random_state=args.seed)

    fold_accs = []
    fold_f1s = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(labels, labels)):
        print(f"\n--- Fold {fold_idx + 1}/{args.num_folds} ---", flush=True)
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        best_acc, best_f1 = run_fold(dataset, train_idx, test_idx, args, device)
        fold_accs.append(best_acc)
        fold_f1s.append(best_f1)
        print(f"Fold {fold_idx + 1}: acc={best_acc:.2f} f1={best_f1:.2f}", flush=True)

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    mean_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)

    print(f"\n10-Fold Results: acc={mean_acc:.2f}+/-{std_acc:.2f} "
          f"f1={mean_f1:.2f}+/-{std_f1:.2f}", flush=True)
    print(f"TEST_METRICS: test_acc={mean_acc:.2f} macro_f1={mean_f1:.2f}", flush=True)


if __name__ == '__main__':
    main()
