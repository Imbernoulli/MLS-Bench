"""
Protein-Ligand Binding Affinity Prediction — Self-contained template.
Predicts binding affinity (-logKd/Ki) on PDBbind benchmarks using
heterogeneous protein-ligand interaction graphs.

Structure:
  Lines 1-105:   FIXED — Imports, constants, PLABatch dataclass
  Lines 106-250: EDITABLE — AffinityModel class (starter: separate GNN + concat readout)
  Lines 251+:    FIXED — Data loading, training loop, evaluation
"""
import os
import sys
import math
import argparse
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# =====================================================================
# Constants — feature dimensions from EHIGN_PLA preprocessing
# =====================================================================

# Atom features: element(10) + degree(7) + valence(7) + hybridization(5) + aromatic(1) + Hs(5) = 35
LIGAND_ATOM_DIM = 35
POCKET_ATOM_DIM = 35

# Intra-molecular edge features: bond_type(4) + conjugated(1) + in_ring(1) + geometric(11) = 17
INTRA_EDGE_DIM = 17

# Inter-molecular edge features: geometric only = 11
INTER_EDGE_DIM = 11


@dataclass
class PLABatch:
    """Batched protein-ligand complex data for binding affinity prediction.

    All graphs in the batch are merged into single tensors with offset indices.
    """
    # Ligand graph
    lig_x: torch.Tensor              # [total_lig_atoms, 35]
    lig_edge_index: torch.Tensor     # [2, total_lig_edges]
    lig_edge_attr: torch.Tensor      # [total_lig_edges, 17]
    lig_batch: torch.Tensor          # [total_lig_atoms] graph assignment

    # Pocket graph
    poc_x: torch.Tensor              # [total_poc_atoms, 35]
    poc_edge_index: torch.Tensor     # [2, total_poc_edges]
    poc_edge_attr: torch.Tensor      # [total_poc_edges, 17]
    poc_batch: torch.Tensor          # [total_poc_atoms] graph assignment

    # Inter-molecular edges (ligand -> pocket)
    l2p_edge_index: torch.Tensor     # [2, total_l2p_edges] (src=lig, dst=poc)
    l2p_edge_attr: torch.Tensor      # [total_l2p_edges, 11]

    # Inter-molecular edges (pocket -> ligand)
    p2l_edge_index: torch.Tensor     # [2, total_p2l_edges] (src=poc, dst=lig)
    p2l_edge_attr: torch.Tensor      # [total_p2l_edges, 11]

    # Metadata
    num_lig_atoms: List[int]         # per-complex ligand atom counts
    num_poc_atoms: List[int]         # per-complex pocket atom counts
    inter_batch: torch.Tensor        # [total_l2p_edges] graph assignment for inter edges

    # Target
    labels: torch.Tensor             # [B]


# Helper: positions for 3D coordinate-based models (not pre-computed in current data)
# Models can use edge_attr geometric features which encode distance/angle info.

# =====================================================================
# FIXED SECTION END (line 95)
# =====================================================================


# Below are some utility functions available in the editable section:

def scatter_mean(src, index, dim_size):
    """Scatter mean: average src values by index."""
    out = torch.zeros(dim_size, src.size(-1), device=src.device)
    count = torch.zeros(dim_size, 1, device=src.device)
    out.index_add_(0, index, src)
    count.index_add_(0, index, torch.ones(src.size(0), 1, device=src.device))
    return out / count.clamp(min=1)


# =====================================================================
# EDITABLE SECTION START — AffinityModel + helper modules
# =====================================================================

class SimpleGNNLayer(nn.Module):
    """Simple message passing layer with edge features."""

    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        msg_input = torch.cat([x[src], edge_attr], dim=-1)
        msg = self.msg_mlp(msg_input)
        agg = torch.zeros(x.size(0), msg.size(-1), device=x.device)
        agg.index_add_(0, dst, msg)
        out = self.update_mlp(torch.cat([x, agg], dim=-1))
        return out


class AffinityModel(nn.Module):
    """Starter model: Separate GNN encoders for ligand/pocket + mean pooling readout.

    A simple baseline that processes ligand and pocket graphs independently
    with message passing, then concatenates their pooled representations
    for final prediction. Does NOT use inter-molecular edges.
    """

    def __init__(self, lig_dim, poc_dim, intra_edge_dim, inter_edge_dim):
        super().__init__()
        hidden_dim = 256
        num_layers = 3

        # Ligand encoder
        self.lig_embed = nn.Linear(lig_dim, hidden_dim)
        self.lig_convs = nn.ModuleList([
            SimpleGNNLayer(hidden_dim, intra_edge_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Pocket encoder
        self.poc_embed = nn.Linear(poc_dim, hidden_dim)
        self.poc_convs = nn.ModuleList([
            SimpleGNNLayer(hidden_dim, intra_edge_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Prediction head
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: PLABatch) -> torch.Tensor:
        """
        Args:
            batch: PLABatch with heterogeneous graph data.
        Returns:
            predictions: [B] binding affinity values
        """
        # Encode ligand
        lig_h = self.lig_embed(batch.lig_x)
        for conv in self.lig_convs:
            lig_h = conv(lig_h, batch.lig_edge_index, batch.lig_edge_attr) + lig_h

        # Encode pocket
        poc_h = self.poc_embed(batch.poc_x)
        for conv in self.poc_convs:
            poc_h = conv(poc_h, batch.poc_edge_index, batch.poc_edge_attr) + poc_h

        # Pool per graph
        num_graphs = batch.labels.size(0)
        lig_pool = scatter_mean(lig_h, batch.lig_batch, num_graphs)
        poc_pool = scatter_mean(poc_h, batch.poc_batch, num_graphs)

        # Concatenate and predict
        combined = torch.cat([lig_pool, poc_pool], dim=-1)
        pred = self.readout(combined).squeeze(-1)
        return pred

# =====================================================================
# EDITABLE SECTION END
# =====================================================================


# =====================================================================
# FIXED — Data loading, collation, training, and evaluation
# =====================================================================

class PLADataset(Dataset):
    """Dataset for protein-ligand binding affinity prediction.
    Loads pre-converted .pt files containing graph tensors.
    """

    def __init__(self, data_path):
        self.data = torch.load(data_path, weights_only=False)
        print(f"Loaded {len(self.data)} complexes from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_pla(batch_list):
    """Collate variable-size protein-ligand complexes into PLABatch."""
    lig_x_list, lig_ei_list, lig_ea_list, lig_batch_list = [], [], [], []
    poc_x_list, poc_ei_list, poc_ea_list, poc_batch_list = [], [], [], []
    l2p_ei_list, l2p_ea_list, p2l_ei_list, p2l_ea_list = [], [], [], []
    inter_batch_list = []
    labels_list = []
    num_lig_list, num_poc_list = [], []

    lig_offset = 0
    poc_offset = 0

    for i, item in enumerate(batch_list):
        n_lig = item['num_lig_atoms']
        n_poc = item['num_poc_atoms']

        lig_x_list.append(item['lig_x'])
        lig_batch_list.append(torch.full((n_lig,), i, dtype=torch.long))

        poc_x_list.append(item['poc_x'])
        poc_batch_list.append(torch.full((n_poc,), i, dtype=torch.long))

        # Offset edge indices
        if item['lig_edge_index'].size(1) > 0:
            lig_ei_list.append(item['lig_edge_index'] + lig_offset)
            lig_ea_list.append(item['lig_edge_attr'])

        if item['poc_edge_index'].size(1) > 0:
            poc_ei_list.append(item['poc_edge_index'] + poc_offset)
            poc_ea_list.append(item['poc_edge_attr'])

        # Inter-molecular edges: src from ligand, dst from pocket (l2p)
        if item['l2p_edge_index'].size(1) > 0:
            l2p_ei = item['l2p_edge_index'].clone()
            l2p_ei[0] += lig_offset  # ligand source
            l2p_ei[1] += poc_offset  # pocket dest
            l2p_ei_list.append(l2p_ei)
            l2p_ea_list.append(item['l2p_edge_attr'])
            inter_batch_list.append(torch.full((l2p_ei.size(1),), i, dtype=torch.long))

        # Inter-molecular edges: src from pocket, dst from ligand (p2l)
        if item['p2l_edge_index'].size(1) > 0:
            p2l_ei = item['p2l_edge_index'].clone()
            p2l_ei[0] += poc_offset  # pocket source
            p2l_ei[1] += lig_offset  # ligand dest
            p2l_ei_list.append(p2l_ei)
            p2l_ea_list.append(item['p2l_edge_attr'])

        labels_list.append(item['label'])
        num_lig_list.append(n_lig)
        num_poc_list.append(n_poc)

        lig_offset += n_lig
        poc_offset += n_poc

    # Concatenate
    lig_x = torch.cat(lig_x_list, dim=0)
    lig_batch = torch.cat(lig_batch_list, dim=0)
    poc_x = torch.cat(poc_x_list, dim=0)
    poc_batch = torch.cat(poc_batch_list, dim=0)

    lig_edge_index = torch.cat(lig_ei_list, dim=1) if lig_ei_list else torch.zeros(2, 0, dtype=torch.long)
    lig_edge_attr = torch.cat(lig_ea_list, dim=0) if lig_ea_list else torch.zeros(0, INTRA_EDGE_DIM)
    poc_edge_index = torch.cat(poc_ei_list, dim=1) if poc_ei_list else torch.zeros(2, 0, dtype=torch.long)
    poc_edge_attr = torch.cat(poc_ea_list, dim=0) if poc_ea_list else torch.zeros(0, INTRA_EDGE_DIM)

    l2p_edge_index = torch.cat(l2p_ei_list, dim=1) if l2p_ei_list else torch.zeros(2, 0, dtype=torch.long)
    l2p_edge_attr = torch.cat(l2p_ea_list, dim=0) if l2p_ea_list else torch.zeros(0, INTER_EDGE_DIM)
    p2l_edge_index = torch.cat(p2l_ei_list, dim=1) if p2l_ei_list else torch.zeros(2, 0, dtype=torch.long)
    p2l_edge_attr = torch.cat(p2l_ea_list, dim=0) if p2l_ea_list else torch.zeros(0, INTER_EDGE_DIM)
    inter_batch = torch.cat(inter_batch_list, dim=0) if inter_batch_list else torch.zeros(0, dtype=torch.long)

    labels = torch.cat(labels_list, dim=0)

    return PLABatch(
        lig_x=lig_x, lig_edge_index=lig_edge_index, lig_edge_attr=lig_edge_attr, lig_batch=lig_batch,
        poc_x=poc_x, poc_edge_index=poc_edge_index, poc_edge_attr=poc_edge_attr, poc_batch=poc_batch,
        l2p_edge_index=l2p_edge_index, l2p_edge_attr=l2p_edge_attr,
        p2l_edge_index=p2l_edge_index, p2l_edge_attr=p2l_edge_attr,
        num_lig_atoms=num_lig_list, num_poc_atoms=num_poc_list,
        inter_batch=inter_batch,
        labels=labels,
    )


def batch_to_device(batch, device):
    return PLABatch(
        lig_x=batch.lig_x.to(device),
        lig_edge_index=batch.lig_edge_index.to(device),
        lig_edge_attr=batch.lig_edge_attr.to(device),
        lig_batch=batch.lig_batch.to(device),
        poc_x=batch.poc_x.to(device),
        poc_edge_index=batch.poc_edge_index.to(device),
        poc_edge_attr=batch.poc_edge_attr.to(device),
        poc_batch=batch.poc_batch.to(device),
        l2p_edge_index=batch.l2p_edge_index.to(device),
        l2p_edge_attr=batch.l2p_edge_attr.to(device),
        p2l_edge_index=batch.p2l_edge_index.to(device),
        p2l_edge_attr=batch.p2l_edge_attr.to(device),
        num_lig_atoms=batch.num_lig_atoms,
        num_poc_atoms=batch.num_poc_atoms,
        inter_batch=batch.inter_batch.to(device),
        labels=batch.labels.to(device),
    )


# =====================================================================
# Training and evaluation
# =====================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch_to_device(batch, device)
        optimizer.zero_grad()

        # Models with custom multi-head losses (e.g. EHIGN's 3-term dual-head loss)
        # can expose `compute_loss(batch, labels)` returning a scalar loss directly.
        # Single-head models fall back to plain MSE on forward() output.
        if hasattr(model, 'compute_loss'):
            loss = model.compute_loss(batch, batch.labels)
        else:
            pred = model(batch)
            loss = F.mse_loss(pred, batch.labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = batch_to_device(batch, device)
        pred = model(batch)
        all_preds.append(pred.cpu().numpy())
        all_labels.append(batch.labels.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    rmse = float(np.sqrt(mean_squared_error(labels, preds)))
    rp = float(pearsonr(preds, labels)[0])

    return rmse, rp


def train_and_evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data_dir = args.data_dir
    train_ds = PLADataset(os.path.join(data_dir, 'train_data.pt'))
    valid_ds = PLADataset(os.path.join(data_dir, 'valid_data.pt'))
    test_ds = PLADataset(os.path.join(data_dir, f'{args.test_set}_data.pt'))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_pla, num_workers=4, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_pla, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_pla, num_workers=4)

    print(f"Train: {len(train_ds)}, Valid: {len(valid_ds)}, Test ({args.test_set}): {len(test_ds)}")

    # Model
    model = AffinityModel(
        lig_dim=LIGAND_ATOM_DIM,
        poc_dim=POCKET_ATOM_DIM,
        intra_edge_dim=INTRA_EDGE_DIM,
        inter_edge_dim=INTER_EDGE_DIM,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    # Training with early stopping
    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_rmse, val_rp = evaluate(model, valid_loader, device)

        print(f"TRAIN_METRICS epoch={epoch} loss={train_loss:.6f} val_rmse={val_rmse:.4f} val_rp={val_rp:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt'), weights_only=True))
    test_rmse, test_rp = evaluate(model, test_loader, device)
    print(f"TEST_METRICS rmse={test_rmse:.6f} rp={test_rp:.6f}")
    print(f"Best val RMSE: {best_val_rmse:.4f} at epoch {best_epoch}")


def main():
    parser = argparse.ArgumentParser(description="Protein-Ligand Binding Affinity Prediction")
    parser.add_argument('--test-set', type=str, required=True,
                        choices=['test2013', 'test2016', 'test2019'])
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='./output')
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_and_evaluate(args)


if __name__ == '__main__':
    main()
