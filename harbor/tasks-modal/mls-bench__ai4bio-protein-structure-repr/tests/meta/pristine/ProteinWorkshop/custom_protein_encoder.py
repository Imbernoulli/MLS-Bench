"""
Protein Structure Representation Learning — Self-contained template.
Trains a geometric GNN encoder for protein structure and evaluates on
downstream classification tasks (EC number, GO-BP, Fold classification).

Structure:
  Lines 1-124:    FIXED — Imports, constants, data loading utilities
  Lines 125-252:  EDITABLE — ProteinEncoder class + helper modules
  Lines 253+:     FIXED — Dataset, decoder head, training loop, evaluation
"""
import os
import sys
import math
import json
import argparse
import warnings
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool, global_add_pool, radius_graph, knn_graph
from torch_geometric.utils import add_self_loops

from torch_scatter import scatter_mean, scatter_add

warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# Constants and Utilities
# =====================================================================

NUM_AMINO_ACIDS = 20
AMINO_ACIDS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
    'THR', 'TRP', 'TYR', 'VAL'
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Node feature dimension: one-hot amino acid (20) + sin/cos positional (2) + dihedrals (6)
SCALAR_NODE_DIM = 28
# Edge feature dimension: distance (1) + direction unit vector (3)
EDGE_FEAT_DIM = 4


def compute_node_features(pos, aa_idx, batch_idx):
    """Compute scalar node features: one-hot AA + positional encoding + pseudo-dihedrals.

    Args:
        pos: (N, 3) alpha-carbon positions
        aa_idx: (N,) amino acid indices [0, 19]
        batch_idx: (N,) batch assignment

    Returns:
        node_feat: (N, SCALAR_NODE_DIM)
    """
    N = pos.size(0)
    device = pos.device

    # One-hot amino acid
    aa_onehot = F.one_hot(aa_idx.clamp(0, NUM_AMINO_ACIDS - 1), NUM_AMINO_ACIDS).float()  # (N, 20)

    # Sequence positional encoding (sin/cos within each graph) — vectorized
    counts = scatter_add(torch.ones(N, device=device), batch_idx, dim=0)
    # Compute per-node offset within its graph using cumsum trick
    ones = torch.ones(N, device=device)
    # For each node, its local index = global_index - start_of_its_graph
    cumcounts = torch.zeros(int(batch_idx.max().item()) + 2, dtype=torch.long, device=device)
    cumcounts[1:len(counts)+1] = counts.long().cumsum(0)
    offsets = torch.arange(N, device=device).float() - cumcounts[batch_idx].float()
    max_len = counts[batch_idx].float().clamp(min=1)
    pos_enc = offsets / max_len
    sin_enc = torch.sin(pos_enc * math.pi).unsqueeze(-1)
    cos_enc = torch.cos(pos_enc * math.pi).unsqueeze(-1)

    # Pseudo-dihedral angles — vectorized (no Python loops)
    dihedrals = torch.zeros(N, 6, device=device)
    # Displacement vectors between consecutive nodes
    d = pos[1:] - pos[:-1]  # (N-1, 3)
    # Mask: consecutive pairs must be in the same graph
    same_graph = (batch_idx[1:] == batch_idx[:-1])  # (N-1,)

    # For dihedral at position j, we need d[j-1], d[j], d[j+1]
    # Valid positions: j in [1, N-3] where j-1, j, j+1 are all within same graph
    if N >= 4:
        v1 = d[:-2]   # d[j-1] for j=1..N-3
        v2 = d[1:-1]  # d[j]   for j=1..N-3
        v3 = d[2:]    # d[j+1] for j=1..N-3
        # Valid mask: all three displacement vectors must be within same graph
        valid = same_graph[:-2] & same_graph[1:-1] & same_graph[2:]  # (N-3,)

        # Compute cross products
        n1 = torch.linalg.cross(v1, v2)  # (N-3, 3)
        n2 = torch.linalg.cross(v2, v3)  # (N-3, 3)
        n1_norm = n1 / (n1.norm(dim=-1, keepdim=True) + 1e-8)
        n2_norm = n2 / (n2.norm(dim=-1, keepdim=True) + 1e-8)

        cos_angle = (n1_norm * n2_norm).sum(dim=-1).clamp(-1, 1)
        sin_angle = torch.linalg.cross(n1_norm, n2_norm).norm(dim=-1).clamp(-1, 1)
        v1_norm = v1.norm(dim=-1)
        v2_norm = v2.norm(dim=-1)
        v1v2_cos = (v1 * v2).sum(dim=-1) / (v1_norm * v2_norm + 1e-8)
        v2v3_cos = (v2 * v3).sum(dim=-1) / (v2_norm * v3.norm(dim=-1) + 1e-8)

        # Target indices in the original array: positions 1 to N-3 (offset by 1)
        target_idx = torch.arange(1, N - 2, device=device)
        valid_idx = target_idx[valid]

        dihedrals[valid_idx, 0] = cos_angle[valid]
        dihedrals[valid_idx, 1] = sin_angle[valid]
        dihedrals[valid_idx, 2] = v1_norm[valid]
        dihedrals[valid_idx, 3] = v2_norm[valid]
        dihedrals[valid_idx, 4] = v1v2_cos[valid]
        dihedrals[valid_idx, 5] = v2v3_cos[valid]

    return torch.cat([aa_onehot, sin_enc, cos_enc, dihedrals], dim=-1)

# =====================================================================
# EDITABLE SECTION START — ProteinEncoder + helper modules
# =====================================================================

class MessagePassingLayer(nn.Module):
    """Basic invariant message passing layer for protein graphs."""

    def __init__(self, hidden_dim, edge_dim=EDGE_FEAT_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index, edge_attr):
        """
        Args:
            h: (N, hidden_dim) node features
            edge_index: (2, E) edge indices
            edge_attr: (E, edge_dim) edge features
        Returns:
            h: (N, hidden_dim) updated node features
        """
        src, dst = edge_index
        edge_input = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        msg = self.edge_mlp(edge_input)
        # Aggregate messages
        agg = scatter_mean(msg, dst, dim=0, dim_size=h.size(0))
        h_new = self.node_mlp(torch.cat([h, agg], dim=-1))
        h = self.norm(h + h_new)
        return h


class ProteinEncoder(nn.Module):
    """Geometric GNN encoder for protein structures.

    Takes alpha-carbon graphs with node features (amino acid type, positional
    encoding, pseudo-dihedrals) and edge features (distance, direction) and
    produces per-node and per-graph embeddings.

    This is the starter implementation using basic invariant message passing.
    The agent should replace this with a more expressive geometric GNN design
    (e.g., equivariant message passing, multi-scale, attention, etc.).

    Args:
        input_dim: Dimension of input node features (default: SCALAR_NODE_DIM=28)
        hidden_dim: Hidden dimension (default: 256)
        out_dim: Output embedding dimension (default: 128)
        num_layers: Number of message passing layers (default: 6)
        dropout: Dropout rate (default: 0.1)
        cutoff: Distance cutoff for edge construction in Angstroms (default: 10.0)
        max_neighbors: Max neighbors in kNN graph (default: 16)
    """

    def __init__(
        self,
        input_dim: int = SCALAR_NODE_DIM,
        hidden_dim: int = 256,
        out_dim: int = 128,
        num_layers: int = 6,
        dropout: float = 0.1,
        cutoff: float = 10.0,
        max_neighbors: int = 16,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors

        self.node_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, EDGE_FEAT_DIM)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def _build_edges(self, pos, batch):
        """Build kNN graph edges with distance and direction features."""
        edge_index = knn_graph(pos, k=self.max_neighbors, batch=batch, loop=False)
        src, dst = edge_index
        diff = pos[dst] - pos[src]
        dist = diff.norm(dim=-1, keepdim=True)
        direction = diff / (dist + 1e-8)
        edge_attr = torch.cat([dist, direction], dim=-1)  # (E, 4)
        return edge_index, edge_attr

    def forward(self, pos, node_feat, batch):
        """
        Args:
            pos: (N, 3) alpha-carbon coordinates
            node_feat: (N, input_dim) node scalar features
            batch: (N,) batch index

        Returns:
            node_emb: (N, out_dim) per-node embeddings
            graph_emb: (B, out_dim) per-graph embeddings (mean pool)
        """
        edge_index, edge_attr = self._build_edges(pos, batch)

        h = self.node_embed(node_feat)

        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
            h = self.dropout(h)

        node_emb = self.out_proj(h)
        graph_emb = global_mean_pool(node_emb, batch)

        return node_emb, graph_emb

# =====================================================================
# EDITABLE SECTION END
# =====================================================================


# =====================================================================
# FIXED — Dataset, decoder head, training loop, evaluation
# =====================================================================

class ProteinGraphDataset(Dataset):
    """Dataset that loads pre-processed protein graph data."""

    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def collate_protein_graphs(batch_list):
    """Collate protein graph Data objects into a Batch."""
    return Batch.from_data_list(batch_list)


class ClassificationHead(nn.Module):
    """MLP classification head on top of graph embeddings."""

    def __init__(self, in_dim, num_classes, hidden_dim=256, task_type='multiclass'):
        super().__init__()
        self.task_type = task_type
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, graph_emb):
        return self.head(graph_emb)


def load_dataset_splits(task_name, data_dir):
    """Load pre-processed protein graph data for a given task.

    The data is expected to be preprocessed during container build into
    /data/ProteinWorkshop/<task_subdir>/processed/ as .pt files.

    Each Data object has:
        - pos: (L, 3) alpha-carbon positions
        - aa_idx: (L,) amino acid indices
        - y: label (int for multiclass, binary vector for multilabel)
        - num_nodes: number of residues
    """
    task_configs = {
        'ec_reaction': {
            'subdir': 'ECReaction',
            'num_classes': 384,
            'task_type': 'multiclass',
        },
        'go_bp': {
            'subdir': 'GeneOntology',
            'num_classes': 1943,
            'task_type': 'multilabel',
        },
        'fold_fold': {
            'subdir': 'FoldClassification',
            'num_classes': 1195,
            'task_type': 'multiclass',
        },
    }

    config = task_configs[task_name]
    base_path = Path(data_dir) / config['subdir'] / 'processed'

    splits = {}
    for split_name in ['train', 'val', 'test']:
        fpath = base_path / f'{split_name}.pt'
        if fpath.exists():
            splits[split_name] = torch.load(fpath, weights_only=False)
        else:
            print(f"Warning: {fpath} not found, skipping {split_name} split")
            splits[split_name] = []

    return splits, config['num_classes'], config['task_type']


def preprocess_and_cache(task_name, data_dir):
    """Use ProteinWorkshop datamodules to load and cache processed data.

    Converts raw PDB/MMTF files into PyG Data objects with:
        - pos: alpha-carbon positions
        - aa_idx: amino acid index per residue
        - y: task label
    """
    processed_dir_map = {
        'ec_reaction': 'ECReaction',
        'go_bp': 'GeneOntology',
        'fold_fold': 'FoldClassification',
    }

    subdir = processed_dir_map[task_name]
    processed_path = Path(data_dir) / subdir / 'processed'

    def _pt_loads(fp):
        """Cheap integrity gate: a present-but-corrupt cache (truncated/bad
        zip member) must NOT be treated as ready, or load_dataset_splits
        crashes with a cryptic PytorchStreamReader error and the rebuild
        below would re-trigger an offline structure download."""
        if not fp.exists():
            return False
        try:
            torch.load(fp, weights_only=False)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"Cached {fp} is unreadable ({type(e).__name__}); regenerating.")
            return False

    if processed_path.exists() and all(
        _pt_loads(processed_path / f'{sp}.pt') for sp in ('train', 'val', 'test')
    ):
        print(f"Processed data found at {processed_path}, skipping preprocessing.")
        return

    processed_path.mkdir(parents=True, exist_ok=True)
    print(f"Preprocessing {task_name} from ProteinWorkshop datamodules...")

    # Import ProteinWorkshop components
    sys.path.insert(0, '/workspace/ProteinWorkshop')
    os.environ['PROTEIN_WORKSHOP_DATA_DIR'] = data_dir

    if task_name == 'ec_reaction':
        from proteinworkshop.datasets.ec_reaction import EnzymeCommissionReactionDataset
        dm = EnzymeCommissionReactionDataset(
            path=str(Path(data_dir) / 'ECReaction'),
            pdb_dir=str(Path(data_dir) / 'pdb'),
            format='pdb',
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            dataset_fraction=1.0,
            shuffle_labels=False,
        )
    elif task_name == 'go_bp':
        from proteinworkshop.datasets.go import GeneOntologyDataset
        dm = GeneOntologyDataset(
            path=str(Path(data_dir) / 'GeneOntology'),
            pdb_dir=str(Path(data_dir) / 'pdb'),
            format='pdb',
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            dataset_fraction=1.0,
            shuffle_labels=False,
            split='BP',
        )
    elif task_name == 'fold_fold':
        from proteinworkshop.datasets.fold_classification import FoldClassificationDataModule
        dm = FoldClassificationDataModule(
            path=str(Path(data_dir) / 'FoldClassification'),
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            dataset_fraction=1.0,
            shuffle_labels=False,
            split='fold',
        )
    else:
        raise ValueError(f"Unknown task: {task_name}")

    dm.setup(stage='fit')
    dm.setup(stage='test')

    # Convert to simple Data objects
    three_to_idx = {}
    for i, aa in enumerate(AMINO_ACIDS):
        three_to_idx[aa] = i

    def convert_batch(dataset_or_loader, split_name):
        data_list = []
        loader = DataLoader(dataset_or_loader, batch_size=1, shuffle=False, num_workers=0)
        skipped = 0
        for batch in loader:
            try:
                # Extract alpha-carbon positions
                if hasattr(batch, 'coords'):
                    # coords shape: (1, L, atoms_per_residue, 3) — take CA (index 1)
                    if batch.coords.dim() == 4:
                        pos = batch.coords[0, :, 1, :]  # CA atom
                    elif batch.coords.dim() == 3:
                        pos = batch.coords[0]
                    else:
                        pos = batch.pos if hasattr(batch, 'pos') else None
                elif hasattr(batch, 'pos'):
                    pos = batch.pos
                else:
                    skipped += 1
                    continue

                if pos is None or pos.size(0) < 4:
                    skipped += 1
                    continue

                # Amino acid indices
                if hasattr(batch, 'residue_type'):
                    aa_idx = batch.residue_type.long()
                    if aa_idx.dim() > 1:
                        aa_idx = aa_idx[0]
                elif hasattr(batch, 'x') and batch.x is not None:
                    # One-hot encoded residue features
                    if batch.x.dim() > 1 and batch.x.size(-1) >= 20:
                        aa_idx = batch.x[:, :20].argmax(dim=-1)
                    else:
                        aa_idx = torch.zeros(pos.size(0), dtype=torch.long)
                else:
                    aa_idx = torch.zeros(pos.size(0), dtype=torch.long)

                # Ensure correct shapes
                if pos.dim() != 2 or pos.size(-1) != 3:
                    skipped += 1
                    continue
                L = pos.size(0)
                if aa_idx.size(0) != L:
                    aa_idx = aa_idx[:L] if aa_idx.size(0) > L else F.pad(aa_idx, (0, L - aa_idx.size(0)))

                # Labels
                if hasattr(batch, 'graph_y'):
                    y = batch.graph_y
                elif hasattr(batch, 'y'):
                    y = batch.y
                else:
                    skipped += 1
                    continue

                if y.dim() > 1:
                    y = y.squeeze(0)

                data = Data(
                    pos=pos.float(),
                    aa_idx=aa_idx.clamp(0, NUM_AMINO_ACIDS - 1),
                    y=y,
                    num_nodes=L,
                )
                data_list.append(data)
            except Exception as e:
                skipped += 1
                continue

        print(f"  {split_name}: {len(data_list)} proteins processed, {skipped} skipped")
        return data_list

    if hasattr(dm, 'train_dataset'):
        train_data = convert_batch(dm.train_dataset(), 'train')
    else:
        train_data = convert_batch(dm.train_dataloader().dataset, 'train')

    if hasattr(dm, 'val_dataset'):
        val_data = convert_batch(dm.val_dataset(), 'val')
    else:
        val_data = convert_batch(dm.val_dataloader().dataset, 'val')

    if hasattr(dm, 'test_dataset'):
        test_data = convert_batch(dm.test_dataset('test_0.95') if task_name == 'go_bp' else dm.test_dataset(), 'test')  # GO test_dataset needs a cutoff split (test_0.95); EC/Fold take none
    else:
        test_dl = dm.test_dataloader()
        if isinstance(test_dl, list):
            test_data = convert_batch(test_dl[0].dataset, 'test')
        else:
            test_data = convert_batch(test_dl.dataset, 'test')

    torch.save(train_data, processed_path / 'train.pt')
    torch.save(val_data, processed_path / 'val.pt')
    torch.save(test_data, processed_path / 'test.pt')
    print(f"Saved processed data to {processed_path}")


def compute_f1_max(pred, target):
    """Compute protein-centric F1 Max (optimal threshold F1).

    Matches the ProteinWorkshop reference implementation: enumerates all
    possible thresholds and picks the one with the maximal F1 score.

    Args:
        pred: (B, C) raw logits
        target: (B, C) binary targets
    Returns:
        f1_max: scalar float
    """
    # Apply sigmoid for multilabel tasks (each class independent)
    pred = torch.sigmoid(pred)

    if target.ndim == 1:
        target = F.one_hot(target.long(), num_classes=pred.shape[1]).float()

    order = pred.argsort(descending=True, dim=1)
    target_sorted = target.gather(1, order).int()
    precision = target_sorted.cumsum(1) / torch.ones_like(target_sorted).cumsum(1)
    recall = target_sorted.cumsum(1) / (target_sorted.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(target_sorted).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)

    all_order = pred.flatten().argsort(descending=True)
    order_flat = (
        order
        + torch.arange(order.shape[0], device=order.device).unsqueeze(1)
        * order.shape[1]
    )
    order_flat = order_flat.flatten()
    inv_order = torch.zeros_like(order_flat)
    inv_order[order_flat] = torch.arange(order_flat.shape[0], device=order_flat.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision = precision.flatten()
    recall = recall.flatten()
    all_precision = precision[all_order] - torch.where(
        is_start, torch.zeros_like(precision[all_order]), precision[all_order - 1]
    )
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - torch.where(
        is_start, torch.zeros_like(recall[all_order]), recall[all_order - 1]
    )
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = (
        2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    )
    return all_f1.max().item()


def compute_accuracy(logits, targets, task_type):
    """Compute accuracy metric appropriate for the task type."""
    if task_type == 'multiclass':
        preds = logits.argmax(dim=-1)
        return (preds == targets).float().mean().item()
    else:  # multilabel
        # Use threshold-sweep F1 max (protein-centric, matches ProteinWorkshop)
        return compute_f1_max(logits, targets)


def compute_loss(logits, targets, task_type):
    """Compute appropriate loss for task type."""
    if task_type == 'multiclass':
        return F.cross_entropy(logits, targets.long())
    else:  # multilabel
        return F.binary_cross_entropy_with_logits(logits, targets.float())


def train_epoch(encoder, head, loader, optimizer, device, task_type):
    encoder.train()
    head.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        node_feat = compute_node_features(batch.pos, batch.aa_idx, batch.batch)
        _, graph_emb = encoder(batch.pos, node_feat, batch.batch)
        logits = head(graph_emb)

        # Reshape y: PyG concatenates y tensors along dim=0
        targets = batch.y
        if targets.dim() == 1 and targets.size(0) != logits.size(0):
            targets = targets.view(logits.size(0), -1)
        # For multiclass with one-hot targets, convert to class indices
        if task_type == 'multiclass' and targets.dim() == 2:
            targets = targets.argmax(dim=-1)

        loss = compute_loss(logits, targets, task_type)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(head.parameters()), 1.0
        )
        optimizer.step()

        total_loss += loss.item() * logits.size(0)
        total_correct += compute_accuracy(logits.detach(), targets, task_type) * logits.size(0)
        total_samples += logits.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(encoder, head, loader, device, task_type):
    encoder.eval()
    head.eval()
    total_loss = 0
    total_samples = 0

    if task_type == 'multilabel':
        # Collect all predictions/targets for proper f1_max over the full set
        all_logits = []
        all_targets = []

    all_correct = 0

    for batch in loader:
        batch = batch.to(device)
        node_feat = compute_node_features(batch.pos, batch.aa_idx, batch.batch)
        _, graph_emb = encoder(batch.pos, node_feat, batch.batch)
        logits = head(graph_emb)

        # Reshape y: PyG concatenates y tensors along dim=0
        targets = batch.y
        if targets.dim() == 1 and targets.size(0) != logits.size(0):
            targets = targets.view(logits.size(0), -1)
        # For multiclass with one-hot targets, convert to class indices
        if task_type == 'multiclass' and targets.dim() == 2:
            targets = targets.argmax(dim=-1)

        loss = compute_loss(logits, targets, task_type)
        total_loss += loss.item() * logits.size(0)
        total_samples += logits.size(0)

        if task_type == 'multilabel':
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
        else:
            preds = logits.argmax(dim=-1)
            all_correct += (preds == targets).float().sum().item()

    avg_loss = total_loss / max(total_samples, 1)

    if task_type == 'multilabel':
        # Compute f1_max over the entire dataset
        cat_logits = torch.cat(all_logits, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)
        avg_acc = compute_f1_max(cat_logits, cat_targets)
    else:
        avg_acc = all_correct / max(total_samples, 1)

    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description='Protein Structure Representation Learning')
    parser.add_argument('--task', type=str, required=True,
                        choices=['ec_reaction', 'go_bp', 'fold_fold'],
                        help='Downstream task name')
    parser.add_argument('--data-dir', type=str,
                        default=os.environ.get('PROTEIN_WORKSHOP_DATA_DIR', '/data/ProteinWorkshop'),
                        help='Base data directory')
    parser.add_argument('--output-dir', type=str,
                        default=os.environ.get('OUTPUT_DIR', './output'),
                        help='Output directory')
    parser.add_argument('--seed', type=int,
                        default=int(os.environ.get('SEED', '42')),
                        help='Random seed')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--out-dim', type=int, default=128,
                        help='Output embedding dimension')
    parser.add_argument('--num-layers', type=int, default=6,
                        help='Number of GNN layers')
    args = parser.parse_args()

    # CONFIG_OVERRIDES: override training hyperparameters for your method.
    # Allowed keys: learning_rate, epochs.
    CONFIG_OVERRIDES = {}

    for _k, _v in CONFIG_OVERRIDES.items():
        if _k == 'learning_rate': args.lr = _v
        elif _k == 'epochs': args.epochs = _v

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Task: {args.task}, Seed: {args.seed}")

    # Preprocess data if needed
    preprocess_and_cache(args.task, args.data_dir)

    # Load data
    splits, num_classes, task_type = load_dataset_splits(args.task, args.data_dir)
    print(f"Dataset loaded: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    print(f"Num classes: {num_classes}, Task type: {task_type}")

    train_loader = DataLoader(
        ProteinGraphDataset(splits['train']),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_protein_graphs,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        ProteinGraphDataset(splits['val']),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_protein_graphs,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        ProteinGraphDataset(splits['test']),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_protein_graphs,
        num_workers=4,
        pin_memory=True,
    )

    # Build model
    encoder = ProteinEncoder(
        input_dim=SCALAR_NODE_DIM,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        num_layers=args.num_layers,
        dropout=0.1,
    ).to(device)

    head = ClassificationHead(
        in_dim=args.out_dim,
        num_classes=num_classes,
        hidden_dim=256,
        task_type=task_type,
    ).to(device)

    total_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in head.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_acc = -1.0
    os.makedirs(args.output_dir, exist_ok=True)
    # Use task-specific checkpoint name to avoid collision when tasks run in parallel
    ckpt_filename = f'best_model_{args.task}.pt'

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(encoder, head, train_loader, optimizer, device, task_type)
        val_loss, val_acc = evaluate(encoder, head, val_loader, device, task_type)
        scheduler.step()

        print(
            f"TRAIN_METRICS epoch={epoch} train_loss={train_loss:.6f} "
            f"train_acc={train_acc:.6f} val_loss={val_loss:.6f} val_acc={val_acc:.6f}",
            flush=True,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'encoder': encoder.state_dict(),
                'head': head.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
            }, os.path.join(args.output_dir, ckpt_filename))

    # Evaluate best model on test set
    best_model_path = os.path.join(args.output_dir, ckpt_filename)
    if not os.path.exists(best_model_path):
        # Save current model as fallback
        torch.save({
            'encoder': encoder.state_dict(),
            'head': head.state_dict(),
            'epoch': args.epochs,
            'val_acc': 0.0,
        }, best_model_path)
    ckpt = torch.load(best_model_path, weights_only=False)
    encoder.load_state_dict(ckpt['encoder'])
    head.load_state_dict(ckpt['head'])
    print(f"Loaded best model from epoch {ckpt['epoch']} with val_acc={ckpt['val_acc']:.6f}")

    test_loss, test_acc = evaluate(encoder, head, test_loader, device, task_type)

    metric_name = 'f1_max' if task_type == 'multilabel' else 'accuracy'
    print(f"TEST_METRICS {metric_name}={test_acc:.6f}", flush=True)
    print(f"TEST_METRICS test_loss={test_loss:.6f}", flush=True)

    # Save results
    results = {
        'task': args.task,
        'seed': args.seed,
        'best_epoch': ckpt['epoch'],
        'best_val_acc': ckpt['val_acc'],
        'test_acc': test_acc,
        'test_loss': test_loss,
        'total_params': total_params,
    }
    results_filename = f'results_{args.task}.json'
    with open(os.path.join(args.output_dir, results_filename), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output_dir}/{results_filename}")


if __name__ == '__main__':
    main()
