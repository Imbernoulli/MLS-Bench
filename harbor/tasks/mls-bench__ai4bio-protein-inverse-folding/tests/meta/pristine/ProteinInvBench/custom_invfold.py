"""
Protein Inverse Folding — Self-contained template.
Given backbone structure (N, CA, C, O coordinates), predict amino acid sequence.

Structure:
  Lines 1-75:    FIXED — Imports, constants, data loading, featurization
  Lines 76-230:  EDITABLE — StructureEncoder + decoder (starter: simple MPNN)
  Lines 231+:    FIXED — Training loop, evaluation, metrics
"""
import os
import sys
import json
import math
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---- Constants ----
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
NUM_AA = 20  # 20 standard amino acids
NUM_BB_ATOMS = 4  # N, CA, C, O

def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    """Radial basis function encoding of distances."""
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    return torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)

def _dihedrals(X, eps=1e-7):
    """Compute backbone dihedral angles (phi, psi, omega) from N-CA-C-O coords.
    X: (B, L, 4, 3) — N, CA, C, O coordinates.
    Returns: (B, L, 6) — sin/cos of 3 dihedral angles.
    """
    X_flat = X[:, :, :3, :].reshape(int(X.shape[0]), -1, 3)  # (B, 3L, 3)
    dX = X_flat[:, 1:, :] - X_flat[:, :-1, :]  # (B, 3L-1, 3)
    U = F.normalize(dX, dim=-1)
    u_2 = U[:, :-2, :]
    u_1 = U[:, 1:-1, :]
    u_0 = U[:, 2:, :]
    n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
    n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)
    cos_d = (n_2 * n_1).sum(-1)
    sin_d = (torch.cross(n_2, n_1, dim=-1) * u_1).sum(-1)
    cos_d = cos_d.clamp(-1 + eps, 1 - eps)
    sin_d = sin_d.clamp(-1 + eps, 1 - eps)
    D = torch.stack([cos_d, sin_d], dim=-1)  # (B, 3L-3, 2)
    # Pad to (B, L, 6) — 3 dihedrals per residue
    D = F.pad(D, (0, 0, 1, 2))  # pad the length
    B, N = int(X.shape[0]), int(X.shape[1])
    D = D.reshape(B, -1, 6)[:, :N, :]
    return D

def _orientations(X):
    """Compute local orientation frames from N-CA-C coords.
    Returns forward and binormal unit vectors. (B, L, 6)
    """
    fwd = F.normalize(X[:, 1:, 1, :] - X[:, :-1, 1, :], dim=-1)  # CA-CA
    fwd = F.pad(fwd, (0, 0, 0, 1))
    u = F.normalize(X[:, :, 2, :] - X[:, :, 1, :], dim=-1)  # C-CA
    b = F.normalize(fwd - (fwd * u).sum(-1, keepdim=True) * u, dim=-1)
    return torch.cat([fwd, b], dim=-1)

def knn_graph(X_ca, mask, k=30):
    """Build k-nearest neighbor graph from CA coordinates.
    X_ca: (B, L, 3), mask: (B, L)
    Returns: E_idx (B, L, K), D_neighbors (B, L, K)
    """
    mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B, L, L)
    dX = X_ca.unsqueeze(1) - X_ca.unsqueeze(2)  # (B, L, L, 3)
    D = mask_2D * torch.sqrt((dX ** 2).sum(-1) + 1e-6) + (1 - mask_2D) * 1e6
    D_neighbors, E_idx = torch.topk(D, min(k, int(D.shape[-1])), dim=-1, largest=False)
    return E_idx, D_neighbors

# =====================================================================
# EDITABLE SECTION START — StructureEncoder + InverseFoldingModel
# =====================================================================

class MPNNEncoderLayer(nn.Module):
    """Message Passing Neural Network layer for protein graphs."""

    def __init__(self, hidden_dim, edge_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Edge message network
        self.W_msg = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Node update network
        self.W_node = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_V, h_E, E_idx, mask):
        """
        h_V: (B, L, D) node features
        h_E: (B, L, K, D_e) edge features
        E_idx: (B, L, K) neighbor indices
        mask: (B, L)
        """
        B, L, K = int(E_idx.shape[0]), int(E_idx.shape[1]), int(E_idx.shape[2])
        # Gather neighbor node features
        D = int(h_V.shape[-1])
        h_V_neighbors = torch.gather(
            h_V.unsqueeze(2).expand(-1, -1, K, -1),
            1,
            E_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        )  # (B, L, K, D)
        h_V_expand = h_V.unsqueeze(2).expand_as(h_V_neighbors)
        # Messages
        msg_input = torch.cat([h_V_expand, h_V_neighbors, h_E], dim=-1)
        messages = self.W_msg(msg_input)  # (B, L, K, D)
        # Mask out invalid neighbors
        mask_attend = torch.gather(mask.unsqueeze(2).expand(-1, -1, K), 1,
                                   E_idx.clamp(0, L-1)).unsqueeze(-1)
        messages = messages * mask_attend
        # Aggregate
        agg = messages.sum(dim=2) / (mask_attend.sum(dim=2).clamp(min=1))
        # Update
        h_V = self.norm1(h_V + self.dropout(agg))
        h_V_upd = self.W_node(torch.cat([h_V, agg], dim=-1))
        h_V = self.norm2(h_V + self.dropout(h_V_upd))
        h_V = h_V * mask.unsqueeze(-1)
        return h_V


class StructureEncoder(nn.Module):
    """GNN encoder for protein backbone structure.
    Takes backbone coordinates and produces per-residue embeddings.
    """

    def __init__(self, hidden_dim=128, num_layers=3, k_neighbors=30, dropout=0.1, num_rbf=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.num_rbf = num_rbf

        # Node input: dihedral features (6) + orientation (6)
        node_input_dim = 12
        # Edge input: RBF distance (num_rbf) + direction (3)
        edge_input_dim = num_rbf + 3

        self.node_embed = nn.Linear(node_input_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_input_dim, hidden_dim)

        self.layers = nn.ModuleList([
            MPNNEncoderLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, X, mask):
        """
        X: (B, L, 4, 3) backbone coordinates (N, CA, C, O)
        mask: (B, L) residue mask
        Returns: h_V (B, L, hidden_dim) per-residue encoder embeddings
        """
        B, L = X.shape[0], X.shape[1]
        X_ca = X[:, :, 1, :]  # CA atoms

        # Build KNN graph
        E_idx, D_neighbors = knn_graph(X_ca, mask, self.k_neighbors)
        K = E_idx.shape[2]

        # Node features: dihedrals + orientations
        dihedrals = _dihedrals(X)  # (B, L, 6)
        orientations = _orientations(X)  # (B, L, 6)
        node_feat = torch.cat([dihedrals, orientations], dim=-1)  # (B, L, 12)

        # Edge features: RBF distances + direction vectors
        rbf = _rbf(D_neighbors, device=X.device)  # (B, L, K, num_rbf)
        # Direction vectors to neighbors
        X_ca_neighbors = torch.gather(
            X_ca.unsqueeze(2).expand(-1, -1, K, -1),
            1,
            E_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        )
        direction = F.normalize(X_ca_neighbors - X_ca.unsqueeze(2), dim=-1)
        edge_feat = torch.cat([rbf, direction], dim=-1)  # (B, L, K, num_rbf+3)

        # Embed
        h_V = self.node_embed(node_feat)  # (B, L, D)
        h_E = self.edge_embed(edge_feat)  # (B, L, K, D)

        # Message passing
        for layer in self.layers:
            h_V = layer(h_V, h_E, E_idx, mask)

        return h_V


class InverseFoldingModel(nn.Module):
    """Protein inverse folding model.
    Encoder: StructureEncoder (editable) produces per-residue embeddings.
    Decoder: simple MLP that predicts amino acid logits from encoder output.
    """

    def __init__(self, hidden_dim=128, num_encoder_layers=3, k_neighbors=30,
                 dropout=0.1, num_rbf=16):
        super().__init__()
        self.encoder = StructureEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            k_neighbors=k_neighbors,
            dropout=dropout,
            num_rbf=num_rbf,
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_AA),
        )

    def forward(self, X, mask):
        """
        X: (B, L, 4, 3) backbone coords
        mask: (B, L) residue mask
        Returns: log_probs (B, L, NUM_AA)
        """
        h_V = self.encoder(X, mask)
        logits = self.decoder(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

# =====================================================================
# EDITABLE SECTION END
# =====================================================================


# ---- Data Loading (uses PInvBench datasets directly) ----

def load_dataset(dataset_name, data_root, split, remove_ts=False):
    """Load protein dataset. Returns list of protein dicts."""
    if dataset_name in ('CATH4.2', 'CATH4.3'):
        version = float(dataset_name.replace('CATH', ''))
        subdir = dataset_name.lower()
        path = os.path.join(data_root, subdir)
        from PInvBench.src.datasets.cath_dataset import CATHDataset
        return CATHDataset(path=path, split=split, max_length=500, version=version, removeTS=int(bool(remove_ts)))
    elif dataset_name == 'TS':
        path = os.path.join(data_root, 'ts')
        from PInvBench.src.datasets.ts_dataset import TSDataset
        ds = TSDataset(path=path, split=split)
        # TSDataset bundles ts50.json + ts500.json (~550 prot). Filter to TS50 only.
        return [d for d in ds if d.get('category') == 'ts50']
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def collate_fn(batch):
    """Collate protein dicts into padded tensors."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    B = len(batch)
    lengths = [len(b['seq']) for b in batch]
    L_max = max(lengths)

    X = np.zeros([B, L_max, 4, 3], dtype=np.float32)
    S = np.zeros([B, L_max], dtype=np.int64)
    mask = np.zeros([B, L_max], dtype=np.float32)

    for i, b in enumerate(batch):
        l = len(b['seq'])
        coords = np.stack([b['N'], b['CA'], b['C'], b['O']], axis=1)  # (L, 4, 3)
        # Handle NaN coordinates: replace with 0 and mask out those residues
        nan_mask = np.isnan(coords).any(axis=(1, 2))  # (L,) True if any NaN in residue
        coords = np.nan_to_num(coords, nan=0.0)
        X[i, :l] = coords
        # Convert sequence to indices
        for j, aa in enumerate(b['seq']):
            if aa in ALPHABET:
                S[i, j] = ALPHABET.index(aa)
            else:
                S[i, j] = 0  # unknown -> Ala
        mask[i, :l] = 1.0
        # Zero out mask for residues with NaN coordinates
        mask[i, :l][nan_mask] = 0.0

    X = torch.from_numpy(X)
    S = torch.from_numpy(S)
    mask = torch.from_numpy(mask)
    return {'X': X, 'S': S, 'mask': mask, 'lengths': lengths}


# ---- Training & Evaluation ----

def compute_recovery(log_probs, S, mask):
    """Compute amino acid sequence recovery rate."""
    pred = log_probs.argmax(dim=-1)
    correct = ((pred == S) * mask).sum()
    total = mask.sum()
    return (correct / total.clamp(min=1)).item()


def compute_perplexity(log_probs, S, mask):
    """Compute per-residue perplexity."""
    nll = F.nll_loss(log_probs.permute(0, 2, 1), S, reduction='none')  # (B, L)
    nll = (nll * mask).sum() / mask.sum().clamp(min=1)
    return torch.exp(nll).item()


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, max_steps=None):
    model.train()
    total_loss = 0.0
    total_recovery = 0.0
    n_batches = 0
    for i, batch in enumerate(dataloader):
        if batch is None:
            continue
        if max_steps is not None and i >= max_steps:
            break
        X = batch['X'].to(device)
        S = batch['S'].to(device)
        mask = batch['mask'].to(device)

        log_probs = model(X, mask)
        loss = F.nll_loss(log_probs.permute(0, 2, 1), S, reduction='none')
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_recovery += compute_recovery(log_probs.detach(), S, mask)
        n_batches += 1

        if (i + 1) % 100 == 0:
            avg_loss = total_loss / n_batches
            avg_rec = total_recovery / n_batches
            print(f"TRAIN_METRICS epoch={epoch} step={i+1} loss={avg_loss:.4f} recovery={avg_rec:.4f}", flush=True)

    if n_batches > 0:
        avg_loss = total_loss / n_batches
        avg_rec = total_recovery / n_batches
        print(f"TRAIN_METRICS epoch={epoch} loss={avg_loss:.4f} recovery={avg_rec:.4f}", flush=True)
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, device, label="val"):
    model.eval()
    total_recovery = 0.0
    total_perplexity = 0.0
    n_batches = 0
    for batch in dataloader:
        if batch is None:
            continue
        X = batch['X'].to(device)
        S = batch['S'].to(device)
        mask = batch['mask'].to(device)

        log_probs = model(X, mask)
        total_recovery += compute_recovery(log_probs, S, mask)
        total_perplexity += compute_perplexity(log_probs, S, mask)
        n_batches += 1

    if n_batches == 0:
        return 0.0, float('inf')
    recovery = total_recovery / n_batches
    perplexity = total_perplexity / n_batches
    return recovery, perplexity


def main():
    parser = argparse.ArgumentParser(description="Protein Inverse Folding")
    parser.add_argument('--dataset', default='CATH4.2', choices=['CATH4.2', 'CATH4.3', 'TS'])
    parser.add_argument('--data-root', default='/workspace/data')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--num-encoder-layers', type=int, default=3)
    parser.add_argument('--k-neighbors', type=int, default=30)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='./output')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-train-hours', type=float, default=3.0)
    args = parser.parse_args()

    # CONFIG_OVERRIDES: override training hyperparameters for your method.
    # Allowed keys: learning_rate, dropout, num_encoder_layers, batch_size.
    CONFIG_OVERRIDES = {}

    for _k, _v in CONFIG_OVERRIDES.items():
        if _k == 'learning_rate': args.lr = _v
        elif _k == 'dropout': args.dropout = _v
        elif _k == 'num_encoder_layers': args.num_encoder_layers = _v
        elif _k == 'batch_size': args.batch_size = _v

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    if args.dataset == 'TS':
        # TS is test-only: train/val on CATH4.2 with remove_ts=True (drops
        # proteins listed in CATH4.2/remove.json that overlap TS50/TS500 by
        # sequence identity), test on TS50 (50 prot, filtered from
        # ts50.json + ts500.json combined dataset).
        train_ds = load_dataset('CATH4.2', args.data_root, 'train', remove_ts=True)
        val_ds = load_dataset('CATH4.2', args.data_root, 'valid', remove_ts=True)
        test_ds = load_dataset('TS', args.data_root, 'test')
    else:
        train_ds = load_dataset(args.dataset, args.data_root, 'train')
        val_ds = load_dataset(args.dataset, args.data_root, 'valid')
        test_ds = load_dataset(args.dataset, args.data_root, 'test')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn,
                            pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn,
                             pin_memory=True)

    # Build model
    model = InverseFoldingModel(
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        k_neighbors=args.k_neighbors,
        dropout=args.dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0,
                                  betas=(0.9, 0.98), eps=1e-8)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, three_phase=False
    )

    best_val_recovery = 0.0
    start_time = time.time()
    max_seconds = args.max_train_hours * 3600

    for epoch in range(1, args.epochs + 1):
        elapsed = time.time() - start_time
        if elapsed > max_seconds:
            print(f"Time limit reached ({args.max_train_hours}h). Stopping training.", flush=True)
            break

        train_epoch(model, train_loader, optimizer, scheduler, device, epoch)

        val_recovery, val_perplexity = evaluate(model, val_loader, device, "val")
        print(f"TRAIN_METRICS epoch={epoch} val_recovery={val_recovery:.4f} val_perplexity={val_perplexity:.4f}", flush=True)

        if val_recovery > best_val_recovery:
            best_val_recovery = val_recovery
            # Use dataset-specific checkpoint name to avoid collision when
            # CATH4.2 and CATH4.3 run in parallel with the same OUTPUT_DIR
            ckpt_name = f'best_model_{args.dataset.replace(".", "_")}.pt'
            ckpt_path = os.path.join(args.output_dir, ckpt_name)
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model (recovery={val_recovery:.4f})", flush=True)

    # Load best model and evaluate on test set
    ckpt_name = f'best_model_{args.dataset.replace(".", "_")}.pt'
    best_path = os.path.join(args.output_dir, ckpt_name)
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    test_recovery, test_perplexity = evaluate(model, test_loader, device, "test")
    print(f"TEST_METRICS recovery={test_recovery:.4f}", flush=True)
    print(f"TEST_METRICS perplexity={test_perplexity:.4f}", flush=True)


if __name__ == '__main__':
    main()
