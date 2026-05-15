"""
Protein Mutation Effect Prediction — Self-contained template.
Predicts DMS fitness scores from frozen ESM-2 embeddings using a supervised model.
Evaluated on ProteinGym DMS assays via Spearman correlation.

Structure:
  Lines 1-107:   FIXED — Imports, data loading, CV fold utilities
  Lines 108-137: EDITABLE — MutationPredictor class (starter: ridge regression)
  Lines 138+:    FIXED — Training loop, evaluation, main
"""
import os
import sys
import math
import argparse
import warnings
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# Constants
# =====================================================================

EMBED_DIM = 1280  # ESM-2 650M embedding dimension


# =====================================================================
# Data loading and CV utilities
# =====================================================================

class DMS_Dataset(Dataset):
    """Dataset for a single DMS assay with precomputed ESM-2 embeddings."""

    def __init__(self, embeddings, scores, wt_embedding, indices=None):
        """
        Args:
            embeddings: [N, EMBED_DIM] ESM-2 mean-pooled embeddings per mutant
            scores: [N] DMS fitness scores
            wt_embedding: [EMBED_DIM] wild-type embedding
            indices: optional subset indices for train/val splits
        """
        if indices is not None:
            self.embeddings = embeddings[indices]
            self.scores = scores[indices]
        else:
            self.embeddings = embeddings
            self.scores = scores
        self.wt_embedding = wt_embedding

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],       # [EMBED_DIM]
            'delta_embedding': self.embeddings[idx] - self.wt_embedding,  # [EMBED_DIM]
            'score': self.scores[idx],                # scalar
        }


def load_dms_data(assay_id, data_dir="/data/esm2_embeddings"):
    """Load precomputed ESM-2 embeddings for a DMS assay."""
    path = os.path.join(data_dir, f"{assay_id}.pt")
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data['embeddings'], data['scores'], data['wt_embedding']


def load_cv_folds(assay_id, cv_dir="/data/proteingym/cv_folds", n_folds=5):
    """Load precomputed random 5-fold CV assignments.
    Returns list of fold indices [0..4] for each mutant.
    Falls back to random assignment if fold file not found.
    """
    import pandas as pd
    fold_col = f"fold_random_{n_folds}"

    # Try to find the fold CSV
    for root, dirs, files in os.walk(cv_dir):
        for fname in files:
            if fname.startswith(assay_id) and fname.endswith('.csv'):
                df = pd.read_csv(os.path.join(root, fname))
                if fold_col in df.columns:
                    # Keep only singles
                    df_singles = df[~df['mutant'].str.contains(':')].reset_index(drop=True)
                    return df_singles[fold_col].values
    # Fallback: random assignment
    return None


def create_random_folds(n_samples, n_folds=5, seed=42):
    """Create random fold assignments."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, n_folds, size=n_samples)


# =====================================================================
# EDITABLE SECTION START — MutationPredictor + helper modules
# =====================================================================

class MutationPredictor(nn.Module):
    """Starter model: Ridge regression (L2-regularized linear model).

    Takes ESM-2 embeddings of mutant sequences and predicts DMS fitness
    scores. This simple linear model serves as a baseline; you should
    design a better prediction architecture.

    The model receives:
      - embedding: [B, 1280] mean-pooled ESM-2 embedding of mutant sequence
      - delta_embedding: [B, 1280] difference from wild-type embedding
    and must return:
      - prediction: [B] predicted fitness scores

    You may use either or both inputs. The delta_embedding highlights
    which residue-level representations changed due to the mutation.
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, embedding, delta_embedding):
        """
        Args:
            embedding: [B, EMBED_DIM] mutant ESM-2 embedding
            delta_embedding: [B, EMBED_DIM] mutant - wildtype embedding
        Returns:
            prediction: [B] predicted fitness scores
        """
        return self.linear(embedding).squeeze(-1)

# =====================================================================
# EDITABLE SECTION END
# =====================================================================


# =====================================================================
# FIXED — Training loop, evaluation, main
# =====================================================================

def collate_fn(batch_list):
    """Collate batch samples."""
    return {
        'embedding': torch.stack([b['embedding'] for b in batch_list]),
        'delta_embedding': torch.stack([b['delta_embedding'] for b in batch_list]),
        'score': torch.stack([b['score'] for b in batch_list]),
    }


def train_epoch(model, loader, optimizer, device, weight_decay=0.01):
    """Train one epoch with MSE loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        emb = batch['embedding'].to(device)
        delta = batch['delta_embedding'].to(device)
        targets = batch['score'].to(device)

        optimizer.zero_grad()
        preds = model(emb, delta)
        loss = F.mse_loss(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model and return Spearman correlation."""
    model.eval()
    all_preds = []
    all_targets = []

    for batch in loader:
        emb = batch['embedding'].to(device)
        delta = batch['delta_embedding'].to(device)
        targets = batch['score']

        preds = model(emb, delta)
        all_preds.append(preds.cpu())
        all_targets.append(targets)

    if not all_preds:
        return 0.0

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    # Spearman correlation
    if len(np.unique(preds)) < 2 or len(np.unique(targets)) < 2:
        return 0.0

    rho, _ = spearmanr(preds, targets)
    return float(rho) if not np.isnan(rho) else 0.0


def train_and_evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    embeddings, scores, wt_embedding = load_dms_data(args.assay_id, args.data_dir)
    n_samples = len(scores)
    print(f"Assay: {args.assay_id}, samples: {n_samples}, embed_dim: {embeddings.shape[1]}")

    # Get CV fold assignments
    folds = load_cv_folds(args.assay_id, args.cv_dir)
    if folds is None:
        print("Using random fold assignment (CV fold file not found)")
        folds = create_random_folds(n_samples, seed=args.seed)
    else:
        # Truncate if sizes don't match (can happen with filtering)
        if len(folds) != n_samples:
            print(f"Warning: fold size ({len(folds)}) != data size ({n_samples}), using random folds")
            folds = create_random_folds(n_samples, seed=args.seed)

    n_folds = 5
    all_test_spearmans = []

    for fold_idx in range(n_folds):
        test_mask = (folds == fold_idx)
        train_mask = ~test_mask

        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        if len(test_indices) == 0 or len(train_indices) == 0:
            continue

        # Further split train into train/val (90/10)
        rng = np.random.RandomState(args.seed + fold_idx)
        rng.shuffle(train_indices)
        val_size = max(1, int(len(train_indices) * 0.1))
        val_indices = train_indices[:val_size]
        actual_train_indices = train_indices[val_size:]

        train_ds = DMS_Dataset(embeddings, scores, wt_embedding, actual_train_indices)
        val_ds = DMS_Dataset(embeddings, scores, wt_embedding, val_indices)
        test_ds = DMS_Dataset(embeddings, scores, wt_embedding, test_indices)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_fn, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=collate_fn)

        # Model
        model = MutationPredictor(embed_dim=EMBED_DIM).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)

        # Training with early stopping
        best_val_spearman = -float('inf')
        best_epoch = 0
        patience_counter = 0
        patience = 20
        best_state = None

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, device,
                                     weight_decay=args.weight_decay)
            val_spearman = evaluate(model, val_loader, device)
            scheduler.step()

            if epoch % 10 == 0 or epoch == 1:
                print(f"TRAIN_METRICS fold={fold_idx} epoch={epoch} "
                      f"loss={train_loss:.6f} val_spearman={val_spearman:.4f}")

            if val_spearman > best_val_spearman:
                best_val_spearman = val_spearman
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping fold {fold_idx} at epoch {epoch}. "
                          f"Best: {best_epoch}")
                    break

        # Load best model and evaluate on test fold
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)
        test_spearman = evaluate(model, test_loader, device)
        all_test_spearmans.append(test_spearman)
        print(f"  Fold {fold_idx}: test_spearman={test_spearman:.4f} "
              f"(best_val={best_val_spearman:.4f} at epoch {best_epoch})")

    # Average across folds
    mean_spearman = float(np.mean(all_test_spearmans))
    std_spearman = float(np.std(all_test_spearmans))
    print(f"\n5-fold CV Results for {args.assay_id}:")
    print(f"  Mean Spearman: {mean_spearman:.4f} +/- {std_spearman:.4f}")
    print(f"TEST_METRICS spearman={mean_spearman:.6f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        'assay_id': args.assay_id,
        'mean_spearman': mean_spearman,
        'std_spearman': std_spearman,
        'fold_spearmans': all_test_spearmans,
    }
    torch.save(results, os.path.join(args.output_dir, 'results.pt'))


def main():
    parser = argparse.ArgumentParser(description="Protein Mutation Effect Prediction")
    parser.add_argument('--assay-id', type=str, required=True,
                        help='DMS assay identifier')
    parser.add_argument('--data-dir', type=str, default='/data/esm2_embeddings',
                        help='Directory with precomputed ESM-2 embeddings')
    parser.add_argument('--cv-dir', type=str,
                        default='/data/proteingym/cv_folds',
                        help='Directory with CV fold assignments')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='./output')
    args = parser.parse_args()

    # =====================================================================
    # EDITABLE SECTION START — CONFIG_OVERRIDES
    # =====================================================================
    # CONFIG_OVERRIDES: override training hyperparameters for your method.
    # Allowed keys: learning_rate, weight_decay.
    CONFIG_OVERRIDES = {}
    # =====================================================================
    # EDITABLE SECTION END
    # =====================================================================

    # =====================================================================
    # FIXED — Apply config overrides and set seeds
    # =====================================================================
    for _k, _v in CONFIG_OVERRIDES.items():
        if _k == 'learning_rate': args.lr = _v
        elif _k == 'weight_decay': args.weight_decay = _v

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_and_evaluate(args)


if __name__ == '__main__':
    main()
