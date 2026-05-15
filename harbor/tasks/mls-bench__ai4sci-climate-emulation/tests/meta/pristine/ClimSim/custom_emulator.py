"""Custom Climate Physics Emulator
Trained on ClimSim low-resolution E3SM data to predict sub-grid physics tendencies.

Input: 556-dim atmospheric state vector (V2 variables)
Output: 368-dim sub-grid physics tendencies
"""

import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset

# ============================================================================
# Data Configuration
# ============================================================================

INPUT_DIM = 556   # V2 input variables (9 multi-level x 60 + 17 single-level - 1)
OUTPUT_DIM = 368  # V2 output variables (6 multi-level x 60 + 8 single-level)
N_LEVELS = 60     # Number of vertical atmospheric levels

# Multi-level input variables (each has 60 levels):
#   state_t, state_q0001, state_q0002, state_q0003,
#   state_u, state_v, pbuf_ozone, pbuf_CH4, pbuf_N2O
# Single-level input variables (17 scalar values):
#   state_ps, pbuf_SOLIN, pbuf_LHFLX, pbuf_SHFLX, pbuf_TAUX, pbuf_TAUY,
#   pbuf_COSZRS, cam_in_ALDIF, cam_in_ALDIR, cam_in_ASDIF, cam_in_ASDIR,
#   cam_in_LWUP, cam_in_ICEFRAC, cam_in_LANDFRAC, cam_in_OCNFRAC,
#   cam_in_SNOWHICE, cam_in_SNOWHLAND
#
# Multi-level output variables (each has 60 levels):
#   ptend_t, ptend_q0001, ptend_q0002, ptend_q0003, ptend_u, ptend_v
# Single-level output variables (8 scalar values):
#   cam_out_NETSW, cam_out_FLWDS, cam_out_PRECSC, cam_out_PRECC,
#   cam_out_SOLS, cam_out_SOLL, cam_out_SOLSD, cam_out_SOLLD

# ============================================================================
# Dataset
# ============================================================================

class ClimSimDataset(Dataset):
    """ClimSim dataset (train-only norm stats; no cross-split leak)."""

    def __init__(self, data_dir, split='train'):
        self.inputs = np.load(os.path.join(data_dir, f'{split}_inputs.npy'))
        self.outputs = np.load(os.path.join(data_dir, f'{split}_outputs.npy'))
        self.inp_mean = np.load(os.path.join(data_dir, 'inp_mean.npy'))
        self.inp_std = np.load(os.path.join(data_dir, 'inp_std.npy'))
        self.out_mean = np.load(os.path.join(data_dir, 'out_mean.npy'))
        self.out_std = np.load(os.path.join(data_dir, 'out_std.npy'))
        self.inputs = (self.inputs - self.inp_mean) / self.inp_std
        self.outputs = (self.outputs - self.out_mean) / self.out_std

        # Adjust dimensions if needed
        actual_inp_dim = self.inputs.shape[1]
        actual_out_dim = self.outputs.shape[1]
        global INPUT_DIM, OUTPUT_DIM
        INPUT_DIM = actual_inp_dim
        OUTPUT_DIM = actual_out_dim

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (torch.tensor(self.inputs[idx], dtype=torch.float32),
                torch.tensor(self.outputs[idx], dtype=torch.float32))


# ================================================================
# EDITABLE REGION — model architecture (lines 86 to 118)
# Modify the Custom model class below. It must:
#   - Accept input_dim and output_dim in __init__
#   - Implement forward(x) -> predictions
#   - Input shape:  (batch_size, input_dim)
#   - Output shape: (batch_size, output_dim)
#   - Anything inside this region may be replaced; keep
#     class name "Custom" so the trainer can find it.
#   - The trainer wraps this model with AdamW + Cosine LR.
#   - Outputs include 360 multi-level + 8 single-level dims.
# ================================================================
class Custom(nn.Module):
    """Neural network for climate physics emulation.

    Default: simple 3-layer MLP baseline.
    Replace with a better architecture to improve prediction accuracy.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        hidden = 512
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).
               First 9*60=540 values are multi-level variables (9 vars x 60 levels),
               remaining values are single-level (scalar) variables.
        Returns:
            Predictions of shape (batch_size, output_dim).
            First 6*60=360 values are multi-level tendencies,
            last 8 values are single-level outputs.
        """
        return self.net(x)
# ================================================================
# END EDITABLE REGION
# ================================================================

# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_nmse(pred, target):
    """Normalized MSE: MSE / Var(target) per variable, averaged over non-constant variables."""
    mse = ((pred - target) ** 2).mean(dim=0)
    var = target.var(dim=0)
    mask = var > 0.01  # skip near-constant dimensions
    if mask.sum() == 0:
        return mse.mean().item()
    nmse = (mse[mask] / var[mask]).mean()
    return nmse.item()


def compute_r2(pred, target):
    """R² averaged across non-constant output variables."""
    ss_res = ((target - pred) ** 2).sum(dim=0)
    ss_tot = ((target - target.mean(dim=0)) ** 2).sum(dim=0)
    mask = ss_tot > 0.01 * target.shape[0]
    if mask.sum() == 0:
        return 0.0
    r2 = 1 - ss_res[mask] / ss_tot[mask]
    return r2.mean().item()


def compute_rmse(pred, target):
    """Root Mean Squared Error, averaged across output variables."""
    rmse_per_var = ((pred - target) ** 2).mean(dim=0).sqrt()
    return rmse_per_var.mean().item()


# ============================================================================
# Training Script
# ============================================================================

if __name__ == '__main__':
    # ── Configuration from environment ──
    output_dir = os.environ.get('OUTPUT_DIR', 'out')
    seed = int(os.environ.get('SEED', 42))
    data_dir = os.environ.get('DATA_DIR', '/data/climsim')
    env_label = os.environ.get('ENV', 'default')

    # Training hyperparameters
    num_epochs = int(os.environ.get('NUM_EPOCHS', 30))
    batch_size = int(os.environ.get('BATCH_SIZE', 1024))
    learning_rate = float(os.environ.get('LEARNING_RATE', 1e-4))
    weight_decay = float(os.environ.get('WEIGHT_DECAY', 1e-5))
    eval_interval = int(os.environ.get('EVAL_INTERVAL', 1))
    patience = 10
    # CONFIG_OVERRIDES: override training hyperparameters for your method.
    # Allowed keys: learning_rate, weight_decay, patience.
    CONFIG_OVERRIDES = {}

    # Apply per-method hyperparameter overrides (fixed infrastructure)
    for _k, _v in CONFIG_OVERRIDES.items():
        if _k == 'learning_rate': learning_rate = _v
        elif _k == 'weight_decay': weight_decay = _v
        elif _k == 'patience': patience = _v

    # ── Setup ──
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(output_dir, exist_ok=True)

    # ── Load Data ──
    # Train/val/test split:
    #   - train: months 0001-02..03 (used for SGD)
    #   - val:   first half of held-out month 0001-07 (early-stopping & model selection)
    #   - test:  second half of held-out month 0001-07 (final reported metrics)
    # The test half is NEVER touched during training — strict held-out evaluation.
    # If a true `test_inputs.npy` is present on disk we use it instead of splitting.
    print(f"Loading data from {data_dir}...")
    train_dataset = ClimSimDataset(data_dir, split='train')

    if os.path.exists(os.path.join(data_dir, 'test_inputs.npy')):
        val_dataset = ClimSimDataset(data_dir, split='val')
        test_dataset = ClimSimDataset(data_dir, split='test')
    else:
        held_out = ClimSimDataset(data_dir, split='val')
        n_held = len(held_out)
        if n_held < 2:
            raise RuntimeError(
                f"Held-out split has only {n_held} samples; cannot derive a test set. "
                "Re-run preprocessing to produce a non-empty val split."
            )
        # Deterministic temporally-contiguous split: first half=val, second half=test
        # (NEVER fallback to train, that would leak training data into evaluation)
        split_idx = n_held // 2
        val_indices = list(range(0, split_idx))
        test_indices = list(range(split_idx, n_held))
        val_dataset = Subset(held_out, val_indices)
        test_dataset = Subset(held_out, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    print(f"Train samples: {len(train_dataset):,}, Val samples: {len(val_dataset):,}, "
          f"Test samples: {len(test_dataset):,}")
    print(f"Input dim: {INPUT_DIM}, Output dim: {OUTPUT_DIM}")
    print(f"Env: {env_label}, Seed: {seed}, Epochs: {num_epochs}")

    # ── Model Init ──
    model = Custom(INPUT_DIM, OUTPUT_DIM).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.MSELoss()

    # ── Training Loop (early stopping uses VAL only — test stays held-out) ──
    best_val_loss = float('inf')
    patience_counter = 0
    t0 = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(n_batches, 1)

        # ── Validation (used for early stopping; NOT for final reporting) ──
        if epoch % eval_interval == 0 or epoch == num_epochs:
            model.eval()
            all_preds, all_targets = [], []
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    predictions = model(inputs)
                    val_loss += criterion(predictions, targets).item()
                    all_preds.append(predictions)
                    all_targets.append(targets)
                    n_val += 1

            avg_val_loss = val_loss / max(n_val, 1)
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            nmse = compute_nmse(all_preds, all_targets)
            r2 = compute_r2(all_preds, all_targets)
            rmse = compute_rmse(all_preds, all_targets)

            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch}/{num_epochs}: train_loss={avg_train_loss:.6f}, "
                  f"val_loss={avg_val_loss:.6f}, nmse={nmse:.6f}, r2={r2:.4f}, "
                  f"rmse={rmse:.6f}, lr={lr_now:.6f}, time={elapsed:.1f}s")
            print(f"TRAIN_METRICS: epoch={epoch}, train_loss={avg_train_loss:.6f}, "
                  f"val_loss={avg_val_loss:.6f}, nmse={nmse:.6f}, r2={r2:.4f}", flush=True)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
            else:
                patience_counter += eval_interval
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch} (patience={patience})")
                    break

    # ── Final Evaluation on the held-out TEST split ──
    print("\n=== Final Evaluation (held-out test split) ===")
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pt'),
                                     weights_only=True))
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            all_preds.append(predictions)
            all_targets.append(targets)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    final_nmse = compute_nmse(all_preds, all_targets)
    final_r2 = compute_r2(all_preds, all_targets)
    final_rmse = compute_rmse(all_preds, all_targets)

    # Per-group metrics: multi-level tendencies vs single-level outputs
    n_ml_out = 6 * N_LEVELS  # 360
    ml_preds, ml_targets = all_preds[:, :n_ml_out], all_targets[:, :n_ml_out]
    sl_preds, sl_targets = all_preds[:, n_ml_out:], all_targets[:, n_ml_out:]
    ml_nmse = compute_nmse(ml_preds, ml_targets)
    sl_nmse = compute_nmse(sl_preds, sl_targets)

    print(f"Final NMSE: {final_nmse:.6f} (ML: {ml_nmse:.6f}, SL: {sl_nmse:.6f})")
    print(f"Final R²: {final_r2:.4f}")
    print(f"Final RMSE: {final_rmse:.6f}")
    print(f"TEST_METRICS: nmse={final_nmse:.6f}, r2={final_r2:.4f}, "
          f"rmse={final_rmse:.6f}, ml_nmse={ml_nmse:.6f}, sl_nmse={sl_nmse:.6f}",
          flush=True)
