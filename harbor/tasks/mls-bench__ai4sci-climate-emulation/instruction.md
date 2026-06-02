# MLS-Bench: ai4sci-climate-emulation

# Climate Physics Emulation: Neural Network Architecture

## Research Question
Design an improved neural network architecture for emulating sub-grid atmospheric physics processes in climate models. Your architecture should achieve lower Normalized MSE (NMSE) than the default MLP baseline on the ClimSim low-resolution dataset.

## Background
Global climate models divide the atmosphere into grid cells, but many critical physical processes (radiation, convection, cloud formation) occur at scales smaller than these grid cells. Traditionally, these sub-grid processes are approximated by parameterization schemes — handcrafted physics-based approximations. Neural network emulators can learn these mappings from high-resolution simulation data, potentially improving both accuracy and computational efficiency.

ClimSim (Yu et al., "ClimSim: A large multi-scale dataset for hybrid physics-ML climate emulation", NeurIPS 2023 Datasets & Benchmarks; arXiv:2306.08754) provides data from the E3SM-MMF multi-scale climate model, where each sample maps an atmospheric column state to the corresponding sub-grid physics tendencies computed by the high-resolution physics module.

## Task
Modify the `Custom` model class in `custom_emulator.py` to implement a better neural network architecture. The model must:

- Accept `input_dim` and `output_dim` in `__init__`.
- Implement `forward(x)` where `x` has shape `(batch_size, input_dim)`.
- Return predictions of shape `(batch_size, output_dim)`.

## Interface

**Input structure** (556-dim vector per atmospheric column):
- 9 multi-level variables × 60 vertical levels = 540 features:
  temperature (`state_t`), specific humidity (`state_q0001`), cloud ice (`state_q0002`),
  cloud liquid (`state_q0003`), zonal wind (`state_u`), meridional wind (`state_v`),
  ozone (`pbuf_ozone`), methane (`pbuf_CH4`), nitrous oxide (`pbuf_N2O`).
- 16–17 single-level (surface/TOA) scalar variables:
  surface pressure, solar insolation, heat fluxes, wind stress, albedos,
  surface type fractions, snow depths.

**Output structure** (368-dim vector):
- 6 multi-level tendency variables × 60 levels = 360 features:
  temperature tendency (`ptend_t`), humidity tendencies (`ptend_q0001`–`q0003`),
  wind tendencies (`ptend_u`, `ptend_v`).
- 8 single-level diagnostic outputs:
  net shortwave, longwave down, snow/rain precipitation, direct/diffuse solar.

## Fixed Pipeline
Dataset loading, input/output normalization, train/val/test splits, optimizer choice and schedule, loss function, and the multi-budget evaluation harness are all fixed by the scaffold. Only the `Custom` architecture is editable.

## Evaluation
- **Primary metric**: Normalized MSE (NMSE = MSE / Var(target), lower is better).
- **Secondary metrics**: R² (higher is better), RMSE, plus separate `ml_nmse` (multi-level) and `sl_nmse` (single-level) breakdowns.
- **Training budgets**: 30 epochs (short), 100 epochs (medium), 200 epochs (long).
- All three training budgets are run; improvements should be consistent across all three.

## Reference Baselines
- **cnn**: 1D convolutional network with residual blocks operating on vertical atmospheric profiles. Multi-level variables are treated as spatial sequences over 60 vertical levels; single-level scalars are broadcast and concatenated. Inspired by the ClimSim CNN baseline (Yu et al., NeurIPS 2023 D&B).
- **ed**: Encoder-decoder (ClimSim ED baseline). Wide 6-layer fully-connected encoder compresses the 556-dim atmospheric state to a 5-node latent bottleneck, then a symmetric 6-layer decoder expands back to the 368-dim tendency output. Layer widths follow the published ClimSim Table A (768/512/384/256/128/64).
- **unet**: 1D U-Net with ResNet-style blocks over 60 vertical levels. Encoder-decoder with skip connections and self-attention at the bottleneck. Adapted from the ClimSim-style stable-ML-parameterization U-Net (arXiv:2407.00124).
- **hsr**: Heteroskedastic regression (ClimSim HSR baseline). Shared MLP backbone with two output heads predicting mean and log-variance per output dimension, trained with Gaussian NLL loss (Nix & Weigend 1994). Inference returns only the mean.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/ClimSim/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `ClimSim/custom_emulator.py`
- editable lines **86–118**
- editable lines **173–175**


Other files you may **read** for context (do not modify):
- `ClimSim/climsim_utils/data_utils.py`


## Readable Context


### `ClimSim/custom_emulator.py`  [EDITABLE — lines 86–118, lines 173–175 only]

```python
     1: """Custom Climate Physics Emulator
     2: Trained on ClimSim low-resolution E3SM data to predict sub-grid physics tendencies.
     3: 
     4: Input: 556-dim atmospheric state vector (V2 variables)
     5: Output: 368-dim sub-grid physics tendencies
     6: """
     7: 
     8: import math
     9: import os
    10: import time
    11: from dataclasses import dataclass
    12: 
    13: import numpy as np
    14: import torch
    15: import torch.nn as nn
    16: from torch.nn import functional as F
    17: from torch.utils.data import Dataset, DataLoader, Subset
    18: 
    19: # ============================================================================
    20: # Data Configuration
    21: # ============================================================================
    22: 
    23: INPUT_DIM = 556   # V2 input variables (9 multi-level x 60 + 17 single-level - 1)
    24: OUTPUT_DIM = 368  # V2 output variables (6 multi-level x 60 + 8 single-level)
    25: N_LEVELS = 60     # Number of vertical atmospheric levels
    26: 
    27: # Multi-level input variables (each has 60 levels):
    28: #   state_t, state_q0001, state_q0002, state_q0003,
    29: #   state_u, state_v, pbuf_ozone, pbuf_CH4, pbuf_N2O
    30: # Single-level input variables (17 scalar values):
    31: #   state_ps, pbuf_SOLIN, pbuf_LHFLX, pbuf_SHFLX, pbuf_TAUX, pbuf_TAUY,
    32: #   pbuf_COSZRS, cam_in_ALDIF, cam_in_ALDIR, cam_in_ASDIF, cam_in_ASDIR,
    33: #   cam_in_LWUP, cam_in_ICEFRAC, cam_in_LANDFRAC, cam_in_OCNFRAC,
    34: #   cam_in_SNOWHICE, cam_in_SNOWHLAND
    35: #
    36: # Multi-level output variables (each has 60 levels):
    37: #   ptend_t, ptend_q0001, ptend_q0002, ptend_q0003, ptend_u, ptend_v
    38: # Single-level output variables (8 scalar values):
    39: #   cam_out_NETSW, cam_out_FLWDS, cam_out_PRECSC, cam_out_PRECC,
    40: #   cam_out_SOLS, cam_out_SOLL, cam_out_SOLSD, cam_out_SOLLD
    41: 
    42: # ============================================================================
    43: # Dataset
    44: # ============================================================================
    45: 
    46: class ClimSimDataset(Dataset):
    47:     """ClimSim dataset (train-only norm stats; no cross-split leak)."""
    48: 
    49:     def __init__(self, data_dir, split='train'):
    50:         self.inputs = np.load(os.path.join(data_dir, f'{split}_inputs.npy'))
    51:         self.outputs = np.load(os.path.join(data_dir, f'{split}_outputs.npy'))
    52:         self.inp_mean = np.load(os.path.join(data_dir, 'inp_mean.npy'))
    53:         self.inp_std = np.load(os.path.join(data_dir, 'inp_std.npy'))
    54:         self.out_mean = np.load(os.path.join(data_dir, 'out_mean.npy'))
    55:         self.out_std = np.load(os.path.join(data_dir, 'out_std.npy'))
    56:         self.inputs = (self.inputs - self.inp_mean) / self.inp_std
    57:         self.outputs = (self.outputs - self.out_mean) / self.out_std
    58: 
    59:         # Adjust dimensions if needed
    60:         actual_inp_dim = self.inputs.shape[1]
    61:         actual_out_dim = self.outputs.shape[1]
    62:         global INPUT_DIM, OUTPUT_DIM
    63:         INPUT_DIM = actual_inp_dim
    64:         OUTPUT_DIM = actual_out_dim
    65: 
    66:     def __len__(self):
    67:         return len(self.inputs)
    68: 
    69:     def __getitem__(self, idx):
    70:         return (torch.tensor(self.inputs[idx], dtype=torch.float32),
    71:                 torch.tensor(self.outputs[idx], dtype=torch.float32))
    72: 
    73: 
    74: # ================================================================
    75: # EDITABLE REGION — model architecture (lines 86 to 118)
    76: # Modify the Custom model class below. It must:
    77: #   - Accept input_dim and output_dim in __init__
    78: #   - Implement forward(x) -> predictions
    79: #   - Input shape:  (batch_size, input_dim)
    80: #   - Output shape: (batch_size, output_dim)
    81: #   - Anything inside this region may be replaced; keep
    82: #     class name "Custom" so the trainer can find it.
    83: #   - The trainer wraps this model with AdamW + Cosine LR.
    84: #   - Outputs include 360 multi-level + 8 single-level dims.
    85: # ================================================================
    86: class Custom(nn.Module):
    87:     """Neural network for climate physics emulation.
    88: 
    89:     Default: simple 3-layer MLP baseline.
    90:     Replace with a better architecture to improve prediction accuracy.
    91:     """
    92: 
    93:     def __init__(self, input_dim, output_dim):
    94:         super().__init__()
    95:         hidden = 512
    96:         self.net = nn.Sequential(
    97:             nn.Linear(input_dim, hidden),
    98:             nn.ReLU(),
    99:             nn.Linear(hidden, hidden),
   100:             nn.ReLU(),
   101:             nn.Linear(hidden, hidden),
   102:             nn.ReLU(),
   103:             nn.Linear(hidden, output_dim),
   104:         )
   105: 
   106:     def forward(self, x):
   107:         """Forward pass.
   108: 
   109:         Args:
   110:             x: Input tensor of shape (batch_size, input_dim).
   111:                First 9*60=540 values are multi-level variables (9 vars x 60 levels),
   112:                remaining values are single-level (scalar) variables.
   113:         Returns:
   114:             Predictions of shape (batch_size, output_dim).
   115:             First 6*60=360 values are multi-level tendencies,
   116:             last 8 values are single-level outputs.
   117:         """
   118:         return self.net(x)
   119: # ================================================================
   120: # END EDITABLE REGION
   121: # ================================================================
   122: 
   123: # ============================================================================
   124: # Evaluation Metrics
   125: # ============================================================================
   126: 
   127: def compute_nmse(pred, target):
   128:     """Normalized MSE: MSE / Var(target) per variable, averaged over non-constant variables."""
   129:     mse = ((pred - target) ** 2).mean(dim=0)
   130:     var = target.var(dim=0)
   131:     mask = var > 0.01  # skip near-constant dimensions
   132:     if mask.sum() == 0:
   133:         return mse.mean().item()
   134:     nmse = (mse[mask] / var[mask]).mean()
   135:     return nmse.item()
   136: 
   137: 
   138: def compute_r2(pred, target):
   139:     """R² averaged across non-constant output variables."""
   140:     ss_res = ((target - pred) ** 2).sum(dim=0)
   141:     ss_tot = ((target - target.mean(dim=0)) ** 2).sum(dim=0)
   142:     mask = ss_tot > 0.01 * target.shape[0]
   143:     if mask.sum() == 0:
   144:         return 0.0
   145:     r2 = 1 - ss_res[mask] / ss_tot[mask]
   146:     return r2.mean().item()
   147: 
   148: 
   149: def compute_rmse(pred, target):
   150:     """Root Mean Squared Error, averaged across output variables."""
   151:     rmse_per_var = ((pred - target) ** 2).mean(dim=0).sqrt()
   152:     return rmse_per_var.mean().item()
   153: 
   154: 
   155: # ============================================================================
   156: # Training Script
   157: # ============================================================================
   158: 
   159: if __name__ == '__main__':
   160:     # ── Configuration from environment ──
   161:     output_dir = os.environ.get('OUTPUT_DIR', 'out')
   162:     seed = int(os.environ.get('SEED', 42))
   163:     data_dir = os.environ.get('DATA_DIR', '/data/climsim')
   164:     env_label = os.environ.get('ENV', 'default')
   165: 
   166:     # Training hyperparameters
   167:     num_epochs = int(os.environ.get('NUM_EPOCHS', 30))
   168:     batch_size = int(os.environ.get('BATCH_SIZE', 1024))
   169:     learning_rate = float(os.environ.get('LEARNING_RATE', 1e-4))
   170:     weight_decay = float(os.environ.get('WEIGHT_DECAY', 1e-5))
   171:     eval_interval = int(os.environ.get('EVAL_INTERVAL', 1))
   172:     patience = 10
   173:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   174:     # Allowed keys: learning_rate, weight_decay, patience.
   175:     CONFIG_OVERRIDES = {}
   176: 
   177:     # Apply per-method hyperparameter overrides (fixed infrastructure)
   178:     for _k, _v in CONFIG_OVERRIDES.items():
   179:         if _k == 'learning_rate': learning_rate = _v
   180:         elif _k == 'weight_decay': weight_decay = _v
   181:         elif _k == 'patience': patience = _v
   182: 
   183:     # ── Setup ──
   184:     torch.manual_seed(seed)
   185:     np.random.seed(seed)
   186:     device = 'cuda' if torch.cuda.is_available() else 'cpu'
   187:     os.makedirs(output_dir, exist_ok=True)
   188: 
   189:     # ── Load Data ──
   190:     # Train/val/test split:
   191:     #   - train: months 0001-02..03 (used for SGD)
   192:     #   - val:   first half of held-out month 0001-07 (early-stopping & model selection)
   193:     #   - test:  second half of held-out month 0001-07 (final reported metrics)
   194:     # The test half is NEVER touched during training — strict held-out evaluation.
   195:     # If a true `test_inputs.npy` is present on disk we use it instead of splitting.
   196:     print(f"Loading data from {data_dir}...")
   197:     train_dataset = ClimSimDataset(data_dir, split='train')
   198: 
   199:     if os.path.exists(os.path.join(data_dir, 'test_inputs.npy')):
   200:         val_dataset = ClimSimDataset(data_dir, split='val')
   201:         test_dataset = ClimSimDataset(data_dir, split='test')
   202:     else:
   203:         held_out = ClimSimDataset(data_dir, split='val')
   204:         n_held = len(held_out)
   205:         if n_held < 2:
   206:             raise RuntimeError(
   207:                 f"Held-out split has only {n_held} samples; cannot derive a test set. "
   208:                 "Re-run preprocessing to produce a non-empty val split."
   209:             )
   210:         # Deterministic temporally-contiguous split: first half=val, second half=test
   211:         # (NEVER fallback to train, that would leak training data into evaluation)
   212:         split_idx = n_held // 2
   213:         val_indices = list(range(0, split_idx))
   214:         test_indices = list(range(split_idx, n_held))
   215:         val_dataset = Subset(held_out, val_indices)
   216:         test_dataset = Subset(held_out, test_indices)
   217: 
   218:     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
   219:                               num_workers=4, pin_memory=True, drop_last=True)
   220:     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
   221:                             num_workers=4, pin_memory=True)
   222:     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
   223:                              num_workers=4, pin_memory=True)
   224: 
   225:     print(f"Train samples: {len(train_dataset):,}, Val samples: {len(val_dataset):,}, "
   226:           f"Test samples: {len(test_dataset):,}")
   227:     print(f"Input dim: {INPUT_DIM}, Output dim: {OUTPUT_DIM}")
   228:     print(f"Env: {env_label}, Seed: {seed}, Epochs: {num_epochs}")
   229: 
   230:     # ── Model Init ──
   231:     model = Custom(INPUT_DIM, OUTPUT_DIM).to(device)
   232:     n_params = sum(p.numel() for p in model.parameters())
   233:     print(f"Model parameters: {n_params:,}")
   234: 
   235:     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
   236:                                   weight_decay=weight_decay)
   237:     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
   238:     criterion = nn.MSELoss()
   239: 
   240:     # ── Training Loop (early stopping uses VAL only — test stays held-out) ──
   241:     best_val_loss = float('inf')
   242:     patience_counter = 0
   243:     t0 = time.time()
   244: 
   245:     for epoch in range(1, num_epochs + 1):
   246:         model.train()
   247:         train_loss = 0.0
   248:         n_batches = 0
   249: 
   250:         for inputs, targets in train_loader:
   251:             inputs, targets = inputs.to(device), targets.to(device)
   252:             optimizer.zero_grad()
   253:             predictions = model(inputs)
   254:             loss = criterion(predictions, targets)
   255:             loss.backward()
   256:             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
   257:             optimizer.step()
   258:             train_loss += loss.item()
   259:             n_batches += 1
   260: 
   261:         scheduler.step()
   262:         avg_train_loss = train_loss / max(n_batches, 1)
   263: 
   264:         # ── Validation (used for early stopping; NOT for final reporting) ──
   265:         if epoch % eval_interval == 0 or epoch == num_epochs:
   266:             model.eval()
   267:             all_preds, all_targets = [], []
   268:             val_loss = 0.0
   269:             n_val = 0
   270:             with torch.no_grad():
   271:                 for inputs, targets in val_loader:
   272:                     inputs, targets = inputs.to(device), targets.to(device)
   273:                     predictions = model(inputs)
   274:                     val_loss += criterion(predictions, targets).item()
   275:                     all_preds.append(predictions)
   276:                     all_targets.append(targets)
   277:                     n_val += 1
   278: 
   279:             avg_val_loss = val_loss / max(n_val, 1)
   280:             all_preds = torch.cat(all_preds, dim=0)
   281:             all_targets = torch.cat(all_targets, dim=0)
   282: 
   283:             nmse = compute_nmse(all_preds, all_targets)
   284:             r2 = compute_r2(all_preds, all_targets)
   285:             rmse = compute_rmse(all_preds, all_targets)
   286: 
   287:             elapsed = time.time() - t0
   288:             lr_now = scheduler.get_last_lr()[0]
   289:             print(f"Epoch {epoch}/{num_epochs}: train_loss={avg_train_loss:.6f}, "
   290:                   f"val_loss={avg_val_loss:.6f}, nmse={nmse:.6f}, r2={r2:.4f}, "
   291:                   f"rmse={rmse:.6f}, lr={lr_now:.6f}, time={elapsed:.1f}s")
   292:             print(f"TRAIN_METRICS: epoch={epoch}, train_loss={avg_train_loss:.6f}, "
   293:                   f"val_loss={avg_val_loss:.6f}, nmse={nmse:.6f}, r2={r2:.4f}", flush=True)
   294: 
   295:             if avg_val_loss < best_val_loss:
   296:                 best_val_loss = avg_val_loss
   297:                 patience_counter = 0
   298:                 torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
   299:             else:
   300:                 patience_counter += eval_interval
   301:                 if patience_counter >= patience:
   302:                     print(f"Early stopping at epoch {epoch} (patience={patience})")
   303:                     break
   304: 
   305:     # ── Final Evaluation on the held-out TEST split ──
   306:     print("\n=== Final Evaluation (held-out test split) ===")
   307:     model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pt'),
   308:                                      weights_only=True))
   309:     model.eval()
   310:     all_preds, all_targets = [], []
   311:     with torch.no_grad():
   312:         for inputs, targets in test_loader:
   313:             inputs, targets = inputs.to(device), targets.to(device)
   314:             predictions = model(inputs)
   315:             all_preds.append(predictions)
   316:             all_targets.append(targets)
   317: 
   318:     all_preds = torch.cat(all_preds, dim=0)
   319:     all_targets = torch.cat(all_targets, dim=0)
   320: 
   321:     final_nmse = compute_nmse(all_preds, all_targets)
   322:     final_r2 = compute_r2(all_preds, all_targets)
   323:     final_rmse = compute_rmse(all_preds, all_targets)
   324: 
   325:     # Per-group metrics: multi-level tendencies vs single-level outputs
   326:     n_ml_out = 6 * N_LEVELS  # 360
   327:     ml_preds, ml_targets = all_preds[:, :n_ml_out], all_targets[:, :n_ml_out]
   328:     sl_preds, sl_targets = all_preds[:, n_ml_out:], all_targets[:, n_ml_out:]
   329:     ml_nmse = compute_nmse(ml_preds, ml_targets)
   330:     sl_nmse = compute_nmse(sl_preds, sl_targets)
   331: 
   332:     print(f"Final NMSE: {final_nmse:.6f} (ML: {ml_nmse:.6f}, SL: {sl_nmse:.6f})")
   333:     print(f"Final R²: {final_r2:.4f}")
   334:     print(f"Final RMSE: {final_rmse:.6f}")
   335:     print(f"TEST_METRICS: nmse={final_nmse:.6f}, r2={final_r2:.4f}, "
   336:           f"rmse={final_rmse:.6f}, ml_nmse={ml_nmse:.6f}, sl_nmse={sl_nmse:.6f}",
   337:           flush=True)
```

## Parameter Budget

This task enforces a parameter-count cap. Your edits will be rejected if
the resulting model exceeds **1.05×** the strongest
baseline's parameter count. The check runs automatically inside the eval
scripts — you don't need to invoke it.

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `cnn` baseline — editable region  [READ-ONLY — reference implementation]

In `ClimSim/custom_emulator.py`:

```python
Lines 86–152:
    83: #   - The trainer wraps this model with AdamW + Cosine LR.
    84: #   - Outputs include 360 multi-level + 8 single-level dims.
    85: # ================================================================
    86: class Custom(nn.Module):
    87:     """1D CNN with residual blocks for climate emulation.
    88: 
    89:     Reshapes input into (n_vars, n_levels) for convolution over vertical profiles,
    90:     then projects back to output space.
    91:     """
    92: 
    93:     def __init__(self, input_dim, output_dim):
    94:         super().__init__()
    95:         self.input_dim = input_dim
    96:         self.output_dim = output_dim
    97: 
    98:         # Input structure: 9 multi-level vars x 60 levels = 540, then 16-17 scalars
    99:         self.n_ml_in = 9
   100:         self.n_levels = 60
   101:         self.n_sl_in = input_dim - self.n_ml_in * self.n_levels
   102: 
   103:         # Project scalar inputs to per-level features
   104:         self.scalar_proj = nn.Linear(self.n_sl_in, self.n_levels)
   105: 
   106:         # Conv channels: n_ml_in + 1 (from scalar projection)
   107:         in_channels = self.n_ml_in + 1
   108:         hidden_channels = 128
   109:         n_blocks = 8
   110: 
   111:         # Initial projection
   112:         self.input_conv = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
   113: 
   114:         # Residual blocks
   115:         self.blocks = nn.ModuleList()
   116:         for _ in range(n_blocks):
   117:             self.blocks.append(nn.Sequential(
   118:                 nn.BatchNorm1d(hidden_channels),
   119:                 nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
   120:                 nn.ReLU(),
   121:                 nn.Dropout(0.1),
   122:                 nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
   123:             ))
   124: 
   125:         # Output: multi-level tendencies
   126:         self.n_ml_out = 6
   127:         self.ml_head = nn.Conv1d(hidden_channels, self.n_ml_out, kernel_size=1)
   128: 
   129:         # Output: single-level scalars from pooled features
   130:         self.sl_head = nn.Sequential(
   131:             nn.AdaptiveAvgPool1d(1),
   132:             nn.Flatten(),
   133:             nn.Linear(hidden_channels, 64),
   134:             nn.ReLU(),
   135:             nn.Linear(64, 8),
   136:         )
   137: 
   138:     def forward(self, x):
   139:         B = x.shape[0]
   140:         # Split multi-level and single-level inputs
   141:         ml_in = x[:, :self.n_ml_in * self.n_levels].view(B, self.n_ml_in, self.n_levels)
   142:         sl_in = x[:, self.n_ml_in * self.n_levels:]
   143:         sl_expanded = self.scalar_proj(sl_in).unsqueeze(1)  # (B, 1, 60)
   144:         h = torch.cat([ml_in, sl_expanded], dim=1)  # (B, n_ml_in+1, 60)
   145: 
   146:         h = F.relu(self.input_conv(h))
   147:         for block in self.blocks:
   148:             h = h + block(h)
   149: 
   150:         ml_out = self.ml_head(h).reshape(B, -1)  # (B, 360)
   151:         sl_out = self.sl_head(h)  # (B, 8)
   152:         return torch.cat([ml_out, sl_out], dim=-1)
   153: # ================================================================
   154: # END EDITABLE REGION
   155: # ================================================================

Lines 207–209:
   204:     weight_decay = float(os.environ.get('WEIGHT_DECAY', 1e-5))
   205:     eval_interval = int(os.environ.get('EVAL_INTERVAL', 1))
   206:     patience = 10
   207:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   208:     # Allowed keys: learning_rate, weight_decay, patience.
   209:     CONFIG_OVERRIDES = {}
   210: 
   211:     # Apply per-method hyperparameter overrides (fixed infrastructure)
   212:     for _k, _v in CONFIG_OVERRIDES.items():
```

### `ed` baseline — editable region  [READ-ONLY — reference implementation]

In `ClimSim/custom_emulator.py`:

```python
Lines 86–141:
    83: #   - The trainer wraps this model with AdamW + Cosine LR.
    84: #   - Outputs include 360 multi-level + 8 single-level dims.
    85: # ================================================================
    86: class _EDBlock(nn.Module):
    87:     """FC + LayerNorm + ELU + Dropout, one rung of the encoder/decoder ladder."""
    88:     def __init__(self, in_dim, out_dim, dropout=0.1):
    89:         super().__init__()
    90:         self.net = nn.Sequential(
    91:             nn.Linear(in_dim, out_dim),
    92:             nn.LayerNorm(out_dim),
    93:             nn.ELU(),
    94:             nn.Dropout(p=dropout),
    95:         )
    96: 
    97:     def forward(self, x):
    98:         return self.net(x)
    99: 
   100: 
   101: class Custom(nn.Module):
   102:     """Wide Encoder-Decoder with 5-node latent bottleneck.
   103: 
   104:     Encoder: 6 FC blocks 556 -> 768 -> 512 -> 384 -> 256 -> 128 -> 5
   105:     Latent:  5 nodes (paper-faithful)
   106:     Decoder: 6 FC blocks 5 -> 128 -> 256 -> 384 -> 512 -> 768 -> 368
   107:     """
   108: 
   109:     LATENT_DIM = 5
   110:     ENC_DIMS = [768, 512, 384, 256, 128]   # 6 FC layers (the 6th = projection to LATENT)
   111:     DEC_DIMS = [128, 256, 384, 512, 768]   # mirrors encoder
   112: 
   113:     def __init__(self, input_dim, output_dim):
   114:         super().__init__()
   115:         self.input_dim = input_dim
   116:         self.output_dim = output_dim
   117: 
   118:         # ---- Encoder: 6 FC blocks ending at the 5-node latent ----
   119:         enc_layers = []
   120:         prev = input_dim
   121:         for d in self.ENC_DIMS:
   122:             enc_layers.append(_EDBlock(prev, d, dropout=0.1))
   123:             prev = d
   124:         # 6th FC: projection into the bottleneck (no nonlinearity → linear code)
   125:         enc_layers.append(nn.Linear(prev, self.LATENT_DIM))
   126:         self.encoder = nn.Sequential(*enc_layers)
   127: 
   128:         # ---- Decoder: 6 FC blocks expanding from the 5-node latent ----
   129:         dec_layers = []
   130:         prev = self.LATENT_DIM
   131:         for d in self.DEC_DIMS:
   132:             dec_layers.append(_EDBlock(prev, d, dropout=0.1))
   133:             prev = d
   134:         # 6th FC: projection to output (linear)
   135:         dec_layers.append(nn.Linear(prev, output_dim))
   136:         self.decoder = nn.Sequential(*dec_layers)
   137: 
   138:     def forward(self, x):
   139:         z = self.encoder(x)              # [B, 5]
   140:         y = self.decoder(z)              # [B, output_dim]
   141:         return y
   142: # ================================================================
   143: # END EDITABLE REGION
   144: # ================================================================

Lines 196–198:
   193:     weight_decay = float(os.environ.get('WEIGHT_DECAY', 1e-5))
   194:     eval_interval = int(os.environ.get('EVAL_INTERVAL', 1))
   195:     patience = 10
   196:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   197:     # Allowed keys: learning_rate, weight_decay, patience.
   198:     CONFIG_OVERRIDES = {}
   199: 
   200:     # Apply per-method hyperparameter overrides (fixed infrastructure)
   201:     for _k, _v in CONFIG_OVERRIDES.items():
```

### `unet` baseline — editable region  [READ-ONLY — reference implementation]

In `ClimSim/custom_emulator.py`:

```python
Lines 86–235:
    83: #   - The trainer wraps this model with AdamW + Cosine LR.
    84: #   - Outputs include 360 multi-level + 8 single-level dims.
    85: # ================================================================
    86: class ResBlock1d(nn.Module):
    87:     """1D residual block: GroupNorm + Conv1d + SiLU + Conv1d + skip."""
    88:     def __init__(self, channels, dropout=0.1):
    89:         super().__init__()
    90:         self.norm1 = nn.GroupNorm(min(32, channels // 4), channels)
    91:         self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
    92:         self.norm2 = nn.GroupNorm(min(32, channels // 4), channels)
    93:         self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
    94:         self.drop = nn.Dropout(dropout)
    95:         nn.init.zeros_(self.conv2.weight)
    96:         nn.init.zeros_(self.conv2.bias)
    97: 
    98:     def forward(self, x):
    99:         h = F.silu(self.norm1(x))
   100:         h = self.conv1(h)
   101:         h = self.drop(F.silu(self.norm2(h)))
   102:         h = self.conv2(h)
   103:         return (x + h) * (0.5 ** 0.5)
   104: 
   105: 
   106: class AttnBlock1d(nn.Module):
   107:     """Self-attention over the sequence (level) dimension."""
   108:     def __init__(self, channels, num_heads=4):
   109:         super().__init__()
   110:         self.norm = nn.GroupNorm(min(32, channels // 4), channels)
   111:         self.qkv = nn.Conv1d(channels, channels * 3, 1)
   112:         self.proj = nn.Conv1d(channels, channels, 1)
   113:         self.num_heads = num_heads
   114:         nn.init.zeros_(self.proj.weight)
   115:         nn.init.zeros_(self.proj.bias)
   116: 
   117:     def forward(self, x):
   118:         B, C, L = x.shape
   119:         h = self.norm(x)
   120:         qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, L)
   121:         q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
   122:         # Scaled dot-product attention
   123:         scale = (C // self.num_heads) ** -0.5
   124:         attn = torch.einsum('bhcl,bhcm->bhlm', q, k) * scale
   125:         attn = attn.softmax(dim=-1)
   126:         out = torch.einsum('bhlm,bhcm->bhcl', attn, v)
   127:         out = out.reshape(B, C, L)
   128:         return (x + self.proj(out)) * (0.5 ** 0.5)
   129: 
   130: 
   131: class Custom(nn.Module):
   132:     """1D U-Net for climate physics emulation (adapted from ClimsimUnet v4).
   133: 
   134:     Architecture:
   135:     - Reshape flat [B, 556] -> [B, num_profile_vars + num_scalar_vars, 60]
   136:       (profile vars naturally span 60 levels; scalars broadcast to all levels)
   137:     - Pad to 64 (power of 2) for clean downsampling
   138:     - Encoder: 3 resolution levels with residual blocks + downsampling
   139:     - Bottleneck: residual block + self-attention
   140:     - Decoder: 3 levels with skip connections + upsampling
   141:     - Output projection back to flat [B, 368]
   142:     """
   143:     N_LEVELS = 60
   144:     N_PROFILE_IN = 9   # 9 multi-level input vars
   145:     N_SCALAR_IN = 16   # 16 single-level input vars
   146:     N_PROFILE_OUT = 6  # 6 multi-level output vars
   147:     N_SCALAR_OUT = 8   # 8 single-level output vars
   148: 
   149:     def __init__(self, input_dim, output_dim):
   150:         super().__init__()
   151:         self.input_dim = input_dim
   152:         self.output_dim = output_dim
   153: 
   154:         in_ch = self.N_PROFILE_IN + self.N_SCALAR_IN  # 25 channels
   155:         base_ch = 128
   156: 
   157:         # Encoder
   158:         self.enc_in = nn.Conv1d(in_ch, base_ch, 3, padding=1)
   159:         self.enc1 = nn.ModuleList([ResBlock1d(base_ch) for _ in range(3)])
   160:         self.down1 = nn.Conv1d(base_ch, base_ch * 2, 2, stride=2)  # 64->32
   161:         self.enc2 = nn.ModuleList([ResBlock1d(base_ch * 2) for _ in range(3)])
   162:         self.down2 = nn.Conv1d(base_ch * 2, base_ch * 2, 2, stride=2)  # 32->16
   163: 
   164:         # Bottleneck with attention
   165:         self.mid1 = ResBlock1d(base_ch * 2)
   166:         self.mid_attn = AttnBlock1d(base_ch * 2, num_heads=4)
   167:         self.mid2 = ResBlock1d(base_ch * 2)
   168: 
   169:         # Decoder
   170:         self.up2 = nn.ConvTranspose1d(base_ch * 2, base_ch * 2, 2, stride=2)  # 16->32
   171:         self.dec2 = nn.ModuleList([ResBlock1d(base_ch * 4)] +
   172:                                   [ResBlock1d(base_ch * 4) for _ in range(2)])
   173:         self.dec2_proj = nn.Conv1d(base_ch * 4, base_ch * 2, 1)
   174:         self.up1 = nn.ConvTranspose1d(base_ch * 2, base_ch, 2, stride=2)  # 32->64
   175:         self.dec1 = nn.ModuleList([ResBlock1d(base_ch * 2)] +
   176:                                   [ResBlock1d(base_ch * 2) for _ in range(2)])
   177:         self.dec1_proj = nn.Conv1d(base_ch * 2, base_ch, 1)
   178: 
   179:         # Output
   180:         self.out_norm = nn.GroupNorm(min(32, base_ch // 4), base_ch)
   181:         self.out_conv = nn.Conv1d(base_ch, self.N_PROFILE_OUT + self.N_SCALAR_OUT, 3, padding=1)
   182: 
   183:     def forward(self, x):
   184:         B = x.shape[0]
   185: 
   186:         # Reshape: split profile (9 vars x 60 levels) and scalar (16 vars)
   187:         x_profile = x[:, :self.N_PROFILE_IN * self.N_LEVELS]
   188:         x_scalar = x[:, self.N_PROFILE_IN * self.N_LEVELS:]
   189: 
   190:         x_profile = x_profile.reshape(B, self.N_PROFILE_IN, self.N_LEVELS)  # [B, 9, 60]
   191:         x_scalar = x_scalar.unsqueeze(2).expand(-1, -1, self.N_LEVELS)      # [B, 16, 60]
   192:         h = torch.cat([x_profile, x_scalar], dim=1)  # [B, 25, 60]
   193: 
   194:         # Pad 60 -> 64 for clean 2x downsampling
   195:         h = F.pad(h, (0, 4))  # [B, 25, 64]
   196: 
   197:         # Encoder
   198:         h = self.enc_in(h)
   199:         for block in self.enc1:
   200:             h = block(h)
   201:         skip1 = h  # [B, 128, 64]
   202:         h = self.down1(h)  # [B, 256, 32]
   203:         for block in self.enc2:
   204:             h = block(h)
   205:         skip2 = h  # [B, 256, 32]
   206:         h = self.down2(h)  # [B, 256, 16]
   207: 
   208:         # Bottleneck
   209:         h = self.mid1(h)
   210:         h = self.mid_attn(h)
   211:         h = self.mid2(h)
   212: 
   213:         # Decoder
   214:         h = self.up2(h)  # [B, 256, 32]
   215:         h = torch.cat([h, skip2], dim=1)  # [B, 512, 32]
   216:         for block in self.dec2:
   217:             h = block(h)
   218:         h = self.dec2_proj(h)  # [B, 256, 32]
   219:         h = self.up1(h)  # [B, 128, 64]
   220:         h = torch.cat([h, skip1], dim=1)  # [B, 256, 64]
   221:         for block in self.dec1:
   222:             h = block(h)
   223:         h = self.dec1_proj(h)  # [B, 128, 64]
   224: 
   225:         # Output
   226:         h = self.out_conv(F.silu(self.out_norm(h)))  # [B, 14, 64]
   227: 
   228:         # Remove padding and reshape
   229:         h = h[:, :, :self.N_LEVELS]  # [B, 14, 60]
   230: 
   231:         y_profile = h[:, :self.N_PROFILE_OUT, :].reshape(B, self.N_PROFILE_OUT * self.N_LEVELS)
   232:         y_scalar = h[:, self.N_PROFILE_OUT:, :].mean(dim=2)  # avg over levels
   233:         y_scalar = F.relu(y_scalar)  # non-negative scalar outputs
   234: 
   235:         return torch.cat([y_profile, y_scalar], dim=1)
   236: # ================================================================
   237: # END EDITABLE REGION
   238: # ================================================================

Lines 290–292:
   287:     weight_decay = float(os.environ.get('WEIGHT_DECAY', 1e-5))
   288:     eval_interval = int(os.environ.get('EVAL_INTERVAL', 1))
   289:     patience = 10
   290:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   291:     # Allowed keys: learning_rate, weight_decay, patience.
   292:     CONFIG_OVERRIDES = {}
   293: 
   294:     # Apply per-method hyperparameter overrides (fixed infrastructure)
   295:     for _k, _v in CONFIG_OVERRIDES.items():
```

### `hsr` baseline — editable region  [READ-ONLY — reference implementation]

In `ClimSim/custom_emulator.py`:

```python
Lines 86–176:
    83: #   - The trainer wraps this model with AdamW + Cosine LR.
    84: #   - Outputs include 360 multi-level + 8 single-level dims.
    85: # ================================================================
    86: class _HSRBlock(nn.Module):
    87:     """Shared-backbone block: Linear + LayerNorm + Dropout + ReLU."""
    88:     def __init__(self, in_dim, out_dim, dropout=0.1):
    89:         super().__init__()
    90:         self.net = nn.Sequential(
    91:             nn.Linear(in_dim, out_dim),
    92:             nn.LayerNorm(out_dim),
    93:             nn.Dropout(p=dropout),
    94:             nn.ReLU(),
    95:         )
    96: 
    97:     def forward(self, x):
    98:         return self.net(x)
    99: 
   100: 
   101: class Custom(nn.Module):
   102:     """Heteroskedastic Regression: single shared backbone + twin heads (mu, log_var).
   103: 
   104:     Trained with Gaussian NLL on (mu, log_var). At inference time only mu is
   105:     returned, matching the ClimSim evaluation protocol where reported metrics
   106:     are computed against the predicted mean.
   107:     """
   108: 
   109:     def __init__(self, input_dim, output_dim):
   110:         super().__init__()
   111:         hidden = 768
   112:         n_layers = 5
   113: 
   114:         # Single shared backbone (one set of weights — paper-faithful)
   115:         layers = []
   116:         for i in range(n_layers):
   117:             layers.append(_HSRBlock(
   118:                 input_dim if i == 0 else hidden, hidden, dropout=0.1
   119:             ))
   120:         self.backbone = nn.Sequential(*layers)
   121: 
   122:         # Twin output heads — both branch off the SAME backbone activation
   123:         self.head_mean = nn.Linear(hidden, output_dim)
   124:         self.head_logvar = nn.Linear(hidden, output_dim)
   125: 
   126:         # Stash for the loss-replacement override
   127:         self._last_logvar = None
   128:         self._last_mean = None
   129: 
   130:     def forward(self, x):
   131:         h = self.backbone(x)
   132:         mu = self.head_mean(h)
   133:         log_var = self.head_logvar(h)
   134:         # Numerical stability: clamp log-variance into a sane range
   135:         log_var = torch.clamp(log_var, min=-10.0, max=10.0)
   136:         # Stash for the NLL surrogate (used during training)
   137:         self._last_mean = mu
   138:         self._last_logvar = log_var
   139:         # Return mean for downstream metric computation (NMSE/R2/RMSE on mu)
   140:         return mu
   141: 
   142:     def gaussian_nll(self, mu, log_var, target):
   143:         """Per-element Gaussian NLL averaged over batch and dims."""
   144:         # 0.5 * (log_var + (y-mu)^2 * exp(-log_var)) [+ const]
   145:         precision = torch.exp(-log_var)
   146:         return 0.5 * (log_var + (target - mu) ** 2 * precision).mean()
   147: 
   148: 
   149: # ---------------------------------------------------------------------------
   150: # Loss-replacement: monkey-patch nn.MSELoss so the trainer's
   151: # ``criterion(predictions, targets)`` uses the Gaussian NLL on the model's
   152: # stashed (mu, log_var) when the active model is a heteroskedastic Custom.
   153: # This keeps the editable-region diff minimal (no trainer changes) while
   154: # producing the paper-faithful NLL training objective.
   155: # ---------------------------------------------------------------------------
   156: _OrigMSELoss = nn.MSELoss
   157: 
   158: class _HSRMSELossShim(_OrigMSELoss):
   159:     _active_model = None  # set after model construction below
   160: 
   161:     def forward(self, predictions, target):
   162:         m = _HSRMSELossShim._active_model
   163:         if m is not None and getattr(m, '_last_logvar', None) is not None \
   164:            and m._last_mean is predictions:
   165:             return m.gaussian_nll(m._last_mean, m._last_logvar, target)
   166:         return super().forward(predictions, target)
   167: 
   168: nn.MSELoss = _HSRMSELossShim
   169: 
   170: _OrigCustomInit = Custom.__init__
   171: 
   172: def _patched_init(self, input_dim, output_dim):
   173:     _OrigCustomInit(self, input_dim, output_dim)
   174:     _HSRMSELossShim._active_model = self
   175: 
   176: Custom.__init__ = _patched_init
   177: # ================================================================
   178: # END EDITABLE REGION
   179: # ================================================================

Lines 231–233:
   228:     weight_decay = float(os.environ.get('WEIGHT_DECAY', 1e-5))
   229:     eval_interval = int(os.environ.get('EVAL_INTERVAL', 1))
   230:     patience = 10
   231:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   232:     # Allowed keys: learning_rate, weight_decay, patience.
   233:     CONFIG_OVERRIDES = {}
   234: 
   235:     # Apply per-method hyperparameter overrides (fixed infrastructure)
   236:     for _k, _v in CONFIG_OVERRIDES.items():
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
