# MLS-Bench: ai4sci-climate-emulation

# Climate Physics Emulation: Neural Network Architecture

## Research Question
Design an improved neural network architecture for emulating sub-grid atmospheric physics processes in climate models.

## Background
Global climate models divide the atmosphere into grid cells, but many critical physical processes (radiation, convection, cloud formation) occur at scales smaller than these grid cells. Traditionally, these sub-grid processes are approximated by parameterization schemes — handcrafted physics-based approximations. Neural network emulators can learn these mappings from high-resolution simulation data, potentially improving both accuracy and computational efficiency.

The data comes from a multi-scale climate model where each sample maps an atmospheric column state to the corresponding sub-grid physics tendencies computed by the high-resolution physics module (Yu et al., "ClimSim: A large multi-scale dataset for hybrid physics-ML climate emulation", NeurIPS 2023 D&B; arXiv:2306.08754).

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
The training and evaluation pipeline (data, normalization, splits, optimizer, schedule, loss, and metrics) is fixed by the harness and not editable. Only the `Custom` architecture is editable.

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
(inclusive, 1-indexed)**. Edits that change code outside these ranges — or creating new files, or
deleting whole files — will cause your submission to be invalid.

The line numbers mark an editable **region**, not a fixed line-count budget: you
may add or remove lines inside it. Only code outside the editable ranges must
stay unchanged.

- `ClimSim/custom_emulator.py`
- editable lines **86–118**
- editable lines **173–175**


Other files you may **read** for context (do not modify):
- `ClimSim/climsim_utils/data_utils.py`


## Readable Context


### `ClimSim/custom_emulator.py`  [EDITABLE — lines 86–118, lines 183–185 only]

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
    56:         self.inputs = np.clip((self.inputs - self.inp_mean) / self.inp_std, -10.0, 10.0)
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
   128:     """Normalized MSE: MSE / Var(target) per variable, averaged over non-constant
   129:     variables, with each dimension capped at 1.0.
   130: 
   131:     The cap makes the score a bounded skill metric in [0, 1]: a per-dimension
   132:     NMSE of 1.0 means "no better than predicting the per-dim mean (climatology)",
   133:     so a handful of near-unpredictable low-variance dims (cloud/momentum
   134:     tendencies ptend_q0002/q0003/u/v) can contribute at most 1.0 each and can
   135:     neither dominate the average nor push it past 1.0. Without the cap those
   136:     dims reach per-dim NMSE 3-7 and make the reported score volatile and
   137:     non-reproducible across seeds/hardware.
   138:     """
   139:     mse = ((pred - target) ** 2).mean(dim=0)
   140:     var = target.var(dim=0)
   141:     mask = var > 0.01  # skip near-constant dimensions
   142:     if mask.sum() == 0:
   143:         return mse.mean().item()
   144:     nmse = (mse[mask] / var[mask]).clamp(max=1.0).mean()
   145:     return nmse.item()
   146: 
   147: 
   148: def compute_r2(pred, target):
   149:     """R² averaged across non-constant output variables."""
   150:     ss_res = ((target - pred) ** 2).sum(dim=0)
   151:     ss_tot = ((target - target.mean(dim=0)) ** 2).sum(dim=0)
   152:     mask = ss_tot > 0.01 * target.shape[0]
   153:     if mask.sum() == 0:
   154:         return 0.0
   155:     r2 = 1 - ss_res[mask] / ss_tot[mask]
   156:     return r2.mean().item()
   157: 
   158: 
   159: def compute_rmse(pred, target):
   160:     """Root Mean Squared Error, averaged across output variables."""
   161:     rmse_per_var = ((pred - target) ** 2).mean(dim=0).sqrt()
   162:     return rmse_per_var.mean().item()
   163: 
   164: 
   165: # ============================================================================
   166: # Training Script
   167: # ============================================================================
   168: 
   169: if __name__ == '__main__':
   170:     # ── Configuration from environment ──
   171:     output_dir = os.environ.get('OUTPUT_DIR', 'out')
   172:     seed = int(os.environ.get('SEED', 42))
   173:     data_dir = os.environ.get('DATA_DIR', '/data/climsim')
   174:     env_label = os.environ.get('ENV', 'default')
   175: 
   176:     # Training hyperparameters
   177:     num_epochs = int(os.environ.get('NUM_EPOCHS', 30))
   178:     batch_size = int(os.environ.get('BATCH_SIZE', 1024))
   179:     learning_rate = float(os.environ.get('LEARNING_RATE', 1e-4))
   180:     weight_decay = float(os.environ.get('WEIGHT_DECAY', 1e-5))
   181:     eval_interval = int(os.environ.get('EVAL_INTERVAL', 1))
   182:     patience = 10
   183:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   184:     # Allowed keys: learning_rate, weight_decay, patience.
   185:     CONFIG_OVERRIDES = {}
   186: 
   187:     # Apply per-method hyperparameter overrides (fixed infrastructure)
   188:     for _k, _v in CONFIG_OVERRIDES.items():
   189:         if _k == 'learning_rate': learning_rate = _v
   190:         elif _k == 'weight_decay': weight_decay = _v
   191:         elif _k == 'patience': patience = _v
   192: 
   193:     # ── Setup ──
   194:     torch.manual_seed(seed)
   195:     np.random.seed(seed)
   196:     device = 'cuda' if torch.cuda.is_available() else 'cpu'
   197:     os.makedirs(output_dir, exist_ok=True)
   198: 
   199:     # ── Load Data ──
   200:     # Train/val/test split (fixed infrastructure, prepared by prepare_data.py):
   201:     #   ClimSim-style temporal holdout — train = simulation year 1, val/test =
   202:     #   year 2 (interleaved). Both span a full annual cycle, so it is an
   203:     #   in-distribution temporal generalization test, not cross-season extrapolation.
   204:     #   - train = SGD; val = early-stopping & model selection (not final metrics)
   205:     #   - test  = held-out, reported metrics (never touched during training)
   206:     print(f"Loading data from {data_dir}...")
   207:     train_dataset = ClimSimDataset(data_dir, split='train')
   208: 
   209:     # prepare_data.py always emits an explicit, independent test split (year-2
   210:     # odd timesteps). If it is missing the data is stale (pre-temporal-holdout),
   211:     # so fail loudly rather than silently scoring on the wrong/old split.
   212:     if not os.path.exists(os.path.join(data_dir, 'test_inputs.npy')):
   213:         raise RuntimeError(
   214:             f"test_inputs.npy missing under {data_dir}: the prepared ClimSim data is "
   215:             "stale. Regenerate with vendor/data_scripts/ClimSim/prepare_data.py (and "
   216:             "rebuild the container image) so val/test are the year-2 timestep splits."
   217:         )
   218:     val_dataset = ClimSimDataset(data_dir, split='val')
   219:     test_dataset = ClimSimDataset(data_dir, split='test')
   220: 
   221:     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
   222:                               num_workers=4, pin_memory=True, drop_last=True)
   223:     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
   224:                             num_workers=4, pin_memory=True)
   225:     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
   226:                              num_workers=4, pin_memory=True)
   227: 
   228:     print(f"Train samples: {len(train_dataset):,}, Val samples: {len(val_dataset):,}, "
   229:           f"Test samples: {len(test_dataset):,}")
   230:     print(f"Input dim: {INPUT_DIM}, Output dim: {OUTPUT_DIM}")
   231:     print(f"Env: {env_label}, Seed: {seed}, Epochs: {num_epochs}")
   232: 
   233:     # ── Model Init ──
   234:     model = Custom(INPUT_DIM, OUTPUT_DIM).to(device)
   235:     n_params = sum(p.numel() for p in model.parameters())
   236:     print(f"Model parameters: {n_params:,}")
   237: 
   238:     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
   239:                                   weight_decay=weight_decay)
   240:     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
   241:     criterion = nn.MSELoss()
   242: 
   243:     # ── Training Loop (early stopping uses VAL only — test stays held-out) ──
   244:     # Select the checkpoint by validation LOSS (smooth MSE), matching the
   245:     # ClimSim reference (which early-stops on val_loss). The per-dimension
   246:     # variance-normalized NMSE is dominated by a handful of near-unpredictable
   247:     # low-variance dims (cloud/momentum tendencies) and is far too volatile to
   248:     # select on: minimizing it over many epochs lands on a different checkpoint
   249:     # run-to-run (seed/cuDNN/host-vs-container), making the reported test metric
   250:     # non-reproducible. val_loss is smooth and its minimum is stable.
   251:     best_val_loss = float('inf')
   252:     patience_counter = 0
   253:     t0 = time.time()
   254: 
   255:     for epoch in range(1, num_epochs + 1):
   256:         model.train()
   257:         train_loss = 0.0
   258:         n_batches = 0
   259: 
   260:         for inputs, targets in train_loader:
   261:             inputs, targets = inputs.to(device), targets.to(device)
   262:             optimizer.zero_grad()
   263:             predictions = model(inputs)
   264:             loss = criterion(predictions, targets)
   265:             loss.backward()
   266:             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
   267:             optimizer.step()
   268:             train_loss += loss.item()
   269:             n_batches += 1
   270: 
   271:         scheduler.step()
   272:         avg_train_loss = train_loss / max(n_batches, 1)
   273: 
   274:         # ── Validation (used for early stopping; NOT for final reporting) ──
   275:         if epoch % eval_interval == 0 or epoch == num_epochs:
   276:             model.eval()
   277:             all_preds, all_targets = [], []
   278:             val_loss = 0.0
   279:             n_val = 0
   280:             with torch.no_grad():
   281:                 for inputs, targets in val_loader:
   282:                     inputs, targets = inputs.to(device), targets.to(device)
   283:                     predictions = model(inputs)
   284:                     val_loss += criterion(predictions, targets).item()
   285:                     all_preds.append(predictions)
   286:                     all_targets.append(targets)
   287:                     n_val += 1
   288: 
   289:             avg_val_loss = val_loss / max(n_val, 1)
   290:             all_preds = torch.cat(all_preds, dim=0)
   291:             all_targets = torch.cat(all_targets, dim=0)
   292: 
   293:             nmse = compute_nmse(all_preds, all_targets)
   294:             r2 = compute_r2(all_preds, all_targets)
   295:             rmse = compute_rmse(all_preds, all_targets)
   296: 
   297:             elapsed = time.time() - t0
   298:             lr_now = scheduler.get_last_lr()[0]
   299:             print(f"Epoch {epoch}/{num_epochs}: train_loss={avg_train_loss:.6f}, "
   300:                   f"val_loss={avg_val_loss:.6f}, nmse={nmse:.6f}, r2={r2:.4f}, "
   301:                   f"rmse={rmse:.6f}, lr={lr_now:.6f}, time={elapsed:.1f}s")
   302:             print(f"TRAIN_METRICS: epoch={epoch}, train_loss={avg_train_loss:.6f}, "
   303:                   f"val_loss={avg_val_loss:.6f}, nmse={nmse:.6f}, r2={r2:.4f}", flush=True)
   304: 
   305:             if avg_val_loss < best_val_loss:
   306:                 best_val_loss = avg_val_loss
   307:                 patience_counter = 0
   308:                 torch.save(model.state_dict(), os.path.join(output_dir, f'best_model_{env_label}.pt'))
   309:             else:
   310:                 patience_counter += 1  # count evaluations, not epochs (budget-independent)
   311:                 if patience_counter >= patience:
   312:                     print(f"Early stopping at epoch {epoch} (patience={patience})")
   313:                     break
   314: 
   315:     # ── Final Evaluation on the held-out TEST split ──
   316:     print("\n=== Final Evaluation (held-out test split) ===")
   317:     model.load_state_dict(torch.load(os.path.join(output_dir, f'best_model_{env_label}.pt'),
   318:                                      weights_only=True))
   319:     model.eval()
   320:     all_preds, all_targets = [], []
   321:     with torch.no_grad():
   322:         for inputs, targets in test_loader:
   323:             inputs, targets = inputs.to(device), targets.to(device)
   324:             predictions = model(inputs)
   325:             all_preds.append(predictions)
   326:             all_targets.append(targets)
   327: 
   328:     all_preds = torch.cat(all_preds, dim=0)
   329:     all_targets = torch.cat(all_targets, dim=0)
   330: 
   331:     final_nmse = compute_nmse(all_preds, all_targets)
   332:     final_r2 = compute_r2(all_preds, all_targets)
   333:     final_rmse = compute_rmse(all_preds, all_targets)
   334: 
   335:     # Per-group metrics: multi-level tendencies vs single-level outputs
   336:     n_ml_out = 6 * N_LEVELS  # 360
   337:     ml_preds, ml_targets = all_preds[:, :n_ml_out], all_targets[:, :n_ml_out]
   338:     sl_preds, sl_targets = all_preds[:, n_ml_out:], all_targets[:, n_ml_out:]
   339:     ml_nmse = compute_nmse(ml_preds, ml_targets)
   340:     sl_nmse = compute_nmse(sl_preds, sl_targets)
   341: 
   342:     print(f"Final NMSE: {final_nmse:.6f} (ML: {ml_nmse:.6f}, SL: {sl_nmse:.6f})")
   343:     print(f"Final R²: {final_r2:.4f}")
   344:     print(f"Final RMSE: {final_rmse:.6f}")
   345:     print(f"TEST_METRICS: nmse={final_nmse:.6f}, r2={final_r2:.4f}, "
   346:           f"rmse={final_rmse:.6f}, ml_nmse={ml_nmse:.6f}, sl_nmse={sl_nmse:.6f}",
   347:           flush=True)
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `cnn` baseline — editable region  [READ-ONLY — reference implementation]

In `ClimSim/custom_emulator.py`:

```python
Lines 86–147:
    83: #   - The trainer wraps this model with AdamW + Cosine LR.
    84: #   - Outputs include 360 multi-level + 8 single-level dims.
    85: # ================================================================
    86: class _CNNResBlock(nn.Module):
    87:     """ClimSim CNN residual block: Conv-ReLU-Drop-Conv-ReLU-Drop + 1x1 skip.
    88: 
    89:     No normalization (reference hp_norm=False)."""
    90:     def __init__(self, channels, kernel=3, dropout=0.175):
    91:         super().__init__()
    92:         pad = kernel // 2
    93:         self.conv1 = nn.Conv1d(channels, channels, kernel, padding=pad)
    94:         self.conv2 = nn.Conv1d(channels, channels, kernel, padding=pad)
    95:         self.skip = nn.Conv1d(channels, channels, 1)
    96:         self.drop = nn.Dropout(dropout)
    97: 
    98:     def forward(self, x):
    99:         h = self.drop(F.relu(self.conv1(x)))
   100:         h = self.drop(F.relu(self.conv2(h)))
   101:         return h + self.skip(x)
   102: 
   103: 
   104: class Custom(nn.Module):
   105:     """1D ResNet CNN over vertical profiles (ClimSim reference: 12 blocks, 406 ch)."""
   106: 
   107:     N_LEVELS = 60
   108:     N_PROFILE_IN = 9
   109:     N_PROFILE_OUT = 6
   110:     N_SCALAR_OUT = 8
   111:     CHANNELS = 406
   112:     N_BLOCKS = 12
   113:     KERNEL = 3
   114:     DROPOUT = 0.175
   115: 
   116:     def __init__(self, input_dim, output_dim):
   117:         super().__init__()
   118:         self.input_dim = input_dim
   119:         self.output_dim = output_dim
   120:         self.n_scalar_in = input_dim - self.N_PROFILE_IN * self.N_LEVELS  # 16
   121:         in_ch = self.N_PROFILE_IN + self.n_scalar_in                      # 25 channels
   122: 
   123:         self.input_conv = nn.Conv1d(in_ch, self.CHANNELS, self.KERNEL, padding=self.KERNEL // 2)
   124:         self.blocks = nn.ModuleList(
   125:             [_CNNResBlock(self.CHANNELS, self.KERNEL, self.DROPOUT) for _ in range(self.N_BLOCKS)]
   126:         )
   127:         # Pre-output ELU projection, then split heads.
   128:         self.out_conv = nn.Conv1d(self.CHANNELS, self.CHANNELS, 1)
   129:         self.ml_head = nn.Conv1d(self.CHANNELS, self.N_PROFILE_OUT, 1)       # linear
   130:         self.sl_head = nn.Sequential(                                        # non-negative scalars
   131:             nn.AdaptiveAvgPool1d(1), nn.Flatten(),
   132:             nn.Linear(self.CHANNELS, self.N_SCALAR_OUT),
   133:         )
   134: 
   135:     def forward(self, x):
   136:         B = x.shape[0]
   137:         ml = x[:, :self.N_PROFILE_IN * self.N_LEVELS].reshape(B, self.N_PROFILE_IN, self.N_LEVELS)
   138:         sl = x[:, self.N_PROFILE_IN * self.N_LEVELS:].unsqueeze(2).expand(-1, -1, self.N_LEVELS)
   139:         h = torch.cat([ml, sl], dim=1)                  # [B, 25, 60]
   140:         h = F.relu(self.input_conv(h))
   141:         for blk in self.blocks:
   142:             h = blk(h)
   143:         h = F.elu(self.out_conv(h))                     # pre-output ELU
   144:         ml_out = self.ml_head(h).reshape(B, -1)         # [B, 360], linear
   145:         # NOTE: outputs are z-normalized (straddle zero) -> scalar head stays linear.
   146:         sl_out = self.sl_head(h)                         # [B, 8]
   147:         return torch.cat([ml_out, sl_out], dim=-1)
   148: # ================================================================
   149: # END EDITABLE REGION
   150: # ================================================================

Lines 212–214:
   209:     weight_decay = float(os.environ.get('WEIGHT_DECAY', 1e-5))
   210:     eval_interval = int(os.environ.get('EVAL_INTERVAL', 1))
   211:     patience = 10
   212:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   213:     # Allowed keys: learning_rate, weight_decay, patience.
   214:     CONFIG_OVERRIDES = {}
   215: 
   216:     # Apply per-method hyperparameter overrides (fixed infrastructure)
   217:     for _k, _v in CONFIG_OVERRIDES.items():
```

### `ed` baseline — editable region  [READ-ONLY — reference implementation]

In `ClimSim/custom_emulator.py`:

```python
Lines 86–117:
    83: #   - The trainer wraps this model with AdamW + Cosine LR.
    84: #   - Outputs include 360 multi-level + 8 single-level dims.
    85: # ================================================================
    86: class Custom(nn.Module):
    87:     """Plain fully-connected Encoder-Decoder with a 5-node latent (ClimSim ED)."""
    88: 
    89:     INTERMEDIATE = 463
    90:     LATENT_DIM = 5
    91:     # intermediate_dim / {1, 1, 2, 4, 8, 16} (floor), matching the reference taper.
    92:     ENC_DIMS = [463, 463, 231, 115, 57, 28]
    93:     DEC_DIMS = [28, 57, 115, 231, 463, 463]
    94: 
    95:     def __init__(self, input_dim, output_dim):
    96:         super().__init__()
    97:         self.input_dim = input_dim
    98:         self.output_dim = output_dim
    99: 
   100:         enc = []
   101:         prev = input_dim
   102:         for d in self.ENC_DIMS:
   103:             enc += [nn.Linear(prev, d), nn.ReLU()]
   104:             prev = d
   105:         enc += [nn.Linear(prev, self.LATENT_DIM), nn.ReLU()]  # ReLU latent (reference)
   106:         self.encoder = nn.Sequential(*enc)
   107: 
   108:         dec = []
   109:         prev = self.LATENT_DIM
   110:         for d in self.DEC_DIMS:
   111:             dec += [nn.Linear(prev, d), nn.ReLU()]
   112:             prev = d
   113:         dec += [nn.Linear(prev, output_dim), nn.ELU()]   # ELU output (reference)
   114:         self.decoder = nn.Sequential(*dec)
   115: 
   116:     def forward(self, x):
   117:         return self.decoder(self.encoder(x))
   118: # ================================================================
   119: # END EDITABLE REGION
   120: # ================================================================

Lines 182–184:
   179:     weight_decay = float(os.environ.get('WEIGHT_DECAY', 1e-5))
   180:     eval_interval = int(os.environ.get('EVAL_INTERVAL', 1))
   181:     patience = 10
   182:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   183:     # Allowed keys: learning_rate, weight_decay, patience.
   184:     CONFIG_OVERRIDES = {}
   185: 
   186:     # Apply per-method hyperparameter overrides (fixed infrastructure)
   187:     for _k, _v in CONFIG_OVERRIDES.items():
```

### `unet` baseline — editable region  [READ-ONLY — reference implementation]

In `ClimSim/custom_emulator.py`:

```python
Lines 86–208:
    83: #   - The trainer wraps this model with AdamW + Cosine LR.
    84: #   - Outputs include 360 multi-level + 8 single-level dims.
    85: # ================================================================
    86: class _UNetResBlock(nn.Module):
    87:     """GroupNorm + SiLU + Conv1d, twice, with a residual connection (no attention)."""
    88:     def __init__(self, channels):
    89:         super().__init__()
    90:         g = min(32, max(1, channels // 4))
    91:         self.norm1 = nn.GroupNorm(g, channels)
    92:         self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
    93:         self.norm2 = nn.GroupNorm(g, channels)
    94:         self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
    95:         nn.init.zeros_(self.conv2.weight); nn.init.zeros_(self.conv2.bias)
    96: 
    97:     def forward(self, x):
    98:         h = self.conv1(F.silu(self.norm1(x)))
    99:         h = self.conv2(F.silu(self.norm2(h)))
   100:         return x + h
   101: 
   102: 
   103: class _UNetAttn(nn.Module):
   104:     """Multi-head self-attention over the vertical-level axis, at the bottleneck
   105:     (the shipped ClimsimUnet has a bottleneck attention block)."""
   106:     def __init__(self, channels, num_heads=4):
   107:         super().__init__()
   108:         self.norm = nn.GroupNorm(min(32, max(1, channels // 4)), channels)
   109:         self.qkv = nn.Conv1d(channels, channels * 3, 1)
   110:         self.proj = nn.Conv1d(channels, channels, 1)
   111:         self.num_heads = num_heads
   112:         nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
   113: 
   114:     def forward(self, x):
   115:         B, C, L = x.shape
   116:         qkv = self.qkv(self.norm(x)).reshape(B, 3, self.num_heads, C // self.num_heads, L)
   117:         q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
   118:         attn = torch.einsum('bhcl,bhcm->bhlm', q, k) * (C // self.num_heads) ** -0.5
   119:         attn = attn.softmax(dim=-1)
   120:         out = torch.einsum('bhlm,bhcm->bhcl', attn, v).reshape(B, C, L)
   121:         return x + self.proj(out)
   122: 
   123: 
   124: class Custom(nn.Module):
   125:     """1D U-Net (ClimsimUnet): depth 4, channels [128,256,256,256], bottleneck attention."""
   126: 
   127:     N_LEVELS = 60
   128:     PAD_LEVELS = 64
   129:     N_PROFILE_IN = 9
   130:     N_SCALAR_IN = 16
   131:     N_PROFILE_OUT = 6
   132:     N_SCALAR_OUT = 8
   133:     CH = [128, 256, 256, 256]   # 4 resolution levels: 64, 32, 16, 8
   134:     ENC_BLOCKS = 4              # reference num_blocks per encoder level
   135:     DEC_BLOCKS = 5             # reference num_blocks + 1 per decoder level
   136: 
   137:     def __init__(self, input_dim, output_dim):
   138:         super().__init__()
   139:         self.input_dim = input_dim
   140:         self.output_dim = output_dim
   141:         in_ch = self.N_PROFILE_IN + self.N_SCALAR_IN     # 25 channels
   142: 
   143:         c0, c1, c2, c3 = self.CH
   144:         mk = lambda c, n: nn.ModuleList([_UNetResBlock(c) for _ in range(n)])
   145: 
   146:         self.enc_in = nn.Conv1d(in_ch, c0, 3, padding=1)
   147:         self.enc0 = mk(c0, self.ENC_BLOCKS)
   148:         self.down0 = nn.Conv1d(c0, c1, 2, stride=2)      # 64 -> 32
   149:         self.enc1 = mk(c1, self.ENC_BLOCKS)
   150:         self.down1 = nn.Conv1d(c1, c2, 2, stride=2)      # 32 -> 16
   151:         self.enc2 = mk(c2, self.ENC_BLOCKS)
   152:         self.down2 = nn.Conv1d(c2, c3, 2, stride=2)      # 16 -> 8
   153:         self.mid = mk(c3, self.ENC_BLOCKS)               # bottleneck
   154:         self.mid_attn = _UNetAttn(c3)                    # bottleneck self-attention
   155: 
   156:         self.up2 = nn.ConvTranspose1d(c3, c2, 2, stride=2)   # 8 -> 16
   157:         self.dec2 = mk(c2, self.DEC_BLOCKS); self.dec2_proj = nn.Conv1d(c2 + c2, c2, 1)
   158:         self.up1 = nn.ConvTranspose1d(c2, c1, 2, stride=2)   # 16 -> 32
   159:         self.dec1 = mk(c1, self.DEC_BLOCKS); self.dec1_proj = nn.Conv1d(c1 + c1, c1, 1)
   160:         self.up0 = nn.ConvTranspose1d(c1, c0, 2, stride=2)   # 32 -> 64
   161:         self.dec0 = mk(c0, self.DEC_BLOCKS); self.dec0_proj = nn.Conv1d(c0 + c0, c0, 1)
   162: 
   163:         self.out_norm = nn.GroupNorm(min(32, max(1, c0 // 4)), c0)
   164:         self.out_conv = nn.Conv1d(c0, self.N_PROFILE_OUT + self.N_SCALAR_OUT, 3, padding=1)
   165: 
   166:     def _run(self, blocks, h):
   167:         for b in blocks:
   168:             h = b(h)
   169:         return h
   170: 
   171:     def forward(self, x):
   172:         B = x.shape[0]
   173:         ml = x[:, :self.N_PROFILE_IN * self.N_LEVELS].reshape(B, self.N_PROFILE_IN, self.N_LEVELS)
   174:         sl = x[:, self.N_PROFILE_IN * self.N_LEVELS:].unsqueeze(2).expand(-1, -1, self.N_LEVELS)
   175:         h = torch.cat([ml, sl], dim=1)                          # [B, 25, 60]
   176:         h = F.pad(h, (0, self.PAD_LEVELS - self.N_LEVELS))      # -> 64
   177: 
   178:         h = self.enc_in(h)
   179:         s0 = self._run(self.enc0, h)                            # [B, c0, 64]
   180:         h = self._run(self.enc1, self.down0(s0)); s1 = h        # [B, c1, 32]
   181:         h = self._run(self.enc2, self.down1(s1)); s2 = h        # [B, c2, 16]
   182:         h = self._run(self.mid, self.down2(s2))                 # [B, c3, 8]
   183:         h = self.mid_attn(h)                                    # bottleneck attention
   184: 
   185:         h = self.dec2_proj(torch.cat([self.up2(h), s2], dim=1))
   186:         h = self._run(self.dec2, h)
   187:         h = self.dec1_proj(torch.cat([self.up1(h), s1], dim=1))
   188:         h = self._run(self.dec1, h)
   189:         h = self.dec0_proj(torch.cat([self.up0(h), s0], dim=1))
   190:         h = self._run(self.dec0, h)
   191: 
   192:         h = self.out_conv(F.silu(self.out_norm(h)))             # [B, 14, 64]
   193:         h = h[:, :, :self.N_LEVELS]                             # [B, 14, 60]
   194:         y_ml = h[:, :self.N_PROFILE_OUT, :].reshape(B, -1)      # [B, 360]
   195:         # outputs are z-normalized (straddle zero) -> scalar head stays linear.
   196:         y_sl = h[:, self.N_PROFILE_OUT:, :].mean(dim=2)         # [B, 8]
   197:         return torch.cat([y_ml, y_sl], dim=1)
   198: 
   199: 
   200: # Reference trains with Huber loss (delta=1); replace the trainer's MSELoss with
   201: # a Huber loss. Use a proper subclass of the canonical MSELoss (not a lambda) so
   202: # that, if all baselines are imported into one process (e.g. budget_check.py),
   203: # a later baseline that subclasses nn.MSELoss still works.
   204: class _UNetHuberLoss(torch.nn.modules.loss.MSELoss):
   205:     def forward(self, pred, target):
   206:         return F.huber_loss(pred, target, delta=1.0)
   207: 
   208: nn.MSELoss = _UNetHuberLoss
   209: # ================================================================
   210: # END EDITABLE REGION
   211: # ================================================================

Lines 273–275:
   270:     weight_decay = float(os.environ.get('WEIGHT_DECAY', 1e-5))
   271:     eval_interval = int(os.environ.get('EVAL_INTERVAL', 1))
   272:     patience = 10
   273:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   274:     # Allowed keys: learning_rate, weight_decay, patience.
   275:     CONFIG_OVERRIDES = {}
   276: 
   277:     # Apply per-method hyperparameter overrides (fixed infrastructure)
   278:     for _k, _v in CONFIG_OVERRIDES.items():
```

### `hsr` baseline — editable region  [READ-ONLY — reference implementation]

In `ClimSim/custom_emulator.py`:

```python
Lines 86–168:
    83: #   - The trainer wraps this model with AdamW + Cosine LR.
    84: #   - Outputs include 360 multi-level + 8 single-level dims.
    85: # ================================================================
    86: class _HSRNet(nn.Module):
    87:     """One MLP head: `layers` x (Linear -> LayerNorm -> Dropout -> ReLU) + Linear."""
    88:     def __init__(self, in_dim, out_dim, hidden=1024, layers=4, dropout=0.0):
    89:         super().__init__()
    90:         blocks = []
    91:         prev = in_dim
    92:         for _ in range(layers):
    93:             blocks += [nn.Linear(prev, hidden), nn.LayerNorm(hidden), nn.Dropout(dropout), nn.ReLU()]
    94:             prev = hidden
    95:         blocks.append(nn.Linear(prev, out_dim))
    96:         self.net = nn.Sequential(*blocks)
    97: 
    98:     def forward(self, x):
    99:         return self.net(x)
   100: 
   101: 
   102: class Custom(nn.Module):
   103:     """Heteroskedastic regression: separate mean and log-precision networks."""
   104: 
   105:     def __init__(self, input_dim, output_dim):
   106:         super().__init__()
   107:         self.input_dim = input_dim
   108:         self.output_dim = output_dim
   109:         self.mean = _HSRNet(input_dim, output_dim, hidden=1024, layers=4, dropout=0.0)
   110:         self.logprec = _HSRNet(input_dim, output_dim, hidden=1024, layers=4, dropout=0.0)
   111:         # Epoch tracking for the MSE -> NLL warm-up (reference: first epochs/3).
   112:         self._epoch = 0
   113:         try:
   114:             # reference switches at epoch < epochs/3 (0-indexed) -> ceil MSE epochs
   115:             self._warmup = max(1, math.ceil(int(os.environ.get('NUM_EPOCHS', 30)) / 3))
   116:         except Exception:
   117:             self._warmup = 1
   118:         self._last_mean = None
   119:         self._last_logprec = None
   120: 
   121:     def train(self, mode=True):
   122:         # The trainer calls model.train() exactly once at the start of each epoch
   123:         # (model.eval() only runs every EVAL_INTERVAL epochs, so a False->True edge
   124:         # is unreliable). Count every train(True) call as one epoch.
   125:         if mode:
   126:             self._epoch += 1
   127:         return super().train(mode)
   128: 
   129:     def forward(self, x):
   130:         mu = self.mean(x)
   131:         logprec = torch.clamp(self.logprec(x), min=-10.0, max=10.0)
   132:         self._last_mean = mu
   133:         self._last_logprec = logprec
   134:         return mu  # inference / metrics use the mean
   135: 
   136:     def hsr_loss(self, mu, logprec, target):
   137:         if self._epoch <= self._warmup:           # MSE warm-up (first 1/3)
   138:             return ((target - mu) ** 2).mean()
   139:         prec = torch.exp(logprec)                 # tau
   140:         nll = (prec * (target - mu) ** 2 - logprec).mean()
   141:         return torch.clamp(nll, min=-1e5, max=1e5)
   142: 
   143: 
   144: # --- inject the HSR objective by overriding the trainer's MSELoss ------------
   145: # Subclass the canonical MSELoss (torch.nn.modules.loss), not nn.MSELoss, which
   146: # another baseline's edit may have rebound when all baselines are imported into
   147: # one process (e.g. budget_check.py).
   148: _OrigMSELoss = torch.nn.modules.loss.MSELoss
   149: 
   150: class _HSRMSELossShim(_OrigMSELoss):
   151:     _active_model = None
   152: 
   153:     def forward(self, predictions, target):
   154:         m = _HSRMSELossShim._active_model
   155:         if m is not None and getattr(m, '_last_logprec', None) is not None \
   156:            and m._last_mean is predictions:
   157:             return m.hsr_loss(m._last_mean, m._last_logprec, target)
   158:         return super().forward(predictions, target)
   159: 
   160: nn.MSELoss = _HSRMSELossShim
   161: 
   162: _OrigCustomInit = Custom.__init__
   163: 
   164: def _patched_init(self, input_dim, output_dim):
   165:     _OrigCustomInit(self, input_dim, output_dim)
   166:     _HSRMSELossShim._active_model = self
   167: 
   168: Custom.__init__ = _patched_init
   169: # ================================================================
   170: # END EDITABLE REGION
   171: # ================================================================

Lines 233–235:
   230:     weight_decay = float(os.environ.get('WEIGHT_DECAY', 1e-5))
   231:     eval_interval = int(os.environ.get('EVAL_INTERVAL', 1))
   232:     patience = 10
   233:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   234:     # Allowed keys: learning_rate, weight_decay, patience.
   235:     CONFIG_OVERRIDES = {}
   236: 
   237:     # Apply per-method hyperparameter overrides (fixed infrastructure)
   238:     for _k, _v in CONFIG_OVERRIDES.items():
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
