# MLS-Bench: ai4bio-mutation-effect-prediction

# Task: Protein Mutation Effect Prediction

## Research Question
Design a supervised prediction architecture that maps pre-computed protein language model (PLM) embeddings to protein fitness scores, improving over simple linear or shallow models for mutation effect prediction.

## Background
Predicting the functional effect of amino acid mutations is a central problem in protein engineering and clinical genetics. Deep mutational scanning (DMS) experiments measure the fitness effect of thousands of mutations in a protein, but are expensive and time-consuming. Computational prediction of these effects can accelerate protein design.

The task uses frozen ESM-2 (650M) protein language model representations (Lin et al., "Evolutionary-scale prediction of atomic-level protein structure with a language model", Science 2023; arXiv:2206.13517 / bioRxiv 2022.07.20.500902) and asks for a supervised prediction head over those embeddings.

Key considerations:
- **Embedding structure**: ESM-2 embeddings encode rich structural and evolutionary information in 1280 dimensions. How best to exploit this high-dimensional representation?
- **Delta features**: The difference between mutant and wild-type embeddings directly encodes what changed due to the mutation.
- **Generalization across folds**: The model must generalize across cross-validation splits, not just memorize training examples.

## What to Implement
Implement the `MutationPredictor` class in `custom_mutation_pred.py`. You must implement:
1. `__init__(self, embed_dim)`: Set up your model architecture. `embed_dim` is 1280 (ESM-2 650M).
2. `forward(self, embedding, delta_embedding) -> Tensor`: Return predictions of shape `[B]`.

## Input Format
The model receives two inputs per mutant:
- `embedding`: `[B, 1280]` — Mean-pooled ESM-2 (650M) representation of the mutant sequence.
- `delta_embedding`: `[B, 1280]` — Difference from wild-type embedding (`mutant_emb - wt_emb`).

## Output Format
- Return a tensor of shape `[B]` with predicted fitness scores (real-valued).

## Fixed Pipeline
The data pipeline, train/test loop, embedding extraction, and cross-validation splits are all fixed by the scaffold. The only learnable degrees of freedom are (a) the `MutationPredictor` architecture and (b) optimizer hyperparameters exposed via `CONFIG_OVERRIDES` in `main()` (allowed keys: `learning_rate`, `weight_decay`).

## Evaluation
The model is evaluated on DMS assays from the ProteinGym benchmark (Notin et al., "ProteinGym: Large-Scale Benchmarks for Protein Fitness Prediction and Design", NeurIPS 2023 Datasets & Benchmarks):

- **BLAT_ECOLX** (Beta-lactamase, OrganismalFitness, 4783 single mutants): Antibiotic resistance enzyme from E. coli.
- **ESTA_BACSU** (Esterase, Stability, 2172 single mutants): Thermostability of a B. subtilis esterase.
- **RASH_HUMAN** (K-Ras GTPase, Activity, 3134 single mutants): Oncogene activity in human cells.

**Metric**: Spearman rank correlation between predicted and true fitness scores, averaged over 5-fold cross-validation using ProteinGym's pre-defined **random** folds. Higher is better.

> ⚠️ **Evaluation protocol note.** ProteinGym's supervised leaderboard averages
> Spearman over three fold strategies — `random`, `modulo` (every 5th residue),
> and `contiguous` (held-out sequence blocks). This task uses **only the
> `random` fold strategy**, which is the easiest of the three and tends to
> give higher Spearman than the published ProteinGym SOTA averages. Numbers
> reported here are therefore not directly comparable to the ProteinGym
> supervised leaderboard; treat them as within-benchmark-relative scores.

## Baselines
Reference baselines on the same fixed pipeline:
- **Ridge regression** on concatenated `[embedding, delta_embedding]` features.
- **MLP** prediction head over the same concatenated features.
- **Reshape-CNN** that reshapes the 1280-dim embedding into a 2D grid and applies small convolutions before regression.

All baselines see the same ESM-2 embeddings, the same CV splits, and the same train/test loop; they differ only in the prediction head.

## Editable Region
The `MutationPredictor` class lives between `EDITABLE SECTION START` and `EDITABLE SECTION END` markers in `custom_mutation_pred.py`. You may define helper classes, layers, or functions within this region. The region must contain a `MutationPredictor` class that is an `nn.Module` with the specified interface.

You may additionally set training-loop hyperparameters by writing into the small `CONFIG_OVERRIDES = {}` dict in `main()` (a small editable region near the bottom of the file). Allowed keys: `learning_rate`, `weight_decay`.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/ProteinGym/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `ProteinGym/custom_mutation_pred.py`
- editable lines **108–137**
- editable lines **345–347**




## Readable Context


### `ProteinGym/custom_mutation_pred.py`  [EDITABLE — lines 108–137, lines 345–347 only]

```python
     1: """
     2: Protein Mutation Effect Prediction — Self-contained template.
     3: Predicts DMS fitness scores from frozen ESM-2 embeddings using a supervised model.
     4: Evaluated on ProteinGym DMS assays via Spearman correlation.
     5: 
     6: Structure:
     7:   Lines 1-107:   FIXED — Imports, data loading, CV fold utilities
     8:   Lines 108-137: EDITABLE — MutationPredictor class (starter: ridge regression)
     9:   Lines 138+:    FIXED — Training loop, evaluation, main
    10: """
    11: import os
    12: import sys
    13: import math
    14: import argparse
    15: import warnings
    16: import numpy as np
    17: from pathlib import Path
    18: from typing import Optional, Dict, List, Tuple
    19: 
    20: import torch
    21: import torch.nn as nn
    22: import torch.nn.functional as F
    23: from torch.utils.data import Dataset, DataLoader
    24: 
    25: from scipy.stats import spearmanr
    26: 
    27: warnings.filterwarnings("ignore", category=UserWarning)
    28: 
    29: # =====================================================================
    30: # Constants
    31: # =====================================================================
    32: 
    33: EMBED_DIM = 1280  # ESM-2 650M embedding dimension
    34: 
    35: 
    36: # =====================================================================
    37: # Data loading and CV utilities
    38: # =====================================================================
    39: 
    40: class DMS_Dataset(Dataset):
    41:     """Dataset for a single DMS assay with precomputed ESM-2 embeddings."""
    42: 
    43:     def __init__(self, embeddings, scores, wt_embedding, indices=None):
    44:         """
    45:         Args:
    46:             embeddings: [N, EMBED_DIM] ESM-2 mean-pooled embeddings per mutant
    47:             scores: [N] DMS fitness scores
    48:             wt_embedding: [EMBED_DIM] wild-type embedding
    49:             indices: optional subset indices for train/val splits
    50:         """
    51:         if indices is not None:
    52:             self.embeddings = embeddings[indices]
    53:             self.scores = scores[indices]
    54:         else:
    55:             self.embeddings = embeddings
    56:             self.scores = scores
    57:         self.wt_embedding = wt_embedding
    58: 
    59:     def __len__(self):
    60:         return len(self.scores)
    61: 
    62:     def __getitem__(self, idx):
    63:         return {
    64:             'embedding': self.embeddings[idx],       # [EMBED_DIM]
    65:             'delta_embedding': self.embeddings[idx] - self.wt_embedding,  # [EMBED_DIM]
    66:             'score': self.scores[idx],                # scalar
    67:         }
    68: 
    69: 
    70: def load_dms_data(assay_id, data_dir="/data/esm2_embeddings"):
    71:     """Load precomputed ESM-2 embeddings for a DMS assay."""
    72:     path = os.path.join(data_dir, f"{assay_id}.pt")
    73:     data = torch.load(path, map_location="cpu", weights_only=False)
    74:     return data['embeddings'], data['scores'], data['wt_embedding']
    75: 
    76: 
    77: def load_cv_folds(assay_id, cv_dir="/data/proteingym/cv_folds", n_folds=5):
    78:     """Load precomputed random 5-fold CV assignments.
    79:     Returns list of fold indices [0..4] for each mutant.
    80:     Falls back to random assignment if fold file not found.
    81:     """
    82:     import pandas as pd
    83:     fold_col = f"fold_random_{n_folds}"
    84: 
    85:     # Try to find the fold CSV
    86:     for root, dirs, files in os.walk(cv_dir):
    87:         for fname in files:
    88:             if fname.startswith(assay_id) and fname.endswith('.csv'):
    89:                 df = pd.read_csv(os.path.join(root, fname))
    90:                 if fold_col in df.columns:
    91:                     # Keep only singles
    92:                     df_singles = df[~df['mutant'].str.contains(':')].reset_index(drop=True)
    93:                     return df_singles[fold_col].values
    94:     # Fallback: random assignment
    95:     return None
    96: 
    97: 
    98: def create_random_folds(n_samples, n_folds=5, seed=42):
    99:     """Create random fold assignments."""
   100:     rng = np.random.RandomState(seed)
   101:     return rng.randint(0, n_folds, size=n_samples)
   102: 
   103: 
   104: # =====================================================================
   105: # EDITABLE SECTION START — MutationPredictor + helper modules
   106: # =====================================================================
   107: 
   108: class MutationPredictor(nn.Module):
   109:     """Starter model: Ridge regression (L2-regularized linear model).
   110: 
   111:     Takes ESM-2 embeddings of mutant sequences and predicts DMS fitness
   112:     scores. This simple linear model serves as a baseline; you should
   113:     design a better prediction architecture.
   114: 
   115:     The model receives:
   116:       - embedding: [B, 1280] mean-pooled ESM-2 embedding of mutant sequence
   117:       - delta_embedding: [B, 1280] difference from wild-type embedding
   118:     and must return:
   119:       - prediction: [B] predicted fitness scores
   120: 
   121:     You may use either or both inputs. The delta_embedding highlights
   122:     which residue-level representations changed due to the mutation.
   123:     """
   124: 
   125:     def __init__(self, embed_dim: int = EMBED_DIM):
   126:         super().__init__()
   127:         self.linear = nn.Linear(embed_dim, 1)
   128: 
   129:     def forward(self, embedding, delta_embedding):
   130:         """
   131:         Args:
   132:             embedding: [B, EMBED_DIM] mutant ESM-2 embedding
   133:             delta_embedding: [B, EMBED_DIM] mutant - wildtype embedding
   134:         Returns:
   135:             prediction: [B] predicted fitness scores
   136:         """
   137:         return self.linear(embedding).squeeze(-1)
   138: 
   139: # =====================================================================
   140: # EDITABLE SECTION END
   141: # =====================================================================
   142: 
   143: 
   144: # =====================================================================
   145: # FIXED — Training loop, evaluation, main
   146: # =====================================================================
   147: 
   148: def collate_fn(batch_list):
   149:     """Collate batch samples."""
   150:     return {
   151:         'embedding': torch.stack([b['embedding'] for b in batch_list]),
   152:         'delta_embedding': torch.stack([b['delta_embedding'] for b in batch_list]),
   153:         'score': torch.stack([b['score'] for b in batch_list]),
   154:     }
   155: 
   156: 
   157: def train_epoch(model, loader, optimizer, device, weight_decay=0.01):
   158:     """Train one epoch with MSE loss."""
   159:     model.train()
   160:     total_loss = 0.0
   161:     n_batches = 0
   162: 
   163:     for batch in loader:
   164:         emb = batch['embedding'].to(device)
   165:         delta = batch['delta_embedding'].to(device)
   166:         targets = batch['score'].to(device)
   167: 
   168:         optimizer.zero_grad()
   169:         preds = model(emb, delta)
   170:         loss = F.mse_loss(preds, targets)
   171:         loss.backward()
   172:         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
   173:         optimizer.step()
   174: 
   175:         total_loss += loss.item()
   176:         n_batches += 1
   177: 
   178:     return total_loss / max(n_batches, 1)
   179: 
   180: 
   181: @torch.no_grad()
   182: def evaluate(model, loader, device):
   183:     """Evaluate model and return Spearman correlation."""
   184:     model.eval()
   185:     all_preds = []
   186:     all_targets = []
   187: 
   188:     for batch in loader:
   189:         emb = batch['embedding'].to(device)
   190:         delta = batch['delta_embedding'].to(device)
   191:         targets = batch['score']
   192: 
   193:         preds = model(emb, delta)
   194:         all_preds.append(preds.cpu())
   195:         all_targets.append(targets)
   196: 
   197:     if not all_preds:
   198:         return 0.0
   199: 
   200:     preds = torch.cat(all_preds).numpy()
   201:     targets = torch.cat(all_targets).numpy()
   202: 
   203:     # Spearman correlation
   204:     if len(np.unique(preds)) < 2 or len(np.unique(targets)) < 2:
   205:         return 0.0
   206: 
   207:     rho, _ = spearmanr(preds, targets)
   208:     return float(rho) if not np.isnan(rho) else 0.0
   209: 
   210: 
   211: def train_and_evaluate(args):
   212:     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   213:     print(f"Using device: {device}")
   214: 
   215:     # Load data
   216:     embeddings, scores, wt_embedding = load_dms_data(args.assay_id, args.data_dir)
   217:     n_samples = len(scores)
   218:     print(f"Assay: {args.assay_id}, samples: {n_samples}, embed_dim: {embeddings.shape[1]}")
   219: 
   220:     # Get CV fold assignments
   221:     folds = load_cv_folds(args.assay_id, args.cv_dir)
   222:     if folds is None:
   223:         print("Using random fold assignment (CV fold file not found)")
   224:         folds = create_random_folds(n_samples, seed=args.seed)
   225:     else:
   226:         # Truncate if sizes don't match (can happen with filtering)
   227:         if len(folds) != n_samples:
   228:             print(f"Warning: fold size ({len(folds)}) != data size ({n_samples}), using random folds")
   229:             folds = create_random_folds(n_samples, seed=args.seed)
   230: 
   231:     n_folds = 5
   232:     all_test_spearmans = []
   233: 
   234:     for fold_idx in range(n_folds):
   235:         test_mask = (folds == fold_idx)
   236:         train_mask = ~test_mask
   237: 
   238:         train_indices = np.where(train_mask)[0]
   239:         test_indices = np.where(test_mask)[0]
   240: 
   241:         if len(test_indices) == 0 or len(train_indices) == 0:
   242:             continue
   243: 
   244:         # Further split train into train/val (90/10)
   245:         rng = np.random.RandomState(args.seed + fold_idx)
   246:         rng.shuffle(train_indices)
   247:         val_size = max(1, int(len(train_indices) * 0.1))
   248:         val_indices = train_indices[:val_size]
   249:         actual_train_indices = train_indices[val_size:]
   250: 
   251:         train_ds = DMS_Dataset(embeddings, scores, wt_embedding, actual_train_indices)
   252:         val_ds = DMS_Dataset(embeddings, scores, wt_embedding, val_indices)
   253:         test_ds = DMS_Dataset(embeddings, scores, wt_embedding, test_indices)
   254: 
   255:         train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
   256:                                   collate_fn=collate_fn, drop_last=False)
   257:         val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
   258:                                 collate_fn=collate_fn)
   259:         test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
   260:                                  collate_fn=collate_fn)
   261: 
   262:         # Model
   263:         model = MutationPredictor(embed_dim=EMBED_DIM).to(device)
   264:         optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
   265:                                       weight_decay=args.weight_decay)
   266:         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
   267:             optimizer, T_max=args.epochs)
   268: 
   269:         # Training with early stopping
   270:         best_val_spearman = -float('inf')
   271:         best_epoch = 0
   272:         patience_counter = 0
   273:         patience = 20
   274:         best_state = None
   275: 
   276:         for epoch in range(1, args.epochs + 1):
   277:             train_loss = train_epoch(model, train_loader, optimizer, device,
   278:                                      weight_decay=args.weight_decay)
   279:             val_spearman = evaluate(model, val_loader, device)
   280:             scheduler.step()
   281: 
   282:             if epoch % 10 == 0 or epoch == 1:
   283:                 print(f"TRAIN_METRICS fold={fold_idx} epoch={epoch} "
   284:                       f"loss={train_loss:.6f} val_spearman={val_spearman:.4f}")
   285: 
   286:             if val_spearman > best_val_spearman:
   287:                 best_val_spearman = val_spearman
   288:                 best_epoch = epoch
   289:                 patience_counter = 0
   290:                 best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
   291:             else:
   292:                 patience_counter += 1
   293:                 if patience_counter >= patience:
   294:                     print(f"  Early stopping fold {fold_idx} at epoch {epoch}. "
   295:                           f"Best: {best_epoch}")
   296:                     break
   297: 
   298:         # Load best model and evaluate on test fold
   299:         if best_state is not None:
   300:             model.load_state_dict(best_state)
   301:             model.to(device)
   302:         test_spearman = evaluate(model, test_loader, device)
   303:         all_test_spearmans.append(test_spearman)
   304:         print(f"  Fold {fold_idx}: test_spearman={test_spearman:.4f} "
   305:               f"(best_val={best_val_spearman:.4f} at epoch {best_epoch})")
   306: 
   307:     # Average across folds
   308:     mean_spearman = float(np.mean(all_test_spearmans))
   309:     std_spearman = float(np.std(all_test_spearmans))
   310:     print(f"\n5-fold CV Results for {args.assay_id}:")
   311:     print(f"  Mean Spearman: {mean_spearman:.4f} +/- {std_spearman:.4f}")
   312:     print(f"TEST_METRICS spearman={mean_spearman:.6f}")
   313: 
   314:     # Save results
   315:     os.makedirs(args.output_dir, exist_ok=True)
   316:     results = {
   317:         'assay_id': args.assay_id,
   318:         'mean_spearman': mean_spearman,
   319:         'std_spearman': std_spearman,
   320:         'fold_spearmans': all_test_spearmans,
   321:     }
   322:     torch.save(results, os.path.join(args.output_dir, 'results.pt'))
   323: 
   324: 
   325: def main():
   326:     parser = argparse.ArgumentParser(description="Protein Mutation Effect Prediction")
   327:     parser.add_argument('--assay-id', type=str, required=True,
   328:                         help='DMS assay identifier')
   329:     parser.add_argument('--data-dir', type=str, default='/data/esm2_embeddings',
   330:                         help='Directory with precomputed ESM-2 embeddings')
   331:     parser.add_argument('--cv-dir', type=str,
   332:                         default='/data/proteingym/cv_folds',
   333:                         help='Directory with CV fold assignments')
   334:     parser.add_argument('--epochs', type=int, default=200)
   335:     parser.add_argument('--batch-size', type=int, default=64)
   336:     parser.add_argument('--lr', type=float, default=1e-3)
   337:     parser.add_argument('--weight-decay', type=float, default=0.05)
   338:     parser.add_argument('--seed', type=int, default=42)
   339:     parser.add_argument('--output-dir', type=str, default='./output')
   340:     args = parser.parse_args()
   341: 
   342:     # =====================================================================
   343:     # EDITABLE SECTION START — CONFIG_OVERRIDES
   344:     # =====================================================================
   345:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   346:     # Allowed keys: learning_rate, weight_decay.
   347:     CONFIG_OVERRIDES = {}
   348:     # =====================================================================
   349:     # EDITABLE SECTION END
   350:     # =====================================================================
   351: 
   352:     # =====================================================================
   353:     # FIXED — Apply config overrides and set seeds
   354:     # =====================================================================
   355:     for _k, _v in CONFIG_OVERRIDES.items():
   356:         if _k == 'learning_rate': args.lr = _v
   357:         elif _k == 'weight_decay': args.weight_decay = _v
   358: 
   359:     # Set seeds
   360:     torch.manual_seed(args.seed)
   361:     np.random.seed(args.seed)
   362:     if torch.cuda.is_available():
   363:         torch.cuda.manual_seed_all(args.seed)
   364: 
   365:     train_and_evaluate(args)
   366: 
   367: 
   368: if __name__ == '__main__':
   369:     main()
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


### `ridge` baseline — editable region  [READ-ONLY — reference implementation]

In `ProteinGym/custom_mutation_pred.py`:

```python
Lines 108–123:
   105: # EDITABLE SECTION START — MutationPredictor + helper modules
   106: # =====================================================================
   107: 
   108: 
   109: class MutationPredictor(nn.Module):
   110:     """Ridge regression as a single nn.Linear, trained with AdamW (wd=5e-2).
   111: 
   112:     Uses delta_embedding (mutant - wildtype) as the input feature, so the
   113:     model learns a linear mapping from the mutation-induced embedding shift
   114:     to the fitness score.
   115:     """
   116: 
   117:     def __init__(self, embed_dim: int = EMBED_DIM):
   118:         super().__init__()
   119:         self.linear = nn.Linear(embed_dim, 1)
   120: 
   121:     def forward(self, embedding, delta_embedding):
   122:         return self.linear(delta_embedding).squeeze(-1)
   123: 
   124: 
   125: # =====================================================================
   126: # EDITABLE SECTION END

Lines 331–331:
   328:     # =====================================================================
   329:     # EDITABLE SECTION START — CONFIG_OVERRIDES
   330:     # =====================================================================
   331:     CONFIG_OVERRIDES = {'weight_decay': 5e-2}
   332:     # =====================================================================
   333:     # EDITABLE SECTION END
   334:     # =====================================================================
```

### `mlp` baseline — editable region  [READ-ONLY — reference implementation]

In `ProteinGym/custom_mutation_pred.py`:

```python
Lines 108–129:
   105: # EDITABLE SECTION START — MutationPredictor + helper modules
   106: # =====================================================================
   107: 
   108: 
   109: class MutationPredictor(nn.Module):
   110:     """Single-hidden-layer MLP over delta_embedding (mutant - WT).
   111: 
   112:     Architecture: Linear(embed_dim, hidden) -> Dropout -> ReLU -> Linear(hidden, 1)
   113:     Uses delta_embedding so the network sees the mutation-induced shift
   114:     in PLM representation space directly.
   115:     """
   116: 
   117:     def __init__(self, embed_dim: int = EMBED_DIM, hidden_dim: int = 512,
   118:                  dropout: float = 0.1):
   119:         super().__init__()
   120:         self.fc1 = nn.Linear(embed_dim, hidden_dim)
   121:         self.dropout = nn.Dropout(dropout)
   122:         self.fc2 = nn.Linear(hidden_dim, 1)
   123: 
   124:     def forward(self, embedding, delta_embedding):
   125:         x = self.fc1(delta_embedding)
   126:         x = self.dropout(x)
   127:         x = F.relu(x)
   128:         return self.fc2(x).squeeze(-1)
   129: 
   130: 
   131: # =====================================================================
   132: # EDITABLE SECTION END

Lines 337–339:
   334:     # =====================================================================
   335:     # EDITABLE SECTION START — CONFIG_OVERRIDES
   336:     # =====================================================================
   337:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   338:     # Allowed keys: learning_rate, weight_decay.
   339:     CONFIG_OVERRIDES = {}
   340:     # =====================================================================
   341:     # EDITABLE SECTION END
   342:     # =====================================================================
```

### `reshape_cnn` baseline — editable region  [READ-ONLY — reference implementation]

In `ProteinGym/custom_mutation_pred.py`:

```python
Lines 108–165:
   105: # EDITABLE SECTION START — MutationPredictor + helper modules
   106: # =====================================================================
   107: 
   108: 
   109: class ConvBlock(nn.Module):
   110:     """1D convolution block with BatchNorm and residual connection."""
   111: 
   112:     def __init__(self, channels, kernel_size, dropout=0.1):
   113:         super().__init__()
   114:         padding = kernel_size // 2
   115:         self.conv = nn.Conv1d(channels, channels, kernel_size, padding=padding)
   116:         self.bn = nn.BatchNorm1d(channels)
   117:         self.dropout = nn.Dropout(dropout)
   118: 
   119:     def forward(self, x):
   120:         residual = x
   121:         x = F.gelu(self.bn(self.conv(x)))
   122:         x = self.dropout(x)
   123:         return x + residual
   124: 
   125: 
   126: class MutationPredictor(nn.Module):
   127:     """Reshape-CNN over mean-pooled ESM-2 features (NOT per-residue).
   128: 
   129:     Concatenates [embedding, delta_embedding] -> [B, 2*EMBED_DIM=2560],
   130:     reshapes to (B, channels=64, length=40), applies a stack of 1D
   131:     convolutions with residual connections over the embedding-channel
   132:     axis, then global-average-pools and predicts.
   133: 
   134:     The reshape axis has NO real sequence structure — see the docstring
   135:     in reshape_cnn.edit.py for why this is not a paper-faithful CNN.
   136:     """
   137: 
   138:     def __init__(self, embed_dim: int = EMBED_DIM):
   139:         super().__init__()
   140:         self.channels = 64
   141:         self.length = (embed_dim * 2) // self.channels  # 40
   142: 
   143:         self.input_proj = nn.Linear(embed_dim * 2, self.channels * self.length)
   144: 
   145:         self.conv_blocks = nn.Sequential(
   146:             ConvBlock(self.channels, kernel_size=3, dropout=0.1),
   147:             ConvBlock(self.channels, kernel_size=5, dropout=0.1),
   148:             ConvBlock(self.channels, kernel_size=7, dropout=0.1),
   149:         )
   150: 
   151:         self.head = nn.Sequential(
   152:             nn.Linear(self.channels, 128),
   153:             nn.GELU(),
   154:             nn.Dropout(0.1),
   155:             nn.Linear(128, 1),
   156:         )
   157: 
   158:     def forward(self, embedding, delta_embedding):
   159:         x = torch.cat([embedding, delta_embedding], dim=-1)  # [B, 2*EMBED_DIM]
   160:         x = F.gelu(self.input_proj(x))                       # [B, C*L]
   161:         x = x.view(x.size(0), self.channels, self.length)    # [B, C, L]
   162:         x = self.conv_blocks(x)                              # [B, C, L]
   163:         x = x.mean(dim=-1)                                   # [B, C]
   164:         return self.head(x).squeeze(-1)                      # [B]
   165: 
   166: 
   167: # =====================================================================
   168: # EDITABLE SECTION END

Lines 373–375:
   370:     # =====================================================================
   371:     # EDITABLE SECTION START — CONFIG_OVERRIDES
   372:     # =====================================================================
   373:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   374:     # Allowed keys: learning_rate, weight_decay.
   375:     CONFIG_OVERRIDES = {}
   376:     # =====================================================================
   377:     # EDITABLE SECTION END
   378:     # =====================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
