# MLS-Bench: ai4bio-protein-inverse-folding

# Task: Protein Inverse Folding — Structure Encoder Design

## Research Question
Design a novel GNN-based structure encoder for protein inverse folding: given backbone atom coordinates (N, CA, C, O), predict the amino acid sequence that would fold into that structure.

## Background
Protein inverse folding (also called computational protein design or fixed-backbone design) is a central problem in structural biology. Given a protein backbone structure, the goal is to predict the amino acid sequence most likely to fold into that structure. This is the inverse of the protein folding problem (predicting structure from sequence).

The key challenge is encoding the 3D protein backbone graph into rich per-residue embeddings that capture local geometry, long-range interactions, and structural motifs. Existing approaches differ primarily in how they encode the protein structure:

- **GVP** (Geometric Vector Perceptron; Jing et al., "Learning from Protein Structure with Geometric Vector Perceptrons", ICLR 2021; arXiv:2009.01411). SE(3)-equivariant message passing with both scalar and vector node/edge features. Code: https://github.com/drorlab/gvp.
- **ProteinMPNN** (Dauparas et al., "Robust deep learning–based protein sequence design using ProteinMPNN", Science 2022, 378(6615):49–56; bioRxiv 2022.06.03.494563). Message-passing encoder with edge updates, followed by an autoregressive decoder with masking. Code: https://github.com/dauparas/ProteinMPNN.
- **PiFold** (Gao et al., "PiFold: Toward Effective and Efficient Protein Inverse Folding", ICLR 2023; arXiv:2209.12643). PiGNN encoder with learnable virtual atoms, multi-scale distance features, and dihedral features, plus a non-autoregressive one-shot decoder. Code: https://github.com/A4Bio/PiFold.

The structure encoder is the critical component: all methods share the same input format (backbone coordinates) and output format (amino acid log-probabilities), but differ in how they transform structure into sequence-informative representations.

## What to Implement
Modify the editable section of `custom_invfold.py`. You must implement:
1. **StructureEncoder**: A GNN module that takes backbone coordinates `X` (B, L, 4, 3) and mask (B, L), and produces per-residue embeddings `h_V` (B, L, hidden_dim).
2. **InverseFoldingModel**: Wraps the encoder with a decoder head that outputs amino acid log-probabilities (B, L, 20).

## Interface
```python
class StructureEncoder(nn.Module):
    def __init__(self, hidden_dim=128, ...):
        ...
    def forward(self, X, mask):
        """
        X: (B, L, 4, 3) backbone coordinates [N, CA, C, O]
        mask: (B, L) binary mask (1 for valid residues, 0 for padding)
        Returns: h_V (B, L, hidden_dim) per-residue embeddings
        """
        ...

class InverseFoldingModel(nn.Module):
    def __init__(self, hidden_dim=128, ...):
        ...
    def forward(self, X, mask):
        """
        Returns: log_probs (B, L, 20) amino acid log-probabilities
        """
        ...
```

Helper functions available in the FIXED section above the editable region:
- `_rbf(D, ...)`: Radial basis function encoding of distances.
- `_dihedrals(X)`: Backbone dihedral angles (phi, psi, omega) as sin/cos features.
- `_orientations(X)`: Local coordinate frame (forward + binormal vectors).
- `knn_graph(X_ca, mask, k)`: Build k-nearest neighbor graph from CA coordinates.

## Fixed Pipeline
Datasets, train/validation/test splits, the training loop, padding/masking, optimizer schedule, loss (per-residue cross-entropy), and evaluation harness are all supplied by the scaffold and not part of the contribution.

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/ProteinInvBench/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `ProteinInvBench/custom_invfold.py`
- editable lines **86–238**
- editable lines **401–403**




## Readable Context


### `ProteinInvBench/custom_invfold.py`  [EDITABLE — lines 86–238, lines 401–403 only]

```python
     1: """
     2: Protein Inverse Folding — Self-contained template.
     3: Given backbone structure (N, CA, C, O coordinates), predict amino acid sequence.
     4: 
     5: Structure:
     6:   Lines 1-75:    FIXED — Imports, constants, data loading, featurization
     7:   Lines 76-230:  EDITABLE — StructureEncoder + decoder (starter: simple MPNN)
     8:   Lines 231+:    FIXED — Training loop, evaluation, metrics
     9: """
    10: import os
    11: import sys
    12: import json
    13: import math
    14: import time
    15: import argparse
    16: import numpy as np
    17: from pathlib import Path
    18: from typing import Dict, Optional, Tuple
    19: 
    20: import torch
    21: import torch.nn as nn
    22: import torch.nn.functional as F
    23: from torch.utils.data import DataLoader
    24: 
    25: # ---- Constants ----
    26: ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
    27: NUM_AA = 20  # 20 standard amino acids
    28: NUM_BB_ATOMS = 4  # N, CA, C, O
    29: 
    30: def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    31:     """Radial basis function encoding of distances."""
    32:     D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    33:     D_mu = D_mu.view([1, -1])
    34:     D_sigma = (D_max - D_min) / D_count
    35:     D_expand = torch.unsqueeze(D, -1)
    36:     return torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    37: 
    38: def _dihedrals(X, eps=1e-7):
    39:     """Compute backbone dihedral angles (phi, psi, omega) from N-CA-C-O coords.
    40:     X: (B, L, 4, 3) — N, CA, C, O coordinates.
    41:     Returns: (B, L, 6) — sin/cos of 3 dihedral angles.
    42:     """
    43:     X_flat = X[:, :, :3, :].reshape(int(X.shape[0]), -1, 3)  # (B, 3L, 3)
    44:     dX = X_flat[:, 1:, :] - X_flat[:, :-1, :]  # (B, 3L-1, 3)
    45:     U = F.normalize(dX, dim=-1)
    46:     u_2 = U[:, :-2, :]
    47:     u_1 = U[:, 1:-1, :]
    48:     u_0 = U[:, 2:, :]
    49:     n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
    50:     n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)
    51:     cos_d = (n_2 * n_1).sum(-1)
    52:     sin_d = (torch.cross(n_2, n_1, dim=-1) * u_1).sum(-1)
    53:     cos_d = cos_d.clamp(-1 + eps, 1 - eps)
    54:     sin_d = sin_d.clamp(-1 + eps, 1 - eps)
    55:     D = torch.stack([cos_d, sin_d], dim=-1)  # (B, 3L-3, 2)
    56:     # Pad to (B, L, 6) — 3 dihedrals per residue
    57:     D = F.pad(D, (0, 0, 1, 2))  # pad the length
    58:     B, N = int(X.shape[0]), int(X.shape[1])
    59:     D = D.reshape(B, -1, 6)[:, :N, :]
    60:     return D
    61: 
    62: def _orientations(X):
    63:     """Compute local orientation frames from N-CA-C coords.
    64:     Returns forward and binormal unit vectors. (B, L, 6)
    65:     """
    66:     fwd = F.normalize(X[:, 1:, 1, :] - X[:, :-1, 1, :], dim=-1)  # CA-CA
    67:     fwd = F.pad(fwd, (0, 0, 0, 1))
    68:     u = F.normalize(X[:, :, 2, :] - X[:, :, 1, :], dim=-1)  # C-CA
    69:     b = F.normalize(fwd - (fwd * u).sum(-1, keepdim=True) * u, dim=-1)
    70:     return torch.cat([fwd, b], dim=-1)
    71: 
    72: def knn_graph(X_ca, mask, k=30):
    73:     """Build k-nearest neighbor graph from CA coordinates.
    74:     X_ca: (B, L, 3), mask: (B, L)
    75:     Returns: E_idx (B, L, K), D_neighbors (B, L, K)
    76:     """
    77:     mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B, L, L)
    78:     dX = X_ca.unsqueeze(1) - X_ca.unsqueeze(2)  # (B, L, L, 3)
    79:     D = mask_2D * torch.sqrt((dX ** 2).sum(-1) + 1e-6) + (1 - mask_2D) * 1e6
    80:     D_neighbors, E_idx = torch.topk(D, min(k, int(D.shape[-1])), dim=-1, largest=False)
    81:     return E_idx, D_neighbors
    82: 
    83: # =====================================================================
    84: # EDITABLE SECTION START — StructureEncoder + InverseFoldingModel
    85: # =====================================================================
    86: 
    87: class MPNNEncoderLayer(nn.Module):
    88:     """Message Passing Neural Network layer for protein graphs."""
    89: 
    90:     def __init__(self, hidden_dim, edge_dim, dropout=0.1):
    91:         super().__init__()
    92:         self.hidden_dim = hidden_dim
    93:         # Edge message network
    94:         self.W_msg = nn.Sequential(
    95:             nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
    96:             nn.ReLU(),
    97:             nn.Linear(hidden_dim, hidden_dim),
    98:         )
    99:         # Node update network
   100:         self.W_node = nn.Sequential(
   101:             nn.Linear(2 * hidden_dim, hidden_dim),
   102:             nn.ReLU(),
   103:             nn.Linear(hidden_dim, hidden_dim),
   104:         )
   105:         self.norm1 = nn.LayerNorm(hidden_dim)
   106:         self.norm2 = nn.LayerNorm(hidden_dim)
   107:         self.dropout = nn.Dropout(dropout)
   108: 
   109:     def forward(self, h_V, h_E, E_idx, mask):
   110:         """
   111:         h_V: (B, L, D) node features
   112:         h_E: (B, L, K, D_e) edge features
   113:         E_idx: (B, L, K) neighbor indices
   114:         mask: (B, L)
   115:         """
   116:         B, L, K = int(E_idx.shape[0]), int(E_idx.shape[1]), int(E_idx.shape[2])
   117:         # Gather neighbor node features
   118:         D = int(h_V.shape[-1])
   119:         h_V_neighbors = torch.gather(
   120:             h_V.unsqueeze(2).expand(-1, -1, K, -1),
   121:             1,
   122:             E_idx.unsqueeze(-1).expand(-1, -1, -1, D)
   123:         )  # (B, L, K, D)
   124:         h_V_expand = h_V.unsqueeze(2).expand_as(h_V_neighbors)
   125:         # Messages
   126:         msg_input = torch.cat([h_V_expand, h_V_neighbors, h_E], dim=-1)
   127:         messages = self.W_msg(msg_input)  # (B, L, K, D)
   128:         # Mask out invalid neighbors
   129:         mask_attend = torch.gather(mask.unsqueeze(2).expand(-1, -1, K), 1,
   130:                                    E_idx.clamp(0, L-1)).unsqueeze(-1)
   131:         messages = messages * mask_attend
   132:         # Aggregate
   133:         agg = messages.sum(dim=2) / (mask_attend.sum(dim=2).clamp(min=1))
   134:         # Update
   135:         h_V = self.norm1(h_V + self.dropout(agg))
   136:         h_V_upd = self.W_node(torch.cat([h_V, agg], dim=-1))
   137:         h_V = self.norm2(h_V + self.dropout(h_V_upd))
   138:         h_V = h_V * mask.unsqueeze(-1)
   139:         return h_V
   140: 
   141: 
   142: class StructureEncoder(nn.Module):
   143:     """GNN encoder for protein backbone structure.
   144:     Takes backbone coordinates and produces per-residue embeddings.
   145:     """
   146: 
   147:     def __init__(self, hidden_dim=128, num_layers=3, k_neighbors=30, dropout=0.1, num_rbf=16):
   148:         super().__init__()
   149:         self.hidden_dim = hidden_dim
   150:         self.k_neighbors = k_neighbors
   151:         self.num_rbf = num_rbf
   152: 
   153:         # Node input: dihedral features (6) + orientation (6)
   154:         node_input_dim = 12
   155:         # Edge input: RBF distance (num_rbf) + direction (3)
   156:         edge_input_dim = num_rbf + 3
   157: 
   158:         self.node_embed = nn.Linear(node_input_dim, hidden_dim)
   159:         self.edge_embed = nn.Linear(edge_input_dim, hidden_dim)
   160: 
   161:         self.layers = nn.ModuleList([
   162:             MPNNEncoderLayer(hidden_dim, hidden_dim, dropout)
   163:             for _ in range(num_layers)
   164:         ])
   165: 
   166:     def forward(self, X, mask):
   167:         """
   168:         X: (B, L, 4, 3) backbone coordinates (N, CA, C, O)
   169:         mask: (B, L) residue mask
   170:         Returns: h_V (B, L, hidden_dim) per-residue encoder embeddings
   171:         """
   172:         B, L = X.shape[0], X.shape[1]
   173:         X_ca = X[:, :, 1, :]  # CA atoms
   174: 
   175:         # Build KNN graph
   176:         E_idx, D_neighbors = knn_graph(X_ca, mask, self.k_neighbors)
   177:         K = E_idx.shape[2]
   178: 
   179:         # Node features: dihedrals + orientations
   180:         dihedrals = _dihedrals(X)  # (B, L, 6)
   181:         orientations = _orientations(X)  # (B, L, 6)
   182:         node_feat = torch.cat([dihedrals, orientations], dim=-1)  # (B, L, 12)
   183: 
   184:         # Edge features: RBF distances + direction vectors
   185:         rbf = _rbf(D_neighbors, device=X.device)  # (B, L, K, num_rbf)
   186:         # Direction vectors to neighbors
   187:         X_ca_neighbors = torch.gather(
   188:             X_ca.unsqueeze(2).expand(-1, -1, K, -1),
   189:             1,
   190:             E_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
   191:         )
   192:         direction = F.normalize(X_ca_neighbors - X_ca.unsqueeze(2), dim=-1)
   193:         edge_feat = torch.cat([rbf, direction], dim=-1)  # (B, L, K, num_rbf+3)
   194: 
   195:         # Embed
   196:         h_V = self.node_embed(node_feat)  # (B, L, D)
   197:         h_E = self.edge_embed(edge_feat)  # (B, L, K, D)
   198: 
   199:         # Message passing
   200:         for layer in self.layers:
   201:             h_V = layer(h_V, h_E, E_idx, mask)
   202: 
   203:         return h_V
   204: 
   205: 
   206: class InverseFoldingModel(nn.Module):
   207:     """Protein inverse folding model.
   208:     Encoder: StructureEncoder (editable) produces per-residue embeddings.
   209:     Decoder: simple MLP that predicts amino acid logits from encoder output.
   210:     """
   211: 
   212:     def __init__(self, hidden_dim=128, num_encoder_layers=3, k_neighbors=30,
   213:                  dropout=0.1, num_rbf=16):
   214:         super().__init__()
   215:         self.encoder = StructureEncoder(
   216:             hidden_dim=hidden_dim,
   217:             num_layers=num_encoder_layers,
   218:             k_neighbors=k_neighbors,
   219:             dropout=dropout,
   220:             num_rbf=num_rbf,
   221:         )
   222:         self.decoder = nn.Sequential(
   223:             nn.Linear(hidden_dim, hidden_dim),
   224:             nn.ReLU(),
   225:             nn.Dropout(dropout),
   226:             nn.Linear(hidden_dim, NUM_AA),
   227:         )
   228: 
   229:     def forward(self, X, mask):
   230:         """
   231:         X: (B, L, 4, 3) backbone coords
   232:         mask: (B, L) residue mask
   233:         Returns: log_probs (B, L, NUM_AA)
   234:         """
   235:         h_V = self.encoder(X, mask)
   236:         logits = self.decoder(h_V)
   237:         log_probs = F.log_softmax(logits, dim=-1)
   238:         return log_probs
   239: 
   240: # =====================================================================
   241: # EDITABLE SECTION END
   242: # =====================================================================
   243: 
   244: 
   245: # ---- Data Loading (uses PInvBench datasets directly) ----
   246: 
   247: def load_dataset(dataset_name, data_root, split, remove_ts=False):
   248:     """Load protein dataset. Returns list of protein dicts."""
   249:     if dataset_name in ('CATH4.2', 'CATH4.3'):
   250:         version = float(dataset_name.replace('CATH', ''))
   251:         subdir = dataset_name.lower()
   252:         path = os.path.join(data_root, subdir)
   253:         from PInvBench.src.datasets.cath_dataset import CATHDataset
   254:         return CATHDataset(path=path, split=split, max_length=500, version=version, removeTS=int(bool(remove_ts)))
   255:     elif dataset_name == 'TS':
   256:         path = os.path.join(data_root, 'ts')
   257:         from PInvBench.src.datasets.ts_dataset import TSDataset
   258:         ds = TSDataset(path=path, split=split)
   259:         # TSDataset bundles ts50.json + ts500.json (~550 prot). Filter to TS50 only.
   260:         return [d for d in ds if d.get('category') == 'ts50']
   261:     else:
   262:         raise ValueError(f"Unknown dataset: {dataset_name}")
   263: 
   264: 
   265: def collate_fn(batch):
   266:     """Collate protein dicts into padded tensors."""
   267:     batch = [b for b in batch if b is not None]
   268:     if len(batch) == 0:
   269:         return None
   270:     B = len(batch)
   271:     lengths = [len(b['seq']) for b in batch]
   272:     L_max = max(lengths)
   273: 
   274:     X = np.zeros([B, L_max, 4, 3], dtype=np.float32)
   275:     S = np.zeros([B, L_max], dtype=np.int64)
   276:     mask = np.zeros([B, L_max], dtype=np.float32)
   277: 
   278:     for i, b in enumerate(batch):
   279:         l = len(b['seq'])
   280:         coords = np.stack([b['N'], b['CA'], b['C'], b['O']], axis=1)  # (L, 4, 3)
   281:         # Handle NaN coordinates: replace with 0 and mask out those residues
   282:         nan_mask = np.isnan(coords).any(axis=(1, 2))  # (L,) True if any NaN in residue
   283:         coords = np.nan_to_num(coords, nan=0.0)
   284:         X[i, :l] = coords
   285:         # Convert sequence to indices
   286:         for j, aa in enumerate(b['seq']):
   287:             if aa in ALPHABET:
   288:                 S[i, j] = ALPHABET.index(aa)
   289:             else:
   290:                 S[i, j] = 0  # unknown -> Ala
   291:         mask[i, :l] = 1.0
   292:         # Zero out mask for residues with NaN coordinates
   293:         mask[i, :l][nan_mask] = 0.0
   294: 
   295:     X = torch.from_numpy(X)
   296:     S = torch.from_numpy(S)
   297:     mask = torch.from_numpy(mask)
   298:     return {'X': X, 'S': S, 'mask': mask, 'lengths': lengths}
   299: 
   300: 
   301: # ---- Training & Evaluation ----
   302: 
   303: def compute_recovery(log_probs, S, mask):
   304:     """Compute amino acid sequence recovery rate."""
   305:     pred = log_probs.argmax(dim=-1)
   306:     correct = ((pred == S) * mask).sum()
   307:     total = mask.sum()
   308:     return (correct / total.clamp(min=1)).item()
   309: 
   310: 
   311: def compute_perplexity(log_probs, S, mask):
   312:     """Compute per-residue perplexity."""
   313:     nll = F.nll_loss(log_probs.permute(0, 2, 1), S, reduction='none')  # (B, L)
   314:     nll = (nll * mask).sum() / mask.sum().clamp(min=1)
   315:     return torch.exp(nll).item()
   316: 
   317: 
   318: def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, max_steps=None):
   319:     model.train()
   320:     total_loss = 0.0
   321:     total_recovery = 0.0
   322:     n_batches = 0
   323:     for i, batch in enumerate(dataloader):
   324:         if batch is None:
   325:             continue
   326:         if max_steps is not None and i >= max_steps:
   327:             break
   328:         X = batch['X'].to(device)
   329:         S = batch['S'].to(device)
   330:         mask = batch['mask'].to(device)
   331: 
   332:         log_probs = model(X, mask)
   333:         loss = F.nll_loss(log_probs.permute(0, 2, 1), S, reduction='none')
   334:         loss = (loss * mask).sum() / mask.sum().clamp(min=1)
   335: 
   336:         optimizer.zero_grad()
   337:         loss.backward()
   338:         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
   339:         optimizer.step()
   340:         if scheduler is not None:
   341:             scheduler.step()
   342: 
   343:         total_loss += loss.item()
   344:         total_recovery += compute_recovery(log_probs.detach(), S, mask)
   345:         n_batches += 1
   346: 
   347:         if (i + 1) % 100 == 0:
   348:             avg_loss = total_loss / n_batches
   349:             avg_rec = total_recovery / n_batches
   350:             print(f"TRAIN_METRICS epoch={epoch} step={i+1} loss={avg_loss:.4f} recovery={avg_rec:.4f}", flush=True)
   351: 
   352:     if n_batches > 0:
   353:         avg_loss = total_loss / n_batches
   354:         avg_rec = total_recovery / n_batches
   355:         print(f"TRAIN_METRICS epoch={epoch} loss={avg_loss:.4f} recovery={avg_rec:.4f}", flush=True)
   356:     return total_loss / max(n_batches, 1)
   357: 
   358: 
   359: @torch.no_grad()
   360: def evaluate(model, dataloader, device, label="val"):
   361:     model.eval()
   362:     total_recovery = 0.0
   363:     total_perplexity = 0.0
   364:     n_batches = 0
   365:     for batch in dataloader:
   366:         if batch is None:
   367:             continue
   368:         X = batch['X'].to(device)
   369:         S = batch['S'].to(device)
   370:         mask = batch['mask'].to(device)
   371: 
   372:         log_probs = model(X, mask)
   373:         total_recovery += compute_recovery(log_probs, S, mask)
   374:         total_perplexity += compute_perplexity(log_probs, S, mask)
   375:         n_batches += 1
   376: 
   377:     if n_batches == 0:
   378:         return 0.0, float('inf')
   379:     recovery = total_recovery / n_batches
   380:     perplexity = total_perplexity / n_batches
   381:     return recovery, perplexity
   382: 
   383: 
   384: def main():
   385:     parser = argparse.ArgumentParser(description="Protein Inverse Folding")
   386:     parser.add_argument('--dataset', default='CATH4.2', choices=['CATH4.2', 'CATH4.3', 'TS'])
   387:     parser.add_argument('--data-root', default='/workspace/data')
   388:     parser.add_argument('--epochs', type=int, default=100)
   389:     parser.add_argument('--batch-size', type=int, default=32)
   390:     parser.add_argument('--lr', type=float, default=1e-3)
   391:     parser.add_argument('--hidden-dim', type=int, default=128)
   392:     parser.add_argument('--num-encoder-layers', type=int, default=3)
   393:     parser.add_argument('--k-neighbors', type=int, default=30)
   394:     parser.add_argument('--dropout', type=float, default=0.1)
   395:     parser.add_argument('--seed', type=int, default=42)
   396:     parser.add_argument('--output-dir', type=str, default='./output')
   397:     parser.add_argument('--num-workers', type=int, default=4)
   398:     parser.add_argument('--max-train-hours', type=float, default=3.0)
   399:     args = parser.parse_args()
   400: 
   401:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   402:     # Allowed keys: learning_rate, dropout, num_encoder_layers, batch_size.
   403:     CONFIG_OVERRIDES = {}
   404: 
   405:     for _k, _v in CONFIG_OVERRIDES.items():
   406:         if _k == 'learning_rate': args.lr = _v
   407:         elif _k == 'dropout': args.dropout = _v
   408:         elif _k == 'num_encoder_layers': args.num_encoder_layers = _v
   409:         elif _k == 'batch_size': args.batch_size = _v
   410: 
   411:     torch.manual_seed(args.seed)
   412:     np.random.seed(args.seed)
   413:     os.makedirs(args.output_dir, exist_ok=True)
   414:     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   415: 
   416:     # Load data
   417:     if args.dataset == 'TS':
   418:         # TS is test-only: train/val on CATH4.2 with remove_ts=True (drops
   419:         # proteins listed in CATH4.2/remove.json that overlap TS50/TS500 by
   420:         # sequence identity), test on TS50 (50 prot, filtered from
   421:         # ts50.json + ts500.json combined dataset).
   422:         train_ds = load_dataset('CATH4.2', args.data_root, 'train', remove_ts=True)
   423:         val_ds = load_dataset('CATH4.2', args.data_root, 'valid', remove_ts=True)
   424:         test_ds = load_dataset('TS', args.data_root, 'test')
   425:     else:
   426:         train_ds = load_dataset(args.dataset, args.data_root, 'train')
   427:         val_ds = load_dataset(args.dataset, args.data_root, 'valid')
   428:         test_ds = load_dataset(args.dataset, args.data_root, 'test')
   429: 
   430:     train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
   431:                               num_workers=args.num_workers, collate_fn=collate_fn,
   432:                               drop_last=True, pin_memory=True)
   433:     val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
   434:                             num_workers=args.num_workers, collate_fn=collate_fn,
   435:                             pin_memory=True)
   436:     test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
   437:                              num_workers=args.num_workers, collate_fn=collate_fn,
   438:                              pin_memory=True)
   439: 
   440:     # Build model
   441:     model = InverseFoldingModel(
   442:         hidden_dim=args.hidden_dim,
   443:         num_encoder_layers=args.num_encoder_layers,
   444:         k_neighbors=args.k_neighbors,
   445:         dropout=args.dropout,
   446:     ).to(device)
   447: 
   448:     param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
   449:     print(f"Model parameters: {param_count:,}", flush=True)
   450: 
   451:     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0,
   452:                                   betas=(0.9, 0.98), eps=1e-8)
   453:     total_steps = len(train_loader) * args.epochs
   454:     scheduler = torch.optim.lr_scheduler.OneCycleLR(
   455:         optimizer, max_lr=args.lr, total_steps=total_steps, three_phase=False
   456:     )
   457: 
   458:     best_val_recovery = 0.0
   459:     start_time = time.time()
   460:     max_seconds = args.max_train_hours * 3600
   461: 
   462:     for epoch in range(1, args.epochs + 1):
   463:         elapsed = time.time() - start_time
   464:         if elapsed > max_seconds:
   465:             print(f"Time limit reached ({args.max_train_hours}h). Stopping training.", flush=True)
   466:             break
   467: 
   468:         train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
   469: 
   470:         val_recovery, val_perplexity = evaluate(model, val_loader, device, "val")
   471:         print(f"TRAIN_METRICS epoch={epoch} val_recovery={val_recovery:.4f} val_perplexity={val_perplexity:.4f}", flush=True)
   472: 
   473:         if val_recovery > best_val_recovery:
   474:             best_val_recovery = val_recovery
   475:             # Use dataset-specific checkpoint name to avoid collision when
   476:             # CATH4.2 and CATH4.3 run in parallel with the same OUTPUT_DIR
   477:             ckpt_name = f'best_model_{args.dataset.replace(".", "_")}.pt'
   478:             ckpt_path = os.path.join(args.output_dir, ckpt_name)
   479:             torch.save(model.state_dict(), ckpt_path)
   480:             print(f"Saved best model (recovery={val_recovery:.4f})", flush=True)
   481: 
   482:     # Load best model and evaluate on test set
   483:     ckpt_name = f'best_model_{args.dataset.replace(".", "_")}.pt'
   484:     best_path = os.path.join(args.output_dir, ckpt_name)
   485:     if os.path.exists(best_path):
   486:         model.load_state_dict(torch.load(best_path, map_location=device))
   487: 
   488:     test_recovery, test_perplexity = evaluate(model, test_loader, device, "test")
   489:     print(f"TEST_METRICS recovery={test_recovery:.4f}", flush=True)
   490:     print(f"TEST_METRICS perplexity={test_perplexity:.4f}", flush=True)
   491: 
   492: 
   493: if __name__ == '__main__':
   494:     main()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `proteinmpnn` baseline — editable region  [READ-ONLY — reference implementation]

In `ProteinInvBench/custom_invfold.py`:

```python
Lines 86–356:
    83: # =====================================================================
    84: # EDITABLE SECTION START — StructureEncoder + InverseFoldingModel
    85: # =====================================================================
    86: # =====================================================================
    87: # EDITABLE SECTION START — ProteinMPNN baseline
    88: # =====================================================================
    89: 
    90: import numpy as np
    91: 
    92: class ProteinFeatures(nn.Module):
    93:     """Extract protein structural features: all-atom pairwise RBFs + positional.
    94: 
    95:     Computes 25 pairwise RBF distance matrices between all backbone atom
    96:     pairs (N, CA, C, O, Cb) following the reference ProteinMPNN implementation.
    97:     Total edge features = 25 * num_rbf + num_pos_emb.
    98:     """
    99:     def __init__(self, edge_features, node_features, num_pos_emb=16, num_rbf=16,
   100:                  top_k=30, augment_eps=0.0):
   101:         super().__init__()
   102:         self.edge_features = edge_features
   103:         self.node_features = node_features
   104:         self.top_k = top_k
   105:         self.augment_eps = augment_eps
   106:         self.num_rbf = num_rbf
   107:         self.num_pos_emb = num_pos_emb
   108: 
   109:         # 25 pairwise RBFs + positional encoding
   110:         edge_in = num_pos_emb + num_rbf * 25
   111:         node_in = 6  # forward + side-chain orientation vectors
   112: 
   113:         self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
   114:         self.norm_edges = nn.LayerNorm(edge_features)
   115:         self.node_embedding = nn.Linear(node_in, node_features, bias=True)
   116:         self.norm_nodes = nn.LayerNorm(node_features)
   117: 
   118:     def _pos_enc(self, E_idx):
   119:         N_nodes = E_idx.size(1)
   120:         ii = torch.arange(N_nodes, dtype=torch.float32, device=E_idx.device).view(1, -1, 1)
   121:         d = (E_idx.float() - ii).unsqueeze(-1)
   122:         frequency = torch.exp(
   123:             torch.arange(0, self.num_pos_emb, 2, dtype=torch.float32, device=E_idx.device)
   124:             * -(np.log(10000.0) / self.num_pos_emb)
   125:         )
   126:         angles = d * frequency.view(1, 1, 1, -1)
   127:         return torch.cat([torch.cos(angles), torch.sin(angles)], -1)
   128: 
   129:     def _dist(self, X, mask, eps=1e-6):
   130:         mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)
   131:         dX = X.unsqueeze(1) - X.unsqueeze(2)
   132:         D = (1. - mask_2D) * 10000 + mask_2D * torch.sqrt((dX ** 2).sum(3) + eps)
   133:         D_max, _ = D.max(-1, keepdim=True)
   134:         D_adjust = D + (1. - mask_2D) * (D_max + 1)
   135:         D_neighbors, E_idx = torch.topk(D_adjust, min(self.top_k, D_adjust.shape[-1]),
   136:                                          dim=-1, largest=False)
   137:         return D_neighbors, E_idx
   138: 
   139:     def _rbf_fn(self, D):
   140:         D_min, D_max, D_count = 2., 22., self.num_rbf
   141:         D_mu = torch.linspace(D_min, D_max, D_count, device=D.device).view(1, 1, 1, -1)
   142:         D_sigma = (D_max - D_min) / D_count
   143:         return torch.exp(-((D.unsqueeze(-1) - D_mu) / D_sigma) ** 2)
   144: 
   145:     def _get_rbf(self, A, B, E_idx):
   146:         """Compute pairwise distances between atoms A and B, gather for neighbors, apply RBF."""
   147:         D_AB = torch.sqrt(torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6)
   148:         # Gather neighbor distances
   149:         B_size, L, K = E_idx.shape
   150:         E_idx_expand = E_idx.unsqueeze(-1)  # (B, L, K, 1)
   151:         D_AB_expand = D_AB.unsqueeze(2).expand(-1, -1, K, -1)  # (B, L, K, L)
   152:         # For each node i and neighbor j = E_idx[i,k], get D_AB[i, j]
   153:         D_AB_neighbors = torch.gather(D_AB_expand, 3, E_idx_expand).squeeze(-1)  # (B, L, K)
   154:         return self._rbf_fn(D_AB_neighbors)
   155: 
   156:     def _orientations(self, X):
   157:         fwd = F.normalize(X[:, 1:, :] - X[:, :-1, :], dim=-1)
   158:         fwd = F.pad(fwd, (0, 0, 0, 1))
   159:         return fwd
   160: 
   161:     def _sidechains(self, X):
   162:         n, ca, c = X[:, :, 0, :], X[:, :, 1, :], X[:, :, 2, :]
   163:         u = F.normalize(n - ca, dim=-1)
   164:         v = F.normalize(c - ca, dim=-1)
   165:         return F.normalize(u - v, dim=-1)
   166: 
   167:     def forward(self, X, mask, residue_idx=None, chain_encoding=None):
   168:         B, L = X.shape[0], X.shape[1]
   169:         N = X[:, :, 0, :]   # N atoms
   170:         Ca = X[:, :, 1, :]  # CA atoms
   171:         C = X[:, :, 2, :]   # C atoms
   172:         O = X[:, :, 3, :]   # O atoms
   173: 
   174:         # Virtual Cb (beta carbon from N-CA-C geometry)
   175:         b = N - Ca
   176:         c = C - Ca
   177:         a = torch.cross(b, c, dim=-1)
   178:         Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca
   179: 
   180:         # KNN based on CA distances
   181:         D_neighbors, E_idx = self._dist(Ca, mask)
   182: 
   183:         # All 25 pairwise RBF distances (matching reference ProteinMPNN)
   184:         RBF_all = []
   185:         RBF_all.append(self._rbf_fn(D_neighbors))  # Ca-Ca
   186:         RBF_all.append(self._get_rbf(N, N, E_idx))
   187:         RBF_all.append(self._get_rbf(C, C, E_idx))
   188:         RBF_all.append(self._get_rbf(O, O, E_idx))
   189:         RBF_all.append(self._get_rbf(Cb, Cb, E_idx))
   190:         RBF_all.append(self._get_rbf(Ca, N, E_idx))
   191:         RBF_all.append(self._get_rbf(Ca, C, E_idx))
   192:         RBF_all.append(self._get_rbf(Ca, O, E_idx))
   193:         RBF_all.append(self._get_rbf(Ca, Cb, E_idx))
   194:         RBF_all.append(self._get_rbf(N, C, E_idx))
   195:         RBF_all.append(self._get_rbf(N, O, E_idx))
   196:         RBF_all.append(self._get_rbf(N, Cb, E_idx))
   197:         RBF_all.append(self._get_rbf(Cb, C, E_idx))
   198:         RBF_all.append(self._get_rbf(Cb, O, E_idx))
   199:         RBF_all.append(self._get_rbf(O, C, E_idx))
   200:         RBF_all.append(self._get_rbf(N, Ca, E_idx))
   201:         RBF_all.append(self._get_rbf(C, Ca, E_idx))
   202:         RBF_all.append(self._get_rbf(O, Ca, E_idx))
   203:         RBF_all.append(self._get_rbf(Cb, Ca, E_idx))
   204:         RBF_all.append(self._get_rbf(C, N, E_idx))
   205:         RBF_all.append(self._get_rbf(O, N, E_idx))
   206:         RBF_all.append(self._get_rbf(Cb, N, E_idx))
   207:         RBF_all.append(self._get_rbf(C, Cb, E_idx))
   208:         RBF_all.append(self._get_rbf(O, Cb, E_idx))
   209:         RBF_all.append(self._get_rbf(C, O, E_idx))
   210:         RBF_all = torch.cat(RBF_all, dim=-1)  # (B, L, K, 25*num_rbf)
   211: 
   212:         # Positional encoding
   213:         O_pos = self._pos_enc(E_idx)  # (B, L, K, num_pos_emb)
   214: 
   215:         # Edge features: positional + all-atom RBFs
   216:         E = torch.cat([O_pos, RBF_all], dim=-1)
   217: 
   218:         # Node features: forward + side-chain orientation vectors
   219:         O_fwd = self._orientations(Ca)
   220:         O_sc = self._sidechains(X)
   221:         V = torch.cat([O_fwd, O_sc], dim=-1)
   222: 
   223:         V = self.norm_nodes(self.node_embedding(V))
   224:         E = self.norm_edges(self.edge_embedding(E))
   225:         return V, E, E_idx
   226: 
   227: 
   228: def gather_nodes(h_V, E_idx):
   229:     """Gather node features for neighbor nodes."""
   230:     B, L, K = E_idx.shape
   231:     D = h_V.shape[-1]
   232:     h_V_expand = h_V.unsqueeze(2).expand(-1, -1, K, -1)
   233:     E_idx_expand = E_idx.unsqueeze(-1).expand(-1, -1, -1, D)
   234:     return torch.gather(h_V_expand, 1, E_idx_expand)
   235: 
   236: 
   237: def cat_neighbors_nodes(h_nodes, h_edges, E_idx):
   238:     """Concatenate neighbor node features with edge features."""
   239:     h_V_neighbors = gather_nodes(h_nodes, E_idx)
   240:     return torch.cat([h_edges, h_V_neighbors], dim=-1)
   241: 
   242: 
   243: class EncLayer(nn.Module):
   244:     """ProteinMPNN encoder layer with node and edge updates."""
   245:     def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
   246:         super().__init__()
   247:         self.num_hidden = num_hidden
   248:         self.scale = scale
   249:         self.dropout1 = nn.Dropout(dropout)
   250:         self.dropout2 = nn.Dropout(dropout)
   251:         self.dropout3 = nn.Dropout(dropout)
   252:         self.norm1 = nn.LayerNorm(num_hidden)
   253:         self.norm2 = nn.LayerNorm(num_hidden)
   254:         self.norm3 = nn.LayerNorm(num_hidden)
   255: 
   256:         self.W1 = nn.Linear(num_hidden + num_in, num_hidden)
   257:         self.W2 = nn.Linear(num_hidden, num_hidden)
   258:         self.W3 = nn.Linear(num_hidden, num_hidden)
   259:         self.W11 = nn.Linear(num_hidden + num_in, num_hidden)
   260:         self.W12 = nn.Linear(num_hidden, num_hidden)
   261:         self.W13 = nn.Linear(num_hidden, num_hidden)
   262:         self.act = nn.GELU()
   263:         self.dense = nn.Sequential(
   264:             nn.Linear(num_hidden, num_hidden * 4),
   265:             nn.GELU(),
   266:             nn.Linear(num_hidden * 4, num_hidden),
   267:         )
   268: 
   269:     def forward(self, h_V, h_E, E_idx, mask, mask_attend):
   270:         h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
   271:         h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
   272:         h_EV = torch.cat([h_V_expand, h_EV], -1)
   273:         h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
   274:         if mask_attend is not None:
   275:             h_message = mask_attend.unsqueeze(-1) * h_message
   276:         dh = h_message.sum(-2) / self.scale
   277:         h_V = self.norm1(h_V + self.dropout1(dh))
   278:         dh = self.dense(h_V)
   279:         h_V = self.norm2(h_V + self.dropout2(dh))
   280:         if mask is not None:
   281:             h_V = mask.unsqueeze(-1) * h_V
   282: 
   283:         h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
   284:         h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
   285:         h_EV = torch.cat([h_V_expand, h_EV], -1)
   286:         h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
   287:         h_E = self.norm3(h_E + self.dropout3(h_message))
   288:         return h_V, h_E
   289: 
   290: 
   291: class StructureEncoder(nn.Module):
   292:     """ProteinMPNN-style structure encoder with all-atom pairwise features."""
   293: 
   294:     def __init__(self, hidden_dim=128, num_layers=3, k_neighbors=30, dropout=0.1, num_rbf=16):
   295:         super().__init__()
   296:         self.hidden_dim = hidden_dim
   297:         self.k_neighbors = k_neighbors
   298: 
   299:         self.features = ProteinFeatures(
   300:             hidden_dim, hidden_dim, top_k=k_neighbors, augment_eps=0.0, num_rbf=num_rbf
   301:         )
   302:         self.W_e = nn.Linear(hidden_dim, hidden_dim, bias=True)
   303: 
   304:         self.encoder_layers = nn.ModuleList([
   305:             EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
   306:             for _ in range(num_layers)
   307:         ])
   308: 
   309:         self._init_params()
   310: 
   311:     def _init_params(self):
   312:         for p in self.parameters():
   313:             if p.dim() > 1:
   314:                 nn.init.xavier_uniform_(p)
   315: 
   316:     def forward(self, X, mask):
   317:         V, E, E_idx = self.features(X, mask)
   318: 
   319:         # Start with zero node features (per reference ProteinMPNN)
   320:         h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
   321:         h_E = self.W_e(E)
   322: 
   323:         mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
   324:         mask_attend = mask.unsqueeze(-1) * mask_attend
   325: 
   326:         for layer in self.encoder_layers:
   327:             h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
   328: 
   329:         return h_V
   330: 
   331: 
   332: class InverseFoldingModel(nn.Module):
   333:     """ProteinMPNN inverse folding model."""
   334: 
   335:     def __init__(self, hidden_dim=128, num_encoder_layers=3, k_neighbors=30,
   336:                  dropout=0.1, num_rbf=16):
   337:         super().__init__()
   338:         self.encoder = StructureEncoder(
   339:             hidden_dim=hidden_dim,
   340:             num_layers=num_encoder_layers,
   341:             k_neighbors=k_neighbors,
   342:             dropout=dropout,
   343:             num_rbf=num_rbf,
   344:         )
   345:         self.decoder = nn.Sequential(
   346:             nn.Linear(hidden_dim, hidden_dim),
   347:             nn.ReLU(),
   348:             nn.Dropout(dropout),
   349:             nn.Linear(hidden_dim, NUM_AA),
   350:         )
   351: 
   352:     def forward(self, X, mask):
   353:         h_V = self.encoder(X, mask)
   354:         logits = self.decoder(h_V)
   355:         log_probs = F.log_softmax(logits, dim=-1)
   356:         return log_probs
   357: 
   358: # =====================================================================
   359: # EDITABLE SECTION END

Lines 519–521:
   516:     parser.add_argument('--max-train-hours', type=float, default=3.0)
   517:     args = parser.parse_args()
   518: 
   519:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   520:     # Allowed keys: learning_rate, dropout, num_encoder_layers, batch_size.
   521:     CONFIG_OVERRIDES = {}
   522: 
   523:     for _k, _v in CONFIG_OVERRIDES.items():
   524:         if _k == 'learning_rate': args.lr = _v
```

### `pifold` baseline — editable region  [READ-ONLY — reference implementation]

In `ProteinInvBench/custom_invfold.py`:

```python
Lines 86–470:
    83: # =====================================================================
    84: # EDITABLE SECTION START — StructureEncoder + InverseFoldingModel
    85: # =====================================================================
    86: # =====================================================================
    87: # EDITABLE SECTION START — PiFold baseline
    88: # =====================================================================
    89: 
    90: import numpy as np
    91: 
    92: 
    93: def gather_nodes_pifold(h_V, E_idx):
    94:     """Gather node features for neighbor nodes. Dense batched version."""
    95:     B, L, K = int(E_idx.shape[0]), int(E_idx.shape[1]), int(E_idx.shape[2])
    96:     D = int(h_V.shape[-1])
    97:     h_V_expand = h_V.unsqueeze(2).expand(-1, -1, K, -1)
    98:     E_idx_expand = E_idx.unsqueeze(-1).expand(-1, -1, -1, D)
    99:     return torch.gather(h_V_expand, 1, E_idx_expand)
   100: 
   101: 
   102: class PiFoldAttention(nn.Module):
   103:     """Attention-based message passing layer inspired by PiFold's NeighborAttention."""
   104: 
   105:     def __init__(self, hidden_dim, edge_dim, num_heads=4, dropout=0.1):
   106:         super().__init__()
   107:         self.num_heads = num_heads
   108:         self.hidden_dim = hidden_dim
   109:         self.d_head = hidden_dim // num_heads
   110: 
   111:         # Value network: processes edge-concatenated features
   112:         self.W_V = nn.Sequential(
   113:             nn.Linear(edge_dim + hidden_dim, hidden_dim),
   114:             nn.GELU(),
   115:             nn.Linear(hidden_dim, hidden_dim),
   116:             nn.GELU(),
   117:             nn.Linear(hidden_dim, hidden_dim),
   118:         )
   119:         # Attention bias from node+edge features
   120:         self.Bias = nn.Sequential(
   121:             nn.Linear(hidden_dim + edge_dim + hidden_dim, hidden_dim),
   122:             nn.ReLU(),
   123:             nn.Linear(hidden_dim, hidden_dim),
   124:             nn.ReLU(),
   125:             nn.Linear(hidden_dim, num_heads),
   126:         )
   127:         self.W_O = nn.Linear(hidden_dim, hidden_dim, bias=False)
   128: 
   129:     def forward(self, h_V, h_E, E_idx, mask, mask_attend):
   130:         """
   131:         h_V: (B, L, D), h_E: (B, L, K, D_e), E_idx: (B, L, K), mask: (B, L)
   132:         """
   133:         B, L, K = int(E_idx.shape[0]), int(E_idx.shape[1]), int(E_idx.shape[2])
   134:         D = self.hidden_dim
   135:         n_heads = self.num_heads
   136:         d = self.d_head
   137: 
   138:         # Gather neighbor features
   139:         h_V_neighbors = gather_nodes_pifold(h_V, E_idx)  # (B, L, K, D)
   140:         h_V_expand = h_V.unsqueeze(2).expand(-1, -1, K, -1)  # (B, L, K, D)
   141: 
   142:         # Edge + neighbor concatenation for value
   143:         val_input = torch.cat([h_E, h_V_neighbors], dim=-1)  # (B, L, K, D_e+D)
   144:         V = self.W_V(val_input).view(B, L, K, n_heads, d)  # (B, L, K, H, d)
   145: 
   146:         # Attention logits
   147:         bias_input = torch.cat([h_V_expand, h_E, h_V_neighbors], dim=-1)
   148:         w = self.Bias(bias_input).view(B, L, K, n_heads, 1) / np.sqrt(d)
   149: 
   150:         # Mask and softmax
   151:         if mask_attend is not None:
   152:             w = w + (1.0 - mask_attend.unsqueeze(-1).unsqueeze(-1)) * (-1e9)
   153:         attend = torch.softmax(w, dim=2)  # (B, L, K, H, 1)
   154: 
   155:         # Aggregate
   156:         h_V_update = (attend * V).sum(dim=2).reshape(B, L, D)  # (B, L, D)
   157:         h_V_update = self.W_O(h_V_update)
   158:         return h_V_update
   159: 
   160: 
   161: class PiFoldEdgeMLP(nn.Module):
   162:     """Edge update network from PiFold."""
   163: 
   164:     def __init__(self, hidden_dim, edge_dim, dropout=0.1):
   165:         super().__init__()
   166:         self.W1 = nn.Linear(hidden_dim + edge_dim + hidden_dim, hidden_dim)
   167:         self.W2 = nn.Linear(hidden_dim, hidden_dim)
   168:         self.W3 = nn.Linear(hidden_dim, hidden_dim)
   169:         self.act = nn.GELU()
   170:         self.norm = nn.BatchNorm1d(hidden_dim)
   171:         self.dropout = nn.Dropout(dropout)
   172: 
   173:     def forward(self, h_V, h_E, E_idx, mask):
   174:         B, L, K = int(E_idx.shape[0]), int(E_idx.shape[1]), int(E_idx.shape[2])
   175:         h_V_neighbors = gather_nodes_pifold(h_V, E_idx)  # (B, L, K, D)
   176:         h_V_expand = h_V.unsqueeze(2).expand(-1, -1, K, -1)
   177:         h_EV = torch.cat([h_V_expand, h_E, h_V_neighbors], dim=-1)
   178:         h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
   179:         # Apply batch norm per-feature
   180:         D_e = int(h_E.shape[-1])
   181:         h_E_flat = h_E.reshape(-1, D_e)
   182:         h_msg_flat = h_message.reshape(-1, D_e)
   183:         h_E = self.norm(h_E_flat + self.dropout(h_msg_flat)).reshape(B, L, K, D_e)
   184:         return h_E
   185: 
   186: 
   187: class PiFoldEncoderLayer(nn.Module):
   188:     """PiFold encoder layer: attention + FFN + edge update + context gating."""
   189: 
   190:     def __init__(self, hidden_dim, edge_dim, num_heads=4, dropout=0.1):
   191:         super().__init__()
   192:         self.attention = PiFoldAttention(hidden_dim, edge_dim, num_heads, dropout)
   193:         self.norm1 = nn.BatchNorm1d(hidden_dim)
   194:         self.norm2 = nn.BatchNorm1d(hidden_dim)
   195:         self.dropout = nn.Dropout(dropout)
   196:         self.ffn = nn.Sequential(
   197:             nn.Linear(hidden_dim, hidden_dim * 4),
   198:             nn.ReLU(),
   199:             nn.Linear(hidden_dim * 4, hidden_dim),
   200:         )
   201:         self.edge_update = PiFoldEdgeMLP(hidden_dim, hidden_dim, dropout)
   202:         # Context gating
   203:         self.context_gate = nn.Sequential(
   204:             nn.Linear(hidden_dim, hidden_dim),
   205:             nn.ReLU(),
   206:             nn.Linear(hidden_dim, hidden_dim),
   207:             nn.ReLU(),
   208:             nn.Linear(hidden_dim, hidden_dim),
   209:             nn.Sigmoid(),
   210:         )
   211: 
   212:     def forward(self, h_V, h_E, E_idx, mask, mask_attend):
   213:         B, L = int(h_V.shape[0]), int(h_V.shape[1])
   214:         # Node update via attention
   215:         D = int(h_V.shape[-1])
   216:         dh = self.attention(h_V, h_E, E_idx, mask, mask_attend)
   217:         h_V_flat = h_V.reshape(-1, D)
   218:         dh_flat = dh.reshape(-1, D)
   219:         h_V = self.norm1(h_V_flat + self.dropout(dh_flat)).reshape(B, L, -1)
   220: 
   221:         dh = self.ffn(h_V)
   222:         h_V_flat = h_V.reshape(-1, D)
   223:         dh_flat = dh.reshape(-1, D)
   224:         h_V = self.norm2(h_V_flat + self.dropout(dh_flat)).reshape(B, L, -1)
   225: 
   226:         # Edge update
   227:         h_E = self.edge_update(h_V, h_E, E_idx, mask)
   228: 
   229:         # Context gating (global information)
   230:         # Mean pool over valid residues for context
   231:         mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
   232:         c_V = (h_V * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / mask_sum.unsqueeze(-1)  # (B, 1, D)
   233:         gate = self.context_gate(c_V.expand_as(h_V))
   234:         h_V = h_V * gate
   235: 
   236:         h_V = h_V * mask.unsqueeze(-1)
   237:         return h_V, h_E
   238: 
   239: 
   240: class StructureEncoder(nn.Module):
   241:     """PiFold-style structure encoder with rich geometric features."""
   242: 
   243:     def __init__(self, hidden_dim=128, num_layers=10, k_neighbors=30, dropout=0.1, num_rbf=16):
   244:         super().__init__()
   245:         self.hidden_dim = hidden_dim
   246:         self.k_neighbors = k_neighbors
   247:         self.num_rbf = num_rbf
   248: 
   249:         # PiFold uses rich multi-atom-pair features
   250:         # Node features: 6 intra-residue atom-pair RBFs + 6 dihedrals + 9 orientations
   251:         # = 6*num_rbf + 6 + 9
   252:         node_input_dim = 6 * num_rbf + 12
   253:         # Edge features: 15 inter-residue atom-pair RBFs + 4 angles + 12 directions + 16 pos_enc
   254:         edge_input_dim = 15 * num_rbf + 4 + 8 + 16
   255: 
   256:         # Virtual atoms (learned positions in local frame, like PiFold)
   257:         prior_matrix = [
   258:             [-0.58273431, 0.56802827, -0.54067466],
   259:             [0.0, 0.83867057, -0.54463904],
   260:             [0.01984028, -0.78380804, -0.54183614],
   261:         ]
   262:         self.virtual_atoms = nn.Parameter(torch.tensor(prior_matrix, dtype=torch.float32))
   263:         n_virtual = 3
   264:         # Add virtual atom pair distances to both node and edge features
   265:         node_input_dim += n_virtual * (n_virtual - 1) * num_rbf  # virtual-virtual pairs
   266:         edge_input_dim += n_virtual * num_rbf + n_virtual * (n_virtual - 1) * num_rbf
   267: 
   268:         self.node_embed = nn.Linear(node_input_dim, hidden_dim)
   269:         self.edge_embed = nn.Linear(edge_input_dim, hidden_dim)
   270:         self.norm_nodes = nn.BatchNorm1d(hidden_dim)
   271:         self.norm_edges = nn.BatchNorm1d(hidden_dim)
   272: 
   273:         self.W_v = nn.Sequential(
   274:             nn.Linear(hidden_dim, hidden_dim),
   275:             nn.LeakyReLU(),
   276:             nn.BatchNorm1d(hidden_dim),
   277:             nn.Linear(hidden_dim, hidden_dim),
   278:             nn.LeakyReLU(),
   279:             nn.BatchNorm1d(hidden_dim),
   280:             nn.Linear(hidden_dim, hidden_dim),
   281:         )
   282:         self.W_e = nn.Linear(hidden_dim, hidden_dim)
   283: 
   284:         self.layers = nn.ModuleList([
   285:             PiFoldEncoderLayer(hidden_dim, hidden_dim, num_heads=4, dropout=dropout)
   286:             for _ in range(num_layers)
   287:         ])
   288: 
   289:         self._init_params()
   290: 
   291:     def _init_params(self):
   292:         for name, p in self.named_parameters():
   293:             if name == 'virtual_atoms':
   294:                 continue
   295:             if p.dim() > 1:
   296:                 nn.init.xavier_uniform_(p)
   297: 
   298:     def _compute_features(self, X, mask, E_idx):
   299:         """Compute PiFold-style rich geometric features."""
   300:         B, L = int(X.shape[0]), int(X.shape[1])
   301:         K = int(E_idx.shape[2])
   302: 
   303:         N_pos = X[:, :, 0, :]
   304:         CA_pos = X[:, :, 1, :]
   305:         C_pos = X[:, :, 2, :]
   306:         O_pos = X[:, :, 3, :]
   307: 
   308:         # Virtual Cb and virtual atoms
   309:         b = CA_pos - N_pos
   310:         c = C_pos - CA_pos
   311:         a = torch.cross(b, c, dim=-1)
   312: 
   313:         va = self.virtual_atoms / torch.norm(self.virtual_atoms, dim=1, keepdim=True)
   314:         virtual_pos = []
   315:         for i in range(int(va.shape[0])):
   316:             vp = va[i, 0] * a + va[i, 1] * b + va[i, 2] * c + CA_pos
   317:             virtual_pos.append(vp)
   318: 
   319:         # --- Node features ---
   320:         def _node_rbf(_src, _dst):
   321:             D = torch.sqrt(((_src - _dst) ** 2).sum(-1) + 1e-6)  # (B, L)
   322:             return _rbf(D.unsqueeze(2), device=X.device).squeeze(2)  # (B, L, num_rbf)
   323: 
   324:         node_dist = []
   325:         for _src, _dst in [(CA_pos, N_pos), (CA_pos, C_pos), (CA_pos, O_pos),
   326:                       (N_pos, C_pos), (N_pos, O_pos), (O_pos, C_pos)]:
   327:             node_dist.append(_node_rbf(_src, _dst))
   328: 
   329:         # Virtual atom pair distances (node-level)
   330:         for i in range(len(virtual_pos)):
   331:             for j in range(i):
   332:                 node_dist.append(_node_rbf(virtual_pos[i], virtual_pos[j]))
   333:                 node_dist.append(_node_rbf(virtual_pos[j], virtual_pos[i]))
   334: 
   335:         V_dist = torch.cat(node_dist, dim=-1)
   336: 
   337:         # Dihedrals and orientations
   338:         V_dihedrals = _dihedrals(X)  # (B, L, 6)
   339:         V_orient = _orientations(X)  # (B, L, 6)
   340: 
   341:         node_feat = torch.cat([V_dist, V_dihedrals, V_orient], dim=-1)
   342: 
   343:         # --- Edge features ---
   344:         def _edge_rbf(_src, _dst, E_idx):
   345:             D = torch.sqrt(((_src[:, :, None, :] - _dst[:, None, :, :]) ** 2).sum(-1) + 1e-6)
   346:             D_neighbors = torch.gather(D, 2, E_idx)
   347:             return _rbf(D_neighbors, device=X.device)
   348: 
   349:         edge_dist = []
   350:         atom_pairs = [
   351:             (CA_pos, CA_pos), (CA_pos, C_pos), (C_pos, CA_pos),
   352:             (CA_pos, N_pos), (N_pos, CA_pos), (CA_pos, O_pos), (O_pos, CA_pos),
   353:             (C_pos, C_pos), (C_pos, N_pos), (N_pos, C_pos),
   354:             (C_pos, O_pos), (O_pos, C_pos), (N_pos, N_pos),
   355:             (N_pos, O_pos), (O_pos, O_pos),
   356:         ]
   357:         for _src, _dst in atom_pairs:
   358:             edge_dist.append(_edge_rbf(_src, _dst, E_idx))
   359: 
   360:         # Virtual atom edge features
   361:         for i in range(len(virtual_pos)):
   362:             edge_dist.append(_edge_rbf(virtual_pos[i], virtual_pos[i], E_idx))
   363:             for j in range(i):
   364:                 edge_dist.append(_edge_rbf(virtual_pos[i], virtual_pos[j], E_idx))
   365:                 edge_dist.append(_edge_rbf(virtual_pos[j], virtual_pos[i], E_idx))
   366: 
   367:         E_dist = torch.cat(edge_dist, dim=-1)
   368: 
   369:         # Edge angles and directions
   370:         CA_neighbors = gather_nodes_pifold(CA_pos, E_idx)  # (B, L, K, 3)
   371:         dX = CA_neighbors - CA_pos.unsqueeze(2)
   372:         dU = F.normalize(dX, dim=-1)
   373: 
   374:         fwd = F.normalize(CA_pos[:, 1:, :] - CA_pos[:, :-1, :], dim=-1)
   375:         fwd = F.pad(fwd, (0, 0, 0, 1))
   376:         n_vec = F.normalize(N_pos - CA_pos, dim=-1)
   377:         c_vec = F.normalize(C_pos - CA_pos, dim=-1)
   378:         o_vec = F.normalize(O_pos - CA_pos, dim=-1)
   379: 
   380:         # Direction features
   381:         E_direct = torch.cat([
   382:             (fwd.unsqueeze(2) * dU).sum(-1, keepdim=True),
   383:             (n_vec.unsqueeze(2) * dU).sum(-1, keepdim=True),
   384:             (c_vec.unsqueeze(2) * dU).sum(-1, keepdim=True),
   385:             (o_vec.unsqueeze(2) * dU).sum(-1, keepdim=True),
   386:             torch.cross(fwd.unsqueeze(2).expand_as(dU), dU, dim=-1).norm(dim=-1, keepdim=True),
   387:             torch.cross(n_vec.unsqueeze(2).expand_as(dU), dU, dim=-1).norm(dim=-1, keepdim=True),
   388:             torch.cross(c_vec.unsqueeze(2).expand_as(dU), dU, dim=-1).norm(dim=-1, keepdim=True),
   389:             torch.cross(o_vec.unsqueeze(2).expand_as(dU), dU, dim=-1).norm(dim=-1, keepdim=True),
   390:         ], dim=-1)  # (B, L, K, 8)
   391: 
   392:         # Edge angles (4): dihedral-like between consecutive neighbors
   393:         E_angles = torch.cat([
   394:             (dU[:, :, :, 0:1] * dU[:, :, :, 1:2]).clamp(-1, 1),
   395:             (dU[:, :, :, 0:1] * dU[:, :, :, 2:3]).clamp(-1, 1),
   396:             dU.norm(dim=-1, keepdim=True),
   397:             dX.norm(dim=-1, keepdim=True) / 20.0,
   398:         ], dim=-1)  # (B, L, K, 4)
   399: 
   400:         # Positional encoding
   401:         residue_idx = torch.arange(L, device=X.device).unsqueeze(0).expand(B, -1)
   402:         offset = residue_idx.unsqueeze(2) - torch.gather(
   403:             residue_idx.unsqueeze(2).expand(-1, -1, K), 1,
   404:             E_idx.clamp(0, L - 1)
   405:         )
   406:         pe_dim = 16
   407:         freq = torch.exp(torch.arange(0, pe_dim, 2, dtype=torch.float32, device=X.device) * -(np.log(10000.0) / pe_dim))
   408:         angles = offset.unsqueeze(-1).float() * freq
   409:         pos_enc = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
   410: 
   411:         edge_feat = torch.cat([E_dist, E_angles, E_direct, pos_enc], dim=-1)
   412: 
   413:         return node_feat, edge_feat
   414: 
   415:     def forward(self, X, mask):
   416:         B, L = int(X.shape[0]), int(X.shape[1])
   417:         X_ca = X[:, :, 1, :]
   418:         E_idx, _ = knn_graph(X_ca, mask, self.k_neighbors)
   419:         K = int(E_idx.shape[2])
   420: 
   421:         # Compute features
   422:         node_feat, edge_feat = self._compute_features(X, mask, E_idx)
   423: 
   424:         # Embed
   425:         h_V_flat = self.node_embed(node_feat).reshape(-1, self.hidden_dim)
   426:         h_V = self.norm_nodes(h_V_flat).reshape(B, L, self.hidden_dim)
   427:         h_V = self.W_v[0](h_V)
   428:         h_V_flat = h_V.reshape(-1, self.hidden_dim)
   429:         h_V = self.W_v[2](self.W_v[1](h_V_flat)).reshape(B, L, self.hidden_dim)
   430:         h_V = self.W_v[3](h_V)
   431:         h_V_flat = h_V.reshape(-1, self.hidden_dim)
   432:         h_V = self.W_v[5](self.W_v[4](h_V_flat)).reshape(B, L, self.hidden_dim)
   433:         h_V = self.W_v[6](h_V)
   434: 
   435:         h_E_flat = self.edge_embed(edge_feat).reshape(-1, self.hidden_dim)
   436:         h_E = self.norm_edges(h_E_flat).reshape(B, L, K, self.hidden_dim)
   437:         h_E = self.W_e(h_E)
   438: 
   439:         # Attention mask
   440:         mask_attend = torch.gather(mask.unsqueeze(2).expand(-1, -1, K), 1,
   441:                                     E_idx.clamp(0, L - 1))
   442:         mask_attend = mask.unsqueeze(-1) * mask_attend
   443: 
   444:         # Message passing
   445:         for layer in self.layers:
   446:             h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
   447: 
   448:         return h_V
   449: 
   450: 
   451: class InverseFoldingModel(nn.Module):
   452:     """PiFold inverse folding model with non-autoregressive MLP decoder."""
   453: 
   454:     def __init__(self, hidden_dim=128, num_encoder_layers=10, k_neighbors=30,
   455:                  dropout=0.1, num_rbf=16):
   456:         super().__init__()
   457:         self.encoder = StructureEncoder(
   458:             hidden_dim=hidden_dim,
   459:             num_layers=num_encoder_layers,
   460:             k_neighbors=k_neighbors,
   461:             dropout=dropout,
   462:             num_rbf=num_rbf,
   463:         )
   464:         self.decoder = nn.Linear(hidden_dim, NUM_AA)
   465: 
   466:     def forward(self, X, mask):
   467:         h_V = self.encoder(X, mask)
   468:         logits = self.decoder(h_V)
   469:         log_probs = F.log_softmax(logits, dim=-1)
   470:         return log_probs
   471: 
   472: # =====================================================================
   473: # EDITABLE SECTION END

Lines 633–635:
   630:     parser.add_argument('--max-train-hours', type=float, default=3.0)
   631:     args = parser.parse_args()
   632: 
   633:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   634:     # Allowed keys: learning_rate, dropout, num_encoder_layers, batch_size.
   635:     CONFIG_OVERRIDES = {'num_encoder_layers': 10, 'batch_size': 8}
   636: 
   637:     for _k, _v in CONFIG_OVERRIDES.items():
   638:         if _k == 'learning_rate': args.lr = _v
```

### `gvp` baseline — editable region  [READ-ONLY — reference implementation]

In `ProteinInvBench/custom_invfold.py`:

```python
Lines 86–411:
    83: # =====================================================================
    84: # EDITABLE SECTION START — StructureEncoder + InverseFoldingModel
    85: # =====================================================================
    86: # =====================================================================
    87: # EDITABLE SECTION START — GVP baseline
    88: # =====================================================================
    89: 
    90: import numpy as np
    91: 
    92: 
    93: def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    94:     """L2 norm clamped above eps."""
    95:     out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    96:     return torch.sqrt(out) if sqrt else out
    97: 
    98: 
    99: def gather_nodes_gvp(h_V, E_idx):
   100:     B, L, K = E_idx.shape
   101:     D = h_V.shape[-1]
   102:     h_V_expand = h_V.unsqueeze(2).expand(-1, -1, K, -1)
   103:     E_idx_expand = E_idx.unsqueeze(-1).expand(-1, -1, -1, D)
   104:     return torch.gather(h_V_expand, 1, E_idx_expand)
   105: 
   106: 
   107: def gather_vectors(V, E_idx):
   108:     """Gather vector features. V: (B, L, n_vec, 3), E_idx: (B, L, K)"""
   109:     B, L, K = E_idx.shape
   110:     nv = V.shape[2]
   111:     E_idx_v = E_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, nv, 3)
   112:     V_expand = V.unsqueeze(2).expand(-1, -1, K, -1, -1)
   113:     return torch.gather(V_expand, 1, E_idx_v)
   114: 
   115: 
   116: class GVPModule(nn.Module):
   117:     """Geometric Vector Perceptron — dense batched version.
   118: 
   119:     Processes tuples of (scalar, vector) features.
   120:     Scalar: (B, L, s_in) -> (B, L, s_out)
   121:     Vector: (B, L, v_in, 3) -> (B, L, v_out, 3)
   122:     """
   123: 
   124:     def __init__(self, in_dims, out_dims, activations=(F.relu, torch.sigmoid)):
   125:         super().__init__()
   126:         self.si, self.vi = in_dims
   127:         self.so, self.vo = out_dims
   128:         self.scalar_act, self.vector_act = activations
   129: 
   130:         if self.vi:
   131:             self.h_dim = max(self.vi, self.vo)
   132:             self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
   133:             self.ws = nn.Linear(self.h_dim + self.si, self.so)
   134:             if self.vo:
   135:                 self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
   136:                 self.wsv = nn.Linear(self.so, self.vo)
   137:         else:
   138:             self.ws = nn.Linear(self.si, self.so)
   139: 
   140:     def forward(self, s, v=None):
   141:         if self.vi and v is not None:
   142:             # v: (*, vi, 3)
   143:             v_t = v.transpose(-1, -2)  # (*, 3, vi)
   144:             vh = self.wh(v_t)  # (*, 3, h_dim)
   145:             vn = _norm_no_nan(vh, axis=-2)  # (*, h_dim)
   146:             s = self.ws(torch.cat([s, vn], -1))
   147:             if self.vo:
   148:                 v_out = self.wv(vh).transpose(-1, -2)  # (*, vo, 3)
   149:                 if self.scalar_act:
   150:                     gate = self.wsv(self.scalar_act(s))
   151:                 else:
   152:                     gate = self.wsv(s)
   153:                 v_out = v_out * torch.sigmoid(gate).unsqueeze(-1)
   154:             else:
   155:                 v_out = None
   156:         else:
   157:             s = self.ws(s)
   158:             v_out = None
   159: 
   160:         if self.scalar_act:
   161:             s = self.scalar_act(s)
   162:         return s, v_out
   163: 
   164: 
   165: class GVPLayerNorm(nn.Module):
   166:     """LayerNorm for GVP scalar+vector tuples."""
   167:     def __init__(self, dims):
   168:         super().__init__()
   169:         self.s_dim, self.v_dim = dims
   170:         self.norm_s = nn.LayerNorm(self.s_dim)
   171: 
   172:     def forward(self, s, v=None):
   173:         s = self.norm_s(s)
   174:         if v is not None and self.v_dim > 0:
   175:             vn = _norm_no_nan(v, axis=-1, keepdims=True)
   176:             v = v / vn.clamp(min=1e-5)  # unit vectors, scaled
   177:             v = v * vn  # restore magnitude (still normalized in mean)
   178:         return s, v
   179: 
   180: 
   181: class GVPConvLayer(nn.Module):
   182:     """GVP convolution layer — dense batched version.
   183: 
   184:     Message passing with GVP for both node and edge updates.
   185:     """
   186: 
   187:     def __init__(self, node_dims, edge_dims, drop_rate=0.1):
   188:         super().__init__()
   189:         self.node_s, self.node_v = node_dims
   190:         self.edge_s, self.edge_v = edge_dims
   191: 
   192:         # Message function: edge_s + 2*node_s, edge_v + 2*node_v -> node_s, node_v
   193:         msg_in_s = self.edge_s + 2 * self.node_s
   194:         msg_in_v = self.edge_v + 2 * self.node_v
   195:         self.msg_gvp = nn.Sequential(
   196:             GVPModule((msg_in_s, msg_in_v), (self.node_s, self.node_v)),
   197:             GVPModule((self.node_s, self.node_v), (self.node_s, self.node_v),
   198:                       activations=(None, None)),
   199:         )
   200: 
   201:         # Node update
   202:         self.ff_gvp = nn.Sequential(
   203:             GVPModule((self.node_s, self.node_v), (self.node_s * 4, self.node_v)),
   204:             GVPModule((self.node_s * 4, self.node_v), (self.node_s, self.node_v),
   205:                       activations=(None, None)),
   206:         )
   207: 
   208:         self.norm1 = GVPLayerNorm(node_dims)
   209:         self.norm2 = GVPLayerNorm(node_dims)
   210:         self.drop = nn.Dropout(drop_rate)
   211: 
   212:     def forward(self, h_s, h_v, e_s, e_v, E_idx, mask, mask_attend):
   213:         """
   214:         h_s: (B, L, node_s), h_v: (B, L, node_v, 3)
   215:         e_s: (B, L, K, edge_s), e_v: (B, L, K, edge_v, 3)
   216:         E_idx: (B, L, K), mask: (B, L), mask_attend: (B, L, K)
   217:         """
   218:         B, L, K = E_idx.shape
   219: 
   220:         # Gather neighbor node features
   221:         h_s_j = gather_nodes_gvp(h_s, E_idx)  # (B, L, K, node_s)
   222:         h_s_i = h_s.unsqueeze(2).expand(-1, -1, K, -1)
   223: 
   224:         # Build message input (scalar)
   225:         msg_s = torch.cat([h_s_i, e_s, h_s_j], dim=-1)  # (B, L, K, msg_in_s)
   226: 
   227:         # Build message input (vector)
   228:         if h_v is not None:
   229:             h_v_j = gather_vectors(h_v, E_idx)  # (B, L, K, node_v, 3)
   230:             h_v_i = h_v.unsqueeze(2).expand(-1, -1, K, -1, -1)
   231:             if e_v is not None:
   232:                 msg_v = torch.cat([h_v_i, e_v, h_v_j], dim=-2)  # (B, L, K, msg_in_v, 3)
   233:             else:
   234:                 msg_v = torch.cat([h_v_i, h_v_j], dim=-2)
   235:         else:
   236:             msg_v = e_v
   237: 
   238:         # Apply message GVP
   239:         for layer in self.msg_gvp:
   240:             msg_s, msg_v = layer(msg_s, msg_v)
   241: 
   242:         # Mask and aggregate
   243:         mask_expand = mask_attend.unsqueeze(-1)
   244:         msg_s = msg_s * mask_expand
   245:         if msg_v is not None:
   246:             msg_v = msg_v * mask_expand.unsqueeze(-1)
   247: 
   248:         # Sum aggregation
   249:         num_neighbors = mask_attend.sum(dim=-1, keepdim=True).clamp(min=1)
   250:         agg_s = msg_s.sum(dim=2) / num_neighbors
   251:         if msg_v is not None:
   252:             agg_v = msg_v.sum(dim=2) / num_neighbors.unsqueeze(-1)
   253:         else:
   254:             agg_v = None
   255: 
   256:         # Residual + norm
   257:         h_s_res, h_v_res = self.norm1(h_s + self.drop(agg_s),
   258:                                         h_v + self.drop(agg_v) if h_v is not None and agg_v is not None else h_v)
   259: 
   260:         # Feed-forward
   261:         ff_s, ff_v = h_s_res, h_v_res
   262:         for layer in self.ff_gvp:
   263:             ff_s, ff_v = layer(ff_s, ff_v)
   264: 
   265:         h_s_out, h_v_out = self.norm2(h_s_res + self.drop(ff_s),
   266:                                         h_v_res + self.drop(ff_v) if h_v_res is not None and ff_v is not None else h_v_res)
   267: 
   268:         # Mask
   269:         h_s_out = h_s_out * mask.unsqueeze(-1)
   270:         if h_v_out is not None:
   271:             h_v_out = h_v_out * mask.unsqueeze(-1).unsqueeze(-1)
   272: 
   273:         return h_s_out, h_v_out
   274: 
   275: 
   276: class StructureEncoder(nn.Module):
   277:     """GVP-based structure encoder.
   278: 
   279:     Uses geometric vector perceptrons for SE(3)-equivariant message passing.
   280:     Node features: scalar (6) = dihedrals; vector (3) = local frame vectors.
   281:     Edge features: scalar (32) = RBF distances + positional; vector (1) = direction.
   282:     """
   283: 
   284:     def __init__(self, hidden_dim=128, num_layers=3, k_neighbors=30, dropout=0.1, num_rbf=16):
   285:         super().__init__()
   286:         self.hidden_dim = hidden_dim
   287:         self.k_neighbors = k_neighbors
   288:         self.num_rbf = num_rbf
   289: 
   290:         # Dimensions
   291:         self.node_s_in = 6    # dihedral sin/cos
   292:         self.node_v_in = 3    # 3 direction vectors
   293:         self.node_s_h = 100   # hidden scalar dim (GVP default)
   294:         self.node_v_h = 16    # hidden vector dim
   295:         self.edge_s_in = num_rbf + 16  # RBF + positional encoding
   296:         self.edge_v_in = 1    # direction unit vector
   297:         self.edge_s_h = 32    # hidden edge scalar
   298:         self.edge_v_h = 1     # hidden edge vector
   299: 
   300:         # Input projections
   301:         self.W_v = GVPModule(
   302:             (self.node_s_in, self.node_v_in),
   303:             (self.node_s_h, self.node_v_h),
   304:             activations=(None, None)
   305:         )
   306:         self.norm_v = GVPLayerNorm((self.node_s_h, self.node_v_h))
   307: 
   308:         self.W_e = GVPModule(
   309:             (self.edge_s_in, self.edge_v_in),
   310:             (self.edge_s_h, self.edge_v_h),
   311:             activations=(None, None)
   312:         )
   313:         self.norm_e = GVPLayerNorm((self.edge_s_h, self.edge_v_h))
   314: 
   315:         # Encoder layers
   316:         self.encoder_layers = nn.ModuleList([
   317:             GVPConvLayer(
   318:                 (self.node_s_h, self.node_v_h),
   319:                 (self.edge_s_h, self.edge_v_h),
   320:                 drop_rate=dropout
   321:             )
   322:             for _ in range(num_layers)
   323:         ])
   324: 
   325:         # Output projection to scalar hidden_dim
   326:         self.out_proj = nn.Linear(self.node_s_h, hidden_dim)
   327: 
   328:     def forward(self, X, mask):
   329:         B, L = int(X.shape[0]), int(X.shape[1])
   330:         X_ca = X[:, :, 1, :]
   331: 
   332:         # Build KNN graph
   333:         E_idx, D_neighbors = knn_graph(X_ca, mask, self.k_neighbors)
   334:         K = int(E_idx.shape[2])
   335: 
   336:         # Node features
   337:         # Scalar: dihedral angles
   338:         node_s = _dihedrals(X)  # (B, L, 6)
   339: 
   340:         # Vector: local frame vectors (CA->N, CA->C, CA->O unit vectors)
   341:         N_pos, CA_pos, C_pos, O_pos = X[:, :, 0], X[:, :, 1], X[:, :, 2], X[:, :, 3]
   342:         v_cn = F.normalize(N_pos - CA_pos, dim=-1)   # (B, L, 3)
   343:         v_cc = F.normalize(C_pos - CA_pos, dim=-1)
   344:         v_co = F.normalize(O_pos - CA_pos, dim=-1)
   345:         node_v = torch.stack([v_cn, v_cc, v_co], dim=2)  # (B, L, 3, 3)
   346: 
   347:         # Edge features
   348:         # Scalar: RBF distances + positional encoding
   349:         rbf = _rbf(D_neighbors, device=X.device)  # (B, L, K, num_rbf)
   350:         residue_idx = torch.arange(L, device=X.device).unsqueeze(0).expand(B, -1)
   351:         offset = residue_idx.unsqueeze(2) - torch.gather(
   352:             residue_idx.unsqueeze(2).expand(-1, -1, K), 1,
   353:             E_idx.clamp(0, L - 1)
   354:         )
   355:         pe_dim = 16
   356:         freq = torch.exp(torch.arange(0, pe_dim, 2, dtype=torch.float32, device=X.device) * -(np.log(10000.0) / pe_dim))
   357:         angles = offset.unsqueeze(-1).float() * freq
   358:         pos_enc = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
   359:         edge_s = torch.cat([rbf, pos_enc], dim=-1)  # (B, L, K, num_rbf+16)
   360: 
   361:         # Vector: direction to neighbors
   362:         CA_neighbors = gather_nodes_gvp(CA_pos, E_idx)  # (B, L, K, 3)
   363:         edge_dir = F.normalize(CA_neighbors - CA_pos.unsqueeze(2), dim=-1)  # (B, L, K, 3)
   364:         edge_v = edge_dir.unsqueeze(3)  # (B, L, K, 1, 3)
   365: 
   366:         # Project inputs
   367:         h_s, h_v = self.W_v(node_s, node_v)
   368:         h_s, h_v = self.norm_v(h_s, h_v)
   369: 
   370:         e_s, e_v = self.W_e(edge_s, edge_v)
   371:         e_s, e_v = self.norm_e(e_s, e_v)
   372: 
   373:         # Attention mask
   374:         mask_attend = torch.gather(mask.unsqueeze(2).expand(-1, -1, K), 1,
   375:                                     E_idx.clamp(0, L - 1))
   376:         mask_attend = mask.unsqueeze(-1) * mask_attend
   377: 
   378:         # Message passing
   379:         for layer in self.encoder_layers:
   380:             h_s, h_v = layer(h_s, h_v, e_s, e_v, E_idx, mask, mask_attend)
   381: 
   382:         # Project to output dim
   383:         h_V = self.out_proj(h_s)
   384:         return h_V
   385: 
   386: 
   387: class InverseFoldingModel(nn.Module):
   388:     """GVP inverse folding model."""
   389: 
   390:     def __init__(self, hidden_dim=128, num_encoder_layers=3, k_neighbors=30,
   391:                  dropout=0.1, num_rbf=16):
   392:         super().__init__()
   393:         self.encoder = StructureEncoder(
   394:             hidden_dim=hidden_dim,
   395:             num_layers=num_encoder_layers,
   396:             k_neighbors=k_neighbors,
   397:             dropout=dropout,
   398:             num_rbf=num_rbf,
   399:         )
   400:         self.decoder = nn.Sequential(
   401:             nn.Linear(hidden_dim, hidden_dim),
   402:             nn.ReLU(),
   403:             nn.Dropout(dropout),
   404:             nn.Linear(hidden_dim, NUM_AA),
   405:         )
   406: 
   407:     def forward(self, X, mask):
   408:         h_V = self.encoder(X, mask)
   409:         logits = self.decoder(h_V)
   410:         log_probs = F.log_softmax(logits, dim=-1)
   411:         return log_probs
   412: 
   413: # =====================================================================
   414: # EDITABLE SECTION END

Lines 574–576:
   571:     parser.add_argument('--max-train-hours', type=float, default=3.0)
   572:     args = parser.parse_args()
   573: 
   574:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   575:     # Allowed keys: learning_rate, dropout, num_encoder_layers, batch_size.
   576:     CONFIG_OVERRIDES = {}
   577: 
   578:     for _k, _v in CONFIG_OVERRIDES.items():
   579:         if _k == 'learning_rate': args.lr = _v
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
