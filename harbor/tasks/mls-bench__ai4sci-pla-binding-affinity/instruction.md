# MLS-Bench: ai4sci-pla-binding-affinity

# Task: Protein-Ligand Binding Affinity Prediction

## Research Question
Design a GNN architecture that effectively models protein-ligand interactions to predict binding affinity (`-logKd/Ki`) from 3D structural data. The goal is to learn representations that capture both intra-molecular structure and inter-molecular interactions between ligand and protein pocket.

## Background
Predicting the binding affinity between a drug molecule (ligand) and its target protein is a central task in structure-based drug design. Given a protein-ligand complex represented as a heterogeneous graph, the model must predict the binding strength (`-logKd/Ki`). Key challenges include:
- **Heterogeneous interactions**: The complex contains two types of molecules (ligand and pocket) with distinct chemistry, connected by non-covalent inter-molecular edges.
- **Geometric features**: Edge features encode rich 3D geometric information (angles, triangle areas, distances between neighboring atoms).
- **Bidirectional modeling**: Inter-molecular interactions can be modeled from ligand→pocket and pocket→ligand perspectives, potentially yielding different insights.

Existing approaches include:
- **EHIGN** (Yang, Zhong, et al., "Interaction-Based Inductive Bias in Graph Neural Networks: Enhancing Protein-Ligand Binding Affinity Predictions From 3D Structures", IEEE TPAMI 2024, vol. 46, pp. 8191–8208). Heterogeneous interaction layers (CIG covalent intra + NIG non-covalent inter) with bidirectional ligand↔pocket prediction. Code: https://github.com/guaguabujianle/EHIGN.
- **GIGN** (Yang, Zhong, Lv, Dong, Chen, "Geometric Interaction Graph Neural Network for Predicting Protein–Ligand Binding Affinities from 3D Structures", J. Phys. Chem. Lett. 2023, 14(8):2020–2033). Single heterogeneous interaction layer that unifies covalent and non-covalent interactions with translation/rotation-invariant geometric features. Code: https://github.com/guaguabujianle/GIGN.
- **SchNet** (Schütt et al., "SchNet: A continuous-filter convolutional neural network for modeling quantum interactions", NeurIPS 2017; arXiv:1706.08566). Continuous-filter convolution with Gaussian-RBF distance expansion, applied here on the heterogeneous complex graph.
- **EGNN** (Satorras, Hoogeboom, Welling, "E(n) Equivariant Graph Neural Networks", ICML 2021; arXiv:2102.09844). E(n)-equivariant message passing using distances as scalar edge features.

## What to Implement
Implement the `AffinityModel` class in `custom_pla.py`. You must implement:
1. `__init__(self, lig_dim, poc_dim, intra_edge_dim, inter_edge_dim)`: Set up your model architecture.
2. `forward(self, batch: PLABatch) -> Tensor`: Return predictions of shape `[B]`.

## Batch Format (PLABatch)
```python
@dataclass
class PLABatch:
    # Ligand graph
    lig_x: Tensor              # [total_lig_atoms, 35] atom features
    lig_edge_index: Tensor     # [2, total_lig_edges] COO format
    lig_edge_attr: Tensor      # [total_lig_edges, 17] bond + geometric features
    lig_batch: Tensor          # [total_lig_atoms] graph assignment (0..B-1)

    # Pocket graph
    poc_x: Tensor              # [total_poc_atoms, 35] atom features
    poc_edge_index: Tensor     # [2, total_poc_edges] COO format
    poc_edge_attr: Tensor      # [total_poc_edges, 17] bond + geometric features
    poc_batch: Tensor          # [total_poc_atoms] graph assignment (0..B-1)

    # Inter-molecular edges (ligand -> pocket)
    l2p_edge_index: Tensor     # [2, total_l2p_edges] (src=ligand, dst=pocket)
    l2p_edge_attr: Tensor      # [total_l2p_edges, 11] geometric features

    # Inter-molecular edges (pocket -> ligand)
    p2l_edge_index: Tensor     # [2, total_p2l_edges] (src=pocket, dst=ligand)
    p2l_edge_attr: Tensor      # [total_p2l_edges, 11] geometric features

    # Metadata
    num_lig_atoms: List[int]   # per-complex ligand atom counts
    num_poc_atoms: List[int]   # per-complex pocket atom counts
    inter_batch: Tensor        # [total_l2p_edges] graph assignment for inter edges

    # Target
    labels: Tensor             # [B] binding affinity (-logKd/Ki)
```

## Atom Features (35 dimensions)
One-hot encodings of: element (C/N/O/S/F/P/Cl/Br/I/Unknown = 10), degree (0–6 = 7), implicit valence (0–6 = 7), hybridization (SP/SP2/SP3/SP3D/SP3D2 = 5), aromatic (1), total Hs (0–4 = 5).

## Intra-molecular Edge Features (17 dimensions)
Bond type (4) + conjugated (1) + in_ring (1) + geometric features (11): angle statistics (max/sum/mean), triangle area statistics (max/sum/mean), neighbor distance statistics (max/sum/mean), pairwise distances (L1, L2).

## Inter-molecular Edge Features (11 dimensions)
Geometric features only (same 11-dim encoding as intra-molecular geometric features): computed between ligand-pocket atom pairs within a 5 Å distance threshold.

## Fixed Pipeline
Graph construction, feature extraction, train/test splits, optimizer, schedule, loss (regression on `-logKd/Ki`), and evaluation harness are all fixed by the scaffold. The contribution is the `AffinityModel` architecture only.

## Evaluation
The model is trained on PDBbind v2020 (general + refined) and tested on three benchmarks:
- **PDBbind 2013 core set** (107 complexes): CASF-2013 benchmark.
- **PDBbind 2016 core set** (285 complexes): CASF-2016 benchmark.
- **PDBbind 2019 holdout** (4366 complexes): Temporal split.

Metrics: **RMSE** (lower is better), **Rp** / Pearson correlation (higher is better).

### Note on Baseline Reproduction
The baselines (EHIGN / GIGN / SchNet / EGNN) are paper-faithful re-implementations on this task's data pipeline (PDBbind **v2020** general+refined → temporal/CASF splits, with intra/inter graph features regenerated from raw PDB/SDF). The original EHIGN and GIGN papers train on PDBbind v2016/v2019 with their own preprocessing, so absolute numbers and the relative ordering between baselines on this leaderboard may differ from the published numbers. The baseline implementations are intentionally NOT tuned to recover the published ordering; they are kept faithful to the published methods.

## Editable Region
The section between `EDITABLE SECTION START` and `EDITABLE SECTION END` markers in `custom_pla.py` is editable. You may define helper classes, layers, or functions within this region. The region must contain an `AffinityModel` class with the specified interface.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/EHIGN_PLA/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `EHIGN_PLA/custom_pla.py`
- editable lines **101–191**




## Readable Context


### `EHIGN_PLA/custom_pla.py`  [EDITABLE — lines 101–191 only]

```python
     1: """
     2: Protein-Ligand Binding Affinity Prediction — Self-contained template.
     3: Predicts binding affinity (-logKd/Ki) on PDBbind benchmarks using
     4: heterogeneous protein-ligand interaction graphs.
     5: 
     6: Structure:
     7:   Lines 1-105:   FIXED — Imports, constants, PLABatch dataclass
     8:   Lines 106-250: EDITABLE — AffinityModel class (starter: separate GNN + concat readout)
     9:   Lines 251+:    FIXED — Data loading, training loop, evaluation
    10: """
    11: import os
    12: import sys
    13: import math
    14: import argparse
    15: import warnings
    16: import numpy as np
    17: import pandas as pd
    18: from dataclasses import dataclass
    19: from typing import Optional, Dict, List, Tuple
    20: 
    21: import torch
    22: import torch.nn as nn
    23: import torch.nn.functional as F
    24: from torch.utils.data import Dataset, DataLoader
    25: from scipy.stats import pearsonr
    26: from sklearn.metrics import mean_squared_error
    27: 
    28: warnings.filterwarnings("ignore")
    29: 
    30: # =====================================================================
    31: # Constants — feature dimensions from EHIGN_PLA preprocessing
    32: # =====================================================================
    33: 
    34: # Atom features: element(10) + degree(7) + valence(7) + hybridization(5) + aromatic(1) + Hs(5) = 35
    35: LIGAND_ATOM_DIM = 35
    36: POCKET_ATOM_DIM = 35
    37: 
    38: # Intra-molecular edge features: bond_type(4) + conjugated(1) + in_ring(1) + geometric(11) = 17
    39: INTRA_EDGE_DIM = 17
    40: 
    41: # Inter-molecular edge features: geometric only = 11
    42: INTER_EDGE_DIM = 11
    43: 
    44: 
    45: @dataclass
    46: class PLABatch:
    47:     """Batched protein-ligand complex data for binding affinity prediction.
    48: 
    49:     All graphs in the batch are merged into single tensors with offset indices.
    50:     """
    51:     # Ligand graph
    52:     lig_x: torch.Tensor              # [total_lig_atoms, 35]
    53:     lig_edge_index: torch.Tensor     # [2, total_lig_edges]
    54:     lig_edge_attr: torch.Tensor      # [total_lig_edges, 17]
    55:     lig_batch: torch.Tensor          # [total_lig_atoms] graph assignment
    56: 
    57:     # Pocket graph
    58:     poc_x: torch.Tensor              # [total_poc_atoms, 35]
    59:     poc_edge_index: torch.Tensor     # [2, total_poc_edges]
    60:     poc_edge_attr: torch.Tensor      # [total_poc_edges, 17]
    61:     poc_batch: torch.Tensor          # [total_poc_atoms] graph assignment
    62: 
    63:     # Inter-molecular edges (ligand -> pocket)
    64:     l2p_edge_index: torch.Tensor     # [2, total_l2p_edges] (src=lig, dst=poc)
    65:     l2p_edge_attr: torch.Tensor      # [total_l2p_edges, 11]
    66: 
    67:     # Inter-molecular edges (pocket -> ligand)
    68:     p2l_edge_index: torch.Tensor     # [2, total_p2l_edges] (src=poc, dst=lig)
    69:     p2l_edge_attr: torch.Tensor      # [total_p2l_edges, 11]
    70: 
    71:     # Metadata
    72:     num_lig_atoms: List[int]         # per-complex ligand atom counts
    73:     num_poc_atoms: List[int]         # per-complex pocket atom counts
    74:     inter_batch: torch.Tensor        # [total_l2p_edges] graph assignment for inter edges
    75: 
    76:     # Target
    77:     labels: torch.Tensor             # [B]
    78: 
    79: 
    80: # Helper: positions for 3D coordinate-based models (not pre-computed in current data)
    81: # Models can use edge_attr geometric features which encode distance/angle info.
    82: 
    83: # =====================================================================
    84: # FIXED SECTION END (line 95)
    85: # =====================================================================
    86: 
    87: 
    88: # Below are some utility functions available in the editable section:
    89: 
    90: def scatter_mean(src, index, dim_size):
    91:     """Scatter mean: average src values by index."""
    92:     out = torch.zeros(dim_size, src.size(-1), device=src.device)
    93:     count = torch.zeros(dim_size, 1, device=src.device)
    94:     out.index_add_(0, index, src)
    95:     count.index_add_(0, index, torch.ones(src.size(0), 1, device=src.device))
    96:     return out / count.clamp(min=1)
    97: 
    98: 
    99: # =====================================================================
   100: # EDITABLE SECTION START — AffinityModel + helper modules
   101: # =====================================================================
   102: 
   103: class SimpleGNNLayer(nn.Module):
   104:     """Simple message passing layer with edge features."""
   105: 
   106:     def __init__(self, node_dim, edge_dim, hidden_dim):
   107:         super().__init__()
   108:         self.msg_mlp = nn.Sequential(
   109:             nn.Linear(node_dim + edge_dim, hidden_dim),
   110:             nn.ReLU(),
   111:             nn.Linear(hidden_dim, hidden_dim),
   112:         )
   113:         self.update_mlp = nn.Sequential(
   114:             nn.Linear(node_dim + hidden_dim, hidden_dim),
   115:             nn.BatchNorm1d(hidden_dim),
   116:             nn.ReLU(),
   117:         )
   118: 
   119:     def forward(self, x, edge_index, edge_attr):
   120:         src, dst = edge_index
   121:         msg_input = torch.cat([x[src], edge_attr], dim=-1)
   122:         msg = self.msg_mlp(msg_input)
   123:         agg = torch.zeros(x.size(0), msg.size(-1), device=x.device)
   124:         agg.index_add_(0, dst, msg)
   125:         out = self.update_mlp(torch.cat([x, agg], dim=-1))
   126:         return out
   127: 
   128: 
   129: class AffinityModel(nn.Module):
   130:     """Starter model: Separate GNN encoders for ligand/pocket + mean pooling readout.
   131: 
   132:     A simple baseline that processes ligand and pocket graphs independently
   133:     with message passing, then concatenates their pooled representations
   134:     for final prediction. Does NOT use inter-molecular edges.
   135:     """
   136: 
   137:     def __init__(self, lig_dim, poc_dim, intra_edge_dim, inter_edge_dim):
   138:         super().__init__()
   139:         hidden_dim = 256
   140:         num_layers = 3
   141: 
   142:         # Ligand encoder
   143:         self.lig_embed = nn.Linear(lig_dim, hidden_dim)
   144:         self.lig_convs = nn.ModuleList([
   145:             SimpleGNNLayer(hidden_dim, intra_edge_dim, hidden_dim)
   146:             for _ in range(num_layers)
   147:         ])
   148: 
   149:         # Pocket encoder
   150:         self.poc_embed = nn.Linear(poc_dim, hidden_dim)
   151:         self.poc_convs = nn.ModuleList([
   152:             SimpleGNNLayer(hidden_dim, intra_edge_dim, hidden_dim)
   153:             for _ in range(num_layers)
   154:         ])
   155: 
   156:         # Prediction head
   157:         self.readout = nn.Sequential(
   158:             nn.Linear(hidden_dim * 2, hidden_dim),
   159:             nn.ReLU(),
   160:             nn.Dropout(0.1),
   161:             nn.Linear(hidden_dim, 1),
   162:         )
   163: 
   164:     def forward(self, batch: PLABatch) -> torch.Tensor:
   165:         """
   166:         Args:
   167:             batch: PLABatch with heterogeneous graph data.
   168:         Returns:
   169:             predictions: [B] binding affinity values
   170:         """
   171:         # Encode ligand
   172:         lig_h = self.lig_embed(batch.lig_x)
   173:         for conv in self.lig_convs:
   174:             lig_h = conv(lig_h, batch.lig_edge_index, batch.lig_edge_attr) + lig_h
   175: 
   176:         # Encode pocket
   177:         poc_h = self.poc_embed(batch.poc_x)
   178:         for conv in self.poc_convs:
   179:             poc_h = conv(poc_h, batch.poc_edge_index, batch.poc_edge_attr) + poc_h
   180: 
   181:         # Pool per graph
   182:         num_graphs = batch.labels.size(0)
   183:         lig_pool = scatter_mean(lig_h, batch.lig_batch, num_graphs)
   184:         poc_pool = scatter_mean(poc_h, batch.poc_batch, num_graphs)
   185: 
   186:         # Concatenate and predict
   187:         combined = torch.cat([lig_pool, poc_pool], dim=-1)
   188:         pred = self.readout(combined).squeeze(-1)
   189:         return pred
   190: 
   191: # =====================================================================
   192: # EDITABLE SECTION END
   193: # =====================================================================
   194: 
   195: 
   196: # =====================================================================
   197: # FIXED — Data loading, collation, training, and evaluation
   198: # =====================================================================
   199: 
   200: class PLADataset(Dataset):
   201:     """Dataset for protein-ligand binding affinity prediction.
   202:     Loads pre-converted .pt files containing graph tensors.
   203:     """
   204: 
   205:     def __init__(self, data_path):
   206:         self.data = torch.load(data_path, weights_only=False)
   207:         print(f"Loaded {len(self.data)} complexes from {data_path}")
   208: 
   209:     def __len__(self):
   210:         return len(self.data)
   211: 
   212:     def __getitem__(self, idx):
   213:         return self.data[idx]
   214: 
   215: 
   216: def collate_pla(batch_list):
   217:     """Collate variable-size protein-ligand complexes into PLABatch."""
   218:     lig_x_list, lig_ei_list, lig_ea_list, lig_batch_list = [], [], [], []
   219:     poc_x_list, poc_ei_list, poc_ea_list, poc_batch_list = [], [], [], []
   220:     l2p_ei_list, l2p_ea_list, p2l_ei_list, p2l_ea_list = [], [], [], []
   221:     inter_batch_list = []
   222:     labels_list = []
   223:     num_lig_list, num_poc_list = [], []
   224: 
   225:     lig_offset = 0
   226:     poc_offset = 0
   227: 
   228:     for i, item in enumerate(batch_list):
   229:         n_lig = item['num_lig_atoms']
   230:         n_poc = item['num_poc_atoms']
   231: 
   232:         lig_x_list.append(item['lig_x'])
   233:         lig_batch_list.append(torch.full((n_lig,), i, dtype=torch.long))
   234: 
   235:         poc_x_list.append(item['poc_x'])
   236:         poc_batch_list.append(torch.full((n_poc,), i, dtype=torch.long))
   237: 
   238:         # Offset edge indices
   239:         if item['lig_edge_index'].size(1) > 0:
   240:             lig_ei_list.append(item['lig_edge_index'] + lig_offset)
   241:             lig_ea_list.append(item['lig_edge_attr'])
   242: 
   243:         if item['poc_edge_index'].size(1) > 0:
   244:             poc_ei_list.append(item['poc_edge_index'] + poc_offset)
   245:             poc_ea_list.append(item['poc_edge_attr'])
   246: 
   247:         # Inter-molecular edges: src from ligand, dst from pocket (l2p)
   248:         if item['l2p_edge_index'].size(1) > 0:
   249:             l2p_ei = item['l2p_edge_index'].clone()
   250:             l2p_ei[0] += lig_offset  # ligand source
   251:             l2p_ei[1] += poc_offset  # pocket dest
   252:             l2p_ei_list.append(l2p_ei)
   253:             l2p_ea_list.append(item['l2p_edge_attr'])
   254:             inter_batch_list.append(torch.full((l2p_ei.size(1),), i, dtype=torch.long))
   255: 
   256:         # Inter-molecular edges: src from pocket, dst from ligand (p2l)
   257:         if item['p2l_edge_index'].size(1) > 0:
   258:             p2l_ei = item['p2l_edge_index'].clone()
   259:             p2l_ei[0] += poc_offset  # pocket source
   260:             p2l_ei[1] += lig_offset  # ligand dest
   261:             p2l_ei_list.append(p2l_ei)
   262:             p2l_ea_list.append(item['p2l_edge_attr'])
   263: 
   264:         labels_list.append(item['label'])
   265:         num_lig_list.append(n_lig)
   266:         num_poc_list.append(n_poc)
   267: 
   268:         lig_offset += n_lig
   269:         poc_offset += n_poc
   270: 
   271:     # Concatenate
   272:     lig_x = torch.cat(lig_x_list, dim=0)
   273:     lig_batch = torch.cat(lig_batch_list, dim=0)
   274:     poc_x = torch.cat(poc_x_list, dim=0)
   275:     poc_batch = torch.cat(poc_batch_list, dim=0)
   276: 
   277:     lig_edge_index = torch.cat(lig_ei_list, dim=1) if lig_ei_list else torch.zeros(2, 0, dtype=torch.long)
   278:     lig_edge_attr = torch.cat(lig_ea_list, dim=0) if lig_ea_list else torch.zeros(0, INTRA_EDGE_DIM)
   279:     poc_edge_index = torch.cat(poc_ei_list, dim=1) if poc_ei_list else torch.zeros(2, 0, dtype=torch.long)
   280:     poc_edge_attr = torch.cat(poc_ea_list, dim=0) if poc_ea_list else torch.zeros(0, INTRA_EDGE_DIM)
   281: 
   282:     l2p_edge_index = torch.cat(l2p_ei_list, dim=1) if l2p_ei_list else torch.zeros(2, 0, dtype=torch.long)
   283:     l2p_edge_attr = torch.cat(l2p_ea_list, dim=0) if l2p_ea_list else torch.zeros(0, INTER_EDGE_DIM)
   284:     p2l_edge_index = torch.cat(p2l_ei_list, dim=1) if p2l_ei_list else torch.zeros(2, 0, dtype=torch.long)
   285:     p2l_edge_attr = torch.cat(p2l_ea_list, dim=0) if p2l_ea_list else torch.zeros(0, INTER_EDGE_DIM)
   286:     inter_batch = torch.cat(inter_batch_list, dim=0) if inter_batch_list else torch.zeros(0, dtype=torch.long)
   287: 
   288:     labels = torch.cat(labels_list, dim=0)
   289: 
   290:     return PLABatch(
   291:         lig_x=lig_x, lig_edge_index=lig_edge_index, lig_edge_attr=lig_edge_attr, lig_batch=lig_batch,
   292:         poc_x=poc_x, poc_edge_index=poc_edge_index, poc_edge_attr=poc_edge_attr, poc_batch=poc_batch,
   293:         l2p_edge_index=l2p_edge_index, l2p_edge_attr=l2p_edge_attr,
   294:         p2l_edge_index=p2l_edge_index, p2l_edge_attr=p2l_edge_attr,
   295:         num_lig_atoms=num_lig_list, num_poc_atoms=num_poc_list,
   296:         inter_batch=inter_batch,
   297:         labels=labels,
   298:     )
   299: 
   300: 
   301: def batch_to_device(batch, device):
   302:     return PLABatch(
   303:         lig_x=batch.lig_x.to(device),
   304:         lig_edge_index=batch.lig_edge_index.to(device),
   305:         lig_edge_attr=batch.lig_edge_attr.to(device),
   306:         lig_batch=batch.lig_batch.to(device),
   307:         poc_x=batch.poc_x.to(device),
   308:         poc_edge_index=batch.poc_edge_index.to(device),
   309:         poc_edge_attr=batch.poc_edge_attr.to(device),
   310:         poc_batch=batch.poc_batch.to(device),
   311:         l2p_edge_index=batch.l2p_edge_index.to(device),
   312:         l2p_edge_attr=batch.l2p_edge_attr.to(device),
   313:         p2l_edge_index=batch.p2l_edge_index.to(device),
   314:         p2l_edge_attr=batch.p2l_edge_attr.to(device),
   315:         num_lig_atoms=batch.num_lig_atoms,
   316:         num_poc_atoms=batch.num_poc_atoms,
   317:         inter_batch=batch.inter_batch.to(device),
   318:         labels=batch.labels.to(device),
   319:     )
   320: 
   321: 
   322: # =====================================================================
   323: # Training and evaluation
   324: # =====================================================================
   325: 
   326: def train_epoch(model, loader, optimizer, device):
   327:     model.train()
   328:     total_loss = 0.0
   329:     n_batches = 0
   330: 
   331:     for batch in loader:
   332:         batch = batch_to_device(batch, device)
   333:         optimizer.zero_grad()
   334: 
   335:         # Models with custom multi-head losses (e.g. EHIGN's 3-term dual-head loss)
   336:         # can expose `compute_loss(batch, labels)` returning a scalar loss directly.
   337:         # Single-head models fall back to plain MSE on forward() output.
   338:         if hasattr(model, 'compute_loss'):
   339:             loss = model.compute_loss(batch, batch.labels)
   340:         else:
   341:             pred = model(batch)
   342:             loss = F.mse_loss(pred, batch.labels)
   343: 
   344:         loss.backward()
   345:         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
   346:         optimizer.step()
   347: 
   348:         total_loss += loss.item()
   349:         n_batches += 1
   350: 
   351:     return total_loss / max(n_batches, 1)
   352: 
   353: 
   354: @torch.no_grad()
   355: def evaluate(model, loader, device):
   356:     model.eval()
   357:     all_preds = []
   358:     all_labels = []
   359: 
   360:     for batch in loader:
   361:         batch = batch_to_device(batch, device)
   362:         pred = model(batch)
   363:         all_preds.append(pred.cpu().numpy())
   364:         all_labels.append(batch.labels.cpu().numpy())
   365: 
   366:     preds = np.concatenate(all_preds)
   367:     labels = np.concatenate(all_labels)
   368: 
   369:     rmse = float(np.sqrt(mean_squared_error(labels, preds)))
   370:     rp = float(pearsonr(preds, labels)[0])
   371: 
   372:     return rmse, rp
   373: 
   374: 
   375: def train_and_evaluate(args):
   376:     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   377:     print(f"Using device: {device}")
   378: 
   379:     # Load data
   380:     data_dir = args.data_dir
   381:     train_ds = PLADataset(os.path.join(data_dir, 'train_data.pt'))
   382:     valid_ds = PLADataset(os.path.join(data_dir, 'valid_data.pt'))
   383:     test_ds = PLADataset(os.path.join(data_dir, f'{args.test_set}_data.pt'))
   384: 
   385:     train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
   386:                               collate_fn=collate_pla, num_workers=4, drop_last=True)
   387:     valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
   388:                               collate_fn=collate_pla, num_workers=4)
   389:     test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
   390:                              collate_fn=collate_pla, num_workers=4)
   391: 
   392:     print(f"Train: {len(train_ds)}, Valid: {len(valid_ds)}, Test ({args.test_set}): {len(test_ds)}")
   393: 
   394:     # Model
   395:     model = AffinityModel(
   396:         lig_dim=LIGAND_ATOM_DIM,
   397:         poc_dim=POCKET_ATOM_DIM,
   398:         intra_edge_dim=INTRA_EDGE_DIM,
   399:         inter_edge_dim=INTER_EDGE_DIM,
   400:     ).to(device)
   401:     print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
   402: 
   403:     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
   404: 
   405:     # Training with early stopping
   406:     best_val_rmse = float('inf')
   407:     best_epoch = 0
   408:     patience_counter = 0
   409: 
   410:     for epoch in range(1, args.epochs + 1):
   411:         train_loss = train_epoch(model, train_loader, optimizer, device)
   412:         val_rmse, val_rp = evaluate(model, valid_loader, device)
   413: 
   414:         print(f"TRAIN_METRICS epoch={epoch} loss={train_loss:.6f} val_rmse={val_rmse:.4f} val_rp={val_rp:.4f}")
   415: 
   416:         if val_rmse < best_val_rmse:
   417:             best_val_rmse = val_rmse
   418:             best_epoch = epoch
   419:             patience_counter = 0
   420:             os.makedirs(args.output_dir, exist_ok=True)
   421:             torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
   422:         else:
   423:             patience_counter += 1
   424:             if patience_counter >= args.patience:
   425:                 print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
   426:                 break
   427: 
   428:     # Load best model and evaluate on test set
   429:     model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt'), weights_only=True))
   430:     test_rmse, test_rp = evaluate(model, test_loader, device)
   431:     print(f"TEST_METRICS rmse={test_rmse:.6f} rp={test_rp:.6f}")
   432:     print(f"Best val RMSE: {best_val_rmse:.4f} at epoch {best_epoch}")
   433: 
   434: 
   435: def main():
   436:     parser = argparse.ArgumentParser(description="Protein-Ligand Binding Affinity Prediction")
   437:     parser.add_argument('--test-set', type=str, required=True,
   438:                         choices=['test2013', 'test2016', 'test2019'])
   439:     parser.add_argument('--data-dir', type=str, required=True)
   440:     parser.add_argument('--epochs', type=int, default=800)
   441:     parser.add_argument('--batch-size', type=int, default=128)
   442:     parser.add_argument('--lr', type=float, default=1e-4)
   443:     parser.add_argument('--patience', type=int, default=50)
   444:     parser.add_argument('--seed', type=int, default=42)
   445:     parser.add_argument('--output-dir', type=str, default='./output')
   446:     args = parser.parse_args()
   447: 
   448:     # Set seeds
   449:     torch.manual_seed(args.seed)
   450:     np.random.seed(args.seed)
   451:     if torch.cuda.is_available():
   452:         torch.cuda.manual_seed_all(args.seed)
   453: 
   454:     train_and_evaluate(args)
   455: 
   456: 
   457: if __name__ == '__main__':
   458:     main()
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **PDBbind2013** — wall-clock budget `00:59:00`, compute share `1.0`
- **PDBbind2016** — wall-clock budget `00:59:00`, compute share `1.0`
- **PDBbind2019** — wall-clock budget `00:59:00`, compute share `1.0`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.

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


### `ehign` baseline — editable region  [READ-ONLY — reference implementation]

In `EHIGN_PLA/custom_pla.py`:

```python
Lines 101–323:
    98: 
    99: # =====================================================================
   100: # EDITABLE SECTION START — AffinityModel + helper modules
   101: # =====================================================================
   102: # EDITABLE SECTION START — EHIGN: Heterogeneous Interaction Graph Network
   103: # =====================================================================
   104: 
   105: class CIGConv(nn.Module):
   106:     """Covalent Interaction Graph Convolution (intra-molecular).
   107:     Message: ReLU(src + edge_feat), sum aggregation, residual, MLP.
   108:     """
   109:     def __init__(self, input_dim, output_dim, drop=0.1):
   110:         super().__init__()
   111:         self.mlp = nn.Sequential(
   112:             nn.Linear(input_dim, output_dim),
   113:             nn.Dropout(drop),
   114:             nn.LeakyReLU(),
   115:             nn.BatchNorm1d(output_dim),
   116:         )
   117: 
   118:     def forward(self, x, edge_index, edge_attr):
   119:         src, dst = edge_index
   120:         msg = F.relu(x[src] + edge_attr)
   121:         agg = torch.zeros_like(x)
   122:         agg.index_add_(0, dst, msg)
   123:         rst = x + agg  # residual
   124:         return self.mlp(rst)
   125: 
   126: 
   127: class NIGConv(nn.Module):
   128:     """Non-covalent Interaction Graph Convolution (inter-molecular).
   129:     Uses edge weights as multiplicative gates on source features, mean aggregation.
   130:     Matches original: when in_feats == out_feats, fc_neigh applied AFTER aggregation.
   131:     """
   132:     def __init__(self, in_feats, out_feats, feat_drop=0.0):
   133:         super().__init__()
   134:         self.feat_drop = nn.Dropout(feat_drop)
   135:         self.fc_neigh = nn.Linear(in_feats, out_feats, bias=False)
   136:         self.fc_self = nn.Linear(in_feats, out_feats, bias=False)
   137:         self.bias = nn.Parameter(torch.zeros(out_feats))
   138:         nn.init.xavier_uniform_(self.fc_self.weight)
   139:         nn.init.xavier_uniform_(self.fc_neigh.weight)
   140: 
   141:     def forward(self, x_src, x_dst, edge_index, edge_weight, num_dst):
   142:         x_src = self.feat_drop(x_src)
   143:         x_dst = self.feat_drop(x_dst)
   144:         src, dst = edge_index
   145:         # Edge-weighted messages: src_feat * edge_weight (element-wise)
   146:         msg = x_src[src] * edge_weight
   147:         # Mean aggregation
   148:         agg = torch.zeros(num_dst, msg.size(-1), device=msg.device)
   149:         count = torch.zeros(num_dst, 1, device=msg.device)
   150:         agg.index_add_(0, dst, msg)
   151:         count.index_add_(0, dst, torch.ones(src.size(0), 1, device=src.device))
   152:         h_neigh = self.fc_neigh(agg / count.clamp(min=1))
   153:         return self.fc_self(x_dst) + h_neigh + self.bias
   154: 
   155: 
   156: class FC(nn.Module):
   157:     """Fully connected prediction head."""
   158:     def __init__(self, d_in, d_hidden, n_layers, dropout, n_out):
   159:         super().__init__()
   160:         layers = []
   161:         for j in range(n_layers):
   162:             if j == 0:
   163:                 layers += [nn.Linear(d_in, d_hidden), nn.Dropout(dropout),
   164:                            nn.LeakyReLU(), nn.BatchNorm1d(d_hidden)]
   165:             if j == n_layers - 1:
   166:                 layers.append(nn.Linear(d_hidden, n_out))
   167:             else:
   168:                 layers += [nn.Linear(d_hidden, d_hidden), nn.Dropout(dropout),
   169:                            nn.LeakyReLU(), nn.BatchNorm1d(d_hidden)]
   170:         self.layers = nn.ModuleList(layers)
   171: 
   172:     def forward(self, h):
   173:         for layer in self.layers:
   174:             h = layer(h)
   175:         return h
   176: 
   177: 
   178: class AffinityModel(nn.Module):
   179:     """EHIGN: Edge-enhanced Heterogeneous Interaction Graph Network.
   180: 
   181:     Uses CIGConv for intra-molecular and NIGConv for inter-molecular message passing.
   182:     HeteroGraphConv pattern: all edge types computed in parallel, outputs summed per node type.
   183:     Dual bidirectional prediction with attention-based bias correction.
   184:     """
   185:     def __init__(self, lig_dim, poc_dim, intra_edge_dim, inter_edge_dim):
   186:         super().__init__()
   187:         H = 256
   188:         num_layers = 3
   189:         self.lin_node_l = nn.Linear(lig_dim, H)
   190:         self.lin_node_p = nn.Linear(poc_dim, H)
   191:         self.lin_edge_ll = nn.Linear(intra_edge_dim, H)
   192:         self.lin_edge_pp = nn.Linear(intra_edge_dim, H)
   193:         self.lin_edge_lp = nn.Linear(inter_edge_dim, H)
   194:         self.lin_edge_pl = nn.Linear(inter_edge_dim, H)
   195: 
   196:         self.cig_l = nn.ModuleList([CIGConv(H, H) for _ in range(num_layers)])
   197:         self.cig_p = nn.ModuleList([CIGConv(H, H) for _ in range(num_layers)])
   198:         self.nig_lp = nn.ModuleList([NIGConv(H, H, 0.1) for _ in range(num_layers)])
   199:         self.nig_pl = nn.ModuleList([NIGConv(H, H, 0.1) for _ in range(num_layers)])
   200: 
   201:         # Atom-atom affinity heads
   202:         self.prj_lp_src = nn.Linear(H, H)
   203:         self.prj_lp_dst = nn.Linear(H, H)
   204:         self.prj_lp_edge = nn.Linear(H, H)
   205:         self.fc_lp = nn.Linear(H, 1)
   206:         self.prj_pl_src = nn.Linear(H, H)
   207:         self.prj_pl_dst = nn.Linear(H, H)
   208:         self.prj_pl_edge = nn.Linear(H, H)
   209:         self.fc_pl = nn.Linear(H, 1)
   210: 
   211:         # Bias correction (L->P direction)
   212:         self.bc_lp_prj_src = nn.Linear(H, H)
   213:         self.bc_lp_prj_dst = nn.Linear(H, H)
   214:         self.bc_lp_prj_edge = nn.Linear(H, H)
   215:         self.bc_lp_att = nn.Sequential(nn.PReLU(), nn.Linear(H, 1))
   216:         self.bc_lp_w_src = nn.Linear(H, H)
   217:         self.bc_lp_w_dst = nn.Linear(H, H)
   218:         self.bc_lp_w_edge = nn.Linear(H, H)
   219:         self.bc_lp_fc = FC(H, 200, 2, 0.1, 1)
   220: 
   221:         # Bias correction (P->L direction)
   222:         self.bc_pl_prj_src = nn.Linear(H, H)
   223:         self.bc_pl_prj_dst = nn.Linear(H, H)
   224:         self.bc_pl_prj_edge = nn.Linear(H, H)
   225:         self.bc_pl_att = nn.Sequential(nn.PReLU(), nn.Linear(H, 1))
   226:         self.bc_pl_w_src = nn.Linear(H, H)
   227:         self.bc_pl_w_dst = nn.Linear(H, H)
   228:         self.bc_pl_w_edge = nn.Linear(H, H)
   229:         self.bc_pl_fc = FC(H, 200, 2, 0.1, 1)
   230: 
   231:     def _edge_softmax(self, scores, batch_idx, num_graphs):
   232:         max_scores = torch.zeros(num_graphs, 1, device=scores.device).fill_(-1e9)
   233:         max_scores.index_reduce_(0, batch_idx, scores, 'amax', include_self=True)
   234:         exp_scores = torch.exp(scores - max_scores[batch_idx])
   235:         sum_exp = torch.zeros(num_graphs, 1, device=scores.device)
   236:         sum_exp.index_add_(0, batch_idx, exp_scores)
   237:         return exp_scores / sum_exp[batch_idx].clamp(min=1e-8)
   238: 
   239:     def _forward_heads(self, batch: PLABatch):
   240:         """Compute both dual prediction heads. Returns (pred_lp, pred_pl) each [B]."""
   241:         B = batch.labels.size(0)
   242:         # Project features
   243:         lig_h = self.lin_node_l(batch.lig_x)
   244:         poc_h = self.lin_node_p(batch.poc_x)
   245:         lig_e = self.lin_edge_ll(batch.lig_edge_attr)
   246:         poc_e = self.lin_edge_pp(batch.poc_edge_attr)
   247:         lp_e = self.lin_edge_lp(batch.l2p_edge_attr)
   248:         pl_e = self.lin_edge_pl(batch.p2l_edge_attr)
   249: 
   250:         # Message passing: HeteroGraphConv pattern — parallel compute, sum aggregate
   251:         for i in range(len(self.cig_l)):
   252:             # Save inputs (all convs use same input features)
   253:             lig_in, poc_in = lig_h, poc_h
   254: 
   255:             # Intra-molecular (CIGConv has internal residual)
   256:             lig_intra = self.cig_l[i](lig_in, batch.lig_edge_index, lig_e)
   257:             poc_intra = self.cig_p[i](poc_in, batch.poc_edge_index, poc_e)
   258: 
   259:             # Inter-molecular (NIGConv with edge weights)
   260:             lig_inter = torch.zeros_like(lig_in)
   261:             poc_inter = torch.zeros_like(poc_in)
   262:             if batch.l2p_edge_index.size(1) > 0:
   263:                 poc_inter = self.nig_lp[i](lig_in, poc_in, batch.l2p_edge_index, lp_e, poc_in.size(0))
   264:             if batch.p2l_edge_index.size(1) > 0:
   265:                 lig_inter = self.nig_pl[i](poc_in, lig_in, batch.p2l_edge_index, pl_e, lig_in.size(0))
   266: 
   267:             # Sum aggregation per destination node type
   268:             lig_h = lig_intra + lig_inter
   269:             poc_h = poc_intra + poc_inter
   270: 
   271:         # Atom-atom affinities (L->P)
   272:         l2p_src, l2p_dst = batch.l2p_edge_index
   273:         i_lp = self.prj_lp_edge(lp_e) * self.prj_lp_src(lig_h)[l2p_src] * self.prj_lp_dst(poc_h)[l2p_dst]
   274:         logit_lp = self.fc_lp(i_lp)
   275:         pred_lp = torch.zeros(B, 1, device=logit_lp.device)
   276:         pred_lp.index_add_(0, batch.inter_batch, logit_lp)
   277: 
   278:         # Atom-atom affinities (P->L)
   279:         p2l_src, p2l_dst = batch.p2l_edge_index
   280:         p2l_batch = batch.lig_batch[p2l_dst]
   281:         i_pl = self.prj_pl_edge(pl_e) * self.prj_pl_src(poc_h)[p2l_src] * self.prj_pl_dst(lig_h)[p2l_dst]
   282:         logit_pl = self.fc_pl(i_pl)
   283:         pred_pl = torch.zeros(B, 1, device=logit_pl.device)
   284:         pred_pl.index_add_(0, p2l_batch, logit_pl)
   285: 
   286:         # Bias correction (L->P)
   287:         w_lp = self.bc_lp_prj_src(lig_h)[l2p_src] + self.bc_lp_prj_dst(poc_h)[l2p_dst] + self.bc_lp_prj_edge(lp_e)
   288:         a_lp = self._edge_softmax(self.bc_lp_att(w_lp), batch.inter_batch, B)
   289:         s_lp = a_lp * self.bc_lp_w_edge(lp_e) * self.bc_lp_w_src(lig_h)[l2p_src] * self.bc_lp_w_dst(poc_h)[l2p_dst]
   290:         bias_lp_agg = torch.zeros(B, s_lp.size(-1), device=s_lp.device)
   291:         bias_lp_agg.index_add_(0, batch.inter_batch, s_lp)
   292:         bias_lp = self.bc_lp_fc(bias_lp_agg)
   293: 
   294:         # Bias correction (P->L)
   295:         w_pl = self.bc_pl_prj_src(poc_h)[p2l_src] + self.bc_pl_prj_dst(lig_h)[p2l_dst] + self.bc_pl_prj_edge(pl_e)
   296:         a_pl = self._edge_softmax(self.bc_pl_att(w_pl), p2l_batch, B)
   297:         s_pl = a_pl * self.bc_pl_w_edge(pl_e) * self.bc_pl_w_src(poc_h)[p2l_src] * self.bc_pl_w_dst(lig_h)[p2l_dst]
   298:         bias_pl_agg = torch.zeros(B, s_pl.size(-1), device=s_pl.device)
   299:         bias_pl_agg.index_add_(0, p2l_batch, s_pl)
   300:         bias_pl = self.bc_pl_fc(bias_pl_agg)
   301: 
   302:         pred_lp_final = (pred_lp - bias_lp).squeeze(-1)
   303:         pred_pl_final = (pred_pl - bias_pl).squeeze(-1)
   304:         return pred_lp_final, pred_pl_final
   305: 
   306:     def forward(self, batch: PLABatch) -> torch.Tensor:
   307:         pred_lp, pred_pl = self._forward_heads(batch)
   308:         return (pred_lp + pred_pl) / 2
   309: 
   310:     def compute_loss(self, batch: PLABatch, labels: torch.Tensor) -> torch.Tensor:
   311:         """EHIGN 3-term dual-head loss (paper: guaguabujianle/EHIGN_PLA train.py#L852):
   312:             loss = (MSE(pred_lp, y) + MSE(pred_pl, y) + MSE(pred_lp, pred_pl)) / 3
   313:         The third term is a consistency regularizer between the two bidirectional heads.
   314:         """
   315:         pred_lp, pred_pl = self._forward_heads(batch)
   316:         loss = (F.mse_loss(pred_lp, labels)
   317:                 + F.mse_loss(pred_pl, labels)
   318:                 + F.mse_loss(pred_lp, pred_pl)) / 3
   319:         return loss
   320: 
   321: # =====================================================================
   322: # EDITABLE SECTION END
   323: # =====================================================================
   324: # EDITABLE SECTION END
   325: # =====================================================================
   326: 
```

### `gign` baseline — editable region  [READ-ONLY — reference implementation]

In `EHIGN_PLA/custom_pla.py`:

```python
Lines 101–213:
    98: 
    99: # =====================================================================
   100: # EDITABLE SECTION START — AffinityModel + helper modules
   101: # =====================================================================
   102: # EDITABLE SECTION START — GIGN: Geometric Interaction Graph Network
   103: # =====================================================================
   104: 
   105: class GINLayer(nn.Module):
   106:     """GIN convolution with edge features."""
   107:     def __init__(self, node_dim, edge_dim, hidden_dim):
   108:         super().__init__()
   109:         self.eps = nn.Parameter(torch.zeros(1))
   110:         self.edge_proj = nn.Linear(edge_dim, node_dim)
   111:         self.mlp = nn.Sequential(
   112:             nn.Linear(node_dim, hidden_dim),
   113:             nn.BatchNorm1d(hidden_dim),
   114:             nn.ReLU(),
   115:             nn.Linear(hidden_dim, hidden_dim),
   116:         )
   117: 
   118:     def forward(self, x, edge_index, edge_attr):
   119:         src, dst = edge_index
   120:         msg = x[src] + self.edge_proj(edge_attr)
   121:         agg = torch.zeros_like(x)
   122:         agg.index_add_(0, dst, msg)
   123:         return self.mlp((1 + self.eps) * x + agg)
   124: 
   125: 
   126: class InterGINLayer(nn.Module):
   127:     """GIN convolution for inter-molecular edges."""
   128:     def __init__(self, src_dim, dst_dim, edge_dim, hidden_dim):
   129:         super().__init__()
   130:         self.edge_proj = nn.Linear(edge_dim, src_dim)
   131:         self.mlp = nn.Sequential(
   132:             nn.Linear(src_dim + dst_dim, hidden_dim),
   133:             nn.BatchNorm1d(hidden_dim),
   134:             nn.ReLU(),
   135:             nn.Linear(hidden_dim, hidden_dim),
   136:         )
   137: 
   138:     def forward(self, x_src, x_dst, edge_index, edge_attr, num_dst):
   139:         src, dst = edge_index
   140:         msg = x_src[src] + self.edge_proj(edge_attr)
   141:         agg = torch.zeros(num_dst, msg.size(-1), device=msg.device)
   142:         count = torch.zeros(num_dst, 1, device=msg.device)
   143:         agg.index_add_(0, dst, msg)
   144:         count.index_add_(0, dst, torch.ones(src.size(0), 1, device=msg.device))
   145:         agg = agg / count.clamp(min=1)
   146:         return self.mlp(torch.cat([x_dst, agg], dim=-1))
   147: 
   148: 
   149: class AffinityModel(nn.Module):
   150:     """GIGN: Geometric Interaction Graph Network.
   151: 
   152:     Uses GIN-style message passing for both intra- and inter-molecular graphs.
   153:     Readout via interaction-weighted sum over inter-molecular edges.
   154:     """
   155:     def __init__(self, lig_dim, poc_dim, intra_edge_dim, inter_edge_dim):
   156:         super().__init__()
   157:         H = 256
   158:         num_layers = 3
   159: 
   160:         self.lig_embed = nn.Linear(lig_dim, H)
   161:         self.poc_embed = nn.Linear(poc_dim, H)
   162: 
   163:         self.lig_convs = nn.ModuleList([GINLayer(H, intra_edge_dim, H) for _ in range(num_layers)])
   164:         self.poc_convs = nn.ModuleList([GINLayer(H, intra_edge_dim, H) for _ in range(num_layers)])
   165:         self.inter_convs = nn.ModuleList([InterGINLayer(H, H, inter_edge_dim, H) for _ in range(num_layers)])
   166: 
   167:         # Interaction readout
   168:         self.edge_readout = nn.Sequential(
   169:             nn.Linear(H * 2 + inter_edge_dim, H),
   170:             nn.ReLU(),
   171:             nn.Linear(H, 1),
   172:         )
   173: 
   174:         # Graph-level readout
   175:         self.graph_readout = nn.Sequential(
   176:             nn.Linear(H * 2, H),
   177:             nn.ReLU(),
   178:             nn.Dropout(0.1),
   179:             nn.Linear(H, 1),
   180:         )
   181: 
   182:     def forward(self, batch: PLABatch) -> torch.Tensor:
   183:         B = batch.labels.size(0)
   184:         lig_h = self.lig_embed(batch.lig_x)
   185:         poc_h = self.poc_embed(batch.poc_x)
   186: 
   187:         for i in range(len(self.lig_convs)):
   188:             lig_h = self.lig_convs[i](lig_h, batch.lig_edge_index, batch.lig_edge_attr) + lig_h
   189:             poc_h = self.poc_convs[i](poc_h, batch.poc_edge_index, batch.poc_edge_attr) + poc_h
   190:             if batch.l2p_edge_index.size(1) > 0:
   191:                 poc_h = self.inter_convs[i](lig_h, poc_h, batch.l2p_edge_index, batch.l2p_edge_attr, poc_h.size(0))
   192: 
   193:         # Interaction-level scoring
   194:         if batch.l2p_edge_index.size(1) > 0:
   195:             l2p_src, l2p_dst = batch.l2p_edge_index
   196:             inter_feat = torch.cat([lig_h[l2p_src], poc_h[l2p_dst], batch.l2p_edge_attr], dim=-1)
   197:             inter_scores = self.edge_readout(inter_feat)
   198:             inter_pred = torch.zeros(B, 1, device=inter_scores.device)
   199:             inter_pred.index_add_(0, batch.inter_batch, inter_scores)
   200:         else:
   201:             inter_pred = torch.zeros(B, 1, device=lig_h.device)
   202: 
   203:         # Graph-level prediction
   204:         lig_pool = scatter_mean(lig_h, batch.lig_batch, B)
   205:         poc_pool = scatter_mean(poc_h, batch.poc_batch, B)
   206:         graph_pred = self.graph_readout(torch.cat([lig_pool, poc_pool], dim=-1))
   207: 
   208:         pred = (inter_pred + graph_pred) / 2
   209:         return pred.squeeze(-1)
   210: 
   211: # =====================================================================
   212: # EDITABLE SECTION END
   213: # =====================================================================
   214: # EDITABLE SECTION END
   215: # =====================================================================
   216: 
```

### `schnet` baseline — editable region  [READ-ONLY — reference implementation]

In `EHIGN_PLA/custom_pla.py`:

```python
Lines 101–297:
    98: 
    99: # =====================================================================
   100: # EDITABLE SECTION START — AffinityModel + helper modules
   101: # =====================================================================
   102: # EDITABLE SECTION START — SchNet: RBF Distance-based Heterogeneous GNN
   103: # =====================================================================
   104: 
   105: class RBFExpansion(nn.Module):
   106:     """Radial basis function expansion of distances."""
   107:     def __init__(self, low=0.0, high=6.0, gap=0.1):
   108:         super().__init__()
   109:         centers = torch.arange(low, high, gap)
   110:         self.register_buffer('centers', centers)
   111:         self.register_buffer('width', torch.tensor(gap))
   112: 
   113:     @property
   114:     def num_features(self):
   115:         return self.centers.size(0)
   116: 
   117:     def forward(self, dist):
   118:         return torch.exp(-0.5 * ((dist - self.centers) / self.width) ** 2)
   119: 
   120: 
   121: class CFConv(nn.Module):
   122:     """Continuous-filter convolution (SchNet interaction block).
   123:     filter_net(rbf) * node_proj(src), sum aggregation, residual, output MLP.
   124:     """
   125:     def __init__(self, node_dim, rbf_dim, hidden_dim):
   126:         super().__init__()
   127:         self.filter_net = nn.Sequential(
   128:             nn.Linear(rbf_dim, hidden_dim),
   129:             nn.Softplus(),
   130:             nn.Linear(hidden_dim, hidden_dim),
   131:         )
   132:         self.node_proj = nn.Linear(node_dim, hidden_dim)
   133:         self.output = nn.Sequential(
   134:             nn.Linear(hidden_dim, hidden_dim),
   135:             nn.Softplus(),
   136:             nn.Linear(hidden_dim, hidden_dim),
   137:         )
   138: 
   139:     def forward(self, x_src, x_dst, edge_index, rbf_feat, num_dst):
   140:         src, dst = edge_index
   141:         W = self.filter_net(rbf_feat)
   142:         msg = self.node_proj(x_src[src]) * W
   143:         agg = torch.zeros(num_dst, msg.size(-1), device=msg.device)
   144:         agg.index_add_(0, dst, msg)
   145:         return x_dst + self.output(agg)
   146: 
   147: 
   148: class FC(nn.Module):
   149:     """Fully connected prediction head."""
   150:     def __init__(self, d_in, d_hidden, n_layers, dropout, n_out):
   151:         super().__init__()
   152:         layers = []
   153:         for j in range(n_layers):
   154:             if j == 0:
   155:                 layers += [nn.Linear(d_in, d_hidden), nn.Dropout(dropout),
   156:                            nn.LeakyReLU(), nn.BatchNorm1d(d_hidden)]
   157:             if j == n_layers - 1:
   158:                 layers.append(nn.Linear(d_hidden, n_out))
   159:             else:
   160:                 layers += [nn.Linear(d_hidden, d_hidden), nn.Dropout(dropout),
   161:                            nn.LeakyReLU(), nn.BatchNorm1d(d_hidden)]
   162:         self.layers = nn.ModuleList(layers)
   163: 
   164:     def forward(self, h):
   165:         for layer in self.layers:
   166:             h = layer(h)
   167:         return h
   168: 
   169: 
   170: class AffinityModel(nn.Module):
   171:     """SchNet-based heterogeneous GNN for binding affinity.
   172: 
   173:     Uses RBF distance expansion and continuous-filter convolution for all edge types.
   174:     HeteroGraphConv pattern: parallel compute, sum aggregate per node type.
   175:     Dual bidirectional prediction with attention-based bias correction.
   176:     """
   177:     def __init__(self, lig_dim, poc_dim, intra_edge_dim, inter_edge_dim):
   178:         super().__init__()
   179:         H = 256
   180:         num_layers = 3
   181:         self.rbf = RBFExpansion(high=6.0, gap=0.1)
   182:         rbf_dim = self.rbf.num_features
   183: 
   184:         self.lin_node_l = nn.Linear(lig_dim, H)
   185:         self.lin_node_p = nn.Linear(poc_dim, H)
   186: 
   187:         self.cf_l = nn.ModuleList([CFConv(H, rbf_dim, H) for _ in range(num_layers)])
   188:         self.cf_p = nn.ModuleList([CFConv(H, rbf_dim, H) for _ in range(num_layers)])
   189:         self.cf_lp = nn.ModuleList([CFConv(H, rbf_dim, H) for _ in range(num_layers)])
   190:         self.cf_pl = nn.ModuleList([CFConv(H, rbf_dim, H) for _ in range(num_layers)])
   191: 
   192:         # Readout via inter-molecular interaction scoring
   193:         self.prj_lp_src = nn.Linear(H, H)
   194:         self.prj_lp_dst = nn.Linear(H, H)
   195:         self.prj_lp_edge = nn.Linear(rbf_dim, H)
   196:         self.fc_lp = nn.Linear(H, 1)
   197:         self.prj_pl_src = nn.Linear(H, H)
   198:         self.prj_pl_dst = nn.Linear(H, H)
   199:         self.prj_pl_edge = nn.Linear(rbf_dim, H)
   200:         self.fc_pl = nn.Linear(H, 1)
   201: 
   202:         # Bias correction (L->P) with attention
   203:         self.bc_lp_prj_src = nn.Linear(H, H)
   204:         self.bc_lp_prj_dst = nn.Linear(H, H)
   205:         self.bc_lp_prj_edge = nn.Linear(rbf_dim, H)
   206:         self.bc_lp_att = nn.Sequential(nn.PReLU(), nn.Linear(H, 1))
   207:         self.bc_lp_w_src = nn.Linear(H, H)
   208:         self.bc_lp_w_dst = nn.Linear(H, H)
   209:         self.bc_lp_w_edge = nn.Linear(rbf_dim, H)
   210:         self.bc_lp_fc = FC(H, 200, 2, 0.1, 1)
   211: 
   212:         # Bias correction (P->L) with attention
   213:         self.bc_pl_prj_src = nn.Linear(H, H)
   214:         self.bc_pl_prj_dst = nn.Linear(H, H)
   215:         self.bc_pl_prj_edge = nn.Linear(rbf_dim, H)
   216:         self.bc_pl_att = nn.Sequential(nn.PReLU(), nn.Linear(H, 1))
   217:         self.bc_pl_w_src = nn.Linear(H, H)
   218:         self.bc_pl_w_dst = nn.Linear(H, H)
   219:         self.bc_pl_w_edge = nn.Linear(rbf_dim, H)
   220:         self.bc_pl_fc = FC(H, 200, 2, 0.1, 1)
   221: 
   222:     def _get_rbf(self, edge_attr):
   223:         dist = edge_attr[:, -1:] * 10
   224:         return self.rbf(dist)
   225: 
   226:     def _edge_softmax(self, scores, batch_idx, num_graphs):
   227:         max_scores = torch.zeros(num_graphs, 1, device=scores.device).fill_(-1e9)
   228:         max_scores.index_reduce_(0, batch_idx, scores, 'amax', include_self=True)
   229:         exp_scores = torch.exp(scores - max_scores[batch_idx])
   230:         sum_exp = torch.zeros(num_graphs, 1, device=scores.device)
   231:         sum_exp.index_add_(0, batch_idx, exp_scores)
   232:         return exp_scores / sum_exp[batch_idx].clamp(min=1e-8)
   233: 
   234:     def forward(self, batch: PLABatch) -> torch.Tensor:
   235:         B = batch.labels.size(0)
   236:         lig_h = self.lin_node_l(batch.lig_x)
   237:         poc_h = self.lin_node_p(batch.poc_x)
   238: 
   239:         lig_rbf = self._get_rbf(batch.lig_edge_attr)
   240:         poc_rbf = self._get_rbf(batch.poc_edge_attr)
   241:         lp_rbf = self._get_rbf(batch.l2p_edge_attr) if batch.l2p_edge_attr.size(0) > 0 else None
   242:         pl_rbf = self._get_rbf(batch.p2l_edge_attr) if batch.p2l_edge_attr.size(0) > 0 else None
   243: 
   244:         # HeteroGraphConv pattern: parallel compute, sum aggregate
   245:         for i in range(len(self.cf_l)):
   246:             lig_in, poc_in = lig_h, poc_h
   247: 
   248:             lig_intra = self.cf_l[i](lig_in, lig_in, batch.lig_edge_index, lig_rbf, lig_in.size(0))
   249:             poc_intra = self.cf_p[i](poc_in, poc_in, batch.poc_edge_index, poc_rbf, poc_in.size(0))
   250: 
   251:             lig_inter = torch.zeros_like(lig_in)
   252:             poc_inter = torch.zeros_like(poc_in)
   253:             if lp_rbf is not None and batch.l2p_edge_index.size(1) > 0:
   254:                 poc_inter = self.cf_lp[i](lig_in, poc_in, batch.l2p_edge_index, lp_rbf, poc_in.size(0))
   255:             if pl_rbf is not None and batch.p2l_edge_index.size(1) > 0:
   256:                 lig_inter = self.cf_pl[i](poc_in, lig_in, batch.p2l_edge_index, pl_rbf, lig_in.size(0))
   257: 
   258:             lig_h = lig_intra + lig_inter
   259:             poc_h = poc_intra + poc_inter
   260: 
   261:         # Scoring (L->P)
   262:         l2p_src, l2p_dst = batch.l2p_edge_index
   263:         i_lp = self.prj_lp_edge(lp_rbf) * self.prj_lp_src(lig_h)[l2p_src] * self.prj_lp_dst(poc_h)[l2p_dst]
   264:         logit_lp = self.fc_lp(i_lp)
   265:         pred_lp = torch.zeros(B, 1, device=logit_lp.device)
   266:         pred_lp.index_add_(0, batch.inter_batch, logit_lp)
   267: 
   268:         # Scoring (P->L)
   269:         p2l_src, p2l_dst = batch.p2l_edge_index
   270:         p2l_batch = batch.lig_batch[p2l_dst]
   271:         i_pl = self.prj_pl_edge(pl_rbf) * self.prj_pl_src(poc_h)[p2l_src] * self.prj_pl_dst(lig_h)[p2l_dst]
   272:         logit_pl = self.fc_pl(i_pl)
   273:         pred_pl = torch.zeros(B, 1, device=logit_pl.device)
   274:         pred_pl.index_add_(0, p2l_batch, logit_pl)
   275: 
   276:         # Bias correction (L->P) with attention
   277:         w_lp = self.bc_lp_prj_src(lig_h)[l2p_src] + self.bc_lp_prj_dst(poc_h)[l2p_dst] + self.bc_lp_prj_edge(lp_rbf)
   278:         a_lp = self._edge_softmax(self.bc_lp_att(w_lp), batch.inter_batch, B)
   279:         s_lp = a_lp * self.bc_lp_w_edge(lp_rbf) * self.bc_lp_w_src(lig_h)[l2p_src] * self.bc_lp_w_dst(poc_h)[l2p_dst]
   280:         bias_lp_agg = torch.zeros(B, s_lp.size(-1), device=s_lp.device)
   281:         bias_lp_agg.index_add_(0, batch.inter_batch, s_lp)
   282:         bias_lp = self.bc_lp_fc(bias_lp_agg)
   283: 
   284:         # Bias correction (P->L) with attention
   285:         w_pl = self.bc_pl_prj_src(poc_h)[p2l_src] + self.bc_pl_prj_dst(lig_h)[p2l_dst] + self.bc_pl_prj_edge(pl_rbf)
   286:         a_pl = self._edge_softmax(self.bc_pl_att(w_pl), p2l_batch, B)
   287:         s_pl = a_pl * self.bc_pl_w_edge(pl_rbf) * self.bc_pl_w_src(poc_h)[p2l_src] * self.bc_pl_w_dst(lig_h)[p2l_dst]
   288:         bias_pl_agg = torch.zeros(B, s_pl.size(-1), device=s_pl.device)
   289:         bias_pl_agg.index_add_(0, p2l_batch, s_pl)
   290:         bias_pl = self.bc_pl_fc(bias_pl_agg)
   291: 
   292:         pred = ((pred_lp - bias_lp) + (pred_pl - bias_pl)) / 2
   293:         return pred.squeeze(-1)
   294: 
   295: # =====================================================================
   296: # EDITABLE SECTION END
   297: # =====================================================================
   298: # EDITABLE SECTION END
   299: # =====================================================================
   300: 
```

### `egnn` baseline — editable region  [READ-ONLY — reference implementation]

In `EHIGN_PLA/custom_pla.py`:

```python
Lines 101–281:
    98: 
    99: # =====================================================================
   100: # EDITABLE SECTION START — AffinityModel + helper modules
   101: # =====================================================================
   102: # EDITABLE SECTION START — EGNN: Equivariant Graph Neural Network
   103: # =====================================================================
   104: 
   105: class EGNNConv(nn.Module):
   106:     """E(n)-equivariant message passing layer using distance as edge feature.
   107:     Message: mlp_u(src) + mlp_v(dst) + mlp_e(dist), sum aggregation,
   108:     then node_mlp(cat[dst, agg]).
   109:     """
   110:     def __init__(self, input_dim, hidden_dim, edge_dim=1):
   111:         super().__init__()
   112:         self.edge_mlp_u = nn.Sequential(
   113:             nn.Linear(input_dim, hidden_dim), nn.SiLU(),
   114:             nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
   115:         self.edge_mlp_v = nn.Sequential(
   116:             nn.Linear(input_dim, hidden_dim), nn.SiLU(),
   117:             nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
   118:         self.edge_mlp_e = nn.Sequential(
   119:             nn.Linear(edge_dim, hidden_dim), nn.SiLU(),
   120:             nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
   121:         self.node_mlp = nn.Sequential(
   122:             nn.Linear(hidden_dim + hidden_dim, hidden_dim), nn.SiLU(),
   123:             nn.Linear(hidden_dim, hidden_dim))
   124: 
   125:     def forward(self, x_src, x_dst, edge_index, edge_feat, num_dst):
   126:         src, dst = edge_index
   127:         msg = self.edge_mlp_u(x_src[src]) + self.edge_mlp_v(x_dst[dst]) + self.edge_mlp_e(edge_feat)
   128:         agg = torch.zeros(num_dst, msg.size(-1), device=msg.device)
   129:         agg.index_add_(0, dst, msg)
   130:         return self.node_mlp(torch.cat([x_dst, agg], dim=-1))
   131: 
   132: 
   133: class FC(nn.Module):
   134:     """Fully connected prediction head."""
   135:     def __init__(self, d_in, d_hidden, n_layers, dropout, n_out):
   136:         super().__init__()
   137:         layers = []
   138:         for j in range(n_layers):
   139:             if j == 0:
   140:                 layers += [nn.Linear(d_in, d_hidden), nn.Dropout(dropout),
   141:                            nn.LeakyReLU(), nn.BatchNorm1d(d_hidden)]
   142:             if j == n_layers - 1:
   143:                 layers.append(nn.Linear(d_hidden, n_out))
   144:             else:
   145:                 layers += [nn.Linear(d_hidden, d_hidden), nn.Dropout(dropout),
   146:                            nn.LeakyReLU(), nn.BatchNorm1d(d_hidden)]
   147:         self.layers = nn.ModuleList(layers)
   148: 
   149:     def forward(self, h):
   150:         for layer in self.layers:
   151:             h = layer(h)
   152:         return h
   153: 
   154: 
   155: class AffinityModel(nn.Module):
   156:     """EGNN-based heterogeneous model for binding affinity.
   157: 
   158:     Uses E(n)-equivariant message passing with distance as scalar edge feature.
   159:     HeteroGraphConv pattern: parallel compute, sum aggregate per node type.
   160:     Dual bidirectional prediction with attention-based bias correction.
   161:     """
   162:     def __init__(self, lig_dim, poc_dim, intra_edge_dim, inter_edge_dim):
   163:         super().__init__()
   164:         H = 256
   165:         num_layers = 3
   166: 
   167:         self.lin_node_l = nn.Linear(lig_dim, H)
   168:         self.lin_node_p = nn.Linear(poc_dim, H)
   169: 
   170:         # EGNN layers for all 4 edge types (using distance as 1-dim edge feat)
   171:         self.egnn_l = nn.ModuleList([EGNNConv(H, H, edge_dim=1) for _ in range(num_layers)])
   172:         self.egnn_p = nn.ModuleList([EGNNConv(H, H, edge_dim=1) for _ in range(num_layers)])
   173:         self.egnn_lp = nn.ModuleList([EGNNConv(H, H, edge_dim=1) for _ in range(num_layers)])
   174:         self.egnn_pl = nn.ModuleList([EGNNConv(H, H, edge_dim=1) for _ in range(num_layers)])
   175: 
   176:         # Interaction scoring (with 1-dim distance edge features)
   177:         self.prj_lp_src = nn.Linear(H, H)
   178:         self.prj_lp_dst = nn.Linear(H, H)
   179:         self.prj_lp_edge = nn.Linear(1, H)
   180:         self.fc_lp = nn.Linear(H, 1)
   181:         self.prj_pl_src = nn.Linear(H, H)
   182:         self.prj_pl_dst = nn.Linear(H, H)
   183:         self.prj_pl_edge = nn.Linear(1, H)
   184:         self.fc_pl = nn.Linear(H, 1)
   185: 
   186:         # Bias correction (L->P)
   187:         self.bc_lp_prj_src = nn.Linear(H, H)
   188:         self.bc_lp_prj_dst = nn.Linear(H, H)
   189:         self.bc_lp_prj_edge = nn.Linear(1, H)
   190:         self.bc_lp_att = nn.Sequential(nn.PReLU(), nn.Linear(H, 1))
   191:         self.bc_lp_w_src = nn.Linear(H, H)
   192:         self.bc_lp_w_dst = nn.Linear(H, H)
   193:         self.bc_lp_w_edge = nn.Linear(1, H)
   194:         self.bc_lp_fc = FC(H, 200, 2, 0.1, 1)
   195: 
   196:         # Bias correction (P->L)
   197:         self.bc_pl_prj_src = nn.Linear(H, H)
   198:         self.bc_pl_prj_dst = nn.Linear(H, H)
   199:         self.bc_pl_prj_edge = nn.Linear(1, H)
   200:         self.bc_pl_att = nn.Sequential(nn.PReLU(), nn.Linear(H, 1))
   201:         self.bc_pl_w_src = nn.Linear(H, H)
   202:         self.bc_pl_w_dst = nn.Linear(H, H)
   203:         self.bc_pl_w_edge = nn.Linear(1, H)
   204:         self.bc_pl_fc = FC(H, 200, 2, 0.1, 1)
   205: 
   206:     def _get_dist(self, edge_attr):
   207:         # Last dim is L2 distance * 0.1, rescale to angstroms
   208:         return edge_attr[:, -1:] * 10
   209: 
   210:     def _edge_softmax(self, scores, batch_idx, num_graphs):
   211:         max_scores = torch.zeros(num_graphs, 1, device=scores.device).fill_(-1e9)
   212:         max_scores.index_reduce_(0, batch_idx, scores, 'amax', include_self=True)
   213:         exp_scores = torch.exp(scores - max_scores[batch_idx])
   214:         sum_exp = torch.zeros(num_graphs, 1, device=scores.device)
   215:         sum_exp.index_add_(0, batch_idx, exp_scores)
   216:         return exp_scores / sum_exp[batch_idx].clamp(min=1e-8)
   217: 
   218:     def forward(self, batch: PLABatch) -> torch.Tensor:
   219:         B = batch.labels.size(0)
   220:         lig_h = self.lin_node_l(batch.lig_x)
   221:         poc_h = self.lin_node_p(batch.poc_x)
   222: 
   223:         lig_dist = self._get_dist(batch.lig_edge_attr)
   224:         poc_dist = self._get_dist(batch.poc_edge_attr)
   225:         lp_dist = self._get_dist(batch.l2p_edge_attr) if batch.l2p_edge_attr.size(0) > 0 else None
   226:         pl_dist = self._get_dist(batch.p2l_edge_attr) if batch.p2l_edge_attr.size(0) > 0 else None
   227: 
   228:         # HeteroGraphConv pattern: parallel compute, sum aggregate
   229:         for i in range(len(self.egnn_l)):
   230:             lig_in, poc_in = lig_h, poc_h
   231: 
   232:             lig_intra = self.egnn_l[i](lig_in, lig_in, batch.lig_edge_index, lig_dist, lig_in.size(0))
   233:             poc_intra = self.egnn_p[i](poc_in, poc_in, batch.poc_edge_index, poc_dist, poc_in.size(0))
   234: 
   235:             lig_inter = torch.zeros_like(lig_in)
   236:             poc_inter = torch.zeros_like(poc_in)
   237:             if lp_dist is not None and batch.l2p_edge_index.size(1) > 0:
   238:                 poc_inter = self.egnn_lp[i](lig_in, poc_in, batch.l2p_edge_index, lp_dist, poc_in.size(0))
   239:             if pl_dist is not None and batch.p2l_edge_index.size(1) > 0:
   240:                 lig_inter = self.egnn_pl[i](poc_in, lig_in, batch.p2l_edge_index, pl_dist, lig_in.size(0))
   241: 
   242:             lig_h = lig_intra + lig_inter
   243:             poc_h = poc_intra + poc_inter
   244: 
   245:         # Atom-atom affinities (L->P) with edge features
   246:         l2p_src, l2p_dst = batch.l2p_edge_index
   247:         i_lp = self.prj_lp_edge(lp_dist) * self.prj_lp_src(lig_h)[l2p_src] * self.prj_lp_dst(poc_h)[l2p_dst]
   248:         logit_lp = self.fc_lp(i_lp)
   249:         pred_lp = torch.zeros(B, 1, device=logit_lp.device)
   250:         pred_lp.index_add_(0, batch.inter_batch, logit_lp)
   251: 
   252:         # Atom-atom affinities (P->L) with edge features
   253:         p2l_src, p2l_dst = batch.p2l_edge_index
   254:         p2l_batch = batch.lig_batch[p2l_dst]
   255:         i_pl = self.prj_pl_edge(pl_dist) * self.prj_pl_src(poc_h)[p2l_src] * self.prj_pl_dst(lig_h)[p2l_dst]
   256:         logit_pl = self.fc_pl(i_pl)
   257:         pred_pl = torch.zeros(B, 1, device=logit_pl.device)
   258:         pred_pl.index_add_(0, p2l_batch, logit_pl)
   259: 
   260:         # Bias correction (L->P) with attention
   261:         w_lp = self.bc_lp_prj_src(lig_h)[l2p_src] + self.bc_lp_prj_dst(poc_h)[l2p_dst] + self.bc_lp_prj_edge(lp_dist)
   262:         a_lp = self._edge_softmax(self.bc_lp_att(w_lp), batch.inter_batch, B)
   263:         s_lp = a_lp * self.bc_lp_w_edge(lp_dist) * self.bc_lp_w_src(lig_h)[l2p_src] * self.bc_lp_w_dst(poc_h)[l2p_dst]
   264:         bias_lp_agg = torch.zeros(B, s_lp.size(-1), device=s_lp.device)
   265:         bias_lp_agg.index_add_(0, batch.inter_batch, s_lp)
   266:         bias_lp = self.bc_lp_fc(bias_lp_agg)
   267: 
   268:         # Bias correction (P->L) with attention
   269:         w_pl = self.bc_pl_prj_src(poc_h)[p2l_src] + self.bc_pl_prj_dst(lig_h)[p2l_dst] + self.bc_pl_prj_edge(pl_dist)
   270:         a_pl = self._edge_softmax(self.bc_pl_att(w_pl), p2l_batch, B)
   271:         s_pl = a_pl * self.bc_pl_w_edge(pl_dist) * self.bc_pl_w_src(poc_h)[p2l_src] * self.bc_pl_w_dst(lig_h)[p2l_dst]
   272:         bias_pl_agg = torch.zeros(B, s_pl.size(-1), device=s_pl.device)
   273:         bias_pl_agg.index_add_(0, p2l_batch, s_pl)
   274:         bias_pl = self.bc_pl_fc(bias_pl_agg)
   275: 
   276:         pred = ((pred_lp - bias_lp) + (pred_pl - bias_pl)) / 2
   277:         return pred.squeeze(-1)
   278: 
   279: # =====================================================================
   280: # EDITABLE SECTION END
   281: # =====================================================================
   282: # EDITABLE SECTION END
   283: # =====================================================================
   284: 
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
