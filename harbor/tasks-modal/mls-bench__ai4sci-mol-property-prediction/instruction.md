# MLS-Bench: ai4sci-mol-property-prediction

# Task: Molecular Property Prediction

## Research Question
Design a molecular representation model for predicting chemical properties (toxicity, blood-brain barrier penetration, enzyme inhibition, etc.) from molecular structure. The goal is to learn effective molecular representations that generalize across diverse property prediction tasks.

## Background
Molecular property prediction is a core task in drug discovery and materials science. Given a molecule (as a SMILES string → molecular graph + optional 3D coordinates), the model must predict one or more chemical properties. Key challenges include:
- **Molecular representation**: How to encode atoms, bonds, and 3D geometry into informative features.
- **Multi-task learning**: Some datasets have multiple targets with missing labels across multiple assays.
- **Scaffold generalization**: The scaffold split ensures the model generalizes to structurally novel molecules.

Existing approaches include:
- **D-MPNN** (Yang et al., "Analyzing Learned Molecular Representations for Property Prediction", J. Chem. Inf. Model. 2019, 59(8):3370–3388; arXiv:1904.01561). Directed message passing on bonds rather than atoms to avoid "message collision". Reference implementation: Chemprop (https://github.com/chemprop/chemprop).
- **GIN** (Xu et al., "How Powerful are Graph Neural Networks?", ICLR 2019; arXiv:1810.00826). Graph Isomorphism Network with sum aggregation that matches the discriminative power of the Weisfeiler–Lehman test.
- **Uni-Mol** (Zhou et al., "Uni-Mol: A Universal 3D Molecular Representation Learning Framework", ICLR 2023; OpenReview 6K2RM6wVqKu; ChemRxiv 628e5b4d5d948517f5ce6d72). SE(3)-invariant Transformer with 3D distance attention bias, pretrained on ~209M molecular conformations. Code: https://github.com/deepmodeling/Uni-Mol.

## What to Implement
Implement the `MoleculeModel` class in `custom_molprop.py`. You must implement:
1. `__init__(self, atom_dim, edge_dim, num_tasks, task_type)`: Set up your model architecture.
2. `forward(self, batch) -> Tensor`: Return predictions of shape `[B, num_tasks]`.

## Batch Format (MolBatch)
```python
@dataclass
class MolBatch:
    # Sparse graph format (for GNN models)
    x: Tensor              # [total_atoms, atom_dim] node features
    edge_index: Tensor     # [2, total_edges] COO format
    edge_attr: Tensor      # [total_edges, edge_dim] bond features
    batch_idx: Tensor      # [total_atoms] graph assignment (0..B-1)

    # Dense format (for Transformer models)
    atom_features: Tensor  # [B, max_atoms, atom_dim] zero-padded
    positions: Tensor      # [B, max_atoms, 3] 3D coordinates
    dist_matrix: Tensor    # [B, max_atoms, max_atoms] pairwise distances
    mask: Tensor           # [B, max_atoms] 1=real atom, 0=padding

    # Uni-Mol specific (from LMDB pipeline)
    atom_tokens: Tensor    # [B, max_tokens] Uni-Mol vocabulary token ids (with [CLS]/[SEP])
    edge_types: Tensor     # [B, max_tokens, max_tokens] atom-pair type ids

    # Targets (normalized for regression tasks)
    targets: Tensor        # [B, num_tasks]
    target_mask: Tensor    # [B, num_tasks] 1=valid label, 0=missing
```

Additional attributes set dynamically on the batch:
- `batch._unimol_dist`: [B, max_tokens, max_tokens] distance matrix for Uni-Mol tokens.
- `batch._unimol_token_mask`: [B, max_tokens] 1=valid token, 0=padding.

## Atom Features (`ATOM_DIM = 136`)
One-hot encodings of: atomic_num (118), degree (6), formal_charge (5), num_Hs (5), hybridization (5), aromatic (1), in_ring (1).

## Bond Features (`EDGE_DIM = 9`)
One-hot encodings of: bond_type (4), stereo (3), conjugated (1), in_ring (1).

## Fixed Pipeline
The training and evaluation pipeline (data preparation, splitting, training loop, optimizer schedule, target normalization, masked loss, test-time augmentation, and metrics) is fixed by the scaffold and not editable.

## Editable Region
The section between `EDITABLE SECTION START` and `EDITABLE SECTION END` markers in `custom_molprop.py` is editable. You may define helper classes, layers, or functions within this region. The region must contain a `MoleculeModel` class with the specified interface.

## Available Resources
- 3D conformers and pre-computed distances/edge types are provided in the batch.
- Uni-Mol vocabulary tokens and edge types are available in the batch.
- Uni-Mol pre-trained weights are available inside the container at the path used by the `unimol` baseline.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/Uni-Mol/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits that change code outside these ranges — or creating new files, or
deleting whole files — will cause your submission to be invalid.

The line numbers mark an editable **region**, not a fixed line-count budget: you
may add or remove lines inside it. Only code outside the editable ranges must
stay unchanged.

- `Uni-Mol/custom_molprop.py`
- editable lines **115–207**




## Readable Context


### `Uni-Mol/custom_molprop.py`  [EDITABLE — lines 115–207 only]

```python
     1: """
     2: Molecular Property Prediction — Self-contained template.
     3: Predicts molecular properties (classification: ROC-AUC, regression: RMSE)
     4: on MoleculeNet benchmarks (BBBP, BACE, Tox21, ESOL, FreeSolv, Lipophilicity).
     5: 
     6: Uses official Uni-Mol pre-split LMDB data with train/valid/test splits
     7: and pre-computed multi-conformer 3D coordinates.  Data pipeline mirrors
     8: Uni-Mol: LMDB -> conformer sample/enumerate -> remove polar H -> normalize
     9: coordinates -> Uni-Mol vocabulary tokenization -> distance matrix + edge types.
    10: 
    11: Structure:
    12:   Lines 1-114:   FIXED — Imports, constants, atom/bond featurization
    13:   Lines 115-207: EDITABLE — MoleculeModel class (starter: simple GIN)
    14:   Lines 208+:    FIXED — Data loading, training loop, evaluation, TTA
    15: """
    16: import os
    17: import sys
    18: import math
    19: import copy
    20: import json
    21: import lmdb
    22: import pickle
    23: import argparse
    24: import warnings
    25: import numpy as np
    26: import pandas as pd
    27: from collections import defaultdict
    28: from dataclasses import dataclass, replace
    29: from typing import Optional, Dict, List, Tuple
    30: from pathlib import Path
    31: from scipy.spatial import distance_matrix as scipy_distance_matrix
    32: 
    33: import torch
    34: import torch.nn as nn
    35: import torch.nn.functional as F
    36: from torch.utils.data import Dataset, DataLoader
    37: 
    38: from rdkit import Chem
    39: from rdkit.Chem import AllChem, Descriptors
    40: 
    41: warnings.filterwarnings("ignore", category=UserWarning)
    42: 
    43: # =====================================================================
    44: # Atom and bond featurization constants (used by GNN-based models)
    45: # =====================================================================
    46: 
    47: ATOM_FEATURES = {
    48:     'atomic_num': list(range(1, 119)),
    49:     'degree': [0, 1, 2, 3, 4, 5],
    50:     'formal_charge': [-2, -1, 0, 1, 2],
    51:     'num_hs': [0, 1, 2, 3, 4],
    52:     'hybridization': [
    53:         Chem.rdchem.HybridizationType.SP,
    54:         Chem.rdchem.HybridizationType.SP2,
    55:         Chem.rdchem.HybridizationType.SP3,
    56:         Chem.rdchem.HybridizationType.SP3D,
    57:         Chem.rdchem.HybridizationType.SP3D2,
    58:     ],
    59: }
    60: 
    61: BOND_FEATURES = {
    62:     'bond_type': [
    63:         Chem.rdchem.BondType.SINGLE,
    64:         Chem.rdchem.BondType.DOUBLE,
    65:         Chem.rdchem.BondType.TRIPLE,
    66:         Chem.rdchem.BondType.AROMATIC,
    67:     ],
    68:     'stereo': [
    69:         Chem.rdchem.BondStereo.STEREONONE,
    70:         Chem.rdchem.BondStereo.STEREOZ,
    71:         Chem.rdchem.BondStereo.STEREOE,
    72:     ],
    73: }
    74: 
    75: ATOM_DIM = len(ATOM_FEATURES['atomic_num']) + len(ATOM_FEATURES['degree']) + \
    76:            len(ATOM_FEATURES['formal_charge']) + len(ATOM_FEATURES['num_hs']) + \
    77:            len(ATOM_FEATURES['hybridization']) + 2  # +2 for aromatic, in_ring
    78: 
    79: EDGE_DIM = len(BOND_FEATURES['bond_type']) + len(BOND_FEATURES['stereo']) + 2  # +2 for conjugated, in_ring
    80: 
    81: 
    82: def one_hot(val, allowable_set):
    83:     """One-hot encode a value. Unknown values map to all-zeros."""
    84:     encoding = [0] * len(allowable_set)
    85:     if val in allowable_set:
    86:         encoding[allowable_set.index(val)] = 1
    87:     return encoding
    88: 
    89: 
    90: def atom_features(atom):
    91:     """Compute atom feature vector."""
    92:     features = []
    93:     features += one_hot(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])
    94:     features += one_hot(atom.GetDegree(), ATOM_FEATURES['degree'])
    95:     features += one_hot(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
    96:     features += one_hot(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs'])
    97:     features += one_hot(atom.GetHybridization(), ATOM_FEATURES['hybridization'])
    98:     features += [int(atom.GetIsAromatic())]
    99:     features += [int(atom.IsInRing())]
   100:     return features
   101: 
   102: 
   103: def bond_features(bond):
   104:     """Compute bond feature vector."""
   105:     features = []
   106:     features += one_hot(bond.GetBondType(), BOND_FEATURES['bond_type'])
   107:     features += one_hot(bond.GetStereo(), BOND_FEATURES['stereo'])
   108:     features += [int(bond.GetIsConjugated())]
   109:     features += [int(bond.IsInRing())]
   110:     return features
   111: 
   112: 
   113: # =====================================================================
   114: # EDITABLE SECTION START — MoleculeModel + helper modules
   115: # =====================================================================
   116: 
   117: class GINConv(nn.Module):
   118:     """Graph Isomorphism Network convolution layer."""
   119: 
   120:     def __init__(self, in_dim, out_dim, edge_dim):
   121:         super().__init__()
   122:         self.mlp = nn.Sequential(
   123:             nn.Linear(in_dim, out_dim),
   124:             nn.BatchNorm1d(out_dim),
   125:             nn.ReLU(),
   126:             nn.Linear(out_dim, out_dim),
   127:         )
   128:         self.edge_proj = nn.Linear(edge_dim, in_dim)
   129:         self.eps = nn.Parameter(torch.zeros(1))
   130: 
   131:     def forward(self, x, edge_index, edge_attr, batch_idx):
   132:         """
   133:         x: [total_atoms, in_dim]
   134:         edge_index: [2, total_edges]
   135:         edge_attr: [total_edges, edge_dim]
   136:         batch_idx: [total_atoms]
   137:         """
   138:         src, dst = edge_index
   139:         edge_msg = self.edge_proj(edge_attr)
   140:         msg = x[src] + edge_msg
   141: 
   142:         # Aggregate messages to destination nodes
   143:         agg = torch.zeros_like(x)
   144:         agg.index_add_(0, dst, msg)
   145: 
   146:         out = self.mlp((1 + self.eps) * x + agg)
   147:         return out
   148: 
   149: 
   150: class MoleculeModel(nn.Module):
   151:     """Starter model: Graph Isomorphism Network (GIN) with mean pooling.
   152: 
   153:     Simple but effective baseline for molecular property prediction.
   154:     Uses message passing on the molecular graph with learned edge features.
   155:     """
   156: 
   157:     def __init__(self, atom_dim: int, edge_dim: int, num_tasks: int, task_type: str):
   158:         super().__init__()
   159:         self.num_tasks = num_tasks
   160:         self.task_type = task_type
   161:         hidden_dim = 256
   162:         num_layers = 4
   163: 
   164:         self.atom_embed = nn.Linear(atom_dim, hidden_dim)
   165:         self.convs = nn.ModuleList([
   166:             GINConv(hidden_dim, hidden_dim, edge_dim) for _ in range(num_layers)
   167:         ])
   168:         self.norms = nn.ModuleList([
   169:             nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
   170:         ])
   171:         self.dropout = nn.Dropout(0.1)
   172: 
   173:         self.readout = nn.Sequential(
   174:             nn.Linear(hidden_dim, hidden_dim),
   175:             nn.ReLU(),
   176:             nn.Dropout(0.1),
   177:             nn.Linear(hidden_dim, num_tasks),
   178:         )
   179: 
   180:     def forward(self, batch):
   181:         """
   182:         Args:
   183:             batch: MolBatch with sparse graph data.
   184:         Returns:
   185:             predictions: [B, num_tasks]
   186:         """
   187:         x = self.atom_embed(batch.x)
   188: 
   189:         for conv, norm in zip(self.convs, self.norms):
   190:             x_new = conv(x, batch.edge_index, batch.edge_attr, batch.batch_idx)
   191:             x_new = norm(x_new)
   192:             x_new = F.relu(x_new)
   193:             x = x + self.dropout(x_new)  # residual
   194: 
   195:         # Mean pooling per graph
   196:         num_graphs = batch.batch_idx.max().item() + 1
   197:         graph_embed = torch.zeros(num_graphs, x.size(-1), device=x.device)
   198:         counts = torch.zeros(num_graphs, 1, device=x.device)
   199:         graph_embed.index_add_(0, batch.batch_idx, x)
   200:         counts.index_add_(0, batch.batch_idx, torch.ones(x.size(0), 1, device=x.device))
   201:         graph_embed = graph_embed / counts.clamp(min=1)
   202: 
   203:         return self.readout(graph_embed)
   204: 
   205: # =====================================================================
   206: # EDITABLE SECTION END
   207: # =====================================================================
   208: 
   209: 
   210: # =====================================================================
   211: # FIXED — Uni-Mol vocabulary, data loading, training, evaluation
   212: # =====================================================================
   213: 
   214: # Uni-Mol atom vocabulary (mirrors dict.txt)
   215: # [PAD]=0, [CLS]=1, [SEP]=2, [UNK]=3, C=4, N=5, O=6, S=7, H=8,
   216: # Cl=9, F=10, Br=11, I=12, Si=13, P=14, B=15, Na=16, K=17, Al=18,
   217: # Ca=19, Sn=20, As=21, Hg=22, Fe=23, Zn=24, Cr=25, Se=26, Gd=27,
   218: # Au=28, Li=29, [MASK]=30
   219: UNIMOL_ELEM_TO_IDX = {
   220:     'C': 4, 'N': 5, 'O': 6, 'S': 7, 'H': 8, 'Cl': 9, 'F': 10,
   221:     'Br': 11, 'I': 12, 'Si': 13, 'P': 14, 'B': 15, 'Na': 16,
   222:     'K': 17, 'Al': 18, 'Ca': 19, 'Sn': 20, 'As': 21, 'Hg': 22,
   223:     'Fe': 23, 'Zn': 24, 'Cr': 25, 'Se': 26, 'Gd': 27, 'Au': 28,
   224:     'Li': 29,
   225: }
   226: UNIMOL_PAD_IDX = 0
   227: UNIMOL_CLS_IDX = 1
   228: UNIMOL_SEP_IDX = 2
   229: UNIMOL_UNK_IDX = 3
   230: UNIMOL_DICT_SIZE = 31  # 30 tokens + [MASK]
   231: 
   232: # Target normalization for regression tasks (from Uni-Mol official)
   233: TARGET_NORM = {
   234:     'esol': {'mean': -3.0501019503546094, 'std': 2.096441210089345},
   235:     'freesolv': {'mean': -3.8030062305295944, 'std': 3.8478201171088138},
   236:     'lipophilicity': {'mean': 2.186336, 'std': 1.203004},
   237: }
   238: 
   239: 
   240: @dataclass
   241: class MolBatch:
   242:     """Molecular batch data for both sparse (GNN) and dense (Transformer) formats."""
   243:     # Sparse graph format
   244:     x: torch.Tensor              # [total_atoms, atom_dim]
   245:     edge_index: torch.Tensor     # [2, total_edges]
   246:     edge_attr: torch.Tensor      # [total_edges, edge_dim]
   247:     batch_idx: torch.Tensor      # [total_atoms] graph assignment
   248: 
   249:     # Dense format (Uni-Mol pipeline: atom tokens, coordinates, distances, edge types)
   250:     atom_features: torch.Tensor  # [B, max_atoms, atom_dim]
   251:     positions: torch.Tensor      # [B, max_atoms, 3]
   252:     dist_matrix: torch.Tensor    # [B, max_atoms, max_atoms]
   253:     mask: torch.Tensor           # [B, max_atoms] boolean
   254: 
   255:     # Uni-Mol specific
   256:     atom_tokens: torch.Tensor    # [B, max_atoms] Uni-Mol vocabulary token ids
   257:     edge_types: torch.Tensor     # [B, max_atoms, max_atoms] atom-pair type ids
   258: 
   259:     # Targets
   260:     targets: torch.Tensor        # [B, num_tasks]
   261:     target_mask: torch.Tensor    # [B, num_tasks] for missing labels
   262: 
   263: 
   264: # =====================================================================
   265: # LMDB data loading (official Uni-Mol pre-split data)
   266: # =====================================================================
   267: 
   268: class LMDBReader:
   269:     """Lazy LMDB reader — opens the environment on first access."""
   270: 
   271:     def __init__(self, lmdb_path):
   272:         self.lmdb_path = lmdb_path
   273:         assert os.path.isfile(lmdb_path), f"LMDB not found: {lmdb_path}"
   274:         env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False,
   275:                         readahead=False, meminit=False, max_readers=256)
   276:         with env.begin() as txn:
   277:             self._len = len(list(txn.cursor().iternext(values=False)))
   278:         env.close()
   279:         self._env = None
   280: 
   281:     def _connect(self):
   282:         if self._env is None:
   283:             self._env = lmdb.open(self.lmdb_path, subdir=False, readonly=True,
   284:                                   lock=False, readahead=False, meminit=False,
   285:                                   max_readers=256)
   286: 
   287:     def __len__(self):
   288:         return self._len
   289: 
   290:     def __getitem__(self, idx):
   291:         self._connect()
   292:         data = self._env.begin().get(f"{idx}".encode("ascii"))
   293:         return pickle.loads(data)
   294: 
   295: 
   296: # Map from our dataset names to the official Uni-Mol directory names
   297: DATASET_LMDB_NAME = {
   298:     'bbbp': 'bbbp',
   299:     'bace': 'bace',
   300:     'tox21': 'tox21',
   301:     'esol': 'esol',
   302:     'freesolv': 'freesolv',
   303:     'lipophilicity': 'lipo',
   304: }
   305: 
   306: DATASET_CONFIG = {
   307:     'bbbp': {
   308:         'target_key': 'target',
   309:         'num_tasks': 1,
   310:         'task_type': 'classification',
   311:     },
   312:     'bace': {
   313:         'target_key': 'target',
   314:         'num_tasks': 1,
   315:         'task_type': 'classification',
   316:     },
   317:     'tox21': {
   318:         'target_key': 'target',
   319:         'num_tasks': 12,
   320:         'task_type': 'classification',
   321:     },
   322:     'esol': {
   323:         'target_key': 'target',
   324:         'num_tasks': 1,
   325:         'task_type': 'regression',
   326:     },
   327:     'freesolv': {
   328:         'target_key': 'target',
   329:         'num_tasks': 1,
   330:         'task_type': 'regression',
   331:     },
   332:     'lipophilicity': {
   333:         'target_key': 'target',
   334:         'num_tasks': 1,
   335:         'task_type': 'regression',
   336:     },
   337: }
   338: 
   339: 
   340: def _remove_polar_hydrogen(atoms, coordinates):
   341:     """Remove trailing polar hydrogen atoms (matches Uni-Mol only_polar=1 mode)."""
   342:     end_idx = 0
   343:     for i, atom in enumerate(atoms[::-1]):
   344:         if atom != 'H':
   345:             break
   346:         else:
   347:             end_idx = i + 1
   348:     if end_idx != 0:
   349:         atoms = atoms[:-end_idx]
   350:         coordinates = coordinates[:-end_idx]
   351:     return atoms, coordinates
   352: 
   353: 
   354: def _tokenize_atoms(atom_symbols):
   355:     """Convert atom element symbols to Uni-Mol vocabulary token ids.
   356:     Prepend [CLS] and append [SEP]."""
   357:     tokens = [UNIMOL_CLS_IDX]
   358:     for sym in atom_symbols:
   359:         tokens.append(UNIMOL_ELEM_TO_IDX.get(sym, UNIMOL_UNK_IDX))
   360:     tokens.append(UNIMOL_SEP_IDX)
   361:     return tokens
   362: 
   363: 
   364: class MoleculeDataset(Dataset):
   365:     """Dataset for molecular property prediction.
   366:     Reads directly from LMDB using the Uni-Mol pipeline:
   367:     atom symbols + multi-conformer coordinates.
   368: 
   369:     Training: randomly sample 1 conformer per molecule.
   370:     Val/Test (TTA): enumerate all conformers; dataset length = N * conf_size.
   371:     """
   372: 
   373:     def __init__(self, lmdb_reader, num_tasks, dataset_name, seed=42,
   374:                  is_train=True, conf_size=11, target_mean=None, target_std=None):
   375:         self.lmdb_reader = lmdb_reader
   376:         self.num_tasks = num_tasks
   377:         self.dataset_name = dataset_name
   378:         self.seed = seed
   379:         self.is_train = is_train
   380:         self.conf_size = conf_size
   381:         self.target_mean = target_mean  # for regression normalization
   382:         self.target_std = target_std
   383:         self.n_molecules = len(lmdb_reader)
   384: 
   385:     def __len__(self):
   386:         if self.is_train:
   387:             return self.n_molecules
   388:         else:
   389:             # TTA: each molecule expanded to conf_size entries
   390:             return self.n_molecules * self.conf_size
   391: 
   392:     def _get_entry_and_conf_idx(self, idx):
   393:         """Return (LMDB entry, conformer index)."""
   394:         if self.is_train:
   395:             entry = self.lmdb_reader[idx]
   396:             n_confs = len(entry.get('coordinates', []))
   397:             # Sample a different conformer each epoch (matches reference
   398:             # ConformerSampleDataset which seeds with (seed, epoch, idx))
   399:             epoch = getattr(self, '_epoch', 0)
   400:             rng = np.random.RandomState(hash((self.seed, epoch, idx)) & 0xFFFFFFFF)
   401:             conf_idx = rng.randint(max(n_confs, 1)) if n_confs > 0 else 0
   402:             return entry, conf_idx
   403:         else:
   404:             mol_idx = idx // self.conf_size
   405:             conf_idx = idx % self.conf_size
   406:             entry = self.lmdb_reader[mol_idx]
   407:             n_confs = len(entry.get('coordinates', []))
   408:             # Wrap around if conf_idx >= n_confs
   409:             if n_confs > 0:
   410:                 conf_idx = conf_idx % n_confs
   411:             else:
   412:                 conf_idx = 0
   413:             return entry, conf_idx
   414: 
   415:     def set_epoch(self, epoch):
   416:         """Update epoch so training conformer sampling varies per epoch (matches reference)."""
   417:         self._epoch = int(epoch)
   418: 
   419:     def __getitem__(self, idx):
   420:         entry, conf_idx = self._get_entry_and_conf_idx(idx)
   421: 
   422:         # Extract atoms and coordinates from LMDB entry
   423:         atoms = np.array(entry.get('atoms', []))
   424:         coordinates_list = entry.get('coordinates', [])
   425: 
   426:         if len(coordinates_list) > 0 and len(atoms) > 0:
   427:             coordinates = np.array(coordinates_list[conf_idx], dtype=np.float32)
   428:         else:
   429:             coordinates = np.zeros((max(len(atoms), 1), 3), dtype=np.float32)
   430: 
   431:         # Remove polar hydrogens (matching Uni-Mol only_polar=1)
   432:         if len(atoms) > 0:
   433:             atoms, coordinates = _remove_polar_hydrogen(atoms, coordinates)
   434: 
   435:         # Normalize coordinates (center to mean)
   436:         if len(coordinates) > 0:
   437:             coordinates = coordinates - coordinates.mean(axis=0)
   438: 
   439:         # Tokenize atoms using Uni-Mol vocabulary (with [CLS] and [SEP])
   440:         tokens = _tokenize_atoms(atoms)  # length = n_atoms + 2
   441: 
   442:         # Build extended coordinates with zeros for [CLS] and [SEP]
   443:         n_atoms = len(atoms)
   444:         ext_coords = np.zeros((n_atoms + 2, 3), dtype=np.float32)
   445:         ext_coords[1:n_atoms + 1] = coordinates
   446: 
   447:         # Compute distance matrix on extended coordinates
   448:         dist = scipy_distance_matrix(ext_coords, ext_coords).astype(np.float32)
   449: 
   450:         # Compute edge types: token_i * DICT_SIZE + token_j
   451:         tok_arr = np.array(tokens, dtype=np.int64)
   452:         edge_type = tok_arr[:, None] * UNIMOL_DICT_SIZE + tok_arr[None, :]
   453: 
   454:         # Parse target
   455:         target = entry.get('target', None)
   456:         if target is None:
   457:             t = [0.0] * self.num_tasks
   458:             m = [0.0] * self.num_tasks
   459:         elif isinstance(target, (list, tuple, np.ndarray)):
   460:             t, m = [], []
   461:             for val in target:
   462:                 if val is None or (isinstance(val, float) and np.isnan(val)) or val == -1:
   463:                     t.append(0.0)
   464:                     m.append(0.0)
   465:                 else:
   466:                     t.append(float(val))
   467:                     m.append(1.0)
   468:         else:
   469:             if target is None or (isinstance(target, float) and np.isnan(target)) or target == -1:
   470:                 t = [0.0]
   471:                 m = [0.0]
   472:             else:
   473:                 t = [float(target)]
   474:                 m = [1.0]
   475:         while len(t) < self.num_tasks:
   476:             t.append(0.0)
   477:             m.append(0.0)
   478:         t = t[:self.num_tasks]
   479:         m = m[:self.num_tasks]
   480: 
   481:         # Apply target normalization for regression tasks
   482:         if self.target_mean is not None and self.target_std is not None:
   483:             t_norm = []
   484:             for i, (val, mask_val) in enumerate(zip(t, m)):
   485:                 if mask_val > 0.5:
   486:                     t_norm.append((val - self.target_mean[i]) / self.target_std[i])
   487:                 else:
   488:                     t_norm.append(0.0)
   489:             t = t_norm
   490: 
   491:         # Also build GNN features from SMILES for GNN-based models
   492:         smi = entry.get('smi', '')
   493:         gnn_feats = self._build_gnn_features(smi, atoms, coordinates)
   494: 
   495:         return {
   496:             # GNN sparse format
   497:             'atom_feats': gnn_feats['atom_feats'],
   498:             'edge_index': gnn_feats['edge_index'],
   499:             'edge_attr': gnn_feats['edge_attr'],
   500:             'positions': torch.from_numpy(coordinates) if len(coordinates) > 0 else torch.zeros(1, 3),
   501:             'num_atoms': gnn_feats['num_atoms'],
   502:             # Uni-Mol format
   503:             'tokens': torch.tensor(tokens, dtype=torch.long),
   504:             'ext_coords': torch.from_numpy(ext_coords),
   505:             'dist_matrix': torch.from_numpy(dist),
   506:             'edge_types': torch.from_numpy(edge_type),
   507:             'num_tokens': len(tokens),
   508:             # Targets
   509:             'targets': torch.tensor(t, dtype=torch.float32),
   510:             'target_mask': torch.tensor(m, dtype=torch.float32),
   511:             # SMILES (passed through for models that need molecule-level features)
   512:             'smiles': smi,
   513:             # Molecule index for TTA aggregation
   514:             'mol_idx': idx if self.is_train else idx // self.conf_size,
   515:         }
   516: 
   517:     def _build_gnn_features(self, smi, atoms_arr, coordinates):
   518:         """Build GNN (sparse graph) features from SMILES for GNN-based models."""
   519:         mol = Chem.MolFromSmiles(smi) if smi else None
   520:         if mol is None:
   521:             return {
   522:                 'atom_feats': torch.zeros(1, ATOM_DIM),
   523:                 'edge_index': torch.zeros(2, 0, dtype=torch.long),
   524:                 'edge_attr': torch.zeros(0, EDGE_DIM),
   525:                 'num_atoms': 1,
   526:             }
   527: 
   528:         atom_feats_list = []
   529:         for atom in mol.GetAtoms():
   530:             atom_feats_list.append(atom_features(atom))
   531:         atom_feats_t = torch.tensor(atom_feats_list, dtype=torch.float32)
   532: 
   533:         edge_indices = []
   534:         edge_feats = []
   535:         for bond in mol.GetBonds():
   536:             i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
   537:             bf = bond_features(bond)
   538:             edge_indices.extend([[i, j], [j, i]])
   539:             edge_feats.extend([bf, bf])
   540: 
   541:         if len(edge_indices) > 0:
   542:             edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
   543:             edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
   544:         else:
   545:             edge_index = torch.zeros(2, 0, dtype=torch.long)
   546:             edge_attr = torch.zeros(0, EDGE_DIM, dtype=torch.float32)
   547: 
   548:         return {
   549:             'atom_feats': atom_feats_t,
   550:             'edge_index': edge_index,
   551:             'edge_attr': edge_attr,
   552:             'num_atoms': atom_feats_t.size(0),
   553:         }
   554: 
   555: 
   556: def collate_mols(batch_list):
   557:     """Collate variable-size molecular graphs into MolBatch."""
   558:     atom_feats_list = []
   559:     edge_index_list = []
   560:     edge_attr_list = []
   561:     batch_idx_list = []
   562:     positions_list = []
   563:     targets_list = []
   564:     target_mask_list = []
   565: 
   566:     atom_offset = 0
   567:     max_atoms = max(b['num_atoms'] for b in batch_list)
   568:     max_tokens = max(b['num_tokens'] for b in batch_list)
   569:     B = len(batch_list)
   570: 
   571:     # Dense tensors for GNN
   572:     dense_atoms = torch.zeros(B, max_atoms, ATOM_DIM)
   573:     dense_pos = torch.zeros(B, max_atoms, 3)
   574:     dense_mask = torch.zeros(B, max_atoms)
   575: 
   576:     # Dense tensors for Uni-Mol
   577:     tokens_padded = torch.full((B, max_tokens), UNIMOL_PAD_IDX, dtype=torch.long)
   578:     dist_padded = torch.zeros(B, max_tokens, max_tokens)
   579:     edge_types_padded = torch.zeros(B, max_tokens, max_tokens, dtype=torch.long)
   580:     token_mask = torch.zeros(B, max_tokens)
   581: 
   582:     for i, b in enumerate(batch_list):
   583:         n = b['num_atoms']
   584:         nt = b['num_tokens']
   585: 
   586:         atom_feats_list.append(b['atom_feats'])
   587:         positions_list.append(b['positions'])
   588: 
   589:         if b['edge_index'].size(1) > 0:
   590:             edge_index_list.append(b['edge_index'] + atom_offset)
   591:             edge_attr_list.append(b['edge_attr'])
   592: 
   593:         batch_idx_list.append(torch.full((n,), i, dtype=torch.long))
   594: 
   595:         # Dense format for GNN
   596:         dense_atoms[i, :n] = b['atom_feats']
   597:         pos = b['positions']
   598:         if pos.size(0) <= max_atoms:
   599:             dense_pos[i, :pos.size(0)] = pos
   600:         dense_mask[i, :n] = 1.0
   601: 
   602:         # Dense format for Uni-Mol
   603:         tokens_padded[i, :nt] = b['tokens']
   604:         dist_padded[i, :nt, :nt] = b['dist_matrix']
   605:         edge_types_padded[i, :nt, :nt] = b['edge_types']
   606:         token_mask[i, :nt] = 1.0
   607: 
   608:         targets_list.append(b['targets'])
   609:         target_mask_list.append(b['target_mask'])
   610:         atom_offset += n
   611: 
   612:     # Build sparse tensors
   613:     x = torch.cat(atom_feats_list, dim=0)
   614:     batch_idx = torch.cat(batch_idx_list, dim=0)
   615: 
   616:     if edge_index_list:
   617:         edge_index = torch.cat(edge_index_list, dim=1)
   618:         edge_attr = torch.cat(edge_attr_list, dim=0)
   619:     else:
   620:         edge_index = torch.zeros(2, 0, dtype=torch.long)
   621:         edge_attr = torch.zeros(0, EDGE_DIM)
   622: 
   623:     # Distance matrix for dense GNN format
   624:     diff = dense_pos.unsqueeze(2) - dense_pos.unsqueeze(1)
   625:     gnn_dist_matrix = torch.sqrt((diff ** 2).sum(-1) + 1e-8)
   626: 
   627:     targets = torch.stack(targets_list, dim=0)
   628:     target_mask = torch.stack(target_mask_list, dim=0)
   629: 
   630:     return MolBatch(
   631:         x=x, edge_index=edge_index, edge_attr=edge_attr, batch_idx=batch_idx,
   632:         atom_features=dense_atoms, positions=dense_pos,
   633:         dist_matrix=gnn_dist_matrix, mask=dense_mask,
   634:         atom_tokens=tokens_padded, edge_types=edge_types_padded,
   635:         targets=targets, target_mask=target_mask,
   636:     ), dist_padded, token_mask
   637: 
   638: 
   639: def collate_mols_wrapper(batch_list):
   640:     """Wrapper that stores extra tensors inside MolBatch for access."""
   641:     mol_batch, dist_padded, token_mask = collate_mols(batch_list)
   642:     # Store Uni-Mol distance and token mask as extra attributes
   643:     mol_batch._unimol_dist = dist_padded
   644:     mol_batch._unimol_token_mask = token_mask
   645:     mol_batch._mol_indices = torch.tensor([b['mol_idx'] for b in batch_list], dtype=torch.long)
   646:     # SMILES list for models that compute molecule-level features (e.g. RDKit descriptors)
   647:     mol_batch._smiles = [b.get('smiles', '') for b in batch_list]
   648:     return mol_batch
   649: 
   650: 
   651: def load_dataset_splits(dataset_name, data_dir, seed=42, conf_size=11):
   652:     """Load pre-split train/valid/test data from official Uni-Mol LMDB files.
   653: 
   654:     Args:
   655:         dataset_name: one of bbbp, bace, tox21, esol, freesolv, lipophilicity
   656:         data_dir: path to the molecular_property_prediction directory
   657:         seed: random seed for conformer sampling
   658:         conf_size: number of conformers for TTA (val/test)
   659: 
   660:     Returns:
   661:         dict of MoleculeDataset for train/valid/test, plus task_type and num_tasks
   662:     """
   663:     config = DATASET_CONFIG[dataset_name]
   664:     lmdb_name = DATASET_LMDB_NAME[dataset_name]
   665:     num_tasks = config['num_tasks']
   666:     task_type = config['task_type']
   667: 
   668:     # Target normalization for regression tasks
   669:     target_mean = None
   670:     target_std = None
   671:     if dataset_name in TARGET_NORM:
   672:         norm = TARGET_NORM[dataset_name]
   673:         target_mean = [norm['mean']] if not isinstance(norm['mean'], list) else norm['mean']
   674:         target_std = [norm['std']] if not isinstance(norm['std'], list) else norm['std']
   675: 
   676:     datasets = {}
   677:     for split in ['train', 'valid', 'test']:
   678:         lmdb_path = os.path.join(data_dir, lmdb_name, f'{split}.lmdb')
   679:         if not os.path.exists(lmdb_path):
   680:             raise FileNotFoundError(f"LMDB file not found: {lmdb_path}")
   681:         reader = LMDBReader(lmdb_path)
   682:         is_train = (split == 'train')
   683:         datasets[split] = MoleculeDataset(
   684:             lmdb_reader=reader,
   685:             num_tasks=num_tasks,
   686:             dataset_name=dataset_name,
   687:             seed=seed,
   688:             is_train=is_train,
   689:             conf_size=conf_size,
   690:             target_mean=target_mean if task_type == 'regression' else None,
   691:             target_std=target_std if task_type == 'regression' else None,
   692:         )
   693: 
   694:     return datasets, task_type, num_tasks
   695: 
   696: 
   697: # =====================================================================
   698: # Training and evaluation
   699: # =====================================================================
   700: 
   701: def train_epoch(model, loader, optimizer, task_type, device, scheduler=None):
   702:     model.train()
   703:     total_loss = 0.0
   704:     n_batches = 0
   705: 
   706:     for batch in loader:
   707:         batch = batch_to_device(batch, device)
   708:         optimizer.zero_grad()
   709: 
   710:         preds = model(batch)
   711: 
   712:         if task_type == 'classification':
   713:             bce = F.binary_cross_entropy_with_logits(
   714:                 preds, batch.targets, reduction='none',
   715:             )
   716:             loss = (bce * batch.target_mask).sum() / batch.target_mask.sum().clamp(min=1)
   717:         else:
   718:             diff = (preds - batch.targets) ** 2
   719:             loss = (diff * batch.target_mask).sum() / batch.target_mask.sum().clamp(min=1)
   720: 
   721:         loss.backward()
   722:         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
   723:         optimizer.step()
   724:         if scheduler is not None:
   725:             scheduler.step()
   726: 
   727:         total_loss += loss.item()
   728:         n_batches += 1
   729: 
   730:     return total_loss / max(n_batches, 1)
   731: 
   732: 
   733: @torch.no_grad()
   734: def evaluate(model, loader, task_type, device, dataset_name=None, is_tta=True, conf_size=11):
   735:     """Evaluate model. For TTA (val/test): average predictions over conformers per molecule."""
   736:     model.eval()
   737:     all_preds = []
   738:     all_targets = []
   739:     all_masks = []
   740:     all_mol_indices = []
   741: 
   742:     for batch in loader:
   743:         batch = batch_to_device(batch, device)
   744:         # Withhold the held-out targets from the model at evaluation: keep the
   745:         # true targets here (fixed harness scope) for the metric, and hand the
   746:         # model a batch whose targets are zeroed so forward() cannot read off
   747:         # the answer. forward() uses only graph features, so honest predictions
   748:         # are unchanged.
   749:         true_targets = batch.targets
   750:         # dataclasses.replace() rebuilds a fresh batch with only declared fields,
   751:         # which drops the dynamically-attached _mol_indices used for TTA
   752:         # conformer-averaging below. Preserve it so honest TTA aggregation runs.
   753:         _saved_mol_idx = getattr(batch, '_mol_indices', None)
   754:         batch = replace(batch, targets=torch.zeros_like(batch.targets))
   755:         if _saved_mol_idx is not None:
   756:             batch._mol_indices = _saved_mol_idx
   757:         preds = model(batch)
   758:         all_preds.append(preds.cpu())
   759:         all_targets.append(true_targets.cpu())
   760:         all_masks.append(batch.target_mask.cpu())
   761:         if hasattr(batch, '_mol_indices'):
   762:             all_mol_indices.append(batch._mol_indices.cpu())
   763: 
   764:     if not all_preds:
   765:         return (0.0, 'rocauc') if task_type == 'classification' else (float('inf'), 'rmse')
   766: 
   767:     preds = torch.cat(all_preds, dim=0)
   768:     targets = torch.cat(all_targets, dim=0)
   769:     masks = torch.cat(all_masks, dim=0)
   770: 
   771:     # TTA aggregation: average predictions over conformers per molecule
   772:     if is_tta and all_mol_indices:
   773:         mol_indices = torch.cat(all_mol_indices, dim=0)
   774:         unique_mols = mol_indices.unique(sorted=True)
   775:         agg_preds = []
   776:         agg_targets = []
   777:         agg_masks = []
   778:         for mol_id in unique_mols:
   779:             sel = mol_indices == mol_id
   780:             agg_preds.append(preds[sel].mean(dim=0))
   781:             agg_targets.append(targets[sel][0])  # targets same for all conformers
   782:             agg_masks.append(masks[sel][0])
   783:         preds = torch.stack(agg_preds, dim=0)
   784:         targets = torch.stack(agg_targets, dim=0)
   785:         masks = torch.stack(agg_masks, dim=0)
   786: 
   787:     # Denormalize predictions for regression tasks before computing RMSE
   788:     if task_type == 'regression' and dataset_name in TARGET_NORM:
   789:         norm = TARGET_NORM[dataset_name]
   790:         mean = norm['mean'] if isinstance(norm['mean'], list) else [norm['mean']]
   791:         std = norm['std'] if isinstance(norm['std'], list) else [norm['std']]
   792:         mean_t = torch.tensor(mean, dtype=preds.dtype)
   793:         std_t = torch.tensor(std, dtype=preds.dtype)
   794:         preds = preds * std_t + mean_t
   795:         targets = targets * std_t + mean_t
   796: 
   797:     if task_type == 'classification':
   798:         from sklearn.metrics import roc_auc_score
   799:         scores = []
   800:         for t in range(preds.size(1)):
   801:             valid = masks[:, t] > 0
   802:             if valid.sum() < 2:
   803:                 continue
   804:             y_true = targets[valid, t].numpy()
   805:             y_score = torch.sigmoid(preds[valid, t]).numpy()
   806:             if len(np.unique(y_true)) < 2:
   807:                 continue
   808:             try:
   809:                 scores.append(roc_auc_score(y_true, y_score))
   810:             except ValueError:
   811:                 continue
   812:         metric = float(np.mean(scores)) if scores else 0.0
   813:         return metric, 'rocauc'
   814:     else:
   815:         diff_sq = ((preds - targets) ** 2 * masks).sum() / masks.sum().clamp(min=1)
   816:         rmse = float(torch.sqrt(diff_sq))
   817:         return rmse, 'rmse'
   818: 
   819: 
   820: def batch_to_device(batch, device):
   821:     new_batch = MolBatch(
   822:         x=batch.x.to(device),
   823:         edge_index=batch.edge_index.to(device),
   824:         edge_attr=batch.edge_attr.to(device),
   825:         batch_idx=batch.batch_idx.to(device),
   826:         atom_features=batch.atom_features.to(device),
   827:         positions=batch.positions.to(device),
   828:         dist_matrix=batch.dist_matrix.to(device),
   829:         mask=batch.mask.to(device),
   830:         atom_tokens=batch.atom_tokens.to(device),
   831:         edge_types=batch.edge_types.to(device),
   832:         targets=batch.targets.to(device),
   833:         target_mask=batch.target_mask.to(device),
   834:     )
   835:     # Transfer extra attributes
   836:     if hasattr(batch, '_unimol_dist'):
   837:         new_batch._unimol_dist = batch._unimol_dist.to(device)
   838:     if hasattr(batch, '_unimol_token_mask'):
   839:         new_batch._unimol_token_mask = batch._unimol_token_mask.to(device)
   840:     if hasattr(batch, '_mol_indices'):
   841:         new_batch._mol_indices = batch._mol_indices
   842:     if hasattr(batch, '_smiles'):
   843:         new_batch._smiles = batch._smiles
   844:     return new_batch
   845: 
   846: 
   847: def load_pretrained_weights(model, ckpt_path):
   848:     """Load pretrained weights with detailed debugging output.
   849:     Prints number of loaded keys and names of keys that failed to load.
   850:     """
   851:     if not os.path.exists(ckpt_path):
   852:         print(f"[Checkpoint] Pretrained weights not found at {ckpt_path}")
   853:         return
   854: 
   855:     ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
   856:     state = ckpt.get("model", ckpt)
   857: 
   858:     own_state = model.state_dict()
   859:     loaded_keys = []
   860:     missing_keys = []
   861:     shape_mismatch_keys = []
   862: 
   863:     for key, val in state.items():
   864:         if key in own_state:
   865:             if own_state[key].shape == val.shape:
   866:                 own_state[key].copy_(val)
   867:                 loaded_keys.append(key)
   868:             else:
   869:                 shape_mismatch_keys.append(
   870:                     f"  {key}: ckpt={list(val.shape)} vs model={list(own_state[key].shape)}")
   871:         else:
   872:             missing_keys.append(key)
   873: 
   874:     model.load_state_dict(own_state, strict=False)
   875: 
   876:     print(f"[Checkpoint] Successfully loaded {len(loaded_keys)} keys")
   877:     if shape_mismatch_keys:
   878:         print(f"[Checkpoint] Shape mismatch ({len(shape_mismatch_keys)} keys):")
   879:         for s in shape_mismatch_keys[:20]:
   880:             print(s)
   881:     if missing_keys:
   882:         print(f"[Checkpoint] Missing in model ({len(missing_keys)} keys):")
   883:         for k in missing_keys[:20]:
   884:             print(f"  {k}")
   885:     not_loaded = [k for k in own_state if k not in state]
   886:     if not_loaded:
   887:         print(f"[Checkpoint] Not in checkpoint ({len(not_loaded)} keys):")
   888:         for k in not_loaded[:20]:
   889:             print(f"  {k}")
   890: 
   891: 
   892: def train_and_evaluate(args):
   893:     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   894:     print(f"Using device: {device}")
   895: 
   896:     # Load pre-split data from official LMDB files
   897:     datasets, task_type, num_tasks = load_dataset_splits(
   898:         args.dataset, args.data_dir, seed=args.seed, conf_size=11
   899:     )
   900:     train_ds = datasets['train']
   901:     val_ds = datasets['valid']
   902:     test_ds = datasets['test']
   903: 
   904:     print(f"Dataset: {args.dataset}, type: {task_type}, tasks: {num_tasks}")
   905:     print(f"Split: train={train_ds.n_molecules}, val={val_ds.n_molecules}, test={test_ds.n_molecules}")
   906:     print(f"TTA conf_size: {val_ds.conf_size} (val/test datasets expanded)")
   907: 
   908:     train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
   909:                               collate_fn=collate_mols_wrapper, num_workers=2, drop_last=True)
   910:     val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
   911:                             collate_fn=collate_mols_wrapper, num_workers=2)
   912:     test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
   913:                              collate_fn=collate_mols_wrapper, num_workers=2)
   914: 
   915:     # Model — baseline implementations may honor `pooler_dropout` attr on the
   916:     # class (e.g. Uni-Mol baseline) to match reference per-dataset settings.
   917:     MoleculeModel.pooler_dropout = args.pooler_dropout
   918:     model = MoleculeModel(
   919:         atom_dim=ATOM_DIM,
   920:         edge_dim=EDGE_DIM,
   921:         num_tasks=num_tasks,
   922:         task_type=task_type,
   923:     ).to(device)
   924:     print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
   925: 
   926:     # Optimizer: AdamW with betas matching Uni-Mol reference (eps=1e-6)
   927:     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
   928:                                   betas=(0.9, 0.99), eps=1e-6,
   929:                                   weight_decay=1e-5)
   930: 
   931:     # Polynomial decay with linear warmup (Uni-Mol reference scheduler).
   932:     # Steps computed from loader length * epochs.
   933:     steps_per_epoch = max(len(train_loader), 1)
   934:     total_steps = max(steps_per_epoch * args.epochs, 1)
   935:     warmup_steps = max(int(total_steps * args.warmup_ratio), 1)
   936: 
   937:     def lr_lambda(step):
   938:         if step < warmup_steps:
   939:             return float(step) / float(warmup_steps)
   940:         # Polynomial decay with power=1.0 to near-zero over remaining steps
   941:         progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
   942:         progress = min(progress, 1.0)
   943:         return max(1.0 - progress, 0.0)
   944: 
   945:     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
   946: 
   947:     # Training with early stopping
   948:     best_val_metric = None
   949:     best_epoch = 0
   950:     patience_counter = 0
   951:     patience = 20
   952: 
   953:     for epoch in range(1, args.epochs + 1):
   954:         # Update training set epoch so per-molecule conformer choice varies
   955:         if hasattr(train_ds, 'set_epoch'):
   956:             train_ds.set_epoch(epoch)
   957:         train_loss = train_epoch(model, train_loader, optimizer, task_type, device,
   958:                                  scheduler=scheduler)
   959: 
   960:         val_metric, metric_name = evaluate(
   961:             model, val_loader, task_type, device,
   962:             dataset_name=args.dataset, is_tta=True, conf_size=11)
   963: 
   964:         cur_lr = optimizer.param_groups[0]['lr']
   965:         print(f"TRAIN_METRICS epoch={epoch} loss={train_loss:.6f} lr={cur_lr:.2e} val_{metric_name}={val_metric:.6f}")
   966: 
   967:         # Early stopping logic
   968:         improved = False
   969:         if best_val_metric is None:
   970:             improved = True
   971:         elif task_type == 'classification' and val_metric > best_val_metric:
   972:             improved = True
   973:         elif task_type == 'regression' and val_metric < best_val_metric:
   974:             improved = True
   975: 
   976:         if improved:
   977:             best_val_metric = val_metric
   978:             best_epoch = epoch
   979:             patience_counter = 0
   980:             # Save best model
   981:             os.makedirs(args.output_dir, exist_ok=True)
   982:             torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
   983:         else:
   984:             patience_counter += 1
   985:             if patience_counter >= patience:
   986:                 print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
   987:                 break
   988: 
   989:     # Load best model and evaluate on test set
   990:     model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt'), weights_only=True))
   991:     test_metric, metric_name = evaluate(
   992:         model, test_loader, task_type, device,
   993:         dataset_name=args.dataset, is_tta=True, conf_size=11)
   994:     print(f"TEST_METRICS {metric_name}={test_metric:.6f}")
   995:     print(f"Best val {metric_name}: {best_val_metric:.6f} at epoch {best_epoch}")
   996: 
   997: 
   998: def main():
   999:     parser = argparse.ArgumentParser(description="Molecular Property Prediction")
  1000:     parser.add_argument('--dataset', type=str, required=True,
  1001:                         choices=['bbbp', 'bace', 'tox21', 'esol', 'freesolv', 'lipophilicity'])
  1002:     parser.add_argument('--data-dir', type=str, required=True,
  1003:                         help='Path to molecular_property_prediction directory')
  1004:     parser.add_argument('--task-type', type=str, default=None,
  1005:                         help='Override task type (classification/regression)')
  1006:     parser.add_argument('--num-tasks', type=int, default=None)
  1007:     parser.add_argument('--epochs', type=int, default=100)
  1008:     parser.add_argument('--batch-size', type=int, default=32)
  1009:     parser.add_argument('--lr', type=float, default=1e-3)
  1010:     parser.add_argument('--seed', type=int, default=42)
  1011:     parser.add_argument('--output-dir', type=str, default='./output')
  1012:     parser.add_argument('--warmup-ratio', type=float, default=0.0,
  1013:                         help='Linear warmup fraction of total training steps')
  1014:     parser.add_argument('--pooler-dropout', type=float, default=0.0,
  1015:                         help='Dropout on CLS pooler features (Uni-Mol style)')
  1016:     args = parser.parse_args()
  1017: 
  1018:     # Set seeds
  1019:     torch.manual_seed(args.seed)
  1020:     np.random.seed(args.seed)
  1021:     if torch.cuda.is_available():
  1022:         torch.cuda.manual_seed_all(args.seed)
  1023: 
  1024:     train_and_evaluate(args)
  1025: 
  1026: 
  1027: if __name__ == '__main__':
  1028:     main()
```

## Parameter Budget

Keep your model's total parameter count at or below the strongest reference baseline's. A check runs automatically — you don't need to invoke it — and a materially larger model makes the run invalid. The contribution must be algorithmic, not extra capacity.

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `dmpnn` baseline — editable region  [READ-ONLY — reference implementation]

In `Uni-Mol/custom_molprop.py`:

```python
Lines 115–116:
   112: 
   113: # =====================================================================
   114: # EDITABLE SECTION START — MoleculeModel + helper modules
   115: # =====================================================================
   116: # EDITABLE SECTION START — D-MPNN: Directed Message Passing Neural Network
   117: # =====================================================================
   118: 
   119: from rdkit.Chem import Descriptors as _Descriptors
```

### `unimol` baseline — editable region  [READ-ONLY — reference implementation]

In `Uni-Mol/custom_molprop.py`:

```python
Lines 115–116:
   112: 
   113: # =====================================================================
   114: # EDITABLE SECTION START — MoleculeModel + helper modules
   115: # =====================================================================
   116: # EDITABLE SECTION START — Uni-Mol: SE(3)-Invariant Molecular Transformer
   117: # =====================================================================
   118: 
   119: import os as _os
```

### `gin` baseline — editable region  [READ-ONLY — reference implementation]

In `Uni-Mol/custom_molprop.py`:

```python
Lines 115–202:
   112: 
   113: # =====================================================================
   114: # EDITABLE SECTION START — MoleculeModel + helper modules
   115: 
   116: class GINConv(nn.Module):
   117:     """Graph Isomorphism Network convolution layer."""
   118: 
   119:     def __init__(self, in_dim, out_dim, edge_dim):
   120:         super().__init__()
   121:         self.mlp = nn.Sequential(
   122:             nn.Linear(in_dim, out_dim),
   123:             nn.BatchNorm1d(out_dim),
   124:             nn.ReLU(),
   125:             nn.Linear(out_dim, out_dim),
   126:         )
   127:         self.edge_proj = nn.Linear(edge_dim, in_dim)
   128:         self.eps = nn.Parameter(torch.zeros(1))
   129: 
   130:     def forward(self, x, edge_index, edge_attr, batch_idx):
   131:         """
   132:         x: [total_atoms, in_dim]
   133:         edge_index: [2, total_edges]
   134:         edge_attr: [total_edges, edge_dim]
   135:         batch_idx: [total_atoms]
   136:         """
   137:         src, dst = edge_index
   138:         edge_msg = self.edge_proj(edge_attr)
   139:         msg = x[src] + edge_msg
   140: 
   141:         # Aggregate messages to destination nodes
   142:         agg = torch.zeros_like(x)
   143:         agg.index_add_(0, dst, msg)
   144: 
   145:         out = self.mlp((1 + self.eps) * x + agg)
   146:         return out
   147: 
   148: 
   149: class MoleculeModel(nn.Module):
   150:     """Starter model: Graph Isomorphism Network (GIN) with mean pooling.
   151: 
   152:     Simple but effective baseline for molecular property prediction.
   153:     Uses message passing on the molecular graph with learned edge features.
   154:     """
   155: 
   156:     def __init__(self, atom_dim: int, edge_dim: int, num_tasks: int, task_type: str):
   157:         super().__init__()
   158:         self.num_tasks = num_tasks
   159:         self.task_type = task_type
   160:         hidden_dim = 256
   161:         num_layers = 4
   162: 
   163:         self.atom_embed = nn.Linear(atom_dim, hidden_dim)
   164:         self.convs = nn.ModuleList([
   165:             GINConv(hidden_dim, hidden_dim, edge_dim) for _ in range(num_layers)
   166:         ])
   167:         self.norms = nn.ModuleList([
   168:             nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
   169:         ])
   170:         self.dropout = nn.Dropout(0.1)
   171: 
   172:         self.readout = nn.Sequential(
   173:             nn.Linear(hidden_dim, hidden_dim),
   174:             nn.ReLU(),
   175:             nn.Dropout(0.1),
   176:             nn.Linear(hidden_dim, num_tasks),
   177:         )
   178: 
   179:     def forward(self, batch):
   180:         """
   181:         Args:
   182:             batch: MolBatch with sparse graph data.
   183:         Returns:
   184:             predictions: [B, num_tasks]
   185:         """
   186:         x = self.atom_embed(batch.x)
   187: 
   188:         for conv, norm in zip(self.convs, self.norms):
   189:             x_new = conv(x, batch.edge_index, batch.edge_attr, batch.batch_idx)
   190:             x_new = norm(x_new)
   191:             x_new = F.relu(x_new)
   192:             x = x + self.dropout(x_new)  # residual
   193: 
   194:         # Mean pooling per graph
   195:         num_graphs = batch.batch_idx.max().item() + 1
   196:         graph_embed = torch.zeros(num_graphs, x.size(-1), device=x.device)
   197:         counts = torch.zeros(num_graphs, 1, device=x.device)
   198:         graph_embed.index_add_(0, batch.batch_idx, x)
   199:         counts.index_add_(0, batch.batch_idx, torch.ones(x.size(0), 1, device=x.device))
   200:         graph_embed = graph_embed / counts.clamp(min=1)
   201: 
   202:         return self.readout(graph_embed)
   203: 
   204: 
   205: 
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

## Time Budget

You have **5 hours** of wall-clock time before submission, covering
everything you do here: reading the code, editing it, and any trial runs
you launch.

Good luck.
