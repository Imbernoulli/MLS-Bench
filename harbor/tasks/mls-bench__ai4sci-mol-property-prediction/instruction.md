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
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

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
    28: from dataclasses import dataclass
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

[truncated: showing at most 500 lines / 60000 bytes from Uni-Mol/custom_molprop.py]
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


### `dmpnn` baseline — editable region  [READ-ONLY — reference implementation]

In `Uni-Mol/custom_molprop.py`:

```python
Lines 115–351:
   112: 
   113: # =====================================================================
   114: # EDITABLE SECTION START — MoleculeModel + helper modules
   115: # =====================================================================
   116: # EDITABLE SECTION START — D-MPNN: Directed Message Passing Neural Network
   117: # =====================================================================
   118: 
   119: from rdkit.Chem import Descriptors as _Descriptors
   120: from rdkit.Chem import rdMolDescriptors as _rdMolDescriptors
   121: from rdkit.Chem import MolFromSmiles as _MolFromSmiles
   122: 
   123: 
   124: # --------------------- RDKit 2D molecular descriptors -----------------
   125: # A compact subset of normalized RDKit 2D descriptors that have been
   126: # shown to improve D-MPNN on physicochemical / biophysical tasks (Yang
   127: # et al. 2019, "rdkit_2d_normalized" features generator).  We compute
   128: # them once per SMILES and per-feature standardize using running stats
   129: # accumulated over the training batches — a robust approximation of
   130: # chemprop's pre-computed Welford normalization.
   131: 
   132: def _rdkit_2d_descriptors(smi):
   133:     """Compute a fixed-length RDKit 2D descriptor vector for a SMILES."""
   134:     if not smi:
   135:         return [0.0] * 17
   136:     mol = _MolFromSmiles(smi)
   137:     if mol is None:
   138:         return [0.0] * 17
   139:     feats = [
   140:         _Descriptors.MolWt(mol),
   141:         _Descriptors.MolLogP(mol),
   142:         _Descriptors.NumHDonors(mol),
   143:         _Descriptors.NumHAcceptors(mol),
   144:         _Descriptors.TPSA(mol),
   145:         _Descriptors.NumRotatableBonds(mol),
   146:         _Descriptors.NumAromaticRings(mol),
   147:         _Descriptors.NumAliphaticRings(mol),
   148:         _Descriptors.HeavyAtomCount(mol),
   149:         _Descriptors.RingCount(mol),
   150:         _Descriptors.FractionCSP3(mol),
   151:         _Descriptors.NumHeteroatoms(mol),
   152:         _rdMolDescriptors.CalcNumSaturatedRings(mol),
   153:         _rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
   154:         _rdMolDescriptors.CalcNumAliphaticHeterocycles(mol),
   155:         _Descriptors.MolMR(mol),
   156:         _Descriptors.LabuteASA(mol),
   157:     ]
   158:     # NaN / inf guard
   159:     cleaned = []
   160:     for v in feats:
   161:         try:
   162:             v = float(v)
   163:             if math.isnan(v) or math.isinf(v):
   164:                 v = 0.0
   165:         except Exception:
   166:             v = 0.0
   167:         cleaned.append(v)
   168:     return cleaned
   169: 
   170: 
   171: _RDKIT_FEAT_DIM = 17
   172: 
   173: 
   174: class _RunningNormalizer(nn.Module):
   175:     """Running mean/std normalizer for RDKit features (BatchNorm-style)."""
   176: 
   177:     def __init__(self, dim, momentum=0.01):
   178:         super().__init__()
   179:         self.dim = dim
   180:         self.momentum = momentum
   181:         self.register_buffer('running_mean', torch.zeros(dim))
   182:         self.register_buffer('running_std', torch.ones(dim))
   183: 
   184:     def forward(self, x):
   185:         if self.training:
   186:             with torch.no_grad():
   187:                 mean = x.mean(dim=0)
   188:                 std = x.std(dim=0).clamp(min=1e-6)
   189:                 self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
   190:                 self.running_std.mul_(1 - self.momentum).add_(self.momentum * std)
   191:         return (x - self.running_mean) / self.running_std.clamp(min=1e-6)
   192: 
   193: 
   194: class DMPNNEncoder(nn.Module):
   195:     """Directed Message Passing Neural Network (Yang et al., 2019).
   196: 
   197:     Bond-level messages flow along directed edges; each message passing step
   198:     computes new edge messages from incoming atom messages minus the reverse
   199:     edge contribution to avoid message collision.
   200:     """
   201: 
   202:     def __init__(self, atom_dim, edge_dim, hidden_dim=300, depth=3, dropout=0.0):
   203:         super().__init__()
   204:         self.hidden_dim = hidden_dim
   205:         self.depth = depth
   206: 
   207:         # Initial bond message: linear over [atom_src || bond_attr]
   208:         self.W_i = nn.Linear(atom_dim + edge_dim, hidden_dim, bias=False)
   209:         # Shared message-update weight (chemprop default)
   210:         self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
   211:         # Final atom-level readout combine
   212:         self.W_o = nn.Linear(atom_dim + hidden_dim, hidden_dim)
   213:         self.dropout = nn.Dropout(dropout)
   214:         self.act = nn.ReLU()
   215: 
   216:     def forward(self, x, edge_index, edge_attr, batch_idx):
   217:         """
   218:         x: [total_atoms, atom_dim]
   219:         edge_index: [2, total_edges] (bidirectional, paired as [i,j],[j,i])
   220:         edge_attr: [total_edges, edge_dim]
   221:         batch_idx: [total_atoms]
   222:         """
   223:         src, dst = edge_index
   224:         num_atoms = x.size(0)
   225:         num_edges = edge_index.size(1)
   226: 
   227:         if num_edges == 0:
   228:             # Fallback for atom-only molecules
   229:             atom_hidden = self.act(self.W_o(torch.cat([x, torch.zeros(num_atoms, self.hidden_dim, device=x.device)], dim=-1)))
   230:             return self.dropout(atom_hidden)
   231: 
   232:         # Reverse edge index: edges are added in pairs (i->j, j->i),
   233:         # so reverse of edge e is e XOR 1.
   234:         rev_edge_idx = torch.arange(num_edges, device=x.device) ^ 1
   235:         rev_edge_idx = rev_edge_idx.clamp(max=num_edges - 1)
   236: 
   237:         # Initial bond input: source atom features concatenated with bond features
   238:         bond_input = torch.cat([x[src], edge_attr], dim=-1)
   239:         h0 = self.act(self.W_i(bond_input))  # [num_edges, hidden]
   240:         h = h0
   241: 
   242:         # Message passing for depth-1 steps (chemprop convention)
   243:         for _ in range(self.depth - 1):
   244:             # Aggregate incoming messages to each atom
   245:             atom_msg = torch.zeros(num_atoms, self.hidden_dim, device=x.device)
   246:             atom_msg.index_add_(0, dst, h)
   247: 
   248:             # New edge message: a_v - h_{v->u}^{rev} (avoid passing back)
   249:             new_h = atom_msg[src] - h[rev_edge_idx]
   250:             new_h = self.W_h(new_h)
   251:             # Residual on h0 (chemprop style)
   252:             new_h = self.act(h0 + new_h)
   253:             new_h = self.dropout(new_h)
   254:             h = new_h
   255: 
   256:         # Final atom messages
   257:         atom_msg = torch.zeros(num_atoms, self.hidden_dim, device=x.device)
   258:         atom_msg.index_add_(0, dst, h)
   259: 
   260:         # Combine atom features with aggregated bond messages
   261:         atom_hidden = self.act(self.W_o(torch.cat([x, atom_msg], dim=-1)))
   262:         atom_hidden = self.dropout(atom_hidden)
   263:         return atom_hidden
   264: 
   265: 
   266: class MoleculeModel(nn.Module):
   267:     """D-MPNN with RDKit 2D normalized molecular descriptors.
   268: 
   269:     Configuration follows Yang et al. 2019 chemprop defaults:
   270:       - hidden_dim = 300
   271:       - depth = 3 message passing steps
   272:       - sum readout per graph
   273:       - 2-layer FFN head with hidden=300
   274:       - RDKit 2D descriptors concatenated at the readout ("+features" mode)
   275:     """
   276: 
   277:     def __init__(self, atom_dim: int, edge_dim: int, num_tasks: int, task_type: str):
   278:         super().__init__()
   279:         self.num_tasks = num_tasks
   280:         self.task_type = task_type
   281:         hidden_dim = 300
   282:         depth = 3
   283:         # `pooler_dropout` may be set by the training driver to vary dropout
   284:         # per dataset (e.g. BACE/Tox21=0.1, BBBP=0.0, regression tasks=0.1-0.2)
   285:         dropout = float(getattr(type(self), "pooler_dropout", 0.0))
   286: 
   287:         self.encoder = DMPNNEncoder(
   288:             atom_dim=atom_dim,
   289:             edge_dim=edge_dim,
   290:             hidden_dim=hidden_dim,
   291:             depth=depth,
   292:             dropout=dropout,
   293:         )
   294: 
   295:         # RDKit 2D descriptor branch
   296:         self.feat_norm = _RunningNormalizer(_RDKIT_FEAT_DIM)
   297: 
   298:         # 2-layer FFN head over [graph_embed || rdkit_features]
   299:         readout_in = hidden_dim + _RDKIT_FEAT_DIM
   300:         self.readout = nn.Sequential(
   301:             nn.Linear(readout_in, hidden_dim),
   302:             nn.ReLU(),
   303:             nn.Dropout(dropout),
   304:             nn.Linear(hidden_dim, num_tasks),
   305:         )
   306: 
   307:         # Lazy SMILES->feature cache (shared across forward calls)
   308:         self._smi_cache = {}
   309: 
   310:     def _batch_rdkit_features(self, batch):
   311:         """Compute RDKit features for the molecules in this batch.
   312: 
   313:         Uses LMDB SMILES via the dataset wrapper.  When SMILES are not
   314:         available (no `_smiles` attr), falls back to a zero vector — the
   315:         running normalizer will then produce zeros, leaving the GNN
   316:         branch unaffected.
   317:         """
   318:         smiles = getattr(batch, "_smiles", None)
   319:         if smiles is None:
   320:             num_graphs = int(batch.batch_idx.max().item()) + 1
   321:             return torch.zeros(num_graphs, _RDKIT_FEAT_DIM,
   322:                                device=batch.x.device)
   323: 
   324:         feats = []
   325:         for smi in smiles:
   326:             if smi in self._smi_cache:
   327:                 feats.append(self._smi_cache[smi])
   328:             else:
   329:                 f = _rdkit_2d_descriptors(smi)
   330:                 self._smi_cache[smi] = f
   331:                 feats.append(f)
   332:         return torch.tensor(feats, dtype=torch.float32, device=batch.x.device)
   333: 
   334:     def forward(self, batch):
   335:         atom_hidden = self.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch_idx)
   336: 
   337:         # Sum pooling per graph (chemprop default)
   338:         num_graphs = int(batch.batch_idx.max().item()) + 1
   339:         graph_embed = torch.zeros(num_graphs, atom_hidden.size(-1), device=atom_hidden.device)
   340:         graph_embed.index_add_(0, batch.batch_idx, atom_hidden)
   341: 
   342:         # RDKit feature branch (per-graph)
   343:         rdkit_feats = self._batch_rdkit_features(batch)
   344:         rdkit_feats = self.feat_norm(rdkit_feats)
   345: 
   346:         combined = torch.cat([graph_embed, rdkit_feats], dim=-1)
   347:         return self.readout(combined)
   348: 
   349: # =====================================================================
   350: # EDITABLE SECTION END
   351: # =====================================================================
   352: 
   353: 
   354: # =====================================================================
```

### `unimol` baseline — editable region  [READ-ONLY — reference implementation]

In `Uni-Mol/custom_molprop.py`:

```python
Lines 115–606:
   112: 
   113: # =====================================================================
   114: # EDITABLE SECTION START — MoleculeModel + helper modules
   115: # =====================================================================
   116: # EDITABLE SECTION START — Uni-Mol: SE(3)-Invariant Molecular Transformer
   117: # =====================================================================
   118: 
   119: import os as _os
   120: import logging as _logging
   121: 
   122: _logger = _logging.getLogger(__name__)
   123: 
   124: # --------------- Uni-Mol dictionary (mirrors dict.txt) ----------------
   125: # The pretrained model uses a token vocabulary.  We map atomic numbers
   126: # from the featurisation to dictionary indices so that we can re-use the
   127: # pretrained ``embed_tokens`` embedding and edge-type Gaussian layer.
   128: # dict.txt ordering:
   129: #   [PAD]=0, [CLS]=1, [SEP]=2, [UNK]=3, C=4, N=5, O=6, S=7, H=8,
   130: #   Cl=9, F=10, Br=11, I=12, Si=13, P=14, B=15, Na=16, K=17, Al=18,
   131: #   Ca=19, Sn=20, As=21, Hg=22, Fe=23, Zn=24, Cr=25, Se=26, Gd=27,
   132: #   Au=28, Li=29
   133: _ELEM_TO_DICT_IDX = {
   134:     6: 4, 7: 5, 8: 6, 16: 7, 1: 8, 17: 9, 9: 10, 35: 11,
   135:     53: 12, 14: 13, 15: 14, 5: 15, 11: 16, 19: 17, 13: 18,
   136:     20: 19, 50: 20, 33: 21, 80: 22, 26: 23, 30: 24, 24: 25,
   137:     34: 26, 64: 27, 79: 28, 3: 29,
   138: }
   139: _DICT_SIZE = 31  # 30 atoms + [MASK] token to match pretrained checkpoint
   140: _PAD_IDX = 0
   141: _CLS_IDX = 1
   142: _SEP_IDX = 2
   143: _UNK_IDX = 3
   144: 
   145: 
   146: def _atomic_num_from_features(atom_feat):
   147:     """Extract atomic number from the one-hot atom feature vector.
   148:     The first 118 elements encode atomic_num (1..118).
   149:     """
   150:     idx = atom_feat[:118].argmax().item()
   151:     if atom_feat[idx].item() < 0.5:
   152:         return 0
   153:     return idx + 1
   154: 
   155: 
   156: def _atoms_to_tokens(atom_features_dense, mask):
   157:     """Convert dense atom features [B, N, D] + mask [B, N] to token ids [B, N+2].
   158:     Prepends [CLS] and appends [SEP], matching the reference data pipeline.
   159:     Padding positions get PAD token.
   160:     """
   161:     B, N, D = atom_features_dense.shape
   162:     device = atom_features_dense.device
   163: 
   164:     tokens = torch.full((B, N), _PAD_IDX, dtype=torch.long, device=device)
   165:     for b in range(B):
   166:         for i in range(N):
   167:             if mask[b, i].item() > 0.5:
   168:                 anum = _atomic_num_from_features(atom_features_dense[b, i])
   169:                 tokens[b, i] = _ELEM_TO_DICT_IDX.get(anum, _UNK_IDX)
   170: 
   171:     cls_col = torch.full((B, 1), _CLS_IDX, dtype=torch.long, device=device)
   172:     sep_col = torch.full((B, 1), _SEP_IDX, dtype=torch.long, device=device)
   173:     tokens = torch.cat([cls_col, tokens, sep_col], dim=1)  # [B, N+2]
   174:     return tokens
   175: 
   176: 
   177: def _extend_dist_and_mask(dist_matrix, mask):
   178:     """Extend distance matrix and mask for [CLS] and [SEP] tokens.
   179:     Returns dist [B, N+2, N+2] and padding_mask [B, N+2] (True=pad).
   180:     """
   181:     B, N, _ = dist_matrix.shape
   182:     device = dist_matrix.device
   183:     dist = torch.zeros(B, N + 2, N + 2, device=device, dtype=dist_matrix.dtype)
   184:     dist[:, 1:N+1, 1:N+1] = dist_matrix
   185: 
   186:     ext_mask = torch.zeros(B, N + 2, device=device, dtype=mask.dtype)
   187:     ext_mask[:, 0] = 1.0    # CLS always valid
   188:     ext_mask[:, 1:N+1] = mask
   189:     ext_mask[:, N+1] = 1.0  # SEP always valid
   190: 
   191:     padding_mask = (ext_mask < 0.5)  # True = padded
   192:     return dist, padding_mask
   193: 
   194: 
   195: # -------------------- Gaussian Distance Encoding ---------------------
   196: 
   197: @torch.jit.script
   198: def _gaussian(x, mean, std):
   199:     pi = 3.14159
   200:     a = (2.0 * pi) ** 0.5
   201:     return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)
   202: 
   203: 
   204: class GaussianLayer(nn.Module):
   205:     """Edge-type-dependent Gaussian distance encoding.
   206:     Each atom-pair type (i,j) has its own learned scaling (mul) and bias.
   207:     This is critical: it lets the model distinguish C-C, C-N, C-O etc.
   208:     """
   209:     def __init__(self, K=128, edge_types=1024):
   210:         super().__init__()
   211:         self.K = K
   212:         self.means = nn.Embedding(1, K)
   213:         self.stds = nn.Embedding(1, K)
   214:         self.mul = nn.Embedding(edge_types, 1)
   215:         self.bias = nn.Embedding(edge_types, 1)
   216:         nn.init.uniform_(self.means.weight, 0, 3)
   217:         nn.init.uniform_(self.stds.weight, 0, 3)
   218:         nn.init.constant_(self.bias.weight, 0)
   219:         nn.init.constant_(self.mul.weight, 1)
   220: 
   221:     def forward(self, x, edge_type):
   222:         mul = self.mul(edge_type).type_as(x)
   223:         bias = self.bias(edge_type).type_as(x)
   224:         x = mul * x.unsqueeze(-1) + bias
   225:         x = x.expand(-1, -1, -1, self.K)
   226:         mean = self.means.weight.float().view(-1)
   227:         std = self.stds.weight.float().view(-1).abs() + 1e-5
   228:         return _gaussian(x.float(), mean, std).type_as(self.means.weight)
   229: 
   230: 
   231: # -------------------- Helper Heads -----------------------------------
   232: 
   233: class NonLinearHead(nn.Module):
   234:     """Two-layer MLP with GELU activation for projecting Gaussian features."""
   235:     def __init__(self, input_dim, out_dim, hidden=None):
   236:         super().__init__()
   237:         hidden = input_dim if not hidden else hidden
   238:         self.linear1 = nn.Linear(input_dim, hidden)
   239:         self.linear2 = nn.Linear(hidden, out_dim)
   240: 
   241:     def forward(self, x):
   242:         return self.linear2(F.gelu(self.linear1(x)))
   243: 
   244: 
   245: class ClassificationHead(nn.Module):
   246:     """Head for molecule-level prediction via [CLS] token.
   247:     Reference Uni-Mol uses simple dropout + linear (no hidden dense/tanh).
   248:     """
   249:     def __init__(self, input_dim, inner_dim, num_classes, pooler_dropout=0.2):
   250:         super().__init__()
   251:         self.dropout = nn.Dropout(p=pooler_dropout)
   252:         self.out_proj = nn.Linear(input_dim, num_classes)
   253: 
   254:     def forward(self, features):
   255:         x = features[:, 0, :]  # [CLS] token
   256:         x = self.dropout(x)
   257:         return self.out_proj(x)
   258: 
   259: 
   260: # -------------------- Transformer Encoder With Pair -------------------
   261: 
   262: class SelfMultiheadAttention(nn.Module):
   263:     """Multi-head self-attention with fused QKV projection."""
   264:     def __init__(self, embed_dim, num_heads, dropout=0.1):
   265:         super().__init__()
   266:         self.embed_dim = embed_dim
   267:         self.num_heads = num_heads
   268:         self.head_dim = embed_dim // num_heads
   269:         self.scaling = self.head_dim ** -0.5
   270:         self.dropout = dropout
   271:         self.in_proj = nn.Linear(embed_dim, embed_dim * 3)
   272:         self.out_proj = nn.Linear(embed_dim, embed_dim)
   273: 
   274:     def forward(self, query, key_padding_mask=None, attn_bias=None,
   275:                 return_attn=False):
   276:         bsz, tgt_len, _ = query.size()
   277:         q, k, v = self.in_proj(query).chunk(3, dim=-1)
   278: 
   279:         def reshape(t):
   280:             return t.view(bsz, -1, self.num_heads, self.head_dim) \
   281:                     .transpose(1, 2).contiguous() \
   282:                     .view(bsz * self.num_heads, -1, self.head_dim)
   283: 
   284:         q = reshape(q) * self.scaling
   285:         k = reshape(k)
   286:         v = reshape(v)
   287:         src_len = k.size(1)
   288: 
   289:         attn_weights = torch.bmm(q, k.transpose(1, 2))
   290: 
   291:         if key_padding_mask is not None:
   292:             attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
   293:             attn_weights.masked_fill_(
   294:                 key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
   295:                 float("-inf"),
   296:             )
   297:             attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
   298: 
   299:         if not return_attn:
   300:             aw = attn_weights
   301:             if attn_bias is not None:
   302:                 aw = aw + attn_bias
   303:             attn = F.dropout(F.softmax(aw, dim=-1),
   304:                              p=self.dropout, training=self.training)
   305:         else:
   306:             attn_weights = attn_weights + attn_bias
   307:             attn = F.dropout(F.softmax(attn_weights.clone(), dim=-1),
   308:                              p=self.dropout, training=self.training)
   309: 
   310:         o = torch.bmm(attn, v)
   311:         o = o.view(bsz, self.num_heads, tgt_len, self.head_dim) \
   312:              .transpose(1, 2).contiguous() \
   313:              .view(bsz, tgt_len, self.embed_dim)
   314:         o = self.out_proj(o)
   315:         if not return_attn:
   316:             return o
   317:         return o, attn_weights, attn
   318: 
   319: 
   320: class UniMolEncoderLayer(nn.Module):
   321:     """Pre-LN Transformer encoder layer (matches reference)."""
   322:     def __init__(self, embed_dim, ffn_embed_dim, attention_heads,
   323:                  dropout=0.1, attention_dropout=0.1, activation_dropout=0.0):
   324:         super().__init__()
   325:         self.dropout = dropout
   326:         self.activation_dropout = activation_dropout
   327:         self.self_attn = SelfMultiheadAttention(
   328:             embed_dim, attention_heads, dropout=attention_dropout)
   329:         self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
   330:         self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
   331:         self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
   332:         self.final_layer_norm = nn.LayerNorm(embed_dim)
   333: 
   334:     def forward(self, x, padding_mask=None, attn_bias=None, return_attn=False):
   335:         residual = x
   336:         x = self.self_attn_layer_norm(x)
   337:         x = self.self_attn(query=x, key_padding_mask=padding_mask,
   338:                            attn_bias=attn_bias, return_attn=return_attn)
   339:         if return_attn:
   340:             x, attn_weights, attn_probs = x
   341:         x = F.dropout(x, p=self.dropout, training=self.training)
   342:         x = residual + x
   343: 
   344:         residual = x
   345:         x = self.final_layer_norm(x)
   346:         x = F.gelu(self.fc1(x))
   347:         x = F.dropout(x, p=self.activation_dropout, training=self.training)
   348:         x = self.fc2(x)
   349:         x = F.dropout(x, p=self.dropout, training=self.training)
   350:         x = residual + x
   351: 
   352:         if not return_attn:
   353:             return x
   354:         return x, attn_weights, attn_probs
   355: 
   356: 
   357: class TransformerEncoderWithPair(nn.Module):
   358:     """Transformer encoder that tracks and updates pair (attention bias)
   359:     representations through layers — a key Uni-Mol architectural feature.
   360:     """
   361:     def __init__(self, encoder_layers, embed_dim, ffn_embed_dim,
   362:                  attention_heads, emb_dropout=0.1, dropout=0.1,
   363:                  attention_dropout=0.1, activation_dropout=0.0):
   364:         super().__init__()
   365:         self.emb_dropout = emb_dropout
   366:         self.embed_dim = embed_dim
   367:         self.attention_heads = attention_heads
   368:         self.emb_layer_norm = nn.LayerNorm(embed_dim)
   369:         self.final_layer_norm = nn.LayerNorm(embed_dim)
   370:         self.final_head_layer_norm = nn.LayerNorm(attention_heads)
   371:         self.layers = nn.ModuleList([
   372:             UniMolEncoderLayer(
   373:                 embed_dim=embed_dim, ffn_embed_dim=ffn_embed_dim,
   374:                 attention_heads=attention_heads, dropout=dropout,
   375:                 attention_dropout=attention_dropout,
   376:                 activation_dropout=activation_dropout,
   377:             )
   378:             for _ in range(encoder_layers)
   379:         ])
   380: 
   381:     def forward(self, emb, attn_mask=None, padding_mask=None):
   382:         bsz, seq_len = emb.size(0), emb.size(1)
   383:         x = self.emb_layer_norm(emb)
   384:         x = F.dropout(x, p=self.emb_dropout, training=self.training)
   385: 
   386:         if padding_mask is not None:
   387:             x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
   388: 
   389:         input_attn_mask = attn_mask
   390:         input_padding_mask = padding_mask
   391: 
   392:         def fill_attn_mask(am, pm, fill_val=float("-inf")):
   393:             if am is not None and pm is not None:
   394:                 am = am.view(bsz, -1, seq_len, seq_len)
   395:                 am.masked_fill_(
   396:                     pm.unsqueeze(1).unsqueeze(2).to(torch.bool), fill_val)
   397:                 am = am.view(-1, seq_len, seq_len)
   398:                 pm = None
   399:             return am, pm
   400: 
   401:         assert attn_mask is not None
   402:         attn_mask, padding_mask = fill_attn_mask(attn_mask, padding_mask)
   403: 
   404:         for layer in self.layers:
   405:             x, attn_mask, _ = layer(
   406:                 x, padding_mask=padding_mask,
   407:                 attn_bias=attn_mask, return_attn=True)
   408: 
   409:         x = self.final_layer_norm(x)
   410: 
   411:         # Compute pair representations
   412:         delta_pair_repr = attn_mask - input_attn_mask
   413:         delta_pair_repr, _ = fill_attn_mask(
   414:             delta_pair_repr, input_padding_mask, 0)
   415:         attn_mask = attn_mask.view(
   416:             bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
   417:         delta_pair_repr = delta_pair_repr.view(
   418:             bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
   419:         delta_pair_repr = self.final_head_layer_norm(delta_pair_repr)
   420: 
   421:         return x, attn_mask, delta_pair_repr
   422: 
   423: 
   424: # -------------------- Main Model ------------------------------------
   425: 
   426: class MoleculeModel(nn.Module):
   427:     """Uni-Mol: SE(3)-invariant Transformer with pretrained weights.
   428: 
   429:     Architecture: 15 layers, 512 hidden dim, 64 attention heads, 2048 FFN
   430:     dim (~86M parameters).  Uses edge-type-dependent Gaussian distance
   431:     encoding and pair representation tracking through the encoder layers.
   432:     Loads pretrained weights from the checkpoint at build time.
   433:     """
   434: 
   435:     def __init__(self, atom_dim: int, edge_dim: int, num_tasks: int,
   436:                  task_type: str):
   437:         super().__init__()
   438:         self.num_tasks = num_tasks
   439:         self.task_type = task_type
   440: 
   441:         # Architecture (matches reference base_architecture)
   442:         embed_dim = 512
   443:         ffn_embed_dim = 2048
   444:         attention_heads = 64
   445:         encoder_layers = 15
   446:         dropout = 0.1
   447:         attention_dropout = 0.1
   448:         K = 128
   449:         n_edge_type = _DICT_SIZE * _DICT_SIZE  # 961
   450: 
   451:         self.embed_dim = embed_dim
   452:         self.attention_heads = attention_heads
   453: 
   454:         # Token embedding (will be loaded from pretrained)
   455:         self.embed_tokens = nn.Embedding(
   456:             _DICT_SIZE, embed_dim, padding_idx=_PAD_IDX)
   457: 
   458:         # Edge-type dependent Gaussian distance encoding
   459:         self.gbf = GaussianLayer(K, n_edge_type)
   460:         self.gbf_proj = NonLinearHead(K, attention_heads)
   461: 
   462:         # Transformer encoder with pair tracking
   463:         self.encoder = TransformerEncoderWithPair(
   464:             encoder_layers=encoder_layers,
   465:             embed_dim=embed_dim,
   466:             ffn_embed_dim=ffn_embed_dim,
   467:             attention_heads=attention_heads,
   468:             emb_dropout=dropout,
   469:             dropout=dropout,
   470:             attention_dropout=attention_dropout,
   471:             activation_dropout=0.0,
   472:         )
   473: 
   474:         # Classification / regression head.  `pooler_dropout` may be set as a
   475:         # class attribute by the training driver to match reference per-dataset
   476:         # settings (e.g. FreeSolv/ESOL use 0.2 per Uni-Mol README).
   477:         pooler_dropout = getattr(type(self), "pooler_dropout", 0.0)
   478:         self.cls_head = ClassificationHead(
   479:             input_dim=embed_dim,
   480:             inner_dim=embed_dim,
   481:             num_classes=num_tasks,
   482:             pooler_dropout=pooler_dropout,
   483:         )
   484: 
   485:         # Initialise, then load pretrained weights
   486:         self.apply(self._init_weights)
   487:         self._load_pretrained()
   488: 
   489:     @staticmethod
   490:     def _init_weights(module):
   491:         """BERT-style weight initialisation."""
   492:         if isinstance(module, nn.Linear):
   493:             nn.init.normal_(module.weight, mean=0.0, std=0.02)
   494:             if module.bias is not None:
   495:                 nn.init.zeros_(module.bias)
   496:         elif isinstance(module, nn.Embedding):
   497:             nn.init.normal_(module.weight, mean=0.0, std=0.02)
   498:             if module.padding_idx is not None:
   499:                 nn.init.zeros_(module.weight[module.padding_idx])
   500:         elif isinstance(module, nn.LayerNorm):
   501:             nn.init.ones_(module.weight)
   502:             nn.init.zeros_(module.bias)
   503: 
   504:     def _load_pretrained(self):
   505:         """Load pretrained encoder + GBF weights from checkpoint."""
   506:         ckpt_path = "/data/unimol_weights/mol_pre_all_h_220816.pt"
   507:         if not _os.path.exists(ckpt_path):
   508:             _logger.warning(
   509:                 "Pretrained weights not found at %s — training from scratch",
   510:                 ckpt_path)
   511:             return
   512: 
   513:         ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
   514:         state = ckpt.get("model", ckpt)
   515: 
   516:         # Remap checkpoint keys from fairseq/unicore format to our flat format
   517:         remapped = {}
   518:         for key, val in state.items():
   519:             if any(s in key for s in ["classification_head", "lm_head",
   520:                                        "dist_head", "pair2coord_proj"]):
   521:                 continue
   522:             new_key = key
   523:             new_key = new_key.replace("encoder.sentence_encoder.", "encoder.")
   524:             new_key = new_key.replace("encoder.gbf", "gbf")
   525:             new_key = new_key.replace("encoder.embed_tokens", "embed_tokens")
   526:             remapped[new_key] = val
   527: 
   528:         own_state = self.state_dict()
   529:         loaded_keys, skipped_shape, skipped_missing = [], [], []
   530: 
   531:         for key, val in remapped.items():
   532:             if key in own_state:
   533:                 if own_state[key].shape == val.shape:
   534:                     own_state[key].copy_(val)
   535:                     loaded_keys.append(key)
   536:                 else:
   537:                     skipped_shape.append(f"  {key}: ckpt={list(val.shape)} model={list(own_state[key].shape)}")
   538:             else:
   539:                 skipped_missing.append(key)
   540: 
   541:         self.load_state_dict(own_state, strict=False)
   542:         print(f"[Checkpoint] Loaded {len(loaded_keys)} keys successfully")
   543:         if skipped_shape:
   544:             print(f"[Checkpoint] Shape mismatch ({len(skipped_shape)} keys):")
   545:             for s in skipped_shape:
   546:                 print(s)
   547:         if skipped_missing:
   548:             print(f"[Checkpoint] Missing in model ({len(skipped_missing)} keys):")
   549:             for k in skipped_missing[:10]:
   550:                 print(f"  {k}")
   551:             if len(skipped_missing) > 10:
   552:                 print(f"  ... and {len(skipped_missing)-10} more")
   553:         # Also show model keys NOT in checkpoint
   554:         not_loaded = [k for k in own_state if k not in set(loaded_keys)]
   555:         if not_loaded:
   556:             print(f"[Checkpoint] Model keys NOT loaded ({len(not_loaded)}):")
   557:             for k in not_loaded[:10]:
   558:                 print(f"  {k}")
   559:             if len(not_loaded) > 10:
   560:                 print(f"  ... and {len(not_loaded)-10} more")
   561: 
   562:     def forward(self, batch):
   563:         """Forward pass using dense batch format.
   564: 
   565:         Args:
   566:             batch: MolBatch with atom_features [B, N, D],
   567:                    dist_matrix [B, N, N], mask [B, N].
   568:         Returns:
   569:             predictions: [B, num_tasks]
   570:         """
   571:         B, N, _ = batch.atom_features.shape
   572:         device = batch.atom_features.device
   573: 
   574:         # Map atom features to dictionary tokens
   575:         tokens = _atoms_to_tokens(batch.atom_features, batch.mask)  # [B, N+2]
   576: 
   577:         # Extend distance matrix / mask for [CLS] and [SEP]
   578:         dist, padding_mask = _extend_dist_and_mask(
   579:             batch.dist_matrix, batch.mask)
   580:         seq_len = N + 2
   581: 
   582:         # Edge types: token_i * dict_size + token_j
   583:         edge_type = tokens.unsqueeze(1) * _DICT_SIZE + tokens.unsqueeze(2)
   584: 
   585:         # Gaussian features with edge-type-dependent scaling
   586:         gbf_feature = self.gbf(dist, edge_type)         # [B, S, S, K]
   587:         attn_bias = self.gbf_proj(gbf_feature)           # [B, S, S, H]
   588:         attn_bias = attn_bias.permute(0, 3, 1, 2).contiguous()  # [B, H, S, S]
   589:         attn_bias = attn_bias.view(-1, seq_len, seq_len)  # [B*H, S, S]
   590: 
   591:         # Token embeddings
   592:         x = self.embed_tokens(tokens)  # [B, S, D]
   593: 
   594:         # Run encoder (with pair representation tracking)
   595:         encoder_rep, pair_rep, delta_pair_repr = self.encoder(
   596:             x, attn_mask=attn_bias,
   597:             padding_mask=padding_mask if padding_mask.any() else None,
   598:         )
   599: 
   600:         # Predict via [CLS] token
   601:         return self.cls_head(encoder_rep)  # [B, num_tasks]
   602: 
   603: 
   604: # =====================================================================
   605: # EDITABLE SECTION END
   606: # =====================================================================
   607: 
   608: 
   609: # =====================================================================
```

### `gin` baseline — editable region  [READ-ONLY — reference implementation]

In `Uni-Mol/custom_molprop.py`:

```python
Lines 115–203:
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
   206: # =====================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
