# MLS-Bench: ai4bio-protein-structure-repr

# Task: Protein Structure Representation Learning

## Research Question
Design a novel geometric GNN encoder for learning protein structure representations from 3D alpha-carbon coordinates. The encoder must capture both local geometric patterns (bond angles, dihedral angles) and global structural motifs to produce informative per-residue and per-protein embeddings.

## Background
Protein function is determined by 3D structure. Geometric GNNs that operate on protein structure graphs (nodes = residues at alpha-carbon positions, edges = spatial/sequential neighbors) have emerged as powerful tools for learning protein representations. Key challenges include:
- **Geometric awareness**: The encoder should leverage 3D spatial information (distances, angles, orientations) beyond simple adjacency.
- **Equivariance/invariance**: Representations should be invariant to rigid body transformations (rotations, translations) of the protein.
- **Multi-scale structure**: Proteins exhibit hierarchical structure (secondary structure elements, domains, global fold) that the encoder should capture.

Existing approaches include:
- **SchNet** (Schütt et al., "SchNet: A continuous-filter convolutional neural network for modeling quantum interactions", NeurIPS 2017; arXiv:1706.08566). Continuous-filter convolutions with Gaussian radial basis function distance expansion. Invariant by design.
- **EGNN** (Satorras, Hoogeboom, Welling, "E(n) Equivariant Graph Neural Networks", ICML 2021; arXiv:2102.09844). E(n)-equivariant message passing that jointly updates node features and coordinates. Code: https://github.com/vgsatorras/egnn.
- **GearNet** (Zhang et al., "Protein Representation Learning by Geometric Structure Pretraining", ICLR 2023; arXiv:2203.06125). Geometry-Aware Relational Graph Neural Network with multiple edge types (sequential, spatial, k-nearest) and relational convolutions, optionally enhanced by edge message passing. Code: https://github.com/DeepGraphLearning/GearNet.

## What to Implement
Implement the `ProteinEncoder` class and any helper modules in `custom_protein_encoder.py`. You must implement:
1. `__init__(self, ...)`: Set up the encoder architecture. The input node features have dimension `SCALAR_NODE_DIM=28` (20-dim amino acid one-hot + 2-dim positional encoding + 6-dim pseudo-dihedral features).
2. `forward(self, pos, node_feat, batch) -> (node_emb, graph_emb)`: Encode the protein graph.
   - `pos`: (N, 3) alpha-carbon coordinates
   - `node_feat`: (N, 28) scalar node features (computed by the fixed `compute_node_features` function)
   - `batch`: (N,) batch assignment indices
   - Returns: `node_emb` (N, out_dim) per-node embeddings, `graph_emb` (B, out_dim) per-graph embeddings

## Fixed Pipeline
Node-feature computation, dataset construction, batching, classifier heads, training/evaluation loops, and metric computation are all fixed. The contribution is the encoder architecture only.

## Editable Region
The section between `EDITABLE SECTION START` and `EDITABLE SECTION END` markers in `custom_protein_encoder.py` is editable. You may define any helper classes, layers, or functions within this region. The region must contain a `ProteinEncoder` class with the interface described above.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/ProteinWorkshop/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `ProteinWorkshop/custom_protein_encoder.py`
- editable lines **125–252**
- editable lines **706–708**




## Readable Context


### `ProteinWorkshop/custom_protein_encoder.py`  [EDITABLE — lines 125–252, lines 706–708 only]

```python
     1: """
     2: Protein Structure Representation Learning — Self-contained template.
     3: Trains a geometric GNN encoder for protein structure and evaluates on
     4: downstream classification tasks (EC number, GO-BP, Fold classification).
     5: 
     6: Structure:
     7:   Lines 1-124:    FIXED — Imports, constants, data loading utilities
     8:   Lines 125-252:  EDITABLE — ProteinEncoder class + helper modules
     9:   Lines 253+:     FIXED — Dataset, decoder head, training loop, evaluation
    10: """
    11: import os
    12: import sys
    13: import math
    14: import json
    15: import argparse
    16: import warnings
    17: import numpy as np
    18: from dataclasses import dataclass
    19: from typing import Optional, Dict, List, Tuple, Union
    20: from pathlib import Path
    21: 
    22: import torch
    23: import torch.nn as nn
    24: import torch.nn.functional as F
    25: from torch.utils.data import Dataset, DataLoader
    26: from torch_geometric.data import Batch, Data
    27: from torch_geometric.nn import global_mean_pool, global_add_pool, radius_graph, knn_graph
    28: from torch_geometric.utils import add_self_loops
    29: 
    30: from torch_scatter import scatter_mean, scatter_add
    31: 
    32: warnings.filterwarnings("ignore", category=UserWarning)
    33: 
    34: # =====================================================================
    35: # Constants and Utilities
    36: # =====================================================================
    37: 
    38: NUM_AMINO_ACIDS = 20
    39: AMINO_ACIDS = [
    40:     'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
    41:     'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
    42:     'THR', 'TRP', 'TYR', 'VAL'
    43: ]
    44: AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    45: 
    46: # Node feature dimension: one-hot amino acid (20) + sin/cos positional (2) + dihedrals (6)
    47: SCALAR_NODE_DIM = 28
    48: # Edge feature dimension: distance (1) + direction unit vector (3)
    49: EDGE_FEAT_DIM = 4
    50: 
    51: 
    52: def compute_node_features(pos, aa_idx, batch_idx):
    53:     """Compute scalar node features: one-hot AA + positional encoding + pseudo-dihedrals.
    54: 
    55:     Args:
    56:         pos: (N, 3) alpha-carbon positions
    57:         aa_idx: (N,) amino acid indices [0, 19]
    58:         batch_idx: (N,) batch assignment
    59: 
    60:     Returns:
    61:         node_feat: (N, SCALAR_NODE_DIM)
    62:     """
    63:     N = pos.size(0)
    64:     device = pos.device
    65: 
    66:     # One-hot amino acid
    67:     aa_onehot = F.one_hot(aa_idx.clamp(0, NUM_AMINO_ACIDS - 1), NUM_AMINO_ACIDS).float()  # (N, 20)
    68: 
    69:     # Sequence positional encoding (sin/cos within each graph) — vectorized
    70:     counts = scatter_add(torch.ones(N, device=device), batch_idx, dim=0)
    71:     # Compute per-node offset within its graph using cumsum trick
    72:     ones = torch.ones(N, device=device)
    73:     # For each node, its local index = global_index - start_of_its_graph
    74:     cumcounts = torch.zeros(int(batch_idx.max().item()) + 2, dtype=torch.long, device=device)
    75:     cumcounts[1:len(counts)+1] = counts.long().cumsum(0)
    76:     offsets = torch.arange(N, device=device).float() - cumcounts[batch_idx].float()
    77:     max_len = counts[batch_idx].float().clamp(min=1)
    78:     pos_enc = offsets / max_len
    79:     sin_enc = torch.sin(pos_enc * math.pi).unsqueeze(-1)
    80:     cos_enc = torch.cos(pos_enc * math.pi).unsqueeze(-1)
    81: 
    82:     # Pseudo-dihedral angles — vectorized (no Python loops)
    83:     dihedrals = torch.zeros(N, 6, device=device)
    84:     # Displacement vectors between consecutive nodes
    85:     d = pos[1:] - pos[:-1]  # (N-1, 3)
    86:     # Mask: consecutive pairs must be in the same graph
    87:     same_graph = (batch_idx[1:] == batch_idx[:-1])  # (N-1,)
    88: 
    89:     # For dihedral at position j, we need d[j-1], d[j], d[j+1]
    90:     # Valid positions: j in [1, N-3] where j-1, j, j+1 are all within same graph
    91:     if N >= 4:
    92:         v1 = d[:-2]   # d[j-1] for j=1..N-3
    93:         v2 = d[1:-1]  # d[j]   for j=1..N-3
    94:         v3 = d[2:]    # d[j+1] for j=1..N-3
    95:         # Valid mask: all three displacement vectors must be within same graph
    96:         valid = same_graph[:-2] & same_graph[1:-1] & same_graph[2:]  # (N-3,)
    97: 
    98:         # Compute cross products
    99:         n1 = torch.linalg.cross(v1, v2)  # (N-3, 3)
   100:         n2 = torch.linalg.cross(v2, v3)  # (N-3, 3)
   101:         n1_norm = n1 / (n1.norm(dim=-1, keepdim=True) + 1e-8)
   102:         n2_norm = n2 / (n2.norm(dim=-1, keepdim=True) + 1e-8)
   103: 
   104:         cos_angle = (n1_norm * n2_norm).sum(dim=-1).clamp(-1, 1)
   105:         sin_angle = torch.linalg.cross(n1_norm, n2_norm).norm(dim=-1).clamp(-1, 1)
   106:         v1_norm = v1.norm(dim=-1)
   107:         v2_norm = v2.norm(dim=-1)
   108:         v1v2_cos = (v1 * v2).sum(dim=-1) / (v1_norm * v2_norm + 1e-8)
   109:         v2v3_cos = (v2 * v3).sum(dim=-1) / (v2_norm * v3.norm(dim=-1) + 1e-8)
   110: 
   111:         # Target indices in the original array: positions 1 to N-3 (offset by 1)
   112:         target_idx = torch.arange(1, N - 2, device=device)
   113:         valid_idx = target_idx[valid]
   114: 
   115:         dihedrals[valid_idx, 0] = cos_angle[valid]
   116:         dihedrals[valid_idx, 1] = sin_angle[valid]
   117:         dihedrals[valid_idx, 2] = v1_norm[valid]
   118:         dihedrals[valid_idx, 3] = v2_norm[valid]
   119:         dihedrals[valid_idx, 4] = v1v2_cos[valid]
   120:         dihedrals[valid_idx, 5] = v2v3_cos[valid]
   121: 
   122:     return torch.cat([aa_onehot, sin_enc, cos_enc, dihedrals], dim=-1)
   123: 
   124: # =====================================================================
   125: # EDITABLE SECTION START — ProteinEncoder + helper modules
   126: # =====================================================================
   127: 
   128: class MessagePassingLayer(nn.Module):
   129:     """Basic invariant message passing layer for protein graphs."""
   130: 
   131:     def __init__(self, hidden_dim, edge_dim=EDGE_FEAT_DIM):
   132:         super().__init__()
   133:         self.hidden_dim = hidden_dim
   134:         self.edge_mlp = nn.Sequential(
   135:             nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
   136:             nn.SiLU(),
   137:             nn.Linear(hidden_dim, hidden_dim),
   138:         )
   139:         self.node_mlp = nn.Sequential(
   140:             nn.Linear(2 * hidden_dim, hidden_dim),
   141:             nn.SiLU(),
   142:             nn.Linear(hidden_dim, hidden_dim),
   143:         )
   144:         self.norm = nn.LayerNorm(hidden_dim)
   145: 
   146:     def forward(self, h, edge_index, edge_attr):
   147:         """
   148:         Args:
   149:             h: (N, hidden_dim) node features
   150:             edge_index: (2, E) edge indices
   151:             edge_attr: (E, edge_dim) edge features
   152:         Returns:
   153:             h: (N, hidden_dim) updated node features
   154:         """
   155:         src, dst = edge_index
   156:         edge_input = torch.cat([h[src], h[dst], edge_attr], dim=-1)
   157:         msg = self.edge_mlp(edge_input)
   158:         # Aggregate messages
   159:         agg = scatter_mean(msg, dst, dim=0, dim_size=h.size(0))
   160:         h_new = self.node_mlp(torch.cat([h, agg], dim=-1))
   161:         h = self.norm(h + h_new)
   162:         return h
   163: 
   164: 
   165: class ProteinEncoder(nn.Module):
   166:     """Geometric GNN encoder for protein structures.
   167: 
   168:     Takes alpha-carbon graphs with node features (amino acid type, positional
   169:     encoding, pseudo-dihedrals) and edge features (distance, direction) and
   170:     produces per-node and per-graph embeddings.
   171: 
   172:     This is the starter implementation using basic invariant message passing.
   173:     The agent should replace this with a more expressive geometric GNN design
   174:     (e.g., equivariant message passing, multi-scale, attention, etc.).
   175: 
   176:     Args:
   177:         input_dim: Dimension of input node features (default: SCALAR_NODE_DIM=28)
   178:         hidden_dim: Hidden dimension (default: 256)
   179:         out_dim: Output embedding dimension (default: 128)
   180:         num_layers: Number of message passing layers (default: 6)
   181:         dropout: Dropout rate (default: 0.1)
   182:         cutoff: Distance cutoff for edge construction in Angstroms (default: 10.0)
   183:         max_neighbors: Max neighbors in kNN graph (default: 16)
   184:     """
   185: 
   186:     def __init__(
   187:         self,
   188:         input_dim: int = SCALAR_NODE_DIM,
   189:         hidden_dim: int = 256,
   190:         out_dim: int = 128,
   191:         num_layers: int = 6,
   192:         dropout: float = 0.1,
   193:         cutoff: float = 10.0,
   194:         max_neighbors: int = 16,
   195:     ):
   196:         super().__init__()
   197:         self.hidden_dim = hidden_dim
   198:         self.out_dim = out_dim
   199:         self.num_layers = num_layers
   200:         self.cutoff = cutoff
   201:         self.max_neighbors = max_neighbors
   202: 
   203:         self.node_embed = nn.Sequential(
   204:             nn.Linear(input_dim, hidden_dim),
   205:             nn.SiLU(),
   206:             nn.Linear(hidden_dim, hidden_dim),
   207:         )
   208: 
   209:         self.layers = nn.ModuleList([
   210:             MessagePassingLayer(hidden_dim, EDGE_FEAT_DIM)
   211:             for _ in range(num_layers)
   212:         ])
   213: 
   214:         self.dropout = nn.Dropout(dropout)
   215:         self.out_proj = nn.Linear(hidden_dim, out_dim)
   216: 
   217:     def _build_edges(self, pos, batch):
   218:         """Build kNN graph edges with distance and direction features."""
   219:         edge_index = knn_graph(pos, k=self.max_neighbors, batch=batch, loop=False)
   220:         src, dst = edge_index
   221:         diff = pos[dst] - pos[src]
   222:         dist = diff.norm(dim=-1, keepdim=True)
   223:         direction = diff / (dist + 1e-8)
   224:         edge_attr = torch.cat([dist, direction], dim=-1)  # (E, 4)
   225:         return edge_index, edge_attr
   226: 
   227:     def forward(self, pos, node_feat, batch):
   228:         """
   229:         Args:
   230:             pos: (N, 3) alpha-carbon coordinates
   231:             node_feat: (N, input_dim) node scalar features
   232:             batch: (N,) batch index
   233: 
   234:         Returns:
   235:             node_emb: (N, out_dim) per-node embeddings
   236:             graph_emb: (B, out_dim) per-graph embeddings (mean pool)
   237:         """
   238:         edge_index, edge_attr = self._build_edges(pos, batch)
   239: 
   240:         h = self.node_embed(node_feat)
   241: 
   242:         for layer in self.layers:
   243:             h = layer(h, edge_index, edge_attr)
   244:             h = self.dropout(h)
   245: 
   246:         node_emb = self.out_proj(h)
   247:         graph_emb = global_mean_pool(node_emb, batch)
   248: 
   249:         return node_emb, graph_emb
   250: 
   251: # =====================================================================
   252: # EDITABLE SECTION END
   253: # =====================================================================
   254: 
   255: 
   256: # =====================================================================
   257: # FIXED — Dataset, decoder head, training loop, evaluation
   258: # =====================================================================
   259: 
   260: class ProteinGraphDataset(Dataset):
   261:     """Dataset that loads pre-processed protein graph data."""
   262: 
   263:     def __init__(self, data_list):
   264:         self.data_list = data_list
   265: 
   266:     def __len__(self):
   267:         return len(self.data_list)
   268: 
   269:     def __getitem__(self, idx):
   270:         return self.data_list[idx]
   271: 
   272: 
   273: def collate_protein_graphs(batch_list):
   274:     """Collate protein graph Data objects into a Batch."""
   275:     return Batch.from_data_list(batch_list)
   276: 
   277: 
   278: class ClassificationHead(nn.Module):
   279:     """MLP classification head on top of graph embeddings."""
   280: 
   281:     def __init__(self, in_dim, num_classes, hidden_dim=256, task_type='multiclass'):
   282:         super().__init__()
   283:         self.task_type = task_type
   284:         self.head = nn.Sequential(
   285:             nn.Linear(in_dim, hidden_dim),
   286:             nn.SiLU(),
   287:             nn.Dropout(0.1),
   288:             nn.Linear(hidden_dim, hidden_dim),
   289:             nn.SiLU(),
   290:             nn.Dropout(0.1),
   291:             nn.Linear(hidden_dim, num_classes),
   292:         )
   293: 
   294:     def forward(self, graph_emb):
   295:         return self.head(graph_emb)
   296: 
   297: 
   298: def load_dataset_splits(task_name, data_dir):
   299:     """Load pre-processed protein graph data for a given task.
   300: 
   301:     The data is expected to be preprocessed during container build into
   302:     /data/ProteinWorkshop/<task_subdir>/processed/ as .pt files.
   303: 
   304:     Each Data object has:
   305:         - pos: (L, 3) alpha-carbon positions
   306:         - aa_idx: (L,) amino acid indices
   307:         - y: label (int for multiclass, binary vector for multilabel)
   308:         - num_nodes: number of residues
   309:     """
   310:     task_configs = {
   311:         'ec_reaction': {
   312:             'subdir': 'ECReaction',
   313:             'num_classes': 384,
   314:             'task_type': 'multiclass',
   315:         },
   316:         'go_bp': {
   317:             'subdir': 'GeneOntology',
   318:             'num_classes': 1943,
   319:             'task_type': 'multilabel',
   320:         },
   321:         'fold_fold': {
   322:             'subdir': 'FoldClassification',
   323:             'num_classes': 1195,
   324:             'task_type': 'multiclass',
   325:         },
   326:     }
   327: 
   328:     config = task_configs[task_name]
   329:     base_path = Path(data_dir) / config['subdir'] / 'processed'
   330: 
   331:     splits = {}
   332:     for split_name in ['train', 'val', 'test']:
   333:         fpath = base_path / f'{split_name}.pt'
   334:         if fpath.exists():
   335:             splits[split_name] = torch.load(fpath, weights_only=False)
   336:         else:
   337:             print(f"Warning: {fpath} not found, skipping {split_name} split")
   338:             splits[split_name] = []
   339: 
   340:     return splits, config['num_classes'], config['task_type']
   341: 
   342: 
   343: def preprocess_and_cache(task_name, data_dir):
   344:     """Use ProteinWorkshop datamodules to load and cache processed data.
   345: 
   346:     Converts raw PDB/MMTF files into PyG Data objects with:
   347:         - pos: alpha-carbon positions
   348:         - aa_idx: amino acid index per residue
   349:         - y: task label
   350:     """
   351:     processed_dir_map = {
   352:         'ec_reaction': 'ECReaction',
   353:         'go_bp': 'GeneOntology',
   354:         'fold_fold': 'FoldClassification',
   355:     }
   356: 
   357:     subdir = processed_dir_map[task_name]
   358:     processed_path = Path(data_dir) / subdir / 'processed'
   359: 
   360:     if processed_path.exists() and (processed_path / 'train.pt').exists():
   361:         print(f"Processed data found at {processed_path}, skipping preprocessing.")
   362:         return
   363: 
   364:     processed_path.mkdir(parents=True, exist_ok=True)
   365:     print(f"Preprocessing {task_name} from ProteinWorkshop datamodules...")
   366: 
   367:     # Import ProteinWorkshop components
   368:     sys.path.insert(0, '/workspace/ProteinWorkshop')
   369:     os.environ['PROTEIN_WORKSHOP_DATA_DIR'] = data_dir
   370: 
   371:     if task_name == 'ec_reaction':
   372:         from proteinworkshop.datasets.ec_reaction import EnzymeCommissionReactionDataset
   373:         dm = EnzymeCommissionReactionDataset(
   374:             path=str(Path(data_dir) / 'ECReaction'),
   375:             pdb_dir=str(Path(data_dir) / 'pdb'),
   376:             format='mmtf',
   377:             batch_size=1,
   378:             num_workers=0,
   379:             pin_memory=False,
   380:             dataset_fraction=1.0,
   381:             shuffle_labels=False,
   382:         )
   383:     elif task_name == 'go_bp':
   384:         from proteinworkshop.datasets.go import GeneOntologyDataset
   385:         dm = GeneOntologyDataset(
   386:             path=str(Path(data_dir) / 'GeneOntology'),
   387:             pdb_dir=str(Path(data_dir) / 'pdb'),
   388:             format='mmtf',
   389:             batch_size=1,
   390:             num_workers=0,
   391:             pin_memory=False,
   392:             dataset_fraction=1.0,
   393:             shuffle_labels=False,
   394:             split='BP',
   395:         )
   396:     elif task_name == 'fold_fold':
   397:         from proteinworkshop.datasets.fold_classification import FoldClassificationDataModule
   398:         dm = FoldClassificationDataModule(
   399:             path=str(Path(data_dir) / 'FoldClassification'),
   400:             batch_size=1,
   401:             num_workers=0,
   402:             pin_memory=False,
   403:             dataset_fraction=1.0,
   404:             shuffle_labels=False,
   405:             split='fold',
   406:         )
   407:     else:
   408:         raise ValueError(f"Unknown task: {task_name}")
   409: 
   410:     dm.setup(stage='fit')
   411:     dm.setup(stage='test')
   412: 
   413:     # Convert to simple Data objects
   414:     three_to_idx = {}
   415:     for i, aa in enumerate(AMINO_ACIDS):
   416:         three_to_idx[aa] = i
   417: 
   418:     def convert_batch(dataset_or_loader, split_name):
   419:         data_list = []
   420:         loader = DataLoader(dataset_or_loader, batch_size=1, shuffle=False, num_workers=0)
   421:         skipped = 0
   422:         for batch in loader:
   423:             try:
   424:                 # Extract alpha-carbon positions
   425:                 if hasattr(batch, 'coords'):
   426:                     # coords shape: (1, L, atoms_per_residue, 3) — take CA (index 1)
   427:                     if batch.coords.dim() == 4:
   428:                         pos = batch.coords[0, :, 1, :]  # CA atom
   429:                     elif batch.coords.dim() == 3:
   430:                         pos = batch.coords[0]
   431:                     else:
   432:                         pos = batch.pos if hasattr(batch, 'pos') else None
   433:                 elif hasattr(batch, 'pos'):
   434:                     pos = batch.pos
   435:                 else:
   436:                     skipped += 1
   437:                     continue
   438: 
   439:                 if pos is None or pos.size(0) < 4:
   440:                     skipped += 1
   441:                     continue
   442: 
   443:                 # Amino acid indices
   444:                 if hasattr(batch, 'residue_type'):
   445:                     aa_idx = batch.residue_type.long()
   446:                     if aa_idx.dim() > 1:
   447:                         aa_idx = aa_idx[0]
   448:                 elif hasattr(batch, 'x') and batch.x is not None:
   449:                     # One-hot encoded residue features
   450:                     if batch.x.dim() > 1 and batch.x.size(-1) >= 20:
   451:                         aa_idx = batch.x[:, :20].argmax(dim=-1)
   452:                     else:
   453:                         aa_idx = torch.zeros(pos.size(0), dtype=torch.long)
   454:                 else:
   455:                     aa_idx = torch.zeros(pos.size(0), dtype=torch.long)
   456: 
   457:                 # Ensure correct shapes
   458:                 if pos.dim() != 2 or pos.size(-1) != 3:
   459:                     skipped += 1
   460:                     continue
   461:                 L = pos.size(0)
   462:                 if aa_idx.size(0) != L:
   463:                     aa_idx = aa_idx[:L] if aa_idx.size(0) > L else F.pad(aa_idx, (0, L - aa_idx.size(0)))
   464: 
   465:                 # Labels
   466:                 if hasattr(batch, 'graph_y'):
   467:                     y = batch.graph_y
   468:                 elif hasattr(batch, 'y'):
   469:                     y = batch.y
   470:                 else:
   471:                     skipped += 1
   472:                     continue
   473: 
   474:                 if y.dim() > 1:
   475:                     y = y.squeeze(0)
   476: 
   477:                 data = Data(
   478:                     pos=pos.float(),
   479:                     aa_idx=aa_idx.clamp(0, NUM_AMINO_ACIDS - 1),
   480:                     y=y,
   481:                     num_nodes=L,
   482:                 )
   483:                 data_list.append(data)
   484:             except Exception as e:
   485:                 skipped += 1
   486:                 continue
   487: 
   488:         print(f"  {split_name}: {len(data_list)} proteins processed, {skipped} skipped")
   489:         return data_list
   490: 
   491:     if hasattr(dm, 'train_dataset'):
   492:         train_data = convert_batch(dm.train_dataset(), 'train')
   493:     else:
   494:         train_data = convert_batch(dm.train_dataloader().dataset, 'train')
   495: 
   496:     if hasattr(dm, 'val_dataset'):
   497:         val_data = convert_batch(dm.val_dataset(), 'val')
   498:     else:
   499:         val_data = convert_batch(dm.val_dataloader().dataset, 'val')
   500: 

[truncated: showing at most 500 lines / 60000 bytes from ProteinWorkshop/custom_protein_encoder.py]
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `schnet` baseline — editable region  [READ-ONLY — reference implementation]

In `ProteinWorkshop/custom_protein_encoder.py`:

```python
Lines 125–229:
   122:     return torch.cat([aa_onehot, sin_enc, cos_enc, dihedrals], dim=-1)
   123: 
   124: # =====================================================================
   125: # =====================================================================
   126: # EDITABLE SECTION START — SchNet encoder (ported from ProteinWorkshop)
   127: # =====================================================================
   128: 
   129: # Import PyG SchNet components used by the reference implementation
   130: from torch_geometric.nn.models.schnet import InteractionBlock, GaussianSmearing, ShiftedSoftplus
   131: 
   132: class ProteinEncoder(nn.Module):
   133:     """SchNet-based protein structure encoder.
   134: 
   135:     Ported directly from ProteinWorkshop SchNetModel.
   136:     Uses continuous-filter convolutions with Gaussian RBF distance expansion.
   137:     Invariant to rotations and translations by design.
   138: 
   139:     Reference hyperparameters (from proteinworkshop/config/encoder/schnet.yaml):
   140:       hidden_channels=512, num_filters=128, num_gaussians=50, cutoff=10.0,
   141:       max_num_neighbors=32, readout="add"
   142:     """
   143:     def __init__(
   144:         self,
   145:         input_dim: int = SCALAR_NODE_DIM,
   146:         hidden_dim: int = 256,
   147:         out_dim: int = 128,
   148:         num_layers: int = 6,
   149:         dropout: float = 0.1,
   150:         cutoff: float = 10.0,
   151:         max_neighbors: int = 16,
   152:     ):
   153:         super().__init__()
   154:         # Override with ProteinWorkshop reference hyperparameters
   155:         hidden_channels = 512
   156:         num_filters = 128
   157:         num_gaussians = 50
   158:         self.cutoff = cutoff
   159:         max_num_neighbors = 32
   160:         readout = "add"
   161: 
   162:         self.hidden_channels = hidden_channels
   163:         self.out_dim = out_dim
   164:         self.max_num_neighbors = max_num_neighbors
   165:         self.readout = readout
   166: 
   167:         # Overwrite embedding to accept arbitrary input features (matching reference LazyLinear)
   168:         self.embedding = nn.Linear(input_dim, hidden_channels)
   169: 
   170:         # Gaussian RBF distance expansion (from PyG SchNet)
   171:         self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
   172: 
   173:         # Stack of InteractionBlocks (from PyG SchNet)
   174:         self.interactions = nn.ModuleList()
   175:         for _ in range(num_layers):
   176:             block = InteractionBlock(
   177:                 hidden_channels, num_gaussians, num_filters, cutoff
   178:             )
   179:             self.interactions.append(block)
   180: 
   181:         # Output MLP: lin1 -> act -> lin2 (matching reference)
   182:         self.lin1 = nn.Linear(hidden_channels, hidden_channels)
   183:         self.act = ShiftedSoftplus()
   184:         self.lin2 = nn.Linear(hidden_channels, out_dim)
   185: 
   186:     def _build_edges(self, pos, batch):
   187:         """Build kNN graph and compute edge weights + RBF features."""
   188:         edge_index = knn_graph(
   189:             pos, k=self.max_num_neighbors, batch=batch, loop=False
   190:         )
   191:         u, v = edge_index
   192:         edge_weight = (pos[u] - pos[v]).norm(dim=-1)
   193:         edge_attr = self.distance_expansion(edge_weight)
   194:         return edge_index, edge_weight, edge_attr
   195: 
   196:     def forward(self, pos, node_feat, batch):
   197:         """Forward pass matching ProteinWorkshop SchNetModel.
   198: 
   199:         Args:
   200:             pos: (N, 3) alpha-carbon coordinates
   201:             node_feat: (N, input_dim) node scalar features
   202:             batch: (N,) batch index
   203: 
   204:         Returns:
   205:             node_emb: (N, out_dim) per-node embeddings
   206:             graph_emb: (B, out_dim) per-graph embeddings
   207:         """
   208:         edge_index, edge_weight, edge_attr = self._build_edges(pos, batch)
   209: 
   210:         # Project input features to hidden dimension
   211:         h = self.embedding(node_feat)
   212: 
   213:         # Message passing with residual connections (matching reference exactly)
   214:         for interaction in self.interactions:
   215:             h = h + interaction(h, edge_index, edge_weight, edge_attr)
   216: 
   217:         # Output projection: lin1 -> act -> lin2 (matching reference)
   218:         h = self.lin1(h)
   219:         h = self.act(h)
   220:         node_emb = self.lin2(h)
   221: 
   222:         # Graph-level readout via scatter (matching reference readout="add")
   223:         graph_emb = scatter_add(node_emb, batch, dim=0)
   224: 
   225:         return node_emb, graph_emb
   226: 
   227: # =====================================================================
   228: # EDITABLE SECTION END
   229: # =====================================================================
   230: # =====================================================================
   231: 
   232: 

Lines 683–685:
   680:                         help='Number of GNN layers')
   681:     args = parser.parse_args()
   682: 
   683:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   684:     # Allowed keys: learning_rate, epochs.
   685:     CONFIG_OVERRIDES = {}
   686: 
   687:     for _k, _v in CONFIG_OVERRIDES.items():
   688:         if _k == 'learning_rate': args.lr = _v
```

### `egnn` baseline — editable region  [READ-ONLY — reference implementation]

In `ProteinWorkshop/custom_protein_encoder.py`:

```python
Lines 125–322:
   122:     return torch.cat([aa_onehot, sin_enc, cos_enc, dihedrals], dim=-1)
   123: 
   124: # =====================================================================
   125: # =====================================================================
   126: # EDITABLE SECTION START — EGNN encoder (ported from ProteinWorkshop)
   127: # =====================================================================
   128: 
   129: import torch_scatter
   130: from torch.nn import Linear, Dropout, Sequential
   131: from torch_geometric.nn import MessagePassing
   132: 
   133: class EGNNLayer(MessagePassing):
   134:     """E(n) Equivariant GNN Layer.
   135: 
   136:     Ported directly from ProteinWorkshop:
   137:       proteinworkshop/models/graph_encoders/layers/egnn.py
   138: 
   139:     Paper: E(n) Equivariant Graph Neural Networks, Satorras et al. (ICML 2021)
   140:     """
   141:     def __init__(self, emb_dim, activation='relu', norm='batch', aggr='sum', dropout=0.1):
   142:         super().__init__(aggr=aggr)
   143: 
   144:         self.emb_dim = emb_dim
   145: 
   146:         # Normalization layer (matching reference)
   147:         norm_cls = {
   148:             'layer': nn.LayerNorm,
   149:             'batch': nn.BatchNorm1d,
   150:         }[norm]
   151: 
   152:         # Helper to create fresh activation instances
   153:         def _make_act():
   154:             if activation == 'relu':
   155:                 return nn.ReLU()
   156:             elif activation in ('silu', 'swish'):
   157:                 return nn.SiLU()
   158:             elif activation == 'elu':
   159:                 return nn.ELU()
   160:             return nn.ReLU()
   161: 
   162:         # MLP psi_h for computing messages m_ij (matching reference exactly)
   163:         self.mlp_msg = Sequential(
   164:             Linear(2 * emb_dim + 1, emb_dim),
   165:             norm_cls(emb_dim),
   166:             _make_act(),
   167:             Dropout(dropout),
   168:             Linear(emb_dim, emb_dim),
   169:             norm_cls(emb_dim),
   170:             _make_act(),
   171:             Dropout(dropout),
   172:         )
   173:         # MLP psi_x for computing coordinate displacement weights
   174:         self.mlp_pos = Sequential(
   175:             Linear(emb_dim, emb_dim),
   176:             norm_cls(emb_dim),
   177:             _make_act(),
   178:             Dropout(dropout),
   179:             Linear(emb_dim, 1),
   180:         )
   181:         # MLP phi for computing updated node features
   182:         self.mlp_upd = Sequential(
   183:             Linear(2 * emb_dim, emb_dim),
   184:             norm_cls(emb_dim),
   185:             _make_act(),
   186:             Dropout(dropout),
   187:             Linear(emb_dim, emb_dim),
   188:             norm_cls(emb_dim),
   189:             _make_act(),
   190:             Dropout(dropout),
   191:         )
   192: 
   193:     def forward(self, h, pos, edge_index):
   194:         """
   195:         Args:
   196:             h: (n, d) - initial node features
   197:             pos: (n, 3) - initial node coordinates
   198:             edge_index: (2, e) - edge indices
   199:         Returns:
   200:             msg_aggr: (n, d) - updated node features delta
   201:             pos_aggr: (n, 3) - coordinate displacement
   202:         """
   203:         msg_aggr, pos_aggr = self.propagate(edge_index, h=h, pos=pos)
   204:         msg_aggr = self.mlp_upd(torch.cat([h, msg_aggr], dim=-1))
   205:         return msg_aggr, pos_aggr
   206: 
   207:     def message(self, h_i, h_j, pos_i, pos_j):
   208:         """Compute messages (matching reference exactly)."""
   209:         pos_diff = pos_i - pos_j
   210:         dists = torch.norm(pos_diff, dim=-1, keepdim=True)
   211:         msg = torch.cat([h_i, h_j, dists], dim=-1)
   212:         msg = self.mlp_msg(msg)
   213:         # Scale displacement vector by learned weight
   214:         pos_diff = pos_diff / (dists + 1) * self.mlp_pos(msg)
   215:         return msg, pos_diff
   216: 
   217:     def aggregate(self, inputs, index):
   218:         """Aggregate messages and position displacements separately (matching reference)."""
   219:         msgs, pos_diffs = inputs
   220:         # Aggregate messages using configured aggr (sum in reference config)
   221:         msg_aggr = torch_scatter.scatter(
   222:             msgs, index, dim=self.node_dim, reduce=self.aggr
   223:         )
   224:         # Aggregate displacement vectors always with mean (matching reference)
   225:         pos_aggr = torch_scatter.scatter(
   226:             pos_diffs, index, dim=self.node_dim, reduce="mean"
   227:         )
   228:         return msg_aggr, pos_aggr
   229: 
   230:     def __repr__(self):
   231:         return f"{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})"
   232: 
   233: 
   234: class ProteinEncoder(nn.Module):
   235:     """EGNN-based protein structure encoder.
   236: 
   237:     Ported directly from ProteinWorkshop EGNNModel.
   238:     E(n)-equivariant: jointly updates node features and coordinates.
   239:     Uses residual connections on both features and coordinates.
   240: 
   241:     Reference hyperparameters (from proteinworkshop/config/encoder/egnn.yaml):
   242:       num_layers=6, emb_dim=512, activation=relu, norm=batch, aggr=sum,
   243:       pool=mean, residual=True, dropout=0.1
   244:     """
   245:     def __init__(
   246:         self,
   247:         input_dim: int = SCALAR_NODE_DIM,
   248:         hidden_dim: int = 256,
   249:         out_dim: int = 128,
   250:         num_layers: int = 6,
   251:         dropout: float = 0.1,
   252:         cutoff: float = 10.0,
   253:         max_neighbors: int = 16,
   254:     ):
   255:         super().__init__()
   256:         # Override with ProteinWorkshop reference hyperparameters
   257:         emb_dim = 512
   258:         activation = 'relu'
   259:         norm = 'batch'
   260:         aggr = 'sum'
   261:         residual = True
   262: 
   263:         self.emb_dim = emb_dim
   264:         self.out_dim = out_dim
   265:         self.cutoff = cutoff
   266:         self.max_neighbors = max_neighbors
   267:         self.residual = residual
   268: 
   269:         # Embedding lookup for initial node features (matching reference LazyLinear)
   270:         self.emb_in = nn.Linear(input_dim, emb_dim)
   271: 
   272:         # Stack of EGNN layers (matching reference)
   273:         self.convs = nn.ModuleList()
   274:         for _ in range(num_layers):
   275:             self.convs.append(EGNNLayer(emb_dim, activation, norm, aggr, dropout))
   276: 
   277:         # Global pooling/readout: mean (matching reference config)
   278:         self.pool = global_mean_pool
   279: 
   280:         # Output projection to match expected out_dim
   281:         self.out_proj = nn.Linear(emb_dim, out_dim)
   282: 
   283:     def _build_edges(self, pos, batch):
   284:         """Build kNN graph for message passing."""
   285:         edge_index = knn_graph(pos, k=self.max_neighbors, batch=batch, loop=False)
   286:         return edge_index
   287: 
   288:     def forward(self, pos, node_feat, batch):
   289:         """Forward pass matching ProteinWorkshop EGNNModel.
   290: 
   291:         Args:
   292:             pos: (N, 3) alpha-carbon coordinates
   293:             node_feat: (N, input_dim) node scalar features
   294:             batch: (N,) batch index
   295: 
   296:         Returns:
   297:             node_emb: (N, out_dim) per-node embeddings
   298:             graph_emb: (B, out_dim) per-graph embeddings
   299:         """
   300:         edge_index = self._build_edges(pos, batch)
   301: 
   302:         h = self.emb_in(node_feat)  # (n, input_dim) -> (n, emb_dim)
   303: 
   304:         for conv in self.convs:
   305:             # Message passing layer
   306:             h_update, pos_update = conv(h, pos, edge_index)
   307: 
   308:             # Update node features with residual (matching reference)
   309:             h = h + h_update if self.residual else h_update
   310: 
   311:             # Update node coordinates with residual (matching reference)
   312:             pos = pos + pos_update if self.residual else pos_update
   313: 
   314:         # Project to output dimension
   315:         node_emb = self.out_proj(h)
   316:         graph_emb = self.pool(node_emb, batch)
   317: 
   318:         return node_emb, graph_emb
   319: 
   320: # =====================================================================
   321: # EDITABLE SECTION END
   322: # =====================================================================
   323: # =====================================================================
   324: 
   325: 

Lines 776–778:
   773:                         help='Number of GNN layers')
   774:     args = parser.parse_args()
   775: 
   776:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   777:     # Allowed keys: learning_rate, epochs.
   778:     CONFIG_OVERRIDES = {}
   779: 
   780:     for _k, _v in CONFIG_OVERRIDES.items():
   781:         if _k == 'learning_rate': args.lr = _v
```

### `gearnet` baseline — editable region  [READ-ONLY — reference implementation]

In `ProteinWorkshop/custom_protein_encoder.py`:

```python
Lines 125–363:
   122:     return torch.cat([aa_onehot, sin_enc, cos_enc, dihedrals], dim=-1)
   123: 
   124: # =====================================================================
   125: # =====================================================================
   126: # EDITABLE SECTION START — GearNet encoder
   127: # =====================================================================
   128: 
   129: class GeometricRelationalConv(nn.Module):
   130:     """Geometric relational graph convolution layer from GearNet.
   131: 
   132:     Handles multiple edge types (relation types) via separate weight matrices
   133:     and incorporates edge features.
   134:     """
   135:     def __init__(self, input_dim, output_dim, num_relation, edge_input_dim=None,
   136:                  batch_norm=True, activation='relu'):
   137:         super().__init__()
   138:         self.input_dim = input_dim
   139:         self.output_dim = output_dim
   140:         self.num_relation = num_relation
   141: 
   142:         # Per-relation linear transforms
   143:         self.linear = nn.Linear(num_relation * input_dim, output_dim)
   144:         self.self_loop = nn.Linear(input_dim, output_dim)
   145: 
   146:         if edge_input_dim is not None:
   147:             self.edge_linear = nn.Linear(edge_input_dim, input_dim)
   148:         else:
   149:             self.edge_linear = None
   150: 
   151:         self.batch_norm_layer = nn.BatchNorm1d(output_dim) if batch_norm else None
   152: 
   153:         if activation == 'relu':
   154:             self.activation = nn.ReLU()
   155:         elif activation == 'silu':
   156:             self.activation = nn.SiLU()
   157:         else:
   158:             self.activation = nn.ReLU()
   159: 
   160:     def forward(self, h, edge_index, edge_type, edge_feat, num_nodes):
   161:         """
   162:         Args:
   163:             h: (N, input_dim) node features
   164:             edge_index: (2, E) edge indices
   165:             edge_type: (E,) relation type per edge
   166:             edge_feat: (E, edge_input_dim) or None
   167:             num_nodes: total number of nodes
   168:         Returns:
   169:             out: (N, output_dim) updated node features
   170:         """
   171:         src, dst = edge_index
   172: 
   173:         # Edge-modulated messages
   174:         msg = h[src]
   175:         if self.edge_linear is not None and edge_feat is not None:
   176:             msg = msg * torch.sigmoid(self.edge_linear(edge_feat))
   177: 
   178:         # Per-relation aggregation
   179:         # Use edge_type to index into relation-specific buckets
   180:         node_out = dst * self.num_relation + edge_type
   181:         update = scatter_add(msg, node_out, dim=0,
   182:                            dim_size=num_nodes * self.num_relation)
   183:         update = update.view(num_nodes, self.num_relation * self.input_dim)
   184:         update = self.linear(update)
   185: 
   186:         # Self-loop
   187:         out = update + self.self_loop(h)
   188:         out = self.activation(out)
   189: 
   190:         if self.batch_norm_layer is not None:
   191:             out = self.batch_norm_layer(out)
   192: 
   193:         return out
   194: 
   195: 
   196: class ProteinEncoder(nn.Module):
   197:     """GearNet-based protein structure encoder.
   198: 
   199:     Geometry-Aware Relational Graph Neural Network that uses multiple
   200:     edge types (sequential bonds, spatial proximity, k-nearest neighbors)
   201:     with relational convolutions and optional short-cut connections.
   202: 
   203:     Reference hyperparameters (from proteinworkshop/config/encoder/gear_net.yaml
   204:     and the GearNet paper, Zhang et al. 2022, arXiv:2203.06125):
   205:       num_layers=6, emb_dim=512, activation=relu, short_cut=True,
   206:       concat_hidden=True, batch_norm=True, pool=sum, num_relation=7
   207:       (5 sequential offsets {-2,-1,0,1,2} + 1 spatial radius + 1 kNN).
   208:     """
   209:     def __init__(
   210:         self,
   211:         input_dim: int = SCALAR_NODE_DIM,
   212:         hidden_dim: int = 512,
   213:         out_dim: int = 128,
   214:         num_layers: int = 6,
   215:         dropout: float = 0.1,
   216:         cutoff: float = 10.0,
   217:         max_neighbors: int = 16,
   218:         num_relation: int = 7,
   219:         short_cut: bool = True,
   220:         concat_hidden: bool = True,
   221:         batch_norm: bool = True,
   222:     ):
   223:         super().__init__()
   224:         self.hidden_dim = hidden_dim
   225:         self.out_dim = out_dim
   226:         self.cutoff = cutoff
   227:         self.max_neighbors = max_neighbors
   228:         self.num_relation = num_relation
   229:         self.short_cut = short_cut
   230:         self.concat_hidden = concat_hidden
   231: 
   232:         # Build layer dimensions
   233:         dims = [input_dim] + [hidden_dim] * num_layers
   234:         edge_input_dim = input_dim * 2 + num_relation + 2  # node_i, node_j, rel_onehot, seq_dist, spatial_dist
   235: 
   236:         self.layers = nn.ModuleList()
   237:         self.batch_norms = nn.ModuleList() if batch_norm else None
   238:         for i in range(num_layers):
   239:             self.layers.append(
   240:                 GeometricRelationalConv(
   241:                     dims[i], dims[i + 1], num_relation,
   242:                     edge_input_dim=edge_input_dim,
   243:                     batch_norm=False,
   244:                     activation='relu',
   245:                 )
   246:             )
   247:             if batch_norm:
   248:                 self.batch_norms.append(nn.BatchNorm1d(dims[i + 1]))
   249: 
   250:         # Output projection
   251:         if concat_hidden:
   252:             total_dim = sum(dims[1:])
   253:         else:
   254:             total_dim = dims[-1]
   255:         self.out_proj = nn.Linear(total_dim, out_dim)
   256:         self.dropout = nn.Dropout(dropout)
   257: 
   258:     def _build_multi_relational_edges(self, pos, node_feat, batch):
   259:         """Build edges with 7 relation types matching GearNet (Zhang et al. 2022):
   260:         0..4: sequential edges with offsets {-2,-1,0,1,2}
   261:               (offset 0 corresponds to a self-loop relation in sequential space)
   262:         5:    spatial proximity (within cutoff radius)
   263:         6:    k-nearest neighbors (k = max_neighbors)
   264:         """
   265:         device = pos.device
   266:         N = pos.size(0)
   267: 
   268:         all_src, all_dst, all_type = [], [], []
   269: 
   270:         # Relations 0..4: sequential edges with offsets {-2, -1, 0, 1, 2}
   271:         # Offsets are within the same protein (same batch index).
   272:         # Bidirectionality is naturally produced by including both negative
   273:         # and positive offsets as distinct relation types.
   274:         seq_offsets = [-2, -1, 0, 1, 2]
   275:         num_graphs = int(batch.max().item()) + 1
   276:         for b in range(num_graphs):
   277:             mask = (batch == b).nonzero(as_tuple=True)[0]
   278:             n_b = len(mask)
   279:             if n_b == 0:
   280:                 continue
   281:             for r_idx, off in enumerate(seq_offsets):
   282:                 if off == 0:
   283:                     # self-loop sequential relation
   284:                     src = mask
   285:                     dst = mask
   286:                 elif off > 0:
   287:                     if n_b <= off:
   288:                         continue
   289:                     src = mask[:-off]
   290:                     dst = mask[off:]
   291:                 else:  # off < 0
   292:                     k = -off
   293:                     if n_b <= k:
   294:                         continue
   295:                     src = mask[k:]
   296:                     dst = mask[:-k]
   297:                 if len(src) == 0:
   298:                     continue
   299:                 all_src.append(src)
   300:                 all_dst.append(dst)
   301:                 all_type.append(torch.full((len(src),), r_idx, dtype=torch.long, device=device))
   302: 
   303:         # Relation 5: spatial proximity within cutoff radius
   304:         rad_edge_index = radius_graph(pos, r=self.cutoff, batch=batch, loop=False,
   305:                                       max_num_neighbors=512)
   306:         rad_src, rad_dst = rad_edge_index
   307:         all_src.append(rad_src)
   308:         all_dst.append(rad_dst)
   309:         all_type.append(torch.full((rad_src.numel(),), 5, dtype=torch.long, device=device))
   310: 
   311:         # Relation 6: k-nearest neighbors
   312:         knn_edge_index = knn_graph(pos, k=self.max_neighbors, batch=batch, loop=False)
   313:         knn_src, knn_dst = knn_edge_index
   314:         all_src.append(knn_src)
   315:         all_dst.append(knn_dst)
   316:         all_type.append(torch.full((knn_src.numel(),), 6, dtype=torch.long, device=device))
   317: 
   318:         edge_index = torch.stack([torch.cat(all_src), torch.cat(all_dst)], dim=0)
   319:         edge_type = torch.cat(all_type)
   320: 
   321:         # Edge features: [node_feat_src, node_feat_dst, rel_onehot, seq_dist, spatial_dist]
   322:         src, dst = edge_index
   323:         ef_node_src = node_feat[src]
   324:         ef_node_dst = node_feat[dst]
   325:         ef_rel = F.one_hot(edge_type, self.num_relation).float()
   326:         ef_seq_dist = torch.abs(src.float() - dst.float()).unsqueeze(-1)
   327:         ef_spatial_dist = (pos[src] - pos[dst]).norm(dim=-1, keepdim=True)
   328:         edge_feat = torch.cat([ef_node_src, ef_node_dst, ef_rel, ef_seq_dist, ef_spatial_dist], dim=-1)
   329: 
   330:         return edge_index, edge_type, edge_feat
   331: 
   332:     def forward(self, pos, node_feat, batch):
   333:         N = pos.size(0)
   334:         edge_index, edge_type, edge_feat = self._build_multi_relational_edges(pos, node_feat, batch)
   335: 
   336:         hiddens = []
   337:         h = node_feat  # start from raw features (input_dim)
   338: 
   339:         for i, layer in enumerate(self.layers):
   340:             hidden = layer(h, edge_index, edge_type, edge_feat, N)
   341:             if self.short_cut and hidden.shape == h.shape:
   342:                 hidden = hidden + h
   343:             if self.batch_norms is not None:
   344:                 hidden = self.batch_norms[i](hidden)
   345:             hidden = self.dropout(hidden)
   346:             hiddens.append(hidden)
   347:             h = hidden
   348: 
   349:         if self.concat_hidden:
   350:             node_feat_out = torch.cat(hiddens, dim=-1)
   351:         else:
   352:             node_feat_out = hiddens[-1]
   353: 
   354:         node_emb = self.out_proj(node_feat_out)
   355:         # Sum pooling matches reference gear_net.yaml (pool=sum) and the
   356:         # GearNet paper (Zhang et al. 2022, arXiv:2203.06125).
   357:         graph_emb = global_add_pool(node_emb, batch)
   358: 
   359:         return node_emb, graph_emb
   360: 
   361: # =====================================================================
   362: # EDITABLE SECTION END
   363: # =====================================================================
   364: # =====================================================================
   365: 
   366: 

Lines 817–819:
   814:                         help='Number of GNN layers')
   815:     args = parser.parse_args()
   816: 
   817:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   818:     # Allowed keys: learning_rate, epochs.
   819:     CONFIG_OVERRIDES = {}
   820: 
   821:     for _k, _v in CONFIG_OVERRIDES.items():
   822:         if _k == 'learning_rate': args.lr = _v
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
