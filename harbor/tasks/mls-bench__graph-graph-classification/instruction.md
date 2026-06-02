# MLS-Bench: graph-graph-classification

# Graph-Level Readout / Pooling for Graph Classification

## Research Question
Design a novel **graph-level readout (pooling) mechanism** that aggregates node
representations from a fixed message-passing backbone into a graph-level
embedding for graph classification, improving classification performance and
generalization across diverse molecular and biological graph datasets.

## Background
Graph classification requires mapping a variable-size graph to a fixed-size
vector for downstream prediction. The standard approach uses simple
permutation-invariant operations (sum, mean, max) over node embeddings, but
these discard structural information and treat all nodes equally. The benchmark
evaluates performance on standard molecular and bio/chemical graph collections
of varying sizes and complexity. Notable prior work:

- **Sum / Mean / Max readout** (basic). Xu, Hu, Leskovec & Jegelka, "How
  Powerful are Graph Neural Networks?", ICLR 2019 (arXiv:1810.00826) shows
  sum readout is most expressive among basic operations and motivates GIN.
- **SortPooling** (Zhang, Cui, Neumann & Chen, "An End-to-End Deep Learning
  Architecture for Graph Classification," AAAI 2018) sorts nodes by structural
  role via WL colors and applies a 1-D convolution.
- **Set2Set** (Vinyals, Bengio & Kudlur, "Order Matters: Sequence to sequence
  for sets," ICLR 2016; arXiv:1511.06391) uses LSTM-based attention over the
  node set.
- **SAGPool** (Lee, Lee & Kang, "Self-Attention Graph Pooling," ICML 2019;
  arXiv:1904.08082) computes self-attention scores for hierarchical top-k node
  selection.
- **DiffPool** (Ying, You, Morris, Ren, Hamilton & Leskovec, "Hierarchical
  Graph Representation Learning with Differentiable Pooling," NeurIPS 2018;
  arXiv:1806.08804) learns differentiable soft cluster assignments for
  hierarchical coarsening.
- **GMT** (Baek, Kang & Hwang, "Accurate Learning of Graph Representations
  with Graph Multiset Pooling," ICLR 2021; arXiv:2102.11533) is a multi-head
  attention based global pooling layer.

There is substantial room to improve graph readout by combining attention,
multi-scale aggregation, structural encodings, or learned pooling strategies.

## What You Can Modify
The `GraphReadout` class in `custom_graph_cls.py`. It receives node embeddings
from a fixed GIN backbone and must produce graph-level embeddings.

You may modify:
- The aggregation function (sum, mean, max, attention, learned weights, ...).
- Hierarchical coarsening (cluster, pool, repeat).
- How to combine multi-layer GNN outputs (jumping knowledge, concatenation,
  attention).
- Self-attention or cross-attention mechanisms over nodes.
- Structural encoding or positional information in the readout.
- Any combination of the above.

Constraints / interface:
- Input: `x` `[N_total, hidden_dim]`, `edge_index` `[2, E_total]`, `batch`
  `[N_total]`, `layer_outputs` list of `[N_total, hidden_dim]`.
- Output: `[B, output_dim]` tensor; set `self.output_dim` in `__init__`.
- Must handle variable graph sizes within a batch.
- Must be permutation equivariant / invariant as appropriate.
- Available imports: `torch`, `torch.nn`, `torch.nn.functional`,
  `torch_geometric.nn`, `torch_geometric.utils`.

A useful method should handle batches of graphs with different sizes, preserve
permutation invariance at the graph level, and generalize across small
molecular graphs and larger bio/chemical graph collections.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/pytorch-geometric/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `pytorch-geometric/custom_graph_cls.py`
- editable lines **41–81**




## Readable Context


### `pytorch-geometric/custom_graph_cls.py`  [EDITABLE — lines 41–81 only]

```python
     1: """Graph Classification Readout/Pooling Benchmark.
     2: 
     3: Train GNN models on TU graph classification datasets (MUTAG, PROTEINS, NCI1)
     4: to evaluate graph-level readout and pooling mechanisms.
     5: 
     6: FIXED: GNN backbone (GIN message-passing layers), data pipeline, training loop.
     7: EDITABLE: GraphReadout class (graph-level pooling/readout mechanism).
     8: 
     9: Usage:
    10:     python custom_graph_cls.py --dataset MUTAG --seed 42 --output-dir ./output
    11: """
    12: 
    13: import argparse
    14: import math
    15: import os
    16: import copy
    17: import random
    18: import time
    19: 
    20: import numpy as np
    21: import torch
    22: import torch.nn as nn
    23: import torch.nn.functional as F
    24: from torch.optim import Adam
    25: from torch.optim.lr_scheduler import CosineAnnealingLR
    26: 
    27: from torch_geometric.datasets import TUDataset
    28: from torch_geometric.loader import DataLoader
    29: from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool
    30: from torch_geometric.utils import degree, to_dense_adj, to_dense_batch
    31: 
    32: from sklearn.metrics import accuracy_score, f1_score
    33: from sklearn.model_selection import StratifiedKFold
    34: 
    35: 
    36: # ============================================================================
    37: # Graph-Level Readout / Pooling Mechanism
    38: # ============================================================================
    39: 
    40: # -- EDITABLE REGION START (lines 41-81) ------------------------------------
    41: class GraphReadout(nn.Module):
    42:     """Custom graph-level readout/pooling mechanism.
    43: 
    44:     Aggregates node-level representations into a single graph-level
    45:     representation for classification. Receives node embeddings from
    46:     a GIN backbone and must produce a fixed-size graph embedding.
    47: 
    48:     Args:
    49:         hidden_dim (int): Dimension of node embeddings from the GNN backbone.
    50:         num_layers (int): Number of GNN layers (for JK-style readout).
    51: 
    52:     Input:
    53:         x (Tensor): Node embeddings [N_total, hidden_dim] (batched).
    54:         edge_index (LongTensor): Edge index [2, E_total] (batched).
    55:         batch (LongTensor): Batch assignment vector [N_total].
    56:         layer_outputs (list[Tensor]): Per-layer node embeddings from GNN,
    57:             each [N_total, hidden_dim]. len == num_layers.
    58: 
    59:     Output:
    60:         Tensor: Graph-level embeddings [B, output_dim].
    61:             output_dim is accessible via self.output_dim attribute.
    62: 
    63:     Design considerations:
    64:         - How to aggregate variable-size node sets into fixed-size vectors
    65:         - Whether to use simple permutation-invariant ops (sum/mean/max)
    66:         - Whether to learn attention weights over nodes
    67:         - Whether to exploit multi-scale information from different GNN layers
    68:         - Whether to use hierarchical coarsening (cluster, pool, repeat)
    69:         - Interaction between pooling and downstream classifier
    70:     """
    71: 
    72:     def __init__(self, hidden_dim, num_layers):
    73:         super().__init__()
    74:         self.hidden_dim = hidden_dim
    75:         self.num_layers = num_layers
    76:         # Default: simple sum pooling over final-layer node embeddings
    77:         self.output_dim = hidden_dim
    78: 
    79:     def forward(self, x, edge_index, batch, layer_outputs):
    80:         # Default: global sum pooling on last-layer embeddings
    81:         return global_add_pool(x, batch)
    82: # -- EDITABLE REGION END (lines 41-81) --------------------------------------
    83: 
    84: 
    85: # ============================================================================
    86: # GIN Backbone (FIXED)
    87: # ============================================================================
    88: 
    89: class GINBackbone(nn.Module):
    90:     """Graph Isomorphism Network backbone (Xu et al., 2019).
    91: 
    92:     Standard 5-layer GIN with batch normalization. Produces per-node
    93:     embeddings at each layer for flexible readout.
    94:     """
    95: 
    96:     def __init__(self, input_dim, hidden_dim, num_layers=5):
    97:         super().__init__()
    98:         self.num_layers = num_layers
    99: 
   100:         self.convs = nn.ModuleList()
   101:         self.bns = nn.ModuleList()
   102: 
   103:         for i in range(num_layers):
   104:             in_dim = input_dim if i == 0 else hidden_dim
   105:             mlp = nn.Sequential(
   106:                 nn.Linear(in_dim, hidden_dim),
   107:                 nn.BatchNorm1d(hidden_dim),
   108:                 nn.ReLU(),
   109:                 nn.Linear(hidden_dim, hidden_dim),
   110:             )
   111:             self.convs.append(GINConv(mlp, train_eps=True))
   112:             self.bns.append(nn.BatchNorm1d(hidden_dim))
   113: 
   114:     def forward(self, x, edge_index, batch):
   115:         layer_outputs = []
   116:         h = x
   117:         for i in range(self.num_layers):
   118:             h = self.convs[i](h, edge_index)
   119:             h = self.bns[i](h)
   120:             h = F.relu(h)
   121:             layer_outputs.append(h)
   122:         return h, layer_outputs
   123: 
   124: 
   125: # ============================================================================
   126: # Full Classifier Model (FIXED)
   127: # ============================================================================
   128: 
   129: class GraphClassifier(nn.Module):
   130:     """GIN backbone + custom readout + MLP classifier."""
   131: 
   132:     def __init__(self, input_dim, hidden_dim, num_classes, num_layers=5,
   133:                  dropout=0.5):
   134:         super().__init__()
   135:         self.backbone = GINBackbone(input_dim, hidden_dim, num_layers)
   136:         self.readout = GraphReadout(hidden_dim, num_layers)
   137:         self.dropout = dropout
   138: 
   139:         # MLP classifier head
   140:         readout_dim = self.readout.output_dim
   141:         self.classifier = nn.Sequential(
   142:             nn.Linear(readout_dim, hidden_dim),
   143:             nn.ReLU(),
   144:             nn.Dropout(dropout),
   145:             nn.Linear(hidden_dim, hidden_dim // 2),
   146:             nn.ReLU(),
   147:             nn.Dropout(dropout),
   148:             nn.Linear(hidden_dim // 2, num_classes),
   149:         )
   150: 
   151:     def forward(self, data):
   152:         x, edge_index, batch = data.x, data.edge_index, data.batch
   153:         node_emb, layer_outputs = self.backbone(x, edge_index, batch)
   154:         graph_emb = self.readout(node_emb, edge_index, batch, layer_outputs)
   155:         return self.classifier(graph_emb)
   156: 
   157: 
   158: # ============================================================================
   159: # Data Loading (FIXED)
   160: # ============================================================================
   161: 
   162: def load_dataset(name, data_root='/data/TUDataset'):
   163:     """Load TU dataset with one-hot degree features if no node features."""
   164:     dataset = TUDataset(root=data_root, name=name, use_node_attr=True)
   165: 
   166:     # If no node features, use one-hot degree encoding
   167:     if dataset.num_node_features == 0:
   168:         max_degree = 0
   169:         for data in dataset:
   170:             d = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
   171:             max_degree = max(max_degree, int(d.max()))
   172: 
   173:         for data in dataset:
   174:             d = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
   175:             data.x = F.one_hot(d, num_classes=max_degree + 1).float()
   176: 
   177:     return dataset
   178: 
   179: 
   180: # ============================================================================
   181: # Training & Evaluation (FIXED)
   182: # ============================================================================
   183: 
   184: def train_epoch(model, loader, optimizer, device):
   185:     """Train one epoch. Returns average loss."""
   186:     model.train()
   187:     total_loss = 0
   188:     total_graphs = 0
   189:     for data in loader:
   190:         data = data.to(device)
   191:         optimizer.zero_grad()
   192:         out = model(data)
   193:         loss = F.cross_entropy(out, data.y)
   194:         loss.backward()
   195:         optimizer.step()
   196:         total_loss += loss.item() * data.num_graphs
   197:         total_graphs += data.num_graphs
   198:     return total_loss / total_graphs
   199: 
   200: 
   201: def evaluate(model, loader, device):
   202:     """Evaluate model. Returns (accuracy%, macro_f1)."""
   203:     model.eval()
   204:     all_preds = []
   205:     all_labels = []
   206:     with torch.no_grad():
   207:         for data in loader:
   208:             data = data.to(device)
   209:             out = model(data)
   210:             pred = out.argmax(dim=1)
   211:             all_preds.extend(pred.cpu().numpy())
   212:             all_labels.extend(data.y.cpu().numpy())
   213: 
   214:     acc = accuracy_score(all_labels, all_preds) * 100.0
   215:     f1 = f1_score(all_labels, all_preds, average='macro') * 100.0
   216:     return acc, f1
   217: 
   218: 
   219: def run_fold(dataset, train_idx, test_idx, args, device):
   220:     """Run training on one fold. Returns test (acc, f1) at epoch of best val acc."""
   221:     # Carve out a stratified 10% validation split from the training fold
   222:     # so that epoch selection does not peek at the test set.
   223:     train_idx = np.asarray(train_idx)
   224:     train_labels = np.array([dataset[int(i)].y.item() for i in train_idx])
   225:     val_frac = 0.1
   226:     rng = np.random.RandomState(args.seed)
   227:     unique_labels = np.unique(train_labels)
   228:     val_mask = np.zeros(len(train_idx), dtype=bool)
   229:     for lbl in unique_labels:
   230:         lbl_positions = np.where(train_labels == lbl)[0]
   231:         rng.shuffle(lbl_positions)
   232:         n_val = max(1, int(round(len(lbl_positions) * val_frac)))
   233:         val_mask[lbl_positions[:n_val]] = True
   234:     # Guard: ensure at least one training sample remains
   235:     if val_mask.all():
   236:         val_mask[0] = False
   237: 
   238:     val_idx = train_idx[val_mask]
   239:     sub_train_idx = train_idx[~val_mask]
   240: 
   241:     train_dataset = dataset[sub_train_idx.tolist()]
   242:     val_dataset = dataset[val_idx.tolist()]
   243:     test_dataset = dataset[test_idx.tolist()]
   244: 
   245:     train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
   246:                               shuffle=True, num_workers=0, drop_last=True)
   247:     val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
   248:                             shuffle=False, num_workers=0)
   249:     test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
   250:                              shuffle=False, num_workers=0)
   251: 
   252:     input_dim = dataset.num_node_features
   253:     num_classes = dataset.num_classes
   254: 
   255:     model = GraphClassifier(
   256:         input_dim=input_dim,
   257:         hidden_dim=args.hidden_dim,
   258:         num_classes=num_classes,
   259:         num_layers=args.num_layers,
   260:         dropout=args.dropout,
   261:     ).to(device)
   262: 
   263:     # ── Parameter Budget Check (first fold only) ──
   264:     # Budget = 1.05x total model params with largest baseline readout.
   265:     # GMT readout: seed(H) + MultiheadAttention(4*H*H+4*H) + 2*LayerNorm(4*H)
   266:     #   + FFN(4*H*H+3*H) = 8*H*H + 12*H
   267:     # Set2Set readout: LSTM(H,H) = 8*H*H+8*H, proj Linear(2H,H) = 2*H*H+H
   268:     #   total = 10*H*H + 9*H
   269:     # Set2Set is larger, so we take the max.
   270:     _H = args.hidden_dim
   271:     _n_params = sum(p.numel() for p in model.parameters())
   272:     _readout_params = sum(p.numel() for p in model.readout.parameters())
   273:     _fixed_params = _n_params - _readout_params
   274:     _gmt_readout = 8 * _H * _H + 12 * _H
   275:     _set2set_readout = 10 * _H * _H + 9 * _H  # LSTM(8H^2+8H) + proj(2H^2+H)
   276:     _max_readout = max(_gmt_readout, _set2set_readout)
   277:     _param_budget = int((_fixed_params + _max_readout) * 1.05)
   278:     print(f"Model parameters: {_n_params:,} (budget: {_param_budget:,})", flush=True)
   279: 
   280:     optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
   281:     scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
   282: 
   283:     best_val_acc = -1.0
   284:     best_test_acc = 0.0
   285:     best_test_f1 = 0.0
   286: 
   287:     for epoch in range(args.epochs):
   288:         train_loss = train_epoch(model, train_loader, optimizer, device)
   289:         val_acc, val_f1 = evaluate(model, val_loader, device)
   290:         scheduler.step()
   291: 
   292:         if val_acc > best_val_acc:
   293:             best_val_acc = val_acc
   294:             # Only touch the test set when val improves; the reported
   295:             # numbers correspond to the epoch selected by val accuracy.
   296:             test_acc, test_f1 = evaluate(model, test_loader, device)
   297:             best_test_acc = test_acc
   298:             best_test_f1 = test_f1
   299: 
   300:         if (epoch + 1) % 50 == 0 or epoch == 0:
   301:             print(
   302:                 f"TRAIN_METRICS: fold_epoch epoch={epoch+1} "
   303:                 f"train_loss={train_loss:.4f} val_acc={val_acc:.2f} "
   304:                 f"val_f1={val_f1:.2f} lr={optimizer.param_groups[0]['lr']:.6f}",
   305:                 flush=True,
   306:             )
   307: 
   308:     return best_test_acc, best_test_f1
   309: 
   310: 
   311: def main():
   312:     parser = argparse.ArgumentParser(description="Graph Classification Readout Benchmark")
   313:     parser.add_argument('--dataset', type=str, required=True,
   314:                         choices=['MUTAG', 'PROTEINS', 'NCI1'])
   315:     parser.add_argument('--data-root', type=str, default='/data/TUDataset')
   316:     parser.add_argument('--hidden-dim', type=int, default=64)
   317:     parser.add_argument('--num-layers', type=int, default=5)
   318:     parser.add_argument('--epochs', type=int, default=350)
   319:     parser.add_argument('--batch-size', type=int, default=32)
   320:     parser.add_argument('--lr', type=float, default=0.01)
   321:     parser.add_argument('--weight-decay', type=float, default=0.0)
   322:     parser.add_argument('--dropout', type=float, default=0.5)
   323:     parser.add_argument('--num-folds', type=int, default=10)
   324:     parser.add_argument('--seed', type=int, default=42)
   325:     parser.add_argument('--output-dir', type=str, default='.')
   326:     args = parser.parse_args()
   327: 
   328:     # Reproducibility
   329:     torch.manual_seed(args.seed)
   330:     np.random.seed(args.seed)
   331:     random.seed(args.seed)
   332:     torch.cuda.manual_seed_all(args.seed)
   333:     torch.backends.cudnn.deterministic = True
   334: 
   335:     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   336: 
   337:     # Load data
   338:     dataset = load_dataset(args.dataset, args.data_root)
   339:     print(f"Dataset: {args.dataset}, Graphs: {len(dataset)}, "
   340:           f"Features: {dataset.num_node_features}, Classes: {dataset.num_classes}",
   341:           flush=True)
   342: 
   343:     # 10-fold stratified cross-validation
   344:     labels = np.array([d.y.item() for d in dataset])
   345:     skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True,
   346:                           random_state=args.seed)
   347: 
   348:     fold_accs = []
   349:     fold_f1s = []
   350: 
   351:     for fold_idx, (train_idx, test_idx) in enumerate(skf.split(labels, labels)):
   352:         print(f"\n--- Fold {fold_idx + 1}/{args.num_folds} ---", flush=True)
   353:         train_idx = np.array(train_idx)
   354:         test_idx = np.array(test_idx)
   355:         best_acc, best_f1 = run_fold(dataset, train_idx, test_idx, args, device)
   356:         fold_accs.append(best_acc)
   357:         fold_f1s.append(best_f1)
   358:         print(f"Fold {fold_idx + 1}: acc={best_acc:.2f} f1={best_f1:.2f}", flush=True)
   359: 
   360:     mean_acc = np.mean(fold_accs)
   361:     std_acc = np.std(fold_accs)
   362:     mean_f1 = np.mean(fold_f1s)
   363:     std_f1 = np.std(fold_f1s)
   364: 
   365:     print(f"\n10-Fold Results: acc={mean_acc:.2f}+/-{std_acc:.2f} "
   366:           f"f1={mean_f1:.2f}+/-{std_f1:.2f}", flush=True)
   367:     print(f"TEST_METRICS: test_acc={mean_acc:.2f} macro_f1={mean_f1:.2f}", flush=True)
   368: 
   369: 
   370: if __name__ == '__main__':
   371:     main()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `gin_sum` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-geometric/custom_graph_cls.py`:

```python
Lines 41–73:
    38: # ============================================================================
    39: 
    40: # -- EDITABLE REGION START (lines 41-81) ------------------------------------
    41: class GraphReadout(nn.Module):
    42:     """GIN JK-Sum Readout (Xu et al., 2019).
    43: 
    44:     Concatenates sum-pooled embeddings from all GIN layers
    45:     (Jumping Knowledge). Each layer's graph embedding is batch-normalized
    46:     before concatenation to stabilize training -- this prevents the
    47:     different-scale representations across layers from causing
    48:     optimization issues (some folds failing to converge).
    49: 
    50:     The output dimension is hidden_dim * num_layers, matching the
    51:     original GIN paper's readout.
    52:     """
    53: 
    54:     def __init__(self, hidden_dim, num_layers):
    55:         super().__init__()
    56:         self.hidden_dim = hidden_dim
    57:         self.num_layers = num_layers
    58:         # Full concatenated dimension -- no projection bottleneck
    59:         self.output_dim = hidden_dim * num_layers
    60:         # Per-layer batch normalization on graph-level embeddings
    61:         self.graph_bns = nn.ModuleList([
    62:             nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
    63:         ])
    64: 
    65:     def forward(self, x, edge_index, batch, layer_outputs):
    66:         # Sum-pool each layer's node embeddings independently
    67:         graph_embs = []
    68:         for i, h in enumerate(layer_outputs):
    69:             g = global_add_pool(h, batch)
    70:             g = self.graph_bns[i](g)
    71:             graph_embs.append(g)
    72:         # Concatenate all layers (Jumping Knowledge)
    73:         return torch.cat(graph_embs, dim=-1)
    74: # -- EDITABLE REGION END (lines 41-81) --------------------------------------
    75: 
    76: 
```

### `sagpool` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-geometric/custom_graph_cls.py`:

```python
Lines 41–78:
    38: # ============================================================================
    39: 
    40: # -- EDITABLE REGION START (lines 41-81) ------------------------------------
    41: class GraphReadout(nn.Module):
    42:     """SAGPool Hierarchical Readout (Lee et al., 2019).
    43: 
    44:     Uses self-attention scores to hierarchically select top-k nodes,
    45:     then applies sum+mean global readout on the coarsened graph.
    46:     Two-level hierarchy: original -> coarsened.
    47:     """
    48: 
    49:     def __init__(self, hidden_dim, num_layers):
    50:         super().__init__()
    51:         self.hidden_dim = hidden_dim
    52:         self.num_layers = num_layers
    53:         from torch_geometric.nn.pool import SAGPooling
    54:         self.pool1 = SAGPooling(hidden_dim, ratio=0.5)
    55:         self.pool2 = SAGPooling(hidden_dim, ratio=0.5)
    56:         # 3 levels (original + 2 coarsened), each with sum+mean
    57:         self.output_dim = hidden_dim * 2 * 3
    58:         self.proj = nn.Linear(self.output_dim, hidden_dim)
    59:         self.output_dim = hidden_dim
    60: 
    61:     def forward(self, x, edge_index, batch, layer_outputs):
    62:         # Level 0: readout on original graph
    63:         r0 = torch.cat([global_add_pool(x, batch),
    64:                          global_mean_pool(x, batch)], dim=-1)
    65: 
    66:         # Level 1: first coarsening
    67:         x1, edge_index1, _, batch1, perm1, score1 = self.pool1(
    68:             x, edge_index, batch=batch)
    69:         r1 = torch.cat([global_add_pool(x1, batch1),
    70:                          global_mean_pool(x1, batch1)], dim=-1)
    71: 
    72:         # Level 2: second coarsening
    73:         x2, edge_index2, _, batch2, perm2, score2 = self.pool2(
    74:             x1, edge_index1, batch=batch1)
    75:         r2 = torch.cat([global_add_pool(x2, batch2),
    76:                          global_mean_pool(x2, batch2)], dim=-1)
    77: 
    78:         return self.proj(torch.cat([r0, r1, r2], dim=-1))
    79: # -- EDITABLE REGION END (lines 41-81) --------------------------------------
    80: 
    81: 
```

### `diffpool` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-geometric/custom_graph_cls.py`:

```python
Lines 41–78:
    38: # ============================================================================
    39: 
    40: # -- EDITABLE REGION START (lines 41-81) ------------------------------------
    41: class GraphReadout(nn.Module):
    42:     """DiffPool Readout (Ying et al., 2018).
    43: 
    44:     Uses a learned soft assignment matrix to cluster nodes into
    45:     a fixed number of super-nodes, then reads out from the
    46:     coarsened graph. Two-level hierarchy.
    47:     """
    48: 
    49:     def __init__(self, hidden_dim, num_layers):
    50:         super().__init__()
    51:         self.hidden_dim = hidden_dim
    52:         self.num_layers = num_layers
    53:         # Assignment network: maps nodes to clusters
    54:         self.max_nodes = 150  # Max nodes per graph (padded)
    55:         self.num_clusters = 25
    56:         self.assign_nn = nn.Sequential(
    57:             nn.Linear(hidden_dim, hidden_dim),
    58:             nn.ReLU(),
    59:             nn.Linear(hidden_dim, self.num_clusters),
    60:         )
    61:         self.output_dim = hidden_dim
    62: 
    63:     def forward(self, x, edge_index, batch, layer_outputs):
    64:         # Convert to dense batch format
    65:         x_dense, mask = to_dense_batch(x, batch)  # [B, N_max, D]
    66:         adj = to_dense_adj(edge_index, batch)  # [B, N_max, N_max]
    67: 
    68:         # Compute soft assignment
    69:         s = self.assign_nn(x_dense)  # [B, N_max, K]
    70:         s = s.masked_fill(~mask.unsqueeze(-1), float('-inf'))
    71:         s = torch.softmax(s, dim=1)
    72:         s = s * mask.unsqueeze(-1).float()
    73: 
    74:         # Pool: X_coarse = S^T @ X, A_coarse = S^T @ A @ S
    75:         x_coarse = torch.bmm(s.transpose(1, 2), x_dense)  # [B, K, D]
    76: 
    77:         # Global mean pool over clusters
    78:         return x_coarse.mean(dim=1)  # [B, D]
    79: # -- EDITABLE REGION END (lines 41-81) --------------------------------------
    80: 
    81: 
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
