# MLS-Bench: graph-link-prediction

# Graph Link Prediction

## Research Question
Design a novel link-prediction method for graphs. The goal is to learn an
encoder that maps nodes to embeddings and a decoder that scores candidate
edges, such that the model accurately predicts missing or future links across
diverse graph types.

## Background
Link prediction is a fundamental graph-learning task: given a partially
observed graph, predict which unobserved edges are likely to exist. It has
applications in social networks (friend recommendation), citation networks
(paper recommendation), knowledge graph completion, and biological interaction
prediction.

Classical approaches:
- **GCN + dot-product decoder**: a GCN encodes nodes and the dot product of
  embeddings scores edges. Simple but often competitive.
- **VGAE** (Variational Graph Auto-Encoder): a probabilistic GCN encoder with
  KL regularization and inner-product decoder. Kipf & Welling, "Variational
  Graph Auto-Encoders," 2016 (arXiv:1611.07308).
- **node2vec**: random-walk based embeddings with biased walks balancing BFS
  and DFS. Grover & Leskovec, KDD 2016 (arXiv:1607.00653).

Recent SOTA methods exploit richer structural information:
- **SEAL** extracts k-hop enclosing subgraphs per edge and uses the DRNL
  labelling trick + GNN for edge classification. Zhang & Chen, "Link Prediction
  Based on Graph Neural Networks," NeurIPS 2018 (arXiv:1802.09691).
- **Neo-GNN** learns neighborhood-overlap features from the adjacency matrix
  to augment GNN predictions. Yun, Kim, Lee, Kang & Kim, NeurIPS 2021
  (arXiv:2206.04216).
- **BUDDY / ELPH** uses subgraph sketching with HyperLogLog and MinHash for
  scalable structural information. Chamberlain, Shirobokov, Rossi, Frasca,
  Markovich, Hammerla, Bronstein & Hansmire, "Graph Neural Networks for Link
  Prediction with Subgraph Sketching," ICLR 2023 (arXiv:2209.15486).

## What to Implement
Implement the `LinkPredictor` class in `custom_linkpred.py`:

```python
class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        ...
    def encode(self, x, edge_index):
        # returns [N, hidden_channels]
        ...
    def decode(self, z_src, z_dst):
        # returns [num_edges] -- scores for given source/dest embeddings
        ...
    def forward(self, x, edge_index, edge_label_index):
        # returns [num_edges] -- end-to-end forward pass
        ...
```

Input format:
- `x`: node features `[N, in_channels]`. Feature dimension varies by dataset.
- `edge_index`: training graph edges `[2, E_train]` in COO format
  (undirected).
- `edge_label_index`: candidate edges to score `[2, num_candidates]`.

Available PyG modules (pre-installed): any of `GCNConv`, `SAGEConv`, `GATConv`,
`GINConv`, `GraphConv`, `MessagePassing`, global pooling, `torch_geometric.utils`
(e.g. `negative_sampling`, `to_undirected`, `degree`),
`torch_geometric.nn`, `torch_geometric.transforms`.

The scientific contribution may improve the encoder, the edge decoder, or the
structural features used for candidate edges. The method should avoid assuming
a fixed feature dimension or graph size and should work for undirected
training graphs with positive and sampled negative candidate edges.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/pytorch-geometric-lp/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `pytorch-geometric-lp/custom_linkpred.py`
- editable lines **127–210**




## Readable Context


### `pytorch-geometric-lp/custom_linkpred.py`  [EDITABLE — lines 127–210 only]

```python
     1: """
     2: Graph Link Prediction — Self-contained template.
     3: Predicts missing links in graphs using learned node representations and a
     4: link scoring function. Evaluated on citation networks (Cora, CiteSeer) and
     5: a collaboration network (ogbl-collab).
     6: 
     7: Structure:
     8:   Lines 1-126:   FIXED — Imports, data loading, negative sampling, evaluation
     9:   Lines 127-210: EDITABLE — LinkPredictor class (model + scoring)
    10:   Lines 211+:    FIXED — Training loop, metric computation, CLI
    11: """
    12: import os
    13: import sys
    14: import math
    15: import argparse
    16: import warnings
    17: import numpy as np
    18: from collections import defaultdict
    19: from typing import Optional, Dict, List, Tuple
    20: 
    21: import torch
    22: import torch.nn as nn
    23: import torch.nn.functional as F
    24: from torch.utils.data import DataLoader
    25: 
    26: import torch_geometric
    27: from torch_geometric.data import Data
    28: from torch_geometric.utils import (
    29:     negative_sampling,
    30:     to_undirected,
    31:     add_self_loops,
    32:     degree,
    33:     coalesce,
    34: )
    35: from torch_geometric.nn import (
    36:     GCNConv, SAGEConv, GATConv, GINConv, GraphConv,
    37:     MessagePassing, global_mean_pool, global_add_pool,
    38: )
    39: from torch_geometric.transforms import RandomLinkSplit
    40: 
    41: from sklearn.metrics import roc_auc_score
    42: 
    43: warnings.filterwarnings("ignore", category=UserWarning)
    44: 
    45: # =====================================================================
    46: # Data loading utilities
    47: # =====================================================================
    48: 
    49: def load_planetoid(name: str, data_dir: str) -> dict:
    50:     """Load Cora or CiteSeer with random 85/5/10 link split."""
    51:     from torch_geometric.datasets import Planetoid
    52:     dataset = Planetoid(root=data_dir, name=name)
    53:     data = dataset[0]
    54: 
    55:     transform = RandomLinkSplit(
    56:         num_val=0.05, num_test=0.10,
    57:         is_undirected=True,
    58:         add_negative_train_samples=True,
    59:         split_labels=True,
    60:     )
    61:     train_data, val_data, test_data = transform(data)
    62:     return {
    63:         "train": train_data, "val": val_data, "test": test_data,
    64:         "num_nodes": data.num_nodes,
    65:         "num_features": data.num_node_features,
    66:         "dataset_type": "planetoid",
    67:     }
    68: 
    69: 
    70: def load_ogbl_collab(data_dir: str) -> dict:
    71:     """Load ogbl-collab with official split."""
    72:     from ogb.linkproppred import PygLinkPropPredDataset
    73:     dataset = PygLinkPropPredDataset(name="ogbl-collab", root=data_dir)
    74:     data = dataset[0]
    75:     split_edge = dataset.get_edge_split()
    76: 
    77:     # Build training graph
    78:     row, col = data.edge_index
    79:     train_edge = split_edge["train"]["edge"]
    80:     train_ei = torch.cat([train_edge, train_edge.flip(1)], dim=0).t()
    81:     train_ei = coalesce(train_ei)
    82:     train_data = Data(x=data.x, edge_index=train_ei, num_nodes=data.num_nodes)
    83: 
    84:     return {
    85:         "train_data": train_data,
    86:         "split_edge": split_edge,
    87:         "num_nodes": data.num_nodes,
    88:         "num_features": data.x.size(1) if data.x is not None else 128,
    89:         "dataset_type": "ogbl",
    90:     }
    91: 
    92: 
    93: def compute_mrr(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
    94:     """Compute MRR: for each positive, rank among all negatives."""
    95:     # pos_scores: [num_pos], neg_scores: [num_pos, num_neg] or [num_neg]
    96:     if neg_scores.dim() == 1:
    97:         neg_scores = neg_scores.unsqueeze(0).expand(pos_scores.size(0), -1)
    98:     # rank = 1 + number of negatives scored higher
    99:     ranks = (neg_scores >= pos_scores.unsqueeze(1)).sum(dim=1) + 1
   100:     return (1.0 / ranks.float()).mean().item()
   101: 
   102: 
   103: def compute_hits_at_k(pos_scores: torch.Tensor, neg_scores: torch.Tensor,
   104:                        k: int = 50) -> float:
   105:     """Compute Hits@K."""
   106:     if neg_scores.dim() == 1:
   107:         neg_scores = neg_scores.unsqueeze(0).expand(pos_scores.size(0), -1)
   108:     kth_neg, _ = neg_scores.kthvalue(max(neg_scores.size(1) - k + 1, 1), dim=1)
   109:     return (pos_scores >= kth_neg).float().mean().item()
   110: 
   111: 
   112: # =====================================================================
   113: # EDITABLE SECTION START — Lines 127-210
   114: # Implement your link prediction model below.
   115: # You MUST define a class named `LinkPredictor` with the following interface:
   116: #   __init__(self, in_channels, hidden_channels, num_layers, dropout)
   117: #   encode(self, x, edge_index) -> node embeddings [N, hidden_channels]
   118: #   decode(self, edge_label_index, z, edge_index=None, num_nodes=None)
   119: #       -> scores [num_edges]
   120: #   forward(self, x, edge_index, edge_label_index) -> scores [num_edges]
   121: #
   122: # Note: `decode` receives the original `edge_label_index` (shape [2, E])
   123: # plus the full node embedding table `z`, so structural features can be
   124: # computed directly from the true node indices (no index recovery needed).
   125: # =====================================================================
   126: 
   127: class LinkPredictor(nn.Module):
   128:     """
   129:     Link prediction model.
   130: 
   131:     Default: 2-layer GCN encoder + dot-product decoder (simple baseline).
   132:     The agent should replace this with a better approach.
   133: 
   134:     Args:
   135:         in_channels: Input feature dimension per node.
   136:         hidden_channels: Hidden dimension.
   137:         num_layers: Number of GNN layers.
   138:         dropout: Dropout rate.
   139:     """
   140:     def __init__(self, in_channels: int, hidden_channels: int = 256,
   141:                  num_layers: int = 2, dropout: float = 0.0):
   142:         super().__init__()
   143:         self.num_layers = num_layers
   144:         self.dropout = dropout
   145: 
   146:         self.convs = nn.ModuleList()
   147:         self.convs.append(GCNConv(in_channels, hidden_channels))
   148:         for _ in range(num_layers - 1):
   149:             self.convs.append(GCNConv(hidden_channels, hidden_channels))
   150: 
   151:         # BN only on intermediate layers — not on the last layer
   152:         # to preserve embedding magnitude for dot-product scoring
   153:         self.bns = nn.ModuleList([
   154:             nn.BatchNorm1d(hidden_channels) for _ in range(num_layers - 1)
   155:         ])
   156: 
   157:     def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
   158:         """Encode nodes into embeddings.
   159: 
   160:         Args:
   161:             x: Node features [N, in_channels].
   162:             edge_index: Graph connectivity [2, E].
   163: 
   164:         Returns:
   165:             Node embeddings [N, hidden_channels].
   166:         """
   167:         for i, conv in enumerate(self.convs):
   168:             x = conv(x, edge_index)
   169:             if i < self.num_layers - 1:
   170:                 x = self.bns[i](x)
   171:                 x = F.relu(x)
   172:                 x = F.dropout(x, p=self.dropout, training=self.training)
   173:         return x
   174: 
   175:     def decode(self, edge_label_index: torch.Tensor, z: torch.Tensor,
   176:                edge_index: Optional[torch.Tensor] = None,
   177:                num_nodes: Optional[int] = None) -> torch.Tensor:
   178:         """Score candidate edges.
   179: 
   180:         Args:
   181:             edge_label_index: Candidate edges [2, num_edges] (original node
   182:                 indices into `z`).
   183:             z: Full node embedding table [N, hidden_channels].
   184:             edge_index: Optional training graph connectivity [2, E], available
   185:                 so structure-aware decoders can compute CN/AA/RA etc.
   186:             num_nodes: Optional number of nodes in the graph.
   187: 
   188:         Returns:
   189:             Edge scores [num_edges].
   190:         """
   191:         z_src = z[edge_label_index[0]]
   192:         z_dst = z[edge_label_index[1]]
   193:         return (z_src * z_dst).sum(dim=-1)
   194: 
   195:     def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
   196:                 edge_label_index: torch.Tensor) -> torch.Tensor:
   197:         """Full forward: encode all nodes, then decode candidate edges.
   198: 
   199:         Args:
   200:             x: Node features [N, in_channels].
   201:             edge_index: Training graph connectivity [2, E_train].
   202:             edge_label_index: Candidate edges to score [2, num_candidates].
   203: 
   204:         Returns:
   205:             Edge scores [num_candidates].
   206:         """
   207:         z = self.encode(x, edge_index)
   208:         return self.decode(edge_label_index, z, edge_index=edge_index,
   209:                            num_nodes=x.size(0))
   210: 
   211: # Helper functions may be defined here as needed.
   212: 
   213: # =====================================================================
   214: # EDITABLE SECTION END
   215: # =====================================================================
   216: 
   217: # =====================================================================
   218: # FIXED — Training loop, evaluation, CLI
   219: # =====================================================================
   220: 
   221: def train_planetoid(model, data_bundle, args, device):
   222:     """Train and evaluate on Planetoid (Cora/CiteSeer)."""
   223:     train_data = data_bundle["train"].to(device)
   224:     val_data = data_bundle["val"].to(device)
   225:     test_data = data_bundle["test"].to(device)
   226: 
   227:     model = model.to(device)
   228:     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
   229:                                   weight_decay=args.weight_decay)
   230: 
   231:     best_val_auc = 0.0
   232:     best_state = None
   233:     patience_counter = 0
   234: 
   235:     for epoch in range(1, args.epochs + 1):
   236:         model.train()
   237:         optimizer.zero_grad()
   238: 
   239:         # Positive edges from split; resample negatives each epoch
   240:         pos_ei = train_data.pos_edge_label_index
   241:         neg_ei = negative_sampling(
   242:             train_data.edge_index,
   243:             num_nodes=train_data.num_nodes,
   244:             num_neg_samples=pos_ei.size(1),
   245:         )
   246: 
   247:         pos_scores = model(train_data.x, train_data.edge_index, pos_ei)
   248:         neg_scores = model(train_data.x, train_data.edge_index, neg_ei)
   249: 
   250:         pos_loss = F.binary_cross_entropy_with_logits(
   251:             pos_scores, torch.ones_like(pos_scores))
   252:         neg_loss = F.binary_cross_entropy_with_logits(
   253:             neg_scores, torch.zeros_like(neg_scores))
   254:         loss = pos_loss + neg_loss
   255:         loss.backward()
   256: 
   257:         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
   258:         optimizer.step()
   259: 
   260:         # Validation
   261:         if epoch % args.eval_every == 0:
   262:             model.eval()
   263:             with torch.no_grad():
   264:                 val_pos = model(val_data.x, train_data.edge_index,
   265:                                 val_data.pos_edge_label_index)
   266:                 val_neg = model(val_data.x, train_data.edge_index,
   267:                                 val_data.neg_edge_label_index)
   268:             val_scores = torch.cat([val_pos, val_neg]).sigmoid().cpu().numpy()
   269:             val_labels = np.concatenate([
   270:                 np.ones(val_pos.size(0)), np.zeros(val_neg.size(0))
   271:             ])
   272:             val_auc = roc_auc_score(val_labels, val_scores) * 100
   273: 
   274:             print(f"TRAIN_METRICS epoch={epoch} loss={loss.item():.4f} "
   275:                   f"val_auc={val_auc:.2f}", flush=True)
   276: 
   277:             if val_auc > best_val_auc:
   278:                 best_val_auc = val_auc
   279:                 best_state = {k: v.clone() for k, v in model.state_dict().items()}
   280:                 patience_counter = 0
   281:             else:
   282:                 patience_counter += 1
   283:                 if patience_counter >= args.patience:
   284:                     print(f"Early stopping at epoch {epoch}.", flush=True)
   285:                     break
   286: 
   287:     # Test evaluation
   288:     if best_state is not None:
   289:         model.load_state_dict(best_state)
   290:     model.eval()
   291:     with torch.no_grad():
   292:         test_pos = model(test_data.x, train_data.edge_index,
   293:                          test_data.pos_edge_label_index)
   294:         test_neg = model(test_data.x, train_data.edge_index,
   295:                          test_data.neg_edge_label_index)
   296: 
   297:     # AUC
   298:     scores = torch.cat([test_pos, test_neg]).sigmoid().cpu().numpy()
   299:     labels = np.concatenate([
   300:         np.ones(test_pos.size(0)), np.zeros(test_neg.size(0))
   301:     ])
   302:     auc = roc_auc_score(labels, scores) * 100
   303: 
   304:     # MRR
   305:     mrr = compute_mrr(test_pos.cpu(), test_neg.cpu()) * 100
   306: 
   307:     # Hits@20
   308:     hits20 = compute_hits_at_k(test_pos.cpu(), test_neg.cpu(), k=20) * 100
   309: 
   310:     print(f"TEST_METRICS AUC={auc:.2f} MRR={mrr:.2f} Hits@20={hits20:.2f}",
   311:           flush=True)
   312: 
   313: 
   314: def train_ogbl(model, data_bundle, args, device):
   315:     """Train and evaluate on ogbl-collab."""
   316:     from ogb.linkproppred import Evaluator
   317:     evaluator = Evaluator(name="ogbl-collab")
   318: 
   319:     train_data = data_bundle["train_data"].to(device)
   320:     split_edge = data_bundle["split_edge"]
   321: 
   322:     model = model.to(device)
   323:     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
   324:                                   weight_decay=args.weight_decay)
   325: 
   326:     best_val_hits = 0.0
   327:     best_state = None
   328:     patience_counter = 0
   329: 
   330:     for epoch in range(1, args.epochs + 1):
   331:         model.train()
   332:         optimizer.zero_grad()
   333: 
   334:         # Sample positive and negative training edges
   335:         pos_train = split_edge["train"]["edge"].to(device)
   336:         # Subsample for efficiency
   337:         n_pos = min(pos_train.size(0), args.batch_size)
   338:         idx = torch.randperm(pos_train.size(0))[:n_pos]
   339:         pos_ei = pos_train[idx].t()  # [2, n_pos]
   340: 
   341:         neg_ei = negative_sampling(
   342:             train_data.edge_index, num_nodes=train_data.num_nodes,
   343:             num_neg_samples=n_pos,
   344:         )
   345: 
   346:         x = train_data.x
   347:         if x is None:
   348:             x = torch.ones(train_data.num_nodes, 1, device=device)
   349: 
   350:         pos_scores = model(x, train_data.edge_index, pos_ei)
   351:         neg_scores = model(x, train_data.edge_index, neg_ei)
   352: 
   353:         pos_loss = F.binary_cross_entropy_with_logits(
   354:             pos_scores, torch.ones_like(pos_scores))
   355:         neg_loss = F.binary_cross_entropy_with_logits(
   356:             neg_scores, torch.zeros_like(neg_scores))
   357:         loss = pos_loss + neg_loss
   358:         loss.backward()
   359: 
   360:         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
   361:         optimizer.step()
   362: 
   363:         # Validation
   364:         if epoch % args.eval_every == 0:
   365:             model.eval()
   366:             with torch.no_grad():
   367:                 z = model.encode(x, train_data.edge_index)
   368: 
   369:                 val_pos = split_edge["valid"]["edge"].to(device)
   370:                 val_neg = split_edge["valid"]["edge_neg"].to(device)
   371: 
   372:                 _N = train_data.num_nodes
   373:                 pos_eli = val_pos.t().contiguous()  # [2, P]
   374:                 val_pos_scores = model.decode(
   375:                     pos_eli, z,
   376:                     edge_index=train_data.edge_index, num_nodes=_N)
   377:                 if val_neg.dim() == 3:
   378:                     vn = val_neg.reshape(-1, 2)
   379:                     neg_eli = vn.t().contiguous()
   380:                     val_neg_scores = model.decode(
   381:                         neg_eli, z,
   382:                         edge_index=train_data.edge_index, num_nodes=_N)
   383:                     val_neg_scores = val_neg_scores.view(val_neg.size(0), val_neg.size(1))
   384:                 elif val_neg.dim() == 2 and val_neg.size(1) == 2:
   385:                     neg_eli = val_neg.t().contiguous()
   386:                     val_neg_scores = model.decode(
   387:                         neg_eli, z,
   388:                         edge_index=train_data.edge_index, num_nodes=_N)
   389:                 else:
   390:                     # [num_pos, K] format: destinations only, source = val_pos source
   391:                     src_rep = val_pos[:, 0].unsqueeze(1).expand_as(val_neg).reshape(-1)
   392:                     dst_rep = val_neg.reshape(-1)
   393:                     neg_eli = torch.stack([src_rep, dst_rep], dim=0)
   394:                     val_neg_scores = model.decode(
   395:                         neg_eli, z,
   396:                         edge_index=train_data.edge_index, num_nodes=_N)
   397:                     val_neg_scores = val_neg_scores.view(val_neg.size(0), val_neg.size(1))
   398: 
   399:             val_hits = compute_hits_at_k(val_pos_scores.cpu(), val_neg_scores.cpu(), k=50) * 100
   400: 
   401:             print(f"TRAIN_METRICS epoch={epoch} loss={loss.item():.4f} "
   402:                   f"val_hits50={val_hits:.2f}", flush=True)
   403: 
   404:             if val_hits > best_val_hits:
   405:                 best_val_hits = val_hits
   406:                 best_state = {k: v.clone() for k, v in model.state_dict().items()}
   407:                 patience_counter = 0
   408:             else:
   409:                 patience_counter += 1
   410:                 if patience_counter >= args.patience:
   411:                     print(f"Early stopping at epoch {epoch}.", flush=True)
   412:                     break
   413: 
   414:     # Test evaluation
   415:     # OGB standard: include validation edges in the adjacency at test time
   416:     if best_state is not None:
   417:         model.load_state_dict(best_state)
   418:     model.eval()
   419:     with torch.no_grad():
   420:         x = train_data.x
   421:         if x is None:
   422:             x = torch.ones(train_data.num_nodes, 1, device=device)
   423: 
   424:         # Build test-time adjacency: train + validation edges
   425:         val_edge = split_edge["valid"]["edge"].to(device)
   426:         val_ei = torch.cat([val_edge, val_edge.flip(1)], dim=0).t()
   427:         test_edge_index = coalesce(
   428:             torch.cat([train_data.edge_index, val_ei], dim=1))
   429: 
   430:         z = model.encode(x, test_edge_index)
   431: 
   432:         test_pos = split_edge["test"]["edge"].to(device)
   433:         test_neg = split_edge["test"]["edge_neg"].to(device)
   434: 
   435:         _N = train_data.num_nodes
   436:         pos_eli = test_pos.t().contiguous()  # [2, P]
   437:         pos_scores = model.decode(
   438:             pos_eli, z, edge_index=test_edge_index, num_nodes=_N)
   439:         if test_neg.dim() == 3:
   440:             tn = test_neg.reshape(-1, 2)
   441:             neg_eli = tn.t().contiguous()
   442:             neg_scores = model.decode(
   443:                 neg_eli, z, edge_index=test_edge_index, num_nodes=_N)
   444:             neg_scores = neg_scores.view(test_neg.size(0), test_neg.size(1))
   445:         elif test_neg.dim() == 2 and test_neg.size(1) == 2:
   446:             neg_eli = test_neg.t().contiguous()
   447:             neg_scores = model.decode(
   448:                 neg_eli, z, edge_index=test_edge_index, num_nodes=_N)
   449:         else:
   450:             src_rep = test_pos[:, 0].unsqueeze(1).expand_as(test_neg).reshape(-1)
   451:             dst_rep = test_neg.reshape(-1)
   452:             neg_eli = torch.stack([src_rep, dst_rep], dim=0)
   453:             neg_scores = model.decode(
   454:                 neg_eli, z, edge_index=test_edge_index, num_nodes=_N)
   455:             neg_scores = neg_scores.view(test_neg.size(0), test_neg.size(1))
   456: 
   457:     hits50 = compute_hits_at_k(pos_scores.cpu(), neg_scores.cpu(), k=50) * 100
   458:     mrr = compute_mrr(pos_scores.cpu(), neg_scores.cpu()) * 100
   459: 
   460:     print(f"TEST_METRICS Hits@50={hits50:.2f} MRR={mrr:.2f}", flush=True)
   461: 
   462: 
   463: def main():
   464:     parser = argparse.ArgumentParser(description="Graph Link Prediction")
   465:     parser.add_argument("--dataset", type=str, required=True,
   466:                         choices=["Cora", "CiteSeer", "ogbl-collab"])
   467:     parser.add_argument("--data-dir", type=str, default="/data")
   468:     parser.add_argument("--hidden-channels", type=int, default=256)
   469:     parser.add_argument("--num-layers", type=int, default=2)
   470:     parser.add_argument("--dropout", type=float, default=0.0)
   471:     parser.add_argument("--lr", type=float, default=0.01)
   472:     parser.add_argument("--weight-decay", type=float, default=0.0)
   473:     parser.add_argument("--epochs", type=int, default=200)
   474:     parser.add_argument("--batch-size", type=int, default=65536)
   475:     parser.add_argument("--eval-every", type=int, default=10)
   476:     parser.add_argument("--patience", type=int, default=20)
   477:     parser.add_argument("--seed", type=int, default=42)
   478:     parser.add_argument("--output-dir", type=str, default="./output")
   479:     args = parser.parse_args()
   480: 
   481:     # Seed
   482:     torch.manual_seed(args.seed)
   483:     np.random.seed(args.seed)
   484:     if torch.cuda.is_available():
   485:         torch.cuda.manual_seed_all(args.seed)
   486: 
   487:     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   488:     os.makedirs(args.output_dir, exist_ok=True)
   489: 
   490:     # ── Parameter Budget Check ──
   491:     # Budget = 1.05x the largest baseline.
   492:     # Neo-GNN is the largest GNN-based baseline, but node2vec uses
   493:     # nn.Embedding(max_num_nodes, H) which can be 50000*H = 12.8M+ params,
   494:     # fundamentally different from GNN methods (~500K params).
   495:     # We take max(gnn_budget, embedding_budget).
   496:     def _check_param_budget(model, in_ch, H):
   497:         # Neo-GNN: GCN encoder(2 layers + 2 BN) + 3 struct_layers + hop_weights + decoder
   498:         _neo_gnn_params = (
   499:             in_ch * H + H * H + 6 * H     # GCN encoder (2 layers + BN)
   500:             + 3 * H * H + 9 * H            # struct_layers (3 NeoGNNLayers)

[truncated: showing at most 500 lines / 60000 bytes from pytorch-geometric-lp/custom_linkpred.py]
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `gcn_dot` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-geometric-lp/custom_linkpred.py`:

```python
Lines 127–181:
   124: # computed directly from the true node indices (no index recovery needed).
   125: # =====================================================================
   126: 
   127: class LinkPredictor(nn.Module):
   128:     """GCN encoder + MLP decoder with pairwise features.
   129: 
   130:     MLP decoder on [z_src, z_dst, z_src*z_dst] matches OGB's standard GCN
   131:     link-prediction baseline; strictly stronger than pure dot product.
   132:     """
   133:     def __init__(self, in_channels: int, hidden_channels: int = 256,
   134:                  num_layers: int = 2, dropout: float = 0.0):
   135:         super().__init__()
   136:         self.num_layers = num_layers
   137:         self.dropout = dropout
   138: 
   139:         self.convs = nn.ModuleList()
   140:         self.convs.append(GCNConv(in_channels, hidden_channels))
   141:         for _ in range(num_layers - 1):
   142:             self.convs.append(GCNConv(hidden_channels, hidden_channels))
   143: 
   144:         self.bns = nn.ModuleList([
   145:             nn.BatchNorm1d(hidden_channels) for _ in range(num_layers - 1)
   146:         ])
   147: 
   148:         # MLP decoder on concatenated pair features
   149:         self.decoder = nn.Sequential(
   150:             nn.Linear(hidden_channels * 3, hidden_channels),
   151:             nn.ReLU(),
   152:             nn.Dropout(dropout),
   153:             nn.Linear(hidden_channels, hidden_channels),
   154:             nn.ReLU(),
   155:             nn.Dropout(dropout),
   156:             nn.Linear(hidden_channels, 1),
   157:         )
   158: 
   159:     def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
   160:         for i, conv in enumerate(self.convs):
   161:             x = conv(x, edge_index)
   162:             if i < self.num_layers - 1:
   163:                 x = self.bns[i](x)
   164:                 x = F.relu(x)
   165:                 x = F.dropout(x, p=self.dropout, training=self.training)
   166:         return x
   167: 
   168:     def decode(self, edge_label_index: torch.Tensor, z: torch.Tensor,
   169:                edge_index: Optional[torch.Tensor] = None,
   170:                num_nodes: Optional[int] = None) -> torch.Tensor:
   171:         z_src = z[edge_label_index[0]]
   172:         z_dst = z[edge_label_index[1]]
   173:         h = torch.cat([z_src, z_dst, z_src * z_dst], dim=-1)
   174:         return self.decoder(h).squeeze(-1)
   175: 
   176:     def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
   177:                 edge_label_index: torch.Tensor) -> torch.Tensor:
   178:         z = self.encode(x, edge_index)
   179:         return self.decode(edge_label_index, z,
   180:                            edge_index=edge_index, num_nodes=x.size(0))
   181: 
   182: # Helper functions may be defined here as needed.
   183: 
   184: # =====================================================================
```

### `vgae` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-geometric-lp/custom_linkpred.py`:

```python
Lines 127–215:
   124: # computed directly from the true node indices (no index recovery needed).
   125: # =====================================================================
   126: 
   127: class LinkPredictor(nn.Module):
   128:     """Variational Graph Auto-Encoder (VGAE).
   129: 
   130:     GCN encoder produces mean + logstd, samples via reparameterization.
   131:     Dot-product decoder. KL regularization is injected into the computation
   132:     graph so that it participates in the backward pass even though the
   133:     external training loop only sees BCE on the returned scores.
   134: 
   135:     The KL term is added to each score as  w * KL / num_nodes  (the
   136:     standard VGAE normalisation), NOT divided by num_scores.  This
   137:     ensures the KL gradient is strong enough to regularise the latent
   138:     space while remaining small enough not to overwhelm the
   139:     reconstruction gradient on any single score.
   140: 
   141:     No BatchNorm on the final (mu/logstd) layers to preserve embedding
   142:     magnitude for dot-product scoring.
   143:     """
   144:     def __init__(self, in_channels: int, hidden_channels: int = 256,
   145:                  num_layers: int = 2, dropout: float = 0.0):
   146:         super().__init__()
   147:         self.dropout = dropout
   148:         # Standard VGAE uses 1/N weighting for KL; we keep a small coefficient
   149:         # because the KL is already normalized per node below.
   150:         self.kl_weight = 0.005
   151: 
   152:         # Shared GCN layers (all but last)
   153:         self.shared_convs = nn.ModuleList()
   154:         self.shared_bns = nn.ModuleList()
   155:         if num_layers > 1:
   156:             self.shared_convs.append(GCNConv(in_channels, hidden_channels))
   157:             self.shared_bns.append(nn.BatchNorm1d(hidden_channels))
   158:             for _ in range(num_layers - 2):
   159:                 self.shared_convs.append(GCNConv(hidden_channels, hidden_channels))
   160:                 self.shared_bns.append(nn.BatchNorm1d(hidden_channels))
   161:             last_in = hidden_channels
   162:         else:
   163:             last_in = in_channels
   164: 
   165:         # Separate heads for mean and log-variance (no BN on these)
   166:         self.conv_mu = GCNConv(last_in, hidden_channels)
   167:         self.conv_logstd = GCNConv(last_in, hidden_channels)
   168: 
   169:         self.__mu = None
   170:         self.__logstd = None
   171:         self.__num_nodes = None
   172: 
   173:     def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
   174:         self.__num_nodes = x.size(0)
   175:         # Shared layers with BN + ReLU
   176:         for conv, bn in zip(self.shared_convs, self.shared_bns):
   177:             x = conv(x, edge_index)
   178:             x = bn(x)
   179:             x = F.relu(x)
   180:             x = F.dropout(x, p=self.dropout, training=self.training)
   181: 
   182:         self.__mu = self.conv_mu(x, edge_index)
   183:         self.__logstd = self.conv_logstd(x, edge_index)
   184: 
   185:         if self.training:
   186:             std = torch.exp(0.5 * self.__logstd)
   187:             eps = torch.randn_like(std)
   188:             return self.__mu + eps * std
   189:         return self.__mu
   190: 
   191:     def decode(self, edge_label_index: torch.Tensor, z: torch.Tensor,
   192:                edge_index: Optional[torch.Tensor] = None,
   193:                num_nodes: Optional[int] = None) -> torch.Tensor:
   194:         z_src = z[edge_label_index[0]]
   195:         z_dst = z[edge_label_index[1]]
   196:         return (z_src * z_dst).sum(dim=-1)
   197: 
   198:     def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
   199:                 edge_label_index: torch.Tensor) -> torch.Tensor:
   200:         z = self.encode(x, edge_index)
   201:         scores = self.decode(edge_label_index, z,
   202:                              edge_index=edge_index, num_nodes=x.size(0))
   203:         # Inject KL divergence into the computation graph so its gradient
   204:         # flows through the encoder during backprop.  We add a uniform
   205:         # per-score shift:  scores + w * KL_per_node.
   206:         # KL_per_node = (1/N) * sum_i KL(q(z_i|X,A) || p(z_i)).
   207:         # The coefficient w controls the strength of the regularisation.
   208:         if self.training and self.__mu is not None:
   209:             kl_per_node = -0.5 * torch.mean(
   210:                 torch.sum(1 + self.__logstd - self.__mu.pow(2)
   211:                           - self.__logstd.exp(), dim=-1)
   212:             )
   213:             scores = scores + self.kl_weight * kl_per_node
   214:         return scores
   215: 
   216: # Helper functions may be defined here as needed.
   217: 
   218: # =====================================================================
```

### `seal` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-geometric-lp/custom_linkpred.py`:

```python
Lines 127–199:
   124: # computed directly from the true node indices (no index recovery needed).
   125: # =====================================================================
   126: 
   127: class StructuralEncoder(nn.Module):
   128:     """GCN encoder augmented with structural node features."""
   129:     def __init__(self, in_channels: int, hidden_channels: int,
   130:                  num_layers: int, dropout: float):
   131:         super().__init__()
   132:         self.num_layers = num_layers
   133:         self.dropout = dropout
   134: 
   135:         self.convs = nn.ModuleList()
   136:         self.convs.append(GCNConv(in_channels, hidden_channels))
   137:         for _ in range(num_layers - 1):
   138:             self.convs.append(GCNConv(hidden_channels, hidden_channels))
   139: 
   140:         self.bns = nn.ModuleList([
   141:             nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
   142:         ])
   143: 
   144:     def forward(self, x, edge_index):
   145:         for i, conv in enumerate(self.convs):
   146:             x = conv(x, edge_index)
   147:             x = self.bns[i](x)
   148:             if i < self.num_layers - 1:
   149:                 x = F.relu(x)
   150:                 x = F.dropout(x, p=self.dropout, training=self.training)
   151:         return x
   152: 
   153: 
   154: class LinkPredictor(nn.Module):
   155:     """SEAL-inspired link predictor.
   156: 
   157:     Uses GCN encoder + pairwise MLP decoder with structural features
   158:     (product, difference, L2 distance) that approximate SEAL's subgraph
   159:     information without the expensive subgraph extraction.
   160:     """
   161:     def __init__(self, in_channels: int, hidden_channels: int = 256,
   162:                  num_layers: int = 2, dropout: float = 0.0):
   163:         super().__init__()
   164:         self.encoder = StructuralEncoder(in_channels, hidden_channels,
   165:                                           num_layers, dropout)
   166:         # SEAL-style pairwise features: concat, hadamard, L1, L2
   167:         # Input: z_src || z_dst || z_src*z_dst || |z_src-z_dst|
   168:         dec_in = hidden_channels * 4
   169:         self.decoder = nn.Sequential(
   170:             nn.Linear(dec_in, hidden_channels),
   171:             nn.ReLU(),
   172:             nn.Dropout(dropout),
   173:             nn.Linear(hidden_channels, hidden_channels),
   174:             nn.ReLU(),
   175:             nn.Dropout(dropout),
   176:             nn.Linear(hidden_channels, 1),
   177:         )
   178: 
   179:     def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
   180:         return self.encoder(x, edge_index)
   181: 
   182:     def decode(self, edge_label_index: torch.Tensor, z: torch.Tensor,
   183:                edge_index: Optional[torch.Tensor] = None,
   184:                num_nodes: Optional[int] = None) -> torch.Tensor:
   185:         z_src = z[edge_label_index[0]]
   186:         z_dst = z[edge_label_index[1]]
   187:         h = torch.cat([
   188:             z_src, z_dst,
   189:             z_src * z_dst,
   190:             torch.abs(z_src - z_dst),
   191:         ], dim=-1)
   192:         return self.decoder(h).squeeze(-1)
   193: 
   194:     def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
   195:                 edge_label_index: torch.Tensor) -> torch.Tensor:
   196:         z = self.encode(x, edge_index)
   197:         return self.decode(edge_label_index, z,
   198:                            edge_index=edge_index, num_nodes=x.size(0))
   199: 
   200: # Helper functions may be defined here as needed.
   201: 
   202: # =====================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
