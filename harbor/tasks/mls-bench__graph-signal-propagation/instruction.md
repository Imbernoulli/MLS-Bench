# MLS-Bench: graph-signal-propagation

# Graph Signal Propagation: Spectral / Spatial Graph Filters

## Research Question
Design a novel **graph signal propagation filter** for node feature aggregation
in graph neural networks. The filter should effectively handle both
**homophilic** graphs (where connected nodes share labels) and **heterophilic**
graphs (where connected nodes often differ).

## Background
GNNs propagate node features through graph structure using graph filters. The
choice of filter is critical: simple low-pass filters such as GCN's first-order
approximation work well on homophilic graphs but fail on heterophilic graphs,
where useful information may live in higher-frequency components. Modern
spectral methods learn polynomial filters in various bases:

- **Monomial basis (GPRGNN)**: `h(A) = sum_k gamma_k A^k`. Simple but can be
  numerically unstable. Chien, Peng, Li & Milenkovic, "Adaptive Universal
  Generalized PageRank Graph Neural Network," ICLR 2021 (arXiv:2006.07988).
- **Bernstein basis (BernNet)**: non-negative, excellent controllability, but
  `O(K^2)` complexity. He, Wei, Huang & Xu, "BernNet: Learning Arbitrary Graph
  Spectral Filters via Bernstein Approximation," NeurIPS 2021
  (arXiv:2106.10994).
- **Chebyshev interpolation (ChebNetII)**: avoids the Runge phenomenon, `O(K)`
  complexity. He, Wei & Wen, "Convolutional Neural Networks on Graphs with
  Chebyshev Approximation, Revisited," NeurIPS 2022 (arXiv:2202.03580).
- **Jacobi polynomials (JacobiConv)**: orthogonal, fast convergence,
  generalizes Chebyshev / Legendre. Wang & Zhang, "How Powerful are Spectral
  Graph Neural Networks?", ICML 2022.

Key design axes include: polynomial basis choice, coefficient initialization
and constraints, normalization (GCN vs Laplacian), and interaction with the
MLP encoder.

## Task
Modify the `CustomProp` (propagation layer) and `CustomFilter` (full model)
classes in `custom_filter.py`. The propagation layer defines how node features
are filtered across the graph; the model wraps it with an MLP encoder and
output head.

```python
class CustomProp(MessagePassing):
    def __init__(self, K, alpha=0.1, **kwargs):
        # K: polynomial order, alpha: teleport probability
        ...
    def forward(self, x, edge_index, edge_weight=None):
        # x: [num_nodes, channels], edge_index: [2, num_edges]
        # returns filtered features [num_nodes, channels]
        ...


class CustomFilter(nn.Module):
    def __init__(self, num_features, num_classes, hidden=64, K=10,
                 alpha=0.1, dropout=0.5, dprate=0.5):
        ...
    def forward(self, data):
        # data: PyG Data object with data.x, data.edge_index
        # returns log_softmax predictions [num_nodes, num_classes]
        ...
```

Available utilities:
- `gcn_norm(edge_index)` -- GCN normalization `D^{-1/2} A D^{-1/2}`.
- `get_laplacian(edge_index, normalization='sym')` -- symmetric normalized
  Laplacian `L = I - D^{-1/2} A D^{-1/2}`.
- `add_self_loops(edge_index, edge_weight, fill_value)` -- add self loops.
- `self.propagate(edge_index, x=x, norm=norm)` -- single-step message passing.
- `cheby(i, x)` -- evaluate the Chebyshev polynomial `T_i(x)`.
- `comb(n, k)` -- binomial coefficient (from scipy).
- Constants: `K`, `ALPHA`, `HIDDEN`, `DROPOUT`, `DPRATE`.

## Evaluation
Datasets (a mix of homophilic and heterophilic graphs):

| Label    | Nodes | Classes | Type         | Source                |
|----------|-------|---------|--------------|-----------------------|
| cora     | 2,708 | 7       | homophilic   | citation network      |
| citeseer | 3,327 | 6       | homophilic   | citation network      |
| texas    | 183   | 5       | heterophilic | WebKB webpage network |
| cornell  | 183   | 5       | heterophilic | WebKB webpage network |

Fixed pipeline: each dataset runs 10 random 60/20/20 train/val/test splits with
early stopping.

Metric: mean test accuracy over the 10 runs, higher-is-better.

The contribution should remain a modular graph filter paired with the fixed
classification pipeline, rather than changing the data split or evaluation
target.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/ChebNetII/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `ChebNetII/main/custom_filter.py`
- editable lines **211–308**




## Readable Context


### `ChebNetII/main/custom_filter.py`  [EDITABLE — lines 211–308 only]

```python
     1: # Custom graph signal propagation filter for MLS-Bench
     2: #
     3: # EDITABLE section: CustomProp (propagation layer) + CustomFilter (full model).
     4: # FIXED sections: everything else (config, data loading, training loop, evaluation).
     5: 
     6: import os
     7: import math
     8: import random
     9: import time
    10: from typing import Optional
    11: 
    12: import numpy as np
    13: import torch
    14: import torch.nn as nn
    15: import torch.nn.functional as F
    16: from torch.nn import Parameter, Linear
    17: from torch_geometric.nn.conv import MessagePassing
    18: from torch_geometric.nn.conv.gcn_conv import gcn_norm
    19: from torch_geometric.nn import GCNConv, APPNP as PyGAPPNP
    20: from torch_geometric.utils import get_laplacian, add_self_loops
    21: from scipy.special import comb
    22: 
    23: # =====================================================================
    24: # FIXED: Configuration
    25: # =====================================================================
    26: SEED = int(os.environ.get("SEED", "42"))
    27: OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
    28: DATASET_NAME = os.environ.get("ENV", "cora")
    29: 
    30: # Training settings
    31: EPOCHS = 1000
    32: EARLY_STOPPING = 200
    33: HIDDEN = 64
    34: K = 10          # polynomial order / propagation steps
    35: ALPHA = 0.1     # teleport probability (for PPR-style init)
    36: DROPOUT = 0.5
    37: DPRATE = 0.0    # no propagation dropout (hurts spectral filters on heterophilic data)
    38: LR = 0.05
    39: WEIGHT_DECAY = 0.0
    40: PROP_LR = 0.01       # lower lr for propagation/filter params (stable learning)
    41: PROP_WD = 0.0        # no weight decay for filter coefficients
    42: TRAIN_RATE = 0.6
    43: VAL_RATE = 0.2
    44: RUNS = 10
    45: 
    46: DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    47: 
    48: # 10 BernNet reference seeds; MLS-Bench SEED offsets them across runs
    49: FIXED_SEEDS = [
    50:     1941488137, 4198936517, 983997847, 4023022221, 4019585660,
    51:     2108550661, 1648766618, 629014539, 3212139042, 2424918363,
    52: ]
    53: 
    54: 
    55: # =====================================================================
    56: # FIXED: Dataset loading (from ChebNetII codebase)
    57: # =====================================================================
    58: import torch_geometric.transforms as T
    59: from torch_geometric.datasets import Planetoid
    60: from torch_sparse import coalesce
    61: from torch_geometric.data import InMemoryDataset, download_url, Data
    62: from torch_geometric.utils.undirected import to_undirected
    63: import os.path as osp
    64: import pickle
    65: 
    66: 
    67: class WebKB(InMemoryDataset):
    68:     """WebKB dataset (Texas, Cornell, Wisconsin, Washington)."""
    69:     url = "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/new_data"
    70: 
    71:     def __init__(self, root, name, transform=None, pre_transform=None):
    72:         self.name = name.lower()
    73:         assert self.name in ["cornell", "texas", "washington", "wisconsin"]
    74:         super(WebKB, self).__init__(root, transform, pre_transform)
    75:         self.data, self.slices = torch.load(self.processed_paths[0])
    76: 
    77:     @property
    78:     def raw_dir(self):
    79:         return osp.join(self.root, self.name, "raw")
    80: 
    81:     @property
    82:     def processed_dir(self):
    83:         return osp.join(self.root, self.name, "processed")
    84: 
    85:     @property
    86:     def raw_file_names(self):
    87:         return ["out1_node_feature_label.txt", "out1_graph_edges.txt"]
    88: 
    89:     @property
    90:     def processed_file_names(self):
    91:         return "data.pt"
    92: 
    93:     def download(self):
    94:         for name in self.raw_file_names:
    95:             download_url(f"{self.url}/{self.name}/{name}", self.raw_dir)
    96: 
    97:     def process(self):
    98:         with open(self.raw_paths[0], "r") as f:
    99:             data = f.read().split("\n")[1:-1]
   100:         x = [[float(v) for v in r.split("\t")[1].split(",")] for r in data]
   101:         x = torch.tensor(x, dtype=torch.float)
   102:         y = [int(r.split("\t")[2]) for r in data]
   103:         y = torch.tensor(y, dtype=torch.long)
   104:         with open(self.raw_paths[1], "r") as f:
   105:             data = f.read().split("\n")[1:-1]
   106:         data = [[int(v) for v in r.split("\t")] for r in data]
   107:         edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
   108:         edge_index = to_undirected(edge_index)
   109:         edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))
   110:         data = Data(x=x, edge_index=edge_index, y=y)
   111:         data = data if self.pre_transform is None else self.pre_transform(data)
   112:         torch.save(self.collate([data]), self.processed_paths[0])
   113: 
   114: 
   115: def load_dataset(name):
   116:     """Load a graph dataset by name."""
   117:     name_lower = name.lower()
   118:     data_root = os.environ.get("MLSBENCH_PKG_DIR", "/workspace/ChebNetII") + "/main/data"
   119:     if name_lower in ["cora", "citeseer", "pubmed"]:
   120:         dataset = Planetoid(osp.join(data_root, name_lower), name_lower,
   121:                             transform=T.NormalizeFeatures())
   122:     elif name_lower in ["texas", "cornell"]:
   123:         dataset = WebKB(root=data_root, name=name_lower,
   124:                         transform=T.NormalizeFeatures())
   125:     else:
   126:         raise ValueError(f"Dataset {name} not supported")
   127:     return dataset
   128: 
   129: 
   130: # =====================================================================
   131: # FIXED: Data splitting utilities
   132: # =====================================================================
   133: def index_to_mask(index, size):
   134:     mask = torch.zeros(size, dtype=torch.bool)
   135:     mask[index] = 1
   136:     return mask
   137: 
   138: 
   139: def random_splits(data, num_classes, percls_trn, val_lb, seed=42):
   140:     """Create random train/val/test splits."""
   141:     index = list(range(data.y.shape[0]))
   142:     train_idx = []
   143:     rnd_state = np.random.RandomState(seed)
   144:     for c in range(num_classes):
   145:         class_idx = np.where(data.y.cpu() == c)[0]
   146:         if len(class_idx) < percls_trn:
   147:             train_idx.extend(class_idx)
   148:         else:
   149:             train_idx.extend(rnd_state.choice(class_idx, percls_trn, replace=False))
   150:     rest_index = [i for i in index if i not in train_idx]
   151:     val_idx = rnd_state.choice(rest_index, val_lb, replace=False)
   152:     test_idx = [i for i in rest_index if i not in val_idx]
   153:     data.train_mask = index_to_mask(train_idx, size=data.num_nodes)
   154:     data.val_mask = index_to_mask(val_idx, size=data.num_nodes)
   155:     data.test_mask = index_to_mask(test_idx, size=data.num_nodes)
   156:     return data
   157: 
   158: 
   159: def set_seed(seed):
   160:     random.seed(seed)
   161:     np.random.seed(seed)
   162:     torch.manual_seed(seed)
   163:     torch.cuda.manual_seed(seed)
   164:     torch.cuda.manual_seed_all(seed)
   165: 
   166: 
   167: def cheby(i, x):
   168:     """Evaluate Chebyshev polynomial T_i(x)."""
   169:     if i == 0:
   170:         return 1
   171:     elif i == 1:
   172:         return x
   173:     else:
   174:         T0, T1 = 1, x
   175:         for _ in range(2, i + 1):
   176:             T2 = 2 * x * T1 - T0
   177:             T0, T1 = T1, T2
   178:         return T2
   179: 
   180: 
   181: # =====================================================================
   182: # FIXED: Training and evaluation functions
   183: # =====================================================================
   184: def train_step(model, optimizer, data):
   185:     model.train()
   186:     optimizer.zero_grad()
   187:     out = model(data)[data.train_mask]
   188:     loss = F.nll_loss(out, data.y[data.train_mask])
   189:     loss.backward()
   190:     optimizer.step()
   191:     return loss.item()
   192: 
   193: 
   194: def evaluate(model, data):
   195:     model.eval()
   196:     logits = model(data)
   197:     accs, losses = [], []
   198:     for mask_name in ["train_mask", "val_mask", "test_mask"]:
   199:         mask = getattr(data, mask_name)
   200:         pred = logits[mask].max(1)[1]
   201:         acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
   202:         loss = F.nll_loss(logits[mask], data.y[mask]).item()
   203:         accs.append(acc)
   204:         losses.append(loss)
   205:     return accs, losses
   206: 
   207: 
   208: # =====================================================================
   209: # EDITABLE: Custom Graph Signal Propagation Filter
   210: # =====================================================================
   211: class CustomProp(MessagePassing):
   212:     """Custom graph signal propagation layer.
   213: 
   214:     This layer defines how node features are propagated (filtered) across
   215:     the graph structure. It operates on the graph Laplacian spectrum.
   216: 
   217:     Design a novel spectral or spatial graph filter here. The filter should:
   218:     1. Accept node features x and edge_index as input
   219:     2. Apply graph-based propagation/filtering
   220:     3. Return filtered node features
   221: 
   222:     Available graph operators (from PyG):
   223:     - get_laplacian(edge_index, normalization='sym') -> (edge_index, norm)
   224:       Returns the symmetric normalized Laplacian L = I - D^{-1/2}AD^{-1/2}
   225:     - add_self_loops(edge_index, edge_weight, fill_value) -> (edge_index, weight)
   226:     - gcn_norm(edge_index) -> (edge_index, norm)
   227:       Returns D^{-1/2}AD^{-1/2} normalization
   228:     - self.propagate(edge_index, x=x, norm=norm) for message passing
   229: 
   230:     Config available: K (polynomial order), ALPHA (teleport probability).
   231: 
   232:     Args:
   233:         K: number of propagation steps / polynomial order
   234:         alpha: teleport probability (for PPR-like initialization)
   235:     """
   236: 
   237:     def __init__(self, K, alpha=0.1, **kwargs):
   238:         super(CustomProp, self).__init__(aggr="add", **kwargs)
   239:         self.K = K
   240:         self.alpha = alpha
   241:         # Learnable polynomial coefficients
   242:         self.temp = Parameter(torch.Tensor(K + 1))
   243:         self.reset_parameters()
   244: 
   245:     def reset_parameters(self):
   246:         # Initialize with PPR-like coefficients
   247:         for k in range(self.K + 1):
   248:             self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
   249:         self.temp.data[-1] = (1 - self.alpha) ** self.K
   250: 
   251:     def forward(self, x, edge_index, edge_weight=None):
   252:         # Compute GCN-normalized adjacency: D^{-1/2}AD^{-1/2}
   253:         edge_index, norm = gcn_norm(
   254:             edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype
   255:         )
   256:         # Weighted sum of K-hop propagations (monomial basis)
   257:         hidden = x * self.temp[0]
   258:         for k in range(self.K):
   259:             x = self.propagate(edge_index, x=x, norm=norm)
   260:             hidden = hidden + self.temp[k + 1] * x
   261:         return hidden
   262: 
   263:     def message(self, x_j, norm):
   264:         return norm.view(-1, 1) * x_j
   265: 
   266: 
   267: class CustomFilter(nn.Module):
   268:     """Full graph filter model: MLP encoder + CustomProp + softmax.
   269: 
   270:     Architecture: input -> dropout -> Linear -> ReLU -> dropout -> Linear
   271:                   -> (optional dprate dropout) -> CustomProp -> log_softmax
   272: 
   273:     Args:
   274:         num_features: input feature dimension
   275:         num_classes: number of output classes
   276:         hidden: hidden layer dimension
   277:         K: polynomial order for propagation
   278:         alpha: teleport probability
   279:         dropout: dropout rate for MLP layers
   280:         dprate: dropout rate for propagation layer
   281:     """
   282: 
   283:     def __init__(self, num_features, num_classes, hidden=64, K=10,
   284:                  alpha=0.1, dropout=0.5, dprate=0.5):
   285:         super(CustomFilter, self).__init__()
   286:         self.lin1 = Linear(num_features, hidden)
   287:         self.lin2 = Linear(hidden, num_classes)
   288:         self.prop = CustomProp(K, alpha)
   289:         self.dropout = dropout
   290:         self.dprate = dprate
   291: 
   292:     def reset_parameters(self):
   293:         self.lin1.reset_parameters()
   294:         self.lin2.reset_parameters()
   295:         self.prop.reset_parameters()
   296: 
   297:     def forward(self, data):
   298:         x, edge_index = data.x, data.edge_index
   299:         x = F.dropout(x, p=self.dropout, training=self.training)
   300:         x = F.relu(self.lin1(x))
   301:         x = F.dropout(x, p=self.dropout, training=self.training)
   302:         x = self.lin2(x)
   303:         if self.dprate == 0.0:
   304:             x = self.prop(x, edge_index)
   305:         else:
   306:             x = F.dropout(x, p=self.dprate, training=self.training)
   307:             x = self.prop(x, edge_index)
   308:         return F.log_softmax(x, dim=1)
   309: 
   310: 
   311: # =====================================================================
   312: # FIXED: Main training and evaluation script
   313: # =====================================================================
   314: if __name__ == "__main__":
   315:     os.makedirs(OUTPUT_DIR, exist_ok=True)
   316:     print(f"Dataset: {DATASET_NAME}, Seed: {SEED}", flush=True)
   317:     print(f"Config: hidden={HIDDEN}, K={K}, alpha={ALPHA}, "
   318:           f"dropout={DROPOUT}, dprate={DPRATE}, lr={LR}", flush=True)
   319: 
   320:     # Load dataset
   321:     dataset = load_dataset(DATASET_NAME)
   322:     data = dataset[0]
   323:     num_features = dataset.num_features
   324:     num_classes = dataset.num_classes
   325: 
   326:     # Compute split sizes
   327:     percls_trn = int(round(TRAIN_RATE * len(data.y) / num_classes))
   328:     val_lb = int(round(VAL_RATE * len(data.y)))
   329: 
   330:     results = []
   331:     for run_idx in range(RUNS):
   332:         run_seed = (FIXED_SEEDS[run_idx] + SEED - 42) & 0xFFFFFFFF
   333:         set_seed(run_seed)
   334: 
   335:         # Create data split
   336:         data_split = random_splits(data, num_classes, percls_trn, val_lb, seed=run_seed)
   337: 
   338:         # Build model
   339:         model = CustomFilter(
   340:             num_features=num_features,
   341:             num_classes=num_classes,
   342:             hidden=HIDDEN,
   343:             K=K,
   344:             alpha=ALPHA,
   345:             dropout=DROPOUT,
   346:             dprate=DPRATE,
   347:         ).to(DEVICE)
   348: 
   349:         data_split = data_split.to(DEVICE)
   350: 
   351:         # Allow model to override training hyperparameters via attributes
   352:         lr = getattr(model, 'custom_lr', LR)
   353:         wd = getattr(model, 'custom_wd', WEIGHT_DECAY)
   354:         prop_lr = getattr(model, 'custom_prop_lr', PROP_LR)
   355:         prop_wd = getattr(model, 'custom_prop_wd', PROP_WD)
   356: 
   357:         # Check if model has separate propagation parameters
   358:         prop_params = []
   359:         other_params = []
   360:         for name, param in model.named_parameters():
   361:             if "prop" in name:
   362:                 prop_params.append(param)
   363:             else:
   364:                 other_params.append(param)
   365: 
   366:         if prop_params:
   367:             optimizer = torch.optim.Adam([
   368:                 {"params": other_params, "lr": lr, "weight_decay": wd},
   369:                 {"params": prop_params, "lr": prop_lr, "weight_decay": prop_wd},
   370:             ])
   371:         else:
   372:             optimizer = torch.optim.Adam(
   373:                 model.parameters(), lr=lr, weight_decay=wd
   374:             )
   375: 
   376:         # Training loop with early stopping
   377:         best_val_loss = float("inf")
   378:         best_test_acc = 0.0
   379:         val_loss_history = []
   380: 
   381:         for epoch in range(EPOCHS):
   382:             train_loss = train_step(model, optimizer, data_split)
   383:             accs, losses = evaluate(model, data_split)
   384:             train_acc, val_acc, test_acc = accs
   385:             _, val_loss, _ = losses
   386: 
   387:             if val_loss < best_val_loss:
   388:                 best_val_loss = val_loss
   389:                 best_test_acc = test_acc
   390: 
   391:             if epoch % 100 == 0:
   392:                 print(
   393:                     f"TRAIN_METRICS run={run_idx} epoch={epoch} "
   394:                     f"train_loss={train_loss:.4f} val_acc={val_acc:.4f} "
   395:                     f"test_acc={test_acc:.4f}",
   396:                     flush=True,
   397:                 )
   398: 
   399:             val_loss_history.append(val_loss)
   400:             if EARLY_STOPPING > 0 and epoch > EARLY_STOPPING:
   401:                 recent = torch.tensor(val_loss_history[-(EARLY_STOPPING + 1):-1])
   402:                 if val_loss > recent.mean().item():
   403:                     break
   404: 
   405:         results.append(best_test_acc)
   406:         print(
   407:             f"TRAIN_METRICS run={run_idx} final best_test_acc={best_test_acc:.4f}",
   408:             flush=True,
   409:         )
   410: 
   411:     # Aggregate results across runs
   412:     mean_acc = np.mean(results)
   413:     std_acc = np.std(results)
   414:     print(f"TEST_METRICS accuracy={mean_acc:.4f} std={std_acc:.4f}", flush=True)
   415:     print(
   416:         f"Result: {DATASET_NAME} accuracy = {100*mean_acc:.2f} +/- {100*std_acc:.2f}%",
   417:         flush=True,
   418:     )
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **cora** — wall-clock budget `00:59:00`, compute share `0.25`
- **citeseer** — wall-clock budget `00:59:00`, compute share `0.25`
- **texas** — wall-clock budget `00:59:00`, compute share `0.25`
- **cornell** — wall-clock budget `00:59:00`, compute share `0.25`


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


### `gprgnn` baseline — editable region  [READ-ONLY — reference implementation]

In `ChebNetII/main/custom_filter.py`:

```python
Lines 211–284:
   208: # =====================================================================
   209: # EDITABLE: Custom Graph Signal Propagation Filter
   210: # =====================================================================
   211: class CustomProp(MessagePassing):
   212:     """GPR propagation: learnable polynomial in the monomial basis.
   213: 
   214:     Filter: h(A) = sum_{k=0}^{K} gamma_k * A^k
   215:     where A is the GCN-normalized adjacency and gamma_k are learnable.
   216: 
   217:     Initialized with uniform coefficients (1/(K+1)) so the filter starts
   218:     as an equal-weight average of all hops. This is dataset-agnostic and
   219:     lets the optimizer freely learn both low-pass (homophilic) and
   220:     high-pass (heterophilic) filters.
   221:     """
   222: 
   223:     def __init__(self, K, alpha=0.1, **kwargs):
   224:         super(CustomProp, self).__init__(aggr="add", **kwargs)
   225:         self.K = K
   226:         self.alpha = alpha
   227:         self.temp = Parameter(torch.Tensor(K + 1))
   228:         self.reset_parameters()
   229: 
   230:     def reset_parameters(self):
   231:         # Uniform initialization for dataset-agnostic starting point.
   232:         nn.init.constant_(self.temp, 1.0 / (self.K + 1))
   233: 
   234:     def forward(self, x, edge_index, edge_weight=None):
   235:         edge_index, norm = gcn_norm(
   236:             edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype
   237:         )
   238:         hidden = x * self.temp[0]
   239:         for k in range(self.K):
   240:             x = self.propagate(edge_index, x=x, norm=norm)
   241:             hidden = hidden + self.temp[k + 1] * x
   242:         return hidden
   243: 
   244:     def message(self, x_j, norm):
   245:         return norm.view(-1, 1) * x_j
   246: 
   247: 
   248: class CustomFilter(nn.Module):
   249:     """GPRGNN: Generalized PageRank GNN (Chien et al., 2021).
   250: 
   251:     MLP encoder + learnable monomial polynomial filter.
   252:     """
   253: 
   254:     def __init__(self, num_features, num_classes, hidden=64, K=10,
   255:                  alpha=0.1, dropout=0.5, dprate=0.5):
   256:         super(CustomFilter, self).__init__()
   257:         self.lin1 = Linear(num_features, hidden)
   258:         self.lin2 = Linear(hidden, num_classes)
   259:         self.prop = CustomProp(K, alpha)
   260:         self.dropout = dropout
   261:         self.dprate = 0.0  # GPRGNN paper: no propagation dropout
   262:         # Override training hyperparams (read by template's training loop)
   263:         self.custom_lr = 0.05
   264:         self.custom_wd = 0.0005
   265:         self.custom_prop_lr = 0.05  # same lr for filter coefficients
   266:         self.custom_prop_wd = 0.0
   267: 
   268:     def reset_parameters(self):
   269:         self.lin1.reset_parameters()
   270:         self.lin2.reset_parameters()
   271:         self.prop.reset_parameters()
   272: 
   273:     def forward(self, data):
   274:         x, edge_index = data.x, data.edge_index
   275:         x = F.dropout(x, p=self.dropout, training=self.training)
   276:         x = F.relu(self.lin1(x))
   277:         x = F.dropout(x, p=self.dropout, training=self.training)
   278:         x = self.lin2(x)
   279:         if self.dprate == 0.0:
   280:             x = self.prop(x, edge_index)
   281:         else:
   282:             x = F.dropout(x, p=self.dprate, training=self.training)
   283:             x = self.prop(x, edge_index)
   284:         return F.log_softmax(x, dim=1)
   285: 
   286: 
   287: # =====================================================================
```

### `bernnet` baseline — editable region  [READ-ONLY — reference implementation]

In `ChebNetII/main/custom_filter.py`:

```python
Lines 211–296:
   208: # =====================================================================
   209: # EDITABLE: Custom Graph Signal Propagation Filter
   210: # =====================================================================
   211: class CustomProp(MessagePassing):
   212:     """Bernstein polynomial propagation layer.
   213: 
   214:     Filter: h(L) = sum_{k=0}^{K} theta_k * C(K,k)/2^K * L^k * (2I-L)^{K-k}
   215:     where theta_k = ReLU(learnable), C(K,k) is binomial coefficient,
   216:     and L is the symmetric normalized Laplacian.
   217:     """
   218: 
   219:     def __init__(self, K, alpha=0.1, **kwargs):
   220:         super(CustomProp, self).__init__(aggr="add", **kwargs)
   221:         self.K = K
   222:         self.temp = Parameter(torch.Tensor(K + 1))
   223:         self.reset_parameters()
   224: 
   225:     def reset_parameters(self):
   226:         self.temp.data.fill_(1.0)
   227: 
   228:     def forward(self, x, edge_index, edge_weight=None):
   229:         TEMP = F.relu(self.temp)
   230: 
   231:         # L = I - D^{-1/2}AD^{-1/2}
   232:         edge_index1, norm1 = get_laplacian(
   233:             edge_index, edge_weight, normalization="sym",
   234:             dtype=x.dtype, num_nodes=x.size(self.node_dim)
   235:         )
   236:         # 2I - L
   237:         edge_index2, norm2 = add_self_loops(
   238:             edge_index1, -norm1, fill_value=2.0,
   239:             num_nodes=x.size(self.node_dim)
   240:         )
   241: 
   242:         # Compute (2I-L)^k * x for k = 0, ..., K
   243:         tmp = [x]
   244:         for i in range(self.K):
   245:             x = self.propagate(edge_index2, x=x, norm=norm2, size=None)
   246:             tmp.append(x)
   247: 
   248:         # Bernstein basis evaluation
   249:         out = (comb(self.K, 0) / (2 ** self.K)) * TEMP[0] * tmp[self.K]
   250: 
   251:         for i in range(self.K):
   252:             x = tmp[self.K - i - 1]
   253:             # Apply L^{i+1}
   254:             x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
   255:             for j in range(i):
   256:                 x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
   257:             out = out + (comb(self.K, i + 1) / (2 ** self.K)) * TEMP[i + 1] * x
   258: 
   259:         return out
   260: 
   261:     def message(self, x_j, norm):
   262:         return norm.view(-1, 1) * x_j
   263: 
   264: 
   265: class CustomFilter(nn.Module):
   266:     """BernNet: Bernstein polynomial graph filter (He et al., 2021).
   267: 
   268:     MLP encoder + Bernstein polynomial propagation.
   269:     """
   270: 
   271:     def __init__(self, num_features, num_classes, hidden=64, K=10,
   272:                  alpha=0.1, dropout=0.5, dprate=0.5):
   273:         super(CustomFilter, self).__init__()
   274:         self.lin1 = Linear(num_features, hidden)
   275:         self.lin2 = Linear(hidden, num_classes)
   276:         self.prop = CustomProp(K)
   277:         self.dropout = dropout
   278:         self.dprate = dprate
   279: 
   280:     def reset_parameters(self):
   281:         self.lin1.reset_parameters()
   282:         self.lin2.reset_parameters()
   283:         self.prop.reset_parameters()
   284: 
   285:     def forward(self, data):
   286:         x, edge_index = data.x, data.edge_index
   287:         x = F.dropout(x, p=self.dropout, training=self.training)
   288:         x = F.relu(self.lin1(x))
   289:         x = F.dropout(x, p=self.dropout, training=self.training)
   290:         x = self.lin2(x)
   291:         if self.dprate == 0.0:
   292:             x = self.prop(x, edge_index)
   293:         else:
   294:             x = F.dropout(x, p=self.dprate, training=self.training)
   295:             x = self.prop(x, edge_index)
   296:         return F.log_softmax(x, dim=1)
   297: 
   298: 
   299: # =====================================================================
```

### `chebnetii` baseline — editable region  [READ-ONLY — reference implementation]

In `ChebNetII/main/custom_filter.py`:

```python
Lines 211–304:
   208: # =====================================================================
   209: # EDITABLE: Custom Graph Signal Propagation Filter
   210: # =====================================================================
   211: class CustomProp(MessagePassing):
   212:     """ChebNetII propagation: Chebyshev interpolation filter.
   213: 
   214:     Learns filter values at Chebyshev interpolation nodes, then converts
   215:     to Chebyshev polynomial coefficients. Uses ReLU to ensure non-negative
   216:     interpolation values.
   217: 
   218:     Filter: h(L_tilde) = sum_{k=0}^{K} c_k * T_k(L_tilde)
   219:     where L_tilde = L - I (shifted Laplacian), T_k is the k-th Chebyshev
   220:     polynomial, and c_k are computed from interpolation values via DCT-like transform.
   221:     """
   222: 
   223:     def __init__(self, K, alpha=0.1, **kwargs):
   224:         super(CustomProp, self).__init__(aggr="add", **kwargs)
   225:         self.K = K
   226:         self.temp = Parameter(torch.Tensor(K + 1))
   227:         self.reset_parameters()
   228: 
   229:     def reset_parameters(self):
   230:         self.temp.data.fill_(1.0)
   231: 
   232:     def forward(self, x, edge_index, edge_weight=None):
   233:         coe_tmp = F.relu(self.temp)
   234:         coe = coe_tmp.clone()
   235: 
   236:         # Convert interpolation values to Chebyshev coefficients
   237:         for i in range(self.K + 1):
   238:             coe[i] = coe_tmp[0] * cheby(i, math.cos((self.K + 0.5) * math.pi / (self.K + 1)))
   239:             for j in range(1, self.K + 1):
   240:                 x_j = math.cos((self.K - j + 0.5) * math.pi / (self.K + 1))
   241:                 coe[i] = coe[i] + coe_tmp[j] * cheby(i, x_j)
   242:             coe[i] = 2 * coe[i] / (self.K + 1)
   243: 
   244:         # L = I - D^{-1/2}AD^{-1/2}
   245:         edge_index1, norm1 = get_laplacian(
   246:             edge_index, edge_weight, normalization="sym",
   247:             dtype=x.dtype, num_nodes=x.size(self.node_dim)
   248:         )
   249:         # L_tilde = L - I (shifted to [-1, 1] range)
   250:         edge_index_tilde, norm_tilde = add_self_loops(
   251:             edge_index1, norm1, fill_value=-1.0,
   252:             num_nodes=x.size(self.node_dim)
   253:         )
   254: 
   255:         # Chebyshev recurrence: T_0(x)=x, T_1(x)=x, T_{k+1}=2xT_k - T_{k-1}
   256:         Tx_0 = x
   257:         Tx_1 = self.propagate(edge_index_tilde, x=x, norm=norm_tilde, size=None)
   258: 
   259:         out = coe[0] / 2 * Tx_0 + coe[1] * Tx_1
   260: 
   261:         for i in range(2, self.K + 1):
   262:             Tx_2 = self.propagate(edge_index_tilde, x=Tx_1, norm=norm_tilde, size=None)
   263:             Tx_2 = 2 * Tx_2 - Tx_0
   264:             out = out + coe[i] * Tx_2
   265:             Tx_0, Tx_1 = Tx_1, Tx_2
   266: 
   267:         return out
   268: 
   269:     def message(self, x_j, norm):
   270:         return norm.view(-1, 1) * x_j
   271: 
   272: 
   273: class CustomFilter(nn.Module):
   274:     """ChebNetII: Chebyshev interpolation graph filter (He et al., 2022).
   275: 
   276:     MLP encoder + ChebNetII propagation with Chebyshev interpolation.
   277:     """
   278: 
   279:     def __init__(self, num_features, num_classes, hidden=64, K=10,
   280:                  alpha=0.1, dropout=0.5, dprate=0.5):
   281:         super(CustomFilter, self).__init__()
   282:         self.lin1 = Linear(num_features, hidden)
   283:         self.lin2 = Linear(hidden, num_classes)
   284:         self.prop = CustomProp(K)
   285:         self.dropout = dropout
   286:         self.dprate = dprate
   287: 
   288:     def reset_parameters(self):
   289:         self.lin1.reset_parameters()
   290:         self.lin2.reset_parameters()
   291:         self.prop.reset_parameters()
   292: 
   293:     def forward(self, data):
   294:         x, edge_index = data.x, data.edge_index
   295:         x = F.dropout(x, p=self.dropout, training=self.training)
   296:         x = F.relu(self.lin1(x))
   297:         x = F.dropout(x, p=self.dropout, training=self.training)
   298:         x = self.lin2(x)
   299:         if self.dprate == 0.0:
   300:             x = self.prop(x, edge_index)
   301:         else:
   302:             x = F.dropout(x, p=self.dprate, training=self.training)
   303:             x = self.prop(x, edge_index)
   304:         return F.log_softmax(x, dim=1)
   305: 
   306: 
   307: # =====================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
