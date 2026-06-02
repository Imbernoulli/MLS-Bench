# MLS-Bench: graph-node-classification

# Graph Neural Network: Node Classification Message Passing

## Research Question
Design a novel **message-passing mechanism** for graph neural networks that
improves node-classification performance across citation network benchmarks.

## Background
Graph neural networks learn node representations by iteratively aggregating
information from neighboring nodes through message passing. The core design
choices are:

- **Message construction**: how to compute messages from source to target
  nodes (e.g., linear transform, attention-weighted, edge-conditioned).
- **Aggregation**: how to combine incoming messages (e.g., sum, mean, max,
  attention-weighted).
- **Update**: how to integrate aggregated messages with the node's own
  representation (residual, gated, concatenation, ...).

Classic approaches include GCN (symmetric normalization), GAT (attention-based
weighting), and GraphSAGE (mean aggregation with self/neighbor separation).
Recent advances include Graph Transformers (GPS) that combine local message
passing with global self-attention, and methods like NAGphormer that use
multi-hop tokenization with Transformer encoders.

## Task
Modify the `CustomMessagePassingLayer` class and `CustomGNN` model in
`custom_nodecls.py` to implement a novel message-passing mechanism. Your
implementation must work within PyTorch Geometric's `MessagePassing` framework.

```python
class CustomMessagePassingLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        # learnable parameters and layers
        ...

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        # x: [num_nodes, in_channels], edge_index: [2, num_edges]
        # returns [num_nodes, out_channels]
        ...

    def message(self, x_j: Tensor, ...) -> Tensor:
        # per-edge message computation
        ...


class CustomGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5):
        ...

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        # returns logits [num_nodes, out_channels]
        ...
```

Available PyG utilities:
- `MessagePassing` base class: `self.propagate(edge_index, ...)` orchestrates
  message / aggregate / update.
- `add_self_loops(edge_index)`: add self-loop edges.
- `degree(index, num_nodes)`: compute node degrees.
- `softmax(src, index)`: sparse softmax over edges.
- Reference convolution layers: `GCNConv`, `GATConv`, `SAGEConv`
  (imported but read-only).

## Evaluation
Trained and evaluated on three citation networks (semi-supervised node
classification with standard Planetoid splits):

| Label    | Nodes  | Edges  | Classes | Features |
|----------|--------|--------|---------|----------|
| Cora     | 2,708  | 5,429  | 7       | 1,433    |
| CiteSeer | 3,327  | 4,732  | 6       | 3,703    |
| PubMed   | 19,717 | 44,338 | 3       | 500      |

Fixed training pipeline: 200 epochs with early stopping (patience=50), Adam,
`lr=0.01`, `weight_decay=5e-4`.

Metrics: test accuracy and macro F1, both higher-is-better.

The research contribution should be the GNN propagation/model design rather
than changing the data split, loss target, or evaluation protocol.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/pytorch-geometric/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `pytorch-geometric/custom_nodecls.py`
- editable lines **48–157**




## Readable Context


### `pytorch-geometric/custom_nodecls.py`  [EDITABLE — lines 48–157 only]

```python
     1: # Custom GNN message passing mechanism for node classification — MLS-Bench
     2: #
     3: # EDITABLE section: CustomMessagePassingLayer + CustomGNN classes (lines 48-157).
     4: # FIXED sections: everything else (config, data loading, training loop, evaluation).
     5: 
     6: import os
     7: import copy
     8: import random
     9: import math
    10: from typing import Optional, Tuple
    11: 
    12: import numpy as np
    13: import torch
    14: import torch.nn as nn
    15: import torch.nn.functional as F
    16: from torch import Tensor
    17: 
    18: from torch_geometric.datasets import Planetoid
    19: from torch_geometric.nn import MessagePassing
    20: from torch_geometric.nn.conv import GCNConv, GATConv, SAGEConv
    21: from torch_geometric.utils import add_self_loops, degree, softmax
    22: from torch_geometric.typing import Adj, OptTensor
    23: import torch_geometric.transforms as T
    24: from sklearn.metrics import f1_score
    25: 
    26: 
    27: # =====================================================================
    28: # FIXED: Configuration
    29: # =====================================================================
    30: SEED = int(os.environ.get("SEED", "42"))
    31: OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
    32: DATASET_NAME = os.environ.get("ENV", "Cora")
    33: 
    34: DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    35: 
    36: # Training hyperparameters
    37: HIDDEN_CHANNELS = 64
    38: NUM_LAYERS = 2
    39: DROPOUT = 0.5
    40: LEARNING_RATE = 0.01
    41: WEIGHT_DECAY = 5e-4
    42: EPOCHS = 200
    43: 
    44: 
    45: # =====================================================================
    46: # EDITABLE: Custom Message Passing Layer and GNN Model (lines 48-157)
    47: # =====================================================================
    48: class CustomMessagePassingLayer(MessagePassing):
    49:     """Custom message passing layer for node classification.
    50: 
    51:     This layer defines how messages are constructed, aggregated, and used
    52:     to update node representations. You should implement a novel message
    53:     passing mechanism here.
    54: 
    55:     The PyG MessagePassing base class provides:
    56:         - self.propagate(edge_index, ...): orchestrates message passing
    57:         - Override message(): define how messages are computed per edge
    58:         - Override aggregate(): define how messages are combined (default: 'add')
    59:         - Override update(): define how node embeddings are updated
    60: 
    61:     Args:
    62:         in_channels: input feature dimension
    63:         out_channels: output feature dimension
    64:     """
    65: 
    66:     def __init__(self, in_channels: int, out_channels: int):
    67:         super().__init__(aggr="add")
    68:         self.lin = nn.Linear(in_channels, out_channels, bias=False)
    69:         self.bias = nn.Parameter(torch.zeros(out_channels))
    70:         self.reset_parameters()
    71: 
    72:     def reset_parameters(self):
    73:         nn.init.xavier_uniform_(self.lin.weight)
    74:         nn.init.zeros_(self.bias)
    75: 
    76:     def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
    77:         """Forward pass.
    78: 
    79:         Args:
    80:             x: node feature matrix [num_nodes, in_channels]
    81:             edge_index: graph connectivity [2, num_edges]
    82: 
    83:         Returns:
    84:             Updated node features [num_nodes, out_channels]
    85:         """
    86:         # Transform features
    87:         x = self.lin(x)
    88: 
    89:         # Add self-loops
    90:         edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
    91: 
    92:         # Compute normalization (symmetric like GCN)
    93:         row, col = edge_index
    94:         deg = degree(col, x.size(0), dtype=x.dtype)
    95:         deg_inv_sqrt = deg.pow(-0.5)
    96:         deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    97:         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    98: 
    99:         # Propagate messages
   100:         out = self.propagate(edge_index, x=x, norm=norm)
   101:         out = out + self.bias
   102:         return out
   103: 
   104:     def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
   105:         """Construct messages from source nodes.
   106: 
   107:         Args:
   108:             x_j: source node features [num_edges, out_channels]
   109:             norm: normalization coefficients [num_edges]
   110: 
   111:         Returns:
   112:             Messages [num_edges, out_channels]
   113:         """
   114:         return norm.view(-1, 1) * x_j
   115: 
   116: 
   117: class CustomGNN(nn.Module):
   118:     """GNN model using CustomMessagePassingLayer for node classification.
   119: 
   120:     Must implement __init__ and forward.
   121:     The model receives the full graph and returns logits for each node.
   122: 
   123:     Args:
   124:         in_channels: number of input features per node
   125:         hidden_channels: hidden layer dimension
   126:         out_channels: number of output classes
   127:         num_layers: number of message passing layers
   128:         dropout: dropout probability
   129:     """
   130: 
   131:     def __init__(self, in_channels: int, hidden_channels: int,
   132:                  out_channels: int, num_layers: int = 2,
   133:                  dropout: float = 0.5):
   134:         super().__init__()
   135:         self.dropout = dropout
   136:         self.convs = nn.ModuleList()
   137:         self.convs.append(CustomMessagePassingLayer(in_channels, hidden_channels))
   138:         for _ in range(num_layers - 2):
   139:             self.convs.append(CustomMessagePassingLayer(hidden_channels, hidden_channels))
   140:         self.convs.append(CustomMessagePassingLayer(hidden_channels, out_channels))
   141: 
   142:     def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
   143:         """Forward pass through the GNN.
   144: 
   145:         Args:
   146:             x: node feature matrix [num_nodes, in_channels]
   147:             edge_index: graph connectivity [2, num_edges]
   148: 
   149:         Returns:
   150:             Node classification logits [num_nodes, out_channels]
   151:         """
   152:         for i, conv in enumerate(self.convs[:-1]):
   153:             x = conv(x, edge_index)
   154:             x = F.relu(x)
   155:             x = F.dropout(x, p=self.dropout, training=self.training)
   156:         x = self.convs[-1](x, edge_index)
   157:         return x
   158: 
   159: 
   160: # =====================================================================
   161: # FIXED: Data loading
   162: # =====================================================================
   163: def load_dataset(name: str):
   164:     """Load a Planetoid citation network dataset.
   165: 
   166:     Args:
   167:         name: one of 'Cora', 'CiteSeer', 'PubMed'
   168: 
   169:     Returns:
   170:         dataset, data (moved to device)
   171:     """
   172:     dataset = Planetoid(root=os.environ.get("DATA_ROOT", "/data") + "/Planetoid", name=name,
   173:                         transform=T.NormalizeFeatures())
   174:     data = dataset[0].to(DEVICE)
   175:     return dataset, data
   176: 
   177: 
   178: # =====================================================================
   179: # FIXED: Training and evaluation
   180: # =====================================================================
   181: def train(model, data, optimizer):
   182:     model.train()
   183:     optimizer.zero_grad()
   184:     out = model(data.x, data.edge_index)
   185:     loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
   186:     loss.backward()
   187:     optimizer.step()
   188:     return float(loss)
   189: 
   190: 
   191: @torch.no_grad()
   192: def evaluate(model, data):
   193:     model.eval()
   194:     out = model(data.x, data.edge_index)
   195:     pred = out.argmax(dim=-1)
   196: 
   197:     results = {}
   198:     for split, mask in [("train", data.train_mask),
   199:                         ("val", data.val_mask),
   200:                         ("test", data.test_mask)]:
   201:         correct = int((pred[mask] == data.y[mask]).sum())
   202:         total = int(mask.sum())
   203:         acc = correct / total
   204:         y_true = data.y[mask].cpu().numpy()
   205:         y_pred = pred[mask].cpu().numpy()
   206:         f1 = f1_score(y_true, y_pred, average="macro")
   207:         results[f"{split}_acc"] = acc
   208:         results[f"{split}_f1"] = f1
   209:     return results
   210: 
   211: 
   212: # =====================================================================
   213: # FIXED: Main
   214: # =====================================================================
   215: if __name__ == "__main__":
   216:     # Reproducibility
   217:     random.seed(SEED)
   218:     np.random.seed(SEED)
   219:     torch.manual_seed(SEED)
   220:     torch.cuda.manual_seed_all(SEED)
   221:     torch.backends.cudnn.deterministic = True
   222:     torch.backends.cudnn.benchmark = False
   223: 
   224:     os.makedirs(OUTPUT_DIR, exist_ok=True)
   225: 
   226:     print(f"Dataset: {DATASET_NAME}, Seed: {SEED}", flush=True)
   227: 
   228:     # Load data
   229:     dataset, data = load_dataset(DATASET_NAME)
   230:     in_channels = dataset.num_node_features
   231:     out_channels = dataset.num_classes
   232: 
   233:     print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}, "
   234:           f"Features: {in_channels}, Classes: {out_channels}", flush=True)
   235: 
   236:     # Build model
   237:     model = CustomGNN(
   238:         in_channels=in_channels,
   239:         hidden_channels=HIDDEN_CHANNELS,
   240:         out_channels=out_channels,
   241:         num_layers=NUM_LAYERS,
   242:         dropout=DROPOUT,
   243:     ).to(DEVICE)
   244: 
   245:     num_params = sum(p.numel() for p in model.parameters())
   246:     print(f"Model parameters: {num_params}", flush=True)
   247: 
   248:     # ── Parameter Budget Check ──
   249:     # Budget = 1.05x the largest baseline across all baselines.
   250:     # GraphSAGE uses 2 linear projections per layer (lin_self + lin_neigh),
   251:     # which doubles the input projection cost on high-dim datasets (Cora 1433,
   252:     # CiteSeer 3703), making it the largest baseline on those datasets.
   253:     # GPS uses MultiheadAttention + FFN + 3 LayerNorms per layer.
   254:     # NAGphormer uses Transformer encoder layers.
   255:     # We compute all three and take the max.
   256:     _H = HIDDEN_CHANNELS
   257:     # GraphSAGE: 2 layers, each with lin_self(in,out)+bias + lin_neigh(in,out) no bias
   258:     # Layer 1: 2*in*H + H, Layer 2: 2*H*out + out
   259:     _graphsage_params = (2 * in_channels * _H + _H) + (2 * _H * out_channels + out_channels)
   260:     # GPS: 2 layers + classifier(H*out+out)
   261:     # Layer 1 (in->H): lin_in(in*H+H) + lin_msg(H*H) + lin_update(H*H+H)
   262:     #   + MultiheadAttention(4*H*H+4*H) + FFN(4*H*H+3*H) + 3*LayerNorm(6*H)
   263:     #   = in*H + 10*H*H + 15*H
   264:     # Layer 2 (H->H): no lin_in + same = 10*H*H + 14*H
   265:     _gps_params = (
   266:         in_channels * _H + 10 * _H * _H + 15 * _H  # layer 1
   267:         + 10 * _H * _H + 14 * _H                     # layer 2
   268:         + _H * out_channels + out_channels             # classifier
   269:     )
   270:     # NAGphormer: tokenizer + hop_embed + norm + transformer + attn_vec + classifier
   271:     _nagphormer_params = (
   272:         in_channels * _H + _H            # tokenizer.lin
   273:         + 6 * _H                          # hop_embedding
   274:         + 2 * _H                          # input_norm
   275:         + 16 * _H * _H + 24 * _H         # TransformerEncoder (2 layers + final norm)
   276:         + _H + 1                          # attn_vec
   277:         + _H * _H + _H + _H * out_channels + out_channels  # classifier
   278:     )
   279:     _max_baseline = max(_graphsage_params, _gps_params, _nagphormer_params)
   280:     _param_budget = int(_max_baseline * 1.05)
   281:     print(f"Parameter budget: {num_params:,} / {_param_budget:,} (1.05x largest baseline)", flush=True)
   282: 
   283:     # Allow model to override training hyperparams via attributes
   284:     lr = getattr(model, 'custom_lr', LEARNING_RATE)
   285:     wd = getattr(model, 'custom_wd', WEIGHT_DECAY)
   286:     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
   287: 
   288:     # Training loop
   289:     best_val_acc = 0.0
   290:     best_state = None
   291:     patience = 50
   292:     patience_counter = 0
   293: 
   294:     for epoch in range(1, EPOCHS + 1):
   295:         loss = train(model, data, optimizer)
   296:         results = evaluate(model, data)
   297: 
   298:         if epoch % 10 == 0 or epoch == 1:
   299:             print(f"TRAIN_METRICS epoch={epoch} loss={loss:.4f} "
   300:                   f"train_acc={results['train_acc']:.4f} "
   301:                   f"val_acc={results['val_acc']:.4f} "
   302:                   f"test_acc={results['test_acc']:.4f}", flush=True)
   303: 
   304:         if results["val_acc"] > best_val_acc:
   305:             best_val_acc = results["val_acc"]
   306:             best_state = copy.deepcopy(model.state_dict())
   307:             patience_counter = 0
   308:         else:
   309:             patience_counter += 1
   310: 
   311:         if patience_counter >= patience:
   312:             print(f"Early stopping at epoch {epoch}", flush=True)
   313:             break
   314: 
   315:     # Load best model and final evaluation
   316:     model.load_state_dict(best_state)
   317:     final = evaluate(model, data)
   318: 
   319:     print(f"TEST_METRICS accuracy={final['test_acc']:.4f} "
   320:           f"macro_f1={final['test_f1']:.4f}", flush=True)
   321:     print(f"Final test accuracy: {100 * final['test_acc']:.2f}%", flush=True)
   322:     print(f"Final test macro F1: {100 * final['test_f1']:.2f}%", flush=True)
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


### `gcn` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-geometric/custom_nodecls.py`:

```python
Lines 48–97:
    45: # =====================================================================
    46: # EDITABLE: Custom Message Passing Layer and GNN Model (lines 48-157)
    47: # =====================================================================
    48: class CustomMessagePassingLayer(MessagePassing):
    49:     """GCN baseline: standard graph convolutional layer."""
    50: 
    51:     def __init__(self, in_channels: int, out_channels: int):
    52:         super().__init__(aggr="add")
    53:         self.lin = nn.Linear(in_channels, out_channels, bias=False)
    54:         self.bias = nn.Parameter(torch.zeros(out_channels))
    55:         self.reset_parameters()
    56: 
    57:     def reset_parameters(self):
    58:         nn.init.xavier_uniform_(self.lin.weight)
    59:         nn.init.zeros_(self.bias)
    60: 
    61:     def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
    62:         x = self.lin(x)
    63:         edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
    64:         row, col = edge_index
    65:         deg = degree(col, x.size(0), dtype=x.dtype)
    66:         deg_inv_sqrt = deg.pow(-0.5)
    67:         deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    68:         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    69:         out = self.propagate(edge_index, x=x, norm=norm)
    70:         out = out + self.bias
    71:         return out
    72: 
    73:     def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
    74:         return norm.view(-1, 1) * x_j
    75: 
    76: 
    77: class CustomGNN(nn.Module):
    78:     """GCN model: 2-layer GCN with ReLU and dropout."""
    79: 
    80:     def __init__(self, in_channels: int, hidden_channels: int,
    81:                  out_channels: int, num_layers: int = 2,
    82:                  dropout: float = 0.5):
    83:         super().__init__()
    84:         self.dropout = dropout
    85:         self.convs = nn.ModuleList()
    86:         self.convs.append(CustomMessagePassingLayer(in_channels, hidden_channels))
    87:         for _ in range(num_layers - 2):
    88:             self.convs.append(CustomMessagePassingLayer(hidden_channels, hidden_channels))
    89:         self.convs.append(CustomMessagePassingLayer(hidden_channels, out_channels))
    90: 
    91:     def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
    92:         for i, conv in enumerate(self.convs[:-1]):
    93:             x = conv(x, edge_index)
    94:             x = F.relu(x)
    95:             x = F.dropout(x, p=self.dropout, training=self.training)
    96:         x = self.convs[-1](x, edge_index)
    97:         return x
    98: 
    99: 
   100: # =====================================================================
```

### `gat` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-geometric/custom_nodecls.py`:

```python
Lines 48–133:
    45: # =====================================================================
    46: # EDITABLE: Custom Message Passing Layer and GNN Model (lines 48-157)
    47: # =====================================================================
    48: class CustomMessagePassingLayer(MessagePassing):
    49:     """GAT baseline: graph attention layer with multi-head attention."""
    50: 
    51:     def __init__(self, in_channels: int, out_channels: int,
    52:                  heads: int = 8, concat: bool = True,
    53:                  negative_slope: float = 0.2):
    54:         super().__init__(aggr="add", node_dim=0)
    55:         self.heads = heads
    56:         self.concat = concat
    57:         self.negative_slope = negative_slope
    58: 
    59:         if concat:
    60:             assert out_channels % heads == 0
    61:             self.head_dim = out_channels // heads
    62:         else:
    63:             self.head_dim = out_channels
    64: 
    65:         self.lin = nn.Linear(in_channels, heads * self.head_dim, bias=False)
    66:         self.att_src = nn.Parameter(torch.empty(1, heads, self.head_dim))
    67:         self.att_dst = nn.Parameter(torch.empty(1, heads, self.head_dim))
    68:         self.bias = nn.Parameter(torch.zeros(heads * self.head_dim if concat else out_channels))
    69:         self.reset_parameters()
    70: 
    71:     def reset_parameters(self):
    72:         nn.init.xavier_uniform_(self.lin.weight)
    73:         nn.init.xavier_uniform_(self.att_src)
    74:         nn.init.xavier_uniform_(self.att_dst)
    75:         nn.init.zeros_(self.bias)
    76: 
    77:     def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
    78:         H, D = self.heads, self.head_dim
    79:         x = self.lin(x).view(-1, H, D)
    80: 
    81:         # Compute attention coefficients
    82:         alpha_src = (x * self.att_src).sum(dim=-1)  # [N, H]
    83:         alpha_dst = (x * self.att_dst).sum(dim=-1)  # [N, H]
    84: 
    85:         edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
    86:         out = self.propagate(edge_index, x=x,
    87:                              alpha_src=alpha_src, alpha_dst=alpha_dst)
    88: 
    89:         if self.concat:
    90:             out = out.view(-1, H * D)
    91:         else:
    92:             out = out.mean(dim=1)
    93: 
    94:         out = out + self.bias
    95:         return out
    96: 
    97:     def message(self, x_j: Tensor, alpha_src_i: Tensor,
    98:                 alpha_dst_j: Tensor, index: Tensor,
    99:                 ptr: OptTensor, size_i: Optional[int]) -> Tensor:
   100:         alpha = alpha_src_i + alpha_dst_j
   101:         alpha = F.leaky_relu(alpha, self.negative_slope)
   102:         alpha = softmax(alpha, index, ptr, size_i)
   103:         alpha = F.dropout(alpha, p=0.6, training=self.training)
   104:         return x_j * alpha.unsqueeze(-1)
   105: 
   106: 
   107: class CustomGNN(nn.Module):
   108:     """GAT model: multi-head attention GNN."""
   109: 
   110:     def __init__(self, in_channels: int, hidden_channels: int,
   111:                  out_channels: int, num_layers: int = 2,
   112:                  dropout: float = 0.6):
   113:         super().__init__()
   114:         self.dropout = dropout
   115:         self.convs = nn.ModuleList()
   116:         # First layer: 8 heads, concat
   117:         self.convs.append(CustomMessagePassingLayer(
   118:             in_channels, hidden_channels, heads=8, concat=True))
   119:         for _ in range(num_layers - 2):
   120:             self.convs.append(CustomMessagePassingLayer(
   121:                 hidden_channels, hidden_channels, heads=8, concat=True))
   122:         # Last layer: 1 head, no concat (average)
   123:         self.convs.append(CustomMessagePassingLayer(
   124:             hidden_channels, out_channels, heads=1, concat=False))
   125: 
   126:     def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
   127:         for i, conv in enumerate(self.convs[:-1]):
   128:             x = F.dropout(x, p=self.dropout, training=self.training)
   129:             x = conv(x, edge_index)
   130:             x = F.elu(x)
   131:         x = F.dropout(x, p=self.dropout, training=self.training)
   132:         x = self.convs[-1](x, edge_index)
   133:         return x
   134: 
   135: 
   136: # =====================================================================
```

### `graphsage` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-geometric/custom_nodecls.py`:

```python
Lines 48–94:
    45: # =====================================================================
    46: # EDITABLE: Custom Message Passing Layer and GNN Model (lines 48-157)
    47: # =====================================================================
    48: class CustomMessagePassingLayer(MessagePassing):
    49:     """GraphSAGE baseline: mean-aggregation message passing."""
    50: 
    51:     def __init__(self, in_channels: int, out_channels: int):
    52:         super().__init__(aggr="mean")
    53:         self.lin_self = nn.Linear(in_channels, out_channels, bias=True)
    54:         self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
    55:         self.reset_parameters()
    56: 
    57:     def reset_parameters(self):
    58:         nn.init.xavier_uniform_(self.lin_self.weight)
    59:         nn.init.xavier_uniform_(self.lin_neigh.weight)
    60:         nn.init.zeros_(self.lin_self.bias)
    61: 
    62:     def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
    63:         # Aggregate neighbor features (mean)
    64:         neigh_agg = self.propagate(edge_index, x=x)
    65:         # Combine self and neighbor
    66:         out = self.lin_self(x) + self.lin_neigh(neigh_agg)
    67:         out = F.normalize(out, p=2, dim=-1)
    68:         return out
    69: 
    70:     def message(self, x_j: Tensor) -> Tensor:
    71:         return x_j
    72: 
    73: 
    74: class CustomGNN(nn.Module):
    75:     """GraphSAGE model: mean-aggregation GNN with L2 normalization."""
    76: 
    77:     def __init__(self, in_channels: int, hidden_channels: int,
    78:                  out_channels: int, num_layers: int = 2,
    79:                  dropout: float = 0.5):
    80:         super().__init__()
    81:         self.dropout = dropout
    82:         self.convs = nn.ModuleList()
    83:         self.convs.append(CustomMessagePassingLayer(in_channels, hidden_channels))
    84:         for _ in range(num_layers - 2):
    85:             self.convs.append(CustomMessagePassingLayer(hidden_channels, hidden_channels))
    86:         self.convs.append(CustomMessagePassingLayer(hidden_channels, out_channels))
    87: 
    88:     def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
    89:         for i, conv in enumerate(self.convs[:-1]):
    90:             x = conv(x, edge_index)
    91:             x = F.relu(x)
    92:             x = F.dropout(x, p=self.dropout, training=self.training)
    93:         x = self.convs[-1](x, edge_index)
    94:         return x
    95: 
    96: 
    97: # =====================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
