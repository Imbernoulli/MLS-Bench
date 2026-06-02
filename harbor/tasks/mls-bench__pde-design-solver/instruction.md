# MLS-Bench: pde-design-solver

# Industrial CFD Design: Custom Neural Operator Design

## Objective
Design and implement a custom neural operator for industrial aerodynamic design prediction on 3D unstructured point clouds. Your code goes in the `Model` class in `models/Custom.py`. Reference implementations (PointNet, GraphSAGE, Graph_UNet, Transolver) from Neural-Solver-Library are provided as read-only context.

## Background
The task targets point-cloud / mesh-based neural operators for steady aerodynamic design prediction. Key reference architectures:

- **PointNet** (Qi, Su, Mo, Guibas, "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation", CVPR 2017; arXiv:1612.00593). Per-point MLP with global max-pooling for permutation invariance over point sets. Code: https://github.com/charlesq34/pointnet.
- **GraphSAGE** (Hamilton, Ying, Leskovec, "Inductive Representation Learning on Large Graphs", NeurIPS 2017; arXiv:1706.02216). Inductive node embeddings via sample-and-aggregate over local neighborhoods; used here for message passing on the mesh graph.
- **Graph U-Net** (Gao, Ji, "Graph U-Nets", ICML 2019; arXiv:1905.05178). Encoder-decoder with learnable graph pooling (gPool) and unpooling (gUnpool) operations.
- **Transolver** (Wu, Luo, Wang, Wang, Long, "Transolver: A Fast Transformer Solver for PDEs on General Geometries", ICML 2024; arXiv:2402.02366). Physics-Attention that adaptively splits the discretized domain into learnable slices and computes attention among physical states rather than mesh points. Code: https://github.com/thuml/Transolver.

## Model Interface
Your model receives `args` at initialization and must implement:
```python
forward(self, x, fx, T=None, geo=None) -> output
```
- `x`: 3D spatial coordinates, shape `(1, N, 3)` where N varies per mesh (~5000–10000 points).
- `fx`: input features (boundary conditions + geometry), shape `(1, N, 7)`.
- `T`: unused (always `None`).
- `geo`: **edge_index** tensor for graph connectivity between mesh points (required for graph-based models, can be `None` for non-graph approaches).
- output: predicted flow field, shape `(1, N, out_dim)` (velocity + pressure components), where `out_dim` is provided via `args.out_dim`.

**Note**: Batch size is always 1 (one mesh per forward pass). Graph models (PointNet, GraphSAGE, Graph_UNet) squeeze the batch dimension and use `geo` for message passing. Non-graph models like Transolver ignore `geo`.

Key `args` attributes: `n_hidden`, `n_layers`, `n_heads`, `space_dim` (spatial dimensionality of the mesh), `fun_dim=7`, `out_dim` (number of predicted flow-field channels), `act`, `mlp_ratio`, `dropout`, `geotype` (`unstructured`), `radius` (for graph construction), `slice_num` (for Transolver-style physics attention).

## Hyperparameter Override (`CONFIG_OVERRIDES`)
The per-dataset shell scripts under `scripts/` default to `--n_hidden 128 --slice_num 32`. Different model families need different widths to be competitive — for example Transolver uses 256 in the original paper, while PointNet and Graph_UNet typically use much smaller widths. To set per-method values, edit the `CONFIG_OVERRIDES` dict at the bottom of `models/Custom.py`:

```python
# Allowed keys: n_hidden (int), slice_num (int).
CONFIG_OVERRIDES = {'n_hidden': 256, 'slice_num': 32}
```

Allowed keys are restricted to `n_hidden` and `slice_num`. The shell scripts read these from your file at runtime and pass them through as `--n_hidden` and `--slice_num`.

## Fixed Pipeline
Dataset loaders, OneCycleLR training schedule, loss function, metric computation, and the parameter budget check are all fixed. Only the `Model` class and the two `CONFIG_OVERRIDES` knobs are editable.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/Neural-Solver-Library/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `Neural-Solver-Library/models/Custom.py`
- editable lines **1–64**
- editable lines **74–74**
- `Neural-Solver-Library/layers/Basic.py`
- editable: **entire file**
- `Neural-Solver-Library/layers/Embedding.py`
- editable: **entire file**
- `Neural-Solver-Library/layers/Physics_Attention.py`
- editable: **entire file**


Other files you may **read** for context (do not modify):
- `Neural-Solver-Library/models/Transolver.py`
- `Neural-Solver-Library/models/PointNet.py`
- `Neural-Solver-Library/models/GraphSAGE.py`
- `Neural-Solver-Library/models/Graph_UNet.py`
- `Neural-Solver-Library/models/GNOT.py`


## Readable Context


### `Neural-Solver-Library/models/Custom.py`  [EDITABLE — lines 1–64, lines 74–74 only]

```python
     1: import torch
     2: import torch.nn as nn
     3: import numpy as np
     4: from timm.models.layers import trunc_normal_
     5: from layers.Basic import MLP
     6: from layers.Embedding import unified_pos_embedding
     7: 
     8: 
     9: class Model(nn.Module):
    10:     def __init__(self, args):
    11:         super(Model, self).__init__()
    12:         self.__name__ = 'Custom'
    13:         self.args = args
    14: 
    15:         # Input encoding: spatial coords (3D) + features (7D) -> hidden_dim
    16:         self.encoder = MLP(args.fun_dim + args.space_dim, args.n_hidden * 2, args.n_hidden,
    17:                            n_layers=0, res=False, act=args.act)
    18: 
    19:         # TODO: Define your custom model architecture here.
    20:         # This model operates on UNSTRUCTURED 3D point clouds (car meshes).
    21:         # Each mesh has variable number of points (~5000-10000).
    22:         # Batch size is always 1.
    23:         # args.geotype = 'unstructured'
    24:         #
    25:         # You can use:
    26:         # - Graph neural networks (edge_index available via geo parameter)
    27:         # - Point cloud methods (PointNet-style global pooling)
    28:         # - Transformer-based approaches (self-attention on all points)
    29:         # - Physics-aware methods (Transolver-style slicing)
    30:         #
    31:         # Reference models: PointNet (global pooling), GraphSAGE (message passing),
    32:         # Graph_UNet (multi-scale graph), Transolver (physics attention)
    33: 
    34:         # Output projection: hidden_dim -> out_dim (velocity xyz + pressure)
    35:         self.decoder = MLP(args.n_hidden, args.n_hidden * 2, args.out_dim,
    36:                            n_layers=0, res=False, act=args.act)
    37: 
    38:         self.initialize_weights()
    39: 
    40:     def initialize_weights(self):
    41:         self.apply(self._init_weights)
    42: 
    43:     def _init_weights(self, m):
    44:         if isinstance(m, nn.Linear):
    45:             trunc_normal_(m.weight, std=0.02)
    46:             if isinstance(m, nn.Linear) and m.bias is not None:
    47:                 nn.init.constant_(m.bias, 0)
    48:         elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
    49:             nn.init.constant_(m.bias, 0)
    50:             nn.init.constant_(m.weight, 1.0)
    51: 
    52:     def forward(self, x, fx, T=None, geo=None):
    53:         # x: (1, N, 3) spatial coords, fx: (1, N, 7) features
    54:         # geo: edge_index tensor if graph connectivity is needed (can be None)
    55:         z = torch.cat((x, fx), dim=-1)  # (1, N, 10)
    56:         z = self.encoder(z)  # (1, N, n_hidden)
    57: 
    58:         # TODO: Implement your custom forward pass here.
    59:         # Input z has shape (1, N, n_hidden) where N varies per mesh.
    60:         # Output should have shape (1, N, out_dim) where out_dim=4.
    61: 
    62:         out = self.decoder(z)  # (1, N, 4)
    63:         return out
    64: 
    65: 
    66: # =====================================================================
    67: # CONFIG_OVERRIDES: per-method hyperparameter overrides
    68: # =====================================================================
    69: # Override widths/capacities that depend on the model family.
    70: # Allowed keys: n_hidden (int), slice_num (int).
    71: # Defaults follow the baseline shell scripts (n_hidden=128, slice_num=32),
    72: # matching the GraphSAGE configuration in Neural-Solver-Library/scripts/DesignBench/car/.
    73: # Other paper settings (for reference): PointNet=16, Transolver=256, Graph_UNet=16, GNOT=256.
    74: CONFIG_OVERRIDES = {}
    75: 
    76: 
    77: # =====================================================================
    78: # FIXED: Parameter budget check — do not modify below this line
    79: # =====================================================================
    80: _orig_init = Model.__init__
    81: 
    82: def _patched_init(self, args):
    83:     _orig_init(self, args)
    84:     _total = sum(p.numel() for p in self.parameters())
    85:     print(f"Total params: {_total:,} (task budget enforced by budget_check.py)")
    86: 
    87: Model.__init__ = _patched_init
```

### `Neural-Solver-Library/layers/Basic.py`  [EDITABLE — entire file only]

```python
     1: import torch
     2: import torch.nn as nn
     3: import torch.nn.functional as F
     4: import numpy as np
     5: from timm.models.layers import trunc_normal_
     6: from einops import rearrange, repeat
     7: from torch import einsum
     8: from functools import partial, reduce
     9: 
    10: ACTIVATION = {
    11:     'gelu': nn.GELU,
    12:     'tanh': nn.Tanh,
    13:     'sigmoid': nn.Sigmoid,
    14:     'relu': nn.ReLU,
    15:     'leaky_relu': nn.LeakyReLU(0.1),
    16:     'softplus': nn.Softplus,
    17:     'ELU': nn.ELU,
    18:     'silu': nn.SiLU
    19: }
    20: 
    21: 
    22: class MLP(nn.Module):
    23:     def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
    24:         super(MLP, self).__init__()
    25: 
    26:         if act in ACTIVATION.keys():
    27:             act = ACTIVATION[act]
    28:         else:
    29:             raise NotImplementedError
    30:         self.n_input = n_input
    31:         self.n_hidden = n_hidden
    32:         self.n_output = n_output
    33:         self.n_layers = n_layers
    34:         self.res = res
    35:         self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
    36:         self.linear_post = nn.Linear(n_hidden, n_output)
    37:         self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])
    38: 
    39:     def forward(self, x):
    40:         x = self.linear_pre(x)
    41:         for i in range(self.n_layers):
    42:             if self.res:
    43:                 x = self.linears[i](x) + x
    44:             else:
    45:                 x = self.linears[i](x)
    46:         x = self.linear_post(x)
    47:         return x
    48: 
    49: 
    50: class PreNorm(nn.Module):
    51:     def __init__(self, dim, fn):
    52:         super().__init__()
    53:         self.norm = nn.LayerNorm(dim)
    54:         self.fn = fn
    55: 
    56:     def forward(self, x, **kwargs):
    57:         return self.fn(self.norm(x), **kwargs)
    58: 
    59: 
    60: class Attention(nn.Module):
    61:     def __init__(self, dim, heads=8, dim_head=64, dropout=0., **kwargs):
    62:         super().__init__()
    63:         inner_dim = dim_head * heads
    64:         self.dim_head = dim_head
    65:         self.heads = heads
    66:         self.scale = dim_head ** -0.5
    67:         self.softmax = nn.Softmax(dim=-1)
    68:         self.dropout = nn.Dropout(dropout)
    69:         self.to_q = nn.Linear(dim_head, dim_head, bias=False)
    70:         self.to_k = nn.Linear(dim_head, dim_head, bias=False)
    71:         self.to_v = nn.Linear(dim_head, dim_head, bias=False)
    72:         self.to_out = nn.Sequential(
    73:             nn.Linear(inner_dim, dim),
    74:             nn.Dropout(dropout)
    75:         )
    76: 
    77:     def forward(self, x):
    78:         # B N C
    79:         B, N, C = x.shape
    80:         x = x.reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()  # B H N C
    81:         q = self.to_q(x)
    82:         k = self.to_k(x)
    83:         v = self.to_v(x)
    84:         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
    85:         attn = self.softmax(dots)
    86:         attn = self.dropout(attn)
    87:         res = torch.matmul(attn, v)  # B H G D
    88:         res = rearrange(res, 'b h n d -> b n (h d)')
    89:         return self.to_out(res)
    90: 
    91: class FlashAttention(nn.Module):
    92:     def __init__(self, dim, heads=8, dim_head=64, dropout=0., **kwargs):
    93:         super().__init__()
    94:         inner_dim = dim_head * heads
    95:         self.dim_head = dim_head
    96:         self.heads = heads
    97:         self.scale = dim_head ** -0.5
    98:         self.dropout = nn.Dropout(dropout)
    99:         
   100:         # Separate projection layers for query, key, and value
   101:         self.to_q = nn.Linear(dim, inner_dim, bias=False)
   102:         self.to_k = nn.Linear(dim, inner_dim, bias=False)
   103:         self.to_v = nn.Linear(dim, inner_dim, bias=False)
   104:         self.to_out = nn.Sequential(
   105:             nn.Linear(inner_dim, dim),
   106:             nn.Dropout(dropout)
   107:         )
   108: 
   109:     def forward(self, x):
   110:         # x shape: [batch_size, seq_len, dim]
   111:         batch_size, seq_len, _ = x.shape
   112:         
   113:         # Get query, key, value projections for all heads
   114:         q = self.to_q(x)
   115:         k = self.to_k(x)
   116:         v = self.to_v(x)
   117:         
   118:         # Reshape for multi-head attention
   119:         q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
   120:         k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
   121:         v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
   122:         
   123:         # Flash attention implementation
   124:         attn_output = F.scaled_dot_product_attention(
   125:             q, k, v,
   126:             dropout_p=self.dropout.p if self.training else 0.0,
   127:         )
   128:         out = rearrange(attn_output, 'b h n d -> b n (h d)')
   129:         return self.to_out(out)
   130: 
   131: class Vanilla_Linear_Attention(nn.Module):
   132:     def __init__(self, dim, heads=8, dim_head=64, dropout=0., **kwargs):
   133:         super().__init__()
   134:         inner_dim = dim_head * heads
   135:         self.dim_head = dim_head
   136:         self.heads = heads
   137:         self.softmax = nn.Softmax(dim=-1)
   138:         self.dropout = nn.Dropout(dropout)
   139:         self.to_q = nn.Linear(dim_head, dim_head, bias=False)
   140:         self.to_k = nn.Linear(dim_head, dim_head, bias=False)
   141:         self.to_v = nn.Linear(dim_head, dim_head, bias=False)
   142:         self.to_out = nn.Sequential(
   143:             nn.Linear(inner_dim, dim),
   144:             nn.Dropout(dropout)
   145:         )
   146: 
   147:     def forward(self, x):
   148:         # B N C
   149:         B, N, C = x.shape
   150:         x = x.reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()  # B H N C
   151:         q = self.to_q(x)
   152:         k = self.to_k(x)
   153:         v = self.to_v(x)
   154:         dots = torch.matmul(k.transpose(-1, -2), v) / float(N)
   155:         dots = self.dropout(dots)
   156:         res = torch.matmul(q, dots)  # B H G D
   157:         res = rearrange(res, 'b h n d -> b n (h d)')
   158:         return self.to_out(res)
   159: 
   160: 
   161: class LinearAttention(nn.Module):
   162:     """
   163:     modified from https://github.com/HaoZhongkai/GNOT/blob/master/models/mmgpt.py
   164:     """
   165: 
   166:     def __init__(self, dim, heads=8, dim_head=64, dropout=0., attn_type='l1', **kwargs):
   167:         super(LinearAttention, self).__init__()
   168:         self.key = nn.Linear(dim, dim)
   169:         self.query = nn.Linear(dim, dim)
   170:         self.value = nn.Linear(dim, dim)
   171:         # regularization
   172:         self.attn_drop = nn.Dropout(dropout)
   173:         # output projection
   174:         self.proj = nn.Linear(dim, dim)
   175:         self.n_head = heads
   176:         self.dim_head = dim_head
   177:         self.attn_type = attn_type
   178: 
   179:     def forward(self, x, y=None):
   180:         y = x if y is None else y
   181:         B, T1, C = x.size()
   182:         _, T2, _ = y.size()
   183:         q = self.query(x).view(B, T1, self.n_head, self.dim_head).transpose(1, 2)  # (B, nh, T, hs)
   184:         k = self.key(y).view(B, T2, self.n_head, self.dim_head).transpose(1, 2)  # (B, nh, T, hs)
   185:         v = self.value(y).view(B, T2, self.n_head, self.dim_head).transpose(1, 2)  # (B, nh, T, hs)
   186: 
   187:         if self.attn_type == 'l1':
   188:             q = q.softmax(dim=-1)
   189:             k = k.softmax(dim=-1)
   190:             k_cumsum = k.sum(dim=-2, keepdim=True)
   191:             D_inv = 1. / (q * k_cumsum).sum(dim=-1, keepdim=True)  # normalized
   192:         elif self.attn_type == "galerkin":
   193:             q = q.softmax(dim=-1)
   194:             k = k.softmax(dim=-1)
   195:             D_inv = 1. / T2
   196:         elif self.attn_type == "l2":  # still use l1 normalization
   197:             q = q / q.norm(dim=-1, keepdim=True, p=1)
   198:             k = k / k.norm(dim=-1, keepdim=True, p=1)
   199:             k_cumsum = k.sum(dim=-2, keepdim=True)
   200:             D_inv = 1. / (q * k_cumsum).abs().sum(dim=-1, keepdim=True)  # normalized
   201:         else:
   202:             raise NotImplementedError
   203: 
   204:         context = k.transpose(-2, -1) @ v
   205:         y = self.attn_drop((q @ context) * D_inv + q)
   206: 
   207:         # output projection
   208:         y = rearrange(y, 'b h n d -> b n (h d)')
   209:         y = self.proj(y)
   210:         return y
   211: 
   212: def exists(val):
   213:     return val is not None
   214: 
   215: def default(value, d):
   216:     return d if not exists(value) else value
   217: 
   218: def max_neg_value(tensor):
   219:     return -torch.finfo(tensor.dtype).max
   220: 
   221: def linear_attn(q, k, v, kv_mask = None):
   222:     dim = q.shape[-1]
   223: 
   224:     if exists(kv_mask):
   225:         mask_value = max_neg_value(q)
   226:         mask = kv_mask[:, None, :, None]
   227:         k = k.masked_fill_(~mask, mask_value)
   228:         v = v.masked_fill_(~mask, 0.)
   229:         del mask
   230: 
   231:     q = q.softmax(dim=-1)
   232:     k = k.softmax(dim=-2)
   233: 
   234:     q = q * dim ** -0.5
   235: 
   236:     context = einsum('bhnd,bhne->bhde', k, v)
   237:     attn = einsum('bhnd,bhde->bhne', q, context)
   238:     return attn.reshape(*q.shape)
   239: 
   240: def split_at_index(dim, index, t):
   241:     pre_slices = (slice(None),) * dim
   242:     l = (*pre_slices, slice(None, index))
   243:     r = (*pre_slices, slice(index, None))
   244:     return t[l], t[r]
   245: 
   246: class SelfAttention(nn.Module):
   247:     def __init__(self, dim, heads, dim_head = None,dropout = 0.):
   248:         super().__init__()
   249:         assert dim_head or (dim % heads) == 0, 'embedding dimension must be divisible by number of heads'
   250:         d_heads = default(dim_head, dim // heads)
   251: 
   252:         self.heads = heads
   253:         self.d_heads = d_heads
   254: 
   255:         self.global_attn_heads = heads
   256:         self.global_attn_fn = linear_attn
   257:         
   258: 
   259:         self.to_q = nn.Linear(dim, d_heads * heads, bias = False)
   260: 
   261:         kv_heads = heads
   262: 
   263:         self.kv_heads = kv_heads
   264:         self.to_k = nn.Linear(dim, d_heads * kv_heads, bias = False)
   265:         self.to_v = nn.Linear(dim, d_heads * kv_heads, bias = False)
   266: 
   267:         self.to_out = nn.Linear(d_heads * heads, dim)
   268:         self.dropout = nn.Dropout(dropout)
   269: 
   270:     def forward(self, x):
   271:         q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))
   272: 
   273:         b, t, e, h, dh = *q.shape, self.heads, self.d_heads
   274: 
   275:         merge_heads = lambda x: x.reshape(*x.shape[:2], -1, dh).transpose(1, 2)
   276: 
   277:         q, k, v = map(merge_heads, (q, k, v))
   278: 
   279:         out = []
   280: 
   281:         split_index_fn = partial(split_at_index, 1, 0)
   282: 
   283:         (lq, q), (lk, k), (lv, v) = map(split_index_fn, (q, k, v))
   284: 
   285:         _, has_global = map(lambda x: x.shape[1] > 0, (lq, q))
   286: 
   287:         if has_global:
   288:             global_out = self.global_attn_fn(q, k, v)
   289:             out.append(global_out)
   290: 
   291:         attn = torch.cat(out, dim=1)
   292:         attn = attn.transpose(1, 2).reshape(b, t, -1)
   293:         return self.dropout(self.to_out(attn))
```

### `Neural-Solver-Library/layers/Embedding.py`  [EDITABLE — entire file only]

```python
     1: import math
     2: import torch
     3: import torch.nn as nn
     4: from einops import rearrange
     5: import numpy as np
     6: 
     7: 
     8: def unified_pos_embedding(shapelist, ref, batchsize=1, device='cuda'):
     9:     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    10:     if len(shapelist) == 1:
    11:         size_x = shapelist[0]
    12:         gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    13:         grid = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1]).to(device)  # B N 1
    14:         gridx = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
    15:         grid_ref = gridx.reshape(1, ref, 1).repeat([batchsize, 1, 1]).to(device)  # B N 1
    16:         pos = torch.sqrt(torch.sum((grid[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1)). \
    17:             reshape(batchsize, size_x, ref).contiguous()
    18:     if len(shapelist) == 2:
    19:         size_x, size_y = shapelist[0], shapelist[1]
    20:         gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    21:         gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    22:         gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    23:         gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    24:         grid = torch.cat((gridx, gridy), dim=-1).to(device)  # B H W 2
    25: 
    26:         gridx = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
    27:         gridx = gridx.reshape(1, ref, 1, 1).repeat([batchsize, 1, ref, 1])
    28:         gridy = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
    29:         gridy = gridy.reshape(1, 1, ref, 1).repeat([batchsize, ref, 1, 1])
    30:         grid_ref = torch.cat((gridx, gridy), dim=-1).to(device)  # B H W 8 8 2
    31: 
    32:         pos = torch.sqrt(torch.sum((grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :]) ** 2, dim=-1)). \
    33:             reshape(batchsize, size_x * size_y, ref * ref).contiguous()
    34:     if len(shapelist) == 3:
    35:         size_x, size_y, size_z = shapelist[0], shapelist[1], shapelist[2]
    36:         gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    37:         gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
    38:         gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    39:         gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
    40:         gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
    41:         gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
    42:         grid = torch.cat((gridx, gridy, gridz), dim=-1).to(device)  # B H W D 3
    43: 
    44:         gridx = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
    45:         gridx = gridx.reshape(1, ref, 1, 1, 1).repeat([batchsize, 1, ref, ref, 1])
    46:         gridy = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
    47:         gridy = gridy.reshape(1, 1, ref, 1, 1).repeat([batchsize, ref, 1, ref, 1])
    48:         gridz = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
    49:         gridz = gridz.reshape(1, 1, 1, ref, 1).repeat([batchsize, ref, ref, 1, 1])
    50:         grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).to(device)  # B 4 4 4 3
    51: 
    52:         pos = torch.sqrt(
    53:             torch.sum((grid[:, :, :, :, None, None, None, :] - grid_ref[:, None, None, None, :, :, :, :]) ** 2,
    54:                       dim=-1)). \
    55:             reshape(batchsize, size_x * size_y * size_z, ref * ref * ref).contiguous()
    56:     return pos
    57: 
    58: 
    59: class RotaryEmbedding(nn.Module):
    60:     def __init__(self, dim, min_freq=1 / 2, scale=1.):
    61:         super().__init__()
    62:         inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    63:         self.min_freq = min_freq
    64:         self.scale = scale
    65:         self.register_buffer('inv_freq', inv_freq)
    66: 
    67:     def forward(self, coordinates, device='cuda'):
    68:         # coordinates [b, n]
    69:         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    70:         t = coordinates.to(device).type_as(self.inv_freq)
    71:         t = t * (self.scale / self.min_freq)
    72:         freqs = torch.einsum('... i , j -> ... i j', t, self.inv_freq)  # [b, n, d//2]
    73:         return torch.cat((freqs, freqs), dim=-1)  # [b, n, d]
    74: 
    75: 
    76: def rotate_half(x):
    77:     x = rearrange(x, '... (j d) -> ... j d', j=2)
    78:     x1, x2 = x.unbind(dim=-2)
    79:     return torch.cat((-x2, x1), dim=-1)
    80: 
    81: 
    82: def apply_rotary_pos_emb(t, freqs):
    83:     return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    84: 
    85: 
    86: def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):
    87:     # split t into first half and second half
    88:     # t: [b, h, n, d]
    89:     # freq_x/y: [b, n, d]
    90:     d = t.shape[-1]
    91:     t_x, t_y = t[..., :d // 2], t[..., d // 2:]
    92: 
    93:     return torch.cat((apply_rotary_pos_emb(t_x, freqs_x),
    94:                       apply_rotary_pos_emb(t_y, freqs_y)), dim=-1)
    95: 
    96: 
    97: class PositionalEncoding(nn.Module):
    98:     "Implement the PE function."
    99: 
   100:     def __init__(self, d_model, dropout, max_len=421 * 421):
   101:         super(PositionalEncoding, self).__init__()
   102:         self.dropout = nn.Dropout(p=dropout)
   103: 
   104:         # Compute the positional encodings once in log space.
   105:         pe = torch.zeros(max_len, d_model)
   106:         position = torch.arange(0, max_len).unsqueeze(1)
   107:         div_term = torch.exp(
   108:             torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
   109:         )
   110:         pe[:, 0::2] = torch.sin(position * div_term)
   111:         pe[:, 1::2] = torch.cos(position * div_term)
   112:         pe = pe.unsqueeze(0)
   113:         self.register_buffer("pe", pe)
   114: 
   115:     def forward(self, x):
   116:         x = x + self.pe[:, : x.size(1)].requires_grad_(False)
   117:         return self.dropout(x)
   118: 
   119: 
   120: def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
   121:     """
   122:     Create sinusoidal timestep embeddings.
   123:     :param timesteps: a 1-D Tensor of N indices, one per batch element.
   124:                       These may be fractional.
   125:     :param dim: the dimension of the output.
   126:     :param max_period: controls the minimum frequency of the embeddings.
   127:     :return: an [N x dim] Tensor of positional embeddings.
   128:     """
   129: 
   130:     half = dim // 2
   131:     freqs = torch.exp(
   132:         -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
   133:     ).to(device=timesteps.device)
   134:     args = timesteps[:, None].float() * freqs[None]
   135:     embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
   136:     if dim % 2:
   137:         embedding = torch.cat([embedding, torch.zeros_like(embedding[:,:,:1])], dim=-1)
   138:     return embedding
```

### `Neural-Solver-Library/layers/Physics_Attention.py`  [EDITABLE — entire file only]

```python
     1: import torch.nn as nn
     2: import torch
     3: from einops import rearrange, repeat
     4: 
     5: 
     6: class Physics_Attention_Irregular_Mesh(nn.Module):
     7:     ## for irregular meshes in 1D, 2D or 3D space
     8:     def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64, shapelist=None):
     9:         super().__init__()
    10:         inner_dim = dim_head * heads
    11:         self.dim_head = dim_head
    12:         self.heads = heads
    13:         self.scale = dim_head ** -0.5
    14:         self.softmax = nn.Softmax(dim=-1)
    15:         self.dropout = nn.Dropout(dropout)
    16:         self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
    17: 
    18:         self.in_project_x = nn.Linear(dim, inner_dim)
    19:         self.in_project_fx = nn.Linear(dim, inner_dim)
    20:         self.in_project_slice = nn.Linear(dim_head, slice_num)
    21:         for l in [self.in_project_slice]:
    22:             torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
    23:         self.to_q = nn.Linear(dim_head, dim_head, bias=False)
    24:         self.to_k = nn.Linear(dim_head, dim_head, bias=False)
    25:         self.to_v = nn.Linear(dim_head, dim_head, bias=False)
    26:         self.to_out = nn.Sequential(
    27:             nn.Linear(inner_dim, dim),
    28:             nn.Dropout(dropout)
    29:         )
    30: 
    31:     def forward(self, x):
    32:         # B N C
    33:         B, N, C = x.shape
    34: 
    35:         ### (1) Slice
    36:         fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head) \
    37:             .permute(0, 2, 1, 3).contiguous()  # B H N C
    38:         x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
    39:             .permute(0, 2, 1, 3).contiguous()  # B H N C
    40:         slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
    41:         slice_norm = slice_weights.sum(2)  # B H G
    42:         slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
    43:         slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))
    44: 
    45:         ### (2) Attention among slice tokens
    46:         q_slice_token = self.to_q(slice_token)
    47:         k_slice_token = self.to_k(slice_token)
    48:         v_slice_token = self.to_v(slice_token)
    49:         dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
    50:         attn = self.softmax(dots)
    51:         attn = self.dropout(attn)
    52:         out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D
    53: 
    54:         ### (3) Deslice
    55:         out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
    56:         out_x = rearrange(out_x, 'b h n d -> b n (h d)')
    57:         return self.to_out(out_x)
    58: 
    59: 
    60: class Physics_Attention_Structured_Mesh_1D(nn.Module):
    61:     ## for structured mesh in 1D space
    62:     def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64, shapelist=None, kernel=3):  # kernel=3):
    63:         super().__init__()
    64:         inner_dim = dim_head * heads
    65:         self.dim_head = dim_head
    66:         self.heads = heads
    67:         self.scale = dim_head ** -0.5
    68:         self.softmax = nn.Softmax(dim=-1)
    69:         self.dropout = nn.Dropout(dropout)
    70:         self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
    71:         self.length = shapelist[0]
    72: 
    73:         self.in_project_x = nn.Conv1d(dim, inner_dim, kernel, 1, kernel // 2)
    74:         self.in_project_fx = nn.Conv1d(dim, inner_dim, kernel, 1, kernel // 2)
    75:         self.in_project_slice = nn.Linear(dim_head, slice_num)
    76:         for l in [self.in_project_slice]:
    77:             torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
    78:         self.to_q = nn.Linear(dim_head, dim_head, bias=False)
    79:         self.to_k = nn.Linear(dim_head, dim_head, bias=False)
    80:         self.to_v = nn.Linear(dim_head, dim_head, bias=False)
    81: 
    82:         self.to_out = nn.Sequential(
    83:             nn.Linear(inner_dim, dim),
    84:             nn.Dropout(dropout)
    85:         )
    86: 
    87:     def forward(self, x):
    88:         # B N C
    89:         B, N, C = x.shape
    90:         x = x.reshape(B, self.length, C).contiguous().permute(0, 2, 1).contiguous()  # B C N
    91: 
    92:         ### (1) Slice
    93:         fx_mid = self.in_project_fx(x).permute(0, 2, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
    94:             .permute(0, 2, 1, 3).contiguous()  # B H N C
    95:         x_mid = self.in_project_x(x).permute(0, 2, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
    96:             .permute(0, 2, 1, 3).contiguous()  # B H N G
    97:         slice_weights = self.softmax(
    98:             self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B H N G
    99:         slice_norm = slice_weights.sum(2)  # B H G
   100:         slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
   101:         slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))
   102: 
   103:         ### (2) Attention among slice tokens
   104:         q_slice_token = self.to_q(slice_token)
   105:         k_slice_token = self.to_k(slice_token)
   106:         v_slice_token = self.to_v(slice_token)
   107:         dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
   108:         attn = self.softmax(dots)
   109:         attn = self.dropout(attn)
   110:         out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D
   111: 
   112:         ### (3) Deslice
   113:         out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
   114:         out_x = rearrange(out_x, 'b h n d -> b n (h d)')
   115:         return self.to_out(out_x)
   116: 
   117: 
   118: class Physics_Attention_Structured_Mesh_2D(nn.Module):
   119:     ## for structured mesh in 2D space
   120:     def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64, shapelist=None, kernel=3):
   121:         super().__init__()
   122:         inner_dim = dim_head * heads
   123:         self.dim_head = dim_head
   124:         self.heads = heads
   125:         self.scale = dim_head ** -0.5
   126:         self.softmax = nn.Softmax(dim=-1)
   127:         self.dropout = nn.Dropout(dropout)
   128:         self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
   129:         self.H = shapelist[0]
   130:         self.W = shapelist[1]
   131: 
   132:         self.in_project_x = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
   133:         self.in_project_fx = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
   134:         self.in_project_slice = nn.Linear(dim_head, slice_num)
   135:         for l in [self.in_project_slice]:
   136:             torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
   137:         self.to_q = nn.Linear(dim_head, dim_head, bias=False)
   138:         self.to_k = nn.Linear(dim_head, dim_head, bias=False)
   139:         self.to_v = nn.Linear(dim_head, dim_head, bias=False)
   140: 
   141:         self.to_out = nn.Sequential(
   142:             nn.Linear(inner_dim, dim),
   143:             nn.Dropout(dropout)
   144:         )
   145: 
   146:     def forward(self, x):
   147:         # B N C
   148:         B, N, C = x.shape
   149:         x = x.reshape(B, self.H, self.W, C).contiguous().permute(0, 3, 1, 2).contiguous()  # B C H W
   150: 
   151:         ### (1) Slice
   152:         fx_mid = self.in_project_fx(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
   153:             .permute(0, 2, 1, 3).contiguous()  # B H N C
   154:         x_mid = self.in_project_x(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
   155:             .permute(0, 2, 1, 3).contiguous()  # B H N G
   156:         slice_weights = self.softmax(
   157:             self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B H N G
   158:         slice_norm = slice_weights.sum(2)  # B H G
   159:         slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
   160:         slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))
   161: 
   162:         ### (2) Attention among slice tokens
   163:         q_slice_token = self.to_q(slice_token)
   164:         k_slice_token = self.to_k(slice_token)
   165:         v_slice_token = self.to_v(slice_token)
   166:         dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
   167:         attn = self.softmax(dots)
   168:         attn = self.dropout(attn)
   169:         out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D
   170: 
   171:         ### (3) Deslice
   172:         out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
   173:         out_x = rearrange(out_x, 'b h n d -> b n (h d)')
   174:         return self.to_out(out_x)
   175: 
   176: 
   177: class Physics_Attention_Structured_Mesh_3D(nn.Module):
   178:     ## for structured mesh in 3D space
   179:     def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=32, shapelist=None, kernel=3):
   180:         super().__init__()
   181:         inner_dim = dim_head * heads
   182:         self.dim_head = dim_head
   183:         self.heads = heads
   184:         self.scale = dim_head ** -0.5
   185:         self.softmax = nn.Softmax(dim=-1)
   186:         self.dropout = nn.Dropout(dropout)
   187:         self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
   188:         self.H = shapelist[0]
   189:         self.W = shapelist[1]
   190:         self.D = shapelist[2]
   191: 
   192:         self.in_project_x = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
   193:         self.in_project_fx = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
   194:         self.in_project_slice = nn.Linear(dim_head, slice_num)
   195:         for l in [self.in_project_slice]:
   196:             torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
   197:         self.to_q = nn.Linear(dim_head, dim_head, bias=False)
   198:         self.to_k = nn.Linear(dim_head, dim_head, bias=False)
   199:         self.to_v = nn.Linear(dim_head, dim_head, bias=False)
   200:         self.to_out = nn.Sequential(
   201:             nn.Linear(inner_dim, dim),
   202:             nn.Dropout(dropout)
   203:         )
   204: 
   205:     def forward(self, x):
   206:         # B N C
   207:         B, N, C = x.shape
   208:         x = x.reshape(B, self.H, self.W, self.D, C).contiguous().permute(0, 4, 1, 2, 3).contiguous()  # B C H W
   209: 
   210:         ### (1) Slice
   211:         fx_mid = self.in_project_fx(x).permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
   212:             .permute(0, 2, 1, 3).contiguous()  # B H N C
   213:         x_mid = self.in_project_x(x).permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
   214:             .permute(0, 2, 1, 3).contiguous()  # B H N G
   215:         slice_weights = self.softmax(
   216:             self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B H N G
   217:         slice_norm = slice_weights.sum(2)  # B H G
   218:         slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
   219:         slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))
   220: 
   221:         ### (2) Attention among slice tokens
   222:         q_slice_token = self.to_q(slice_token)
   223:         k_slice_token = self.to_k(slice_token)
   224:         v_slice_token = self.to_v(slice_token)
   225:         dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
   226:         attn = self.softmax(dots)
   227:         attn = self.dropout(attn)
   228:         out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D
   229: 
   230:         ### (3) Deslice
   231:         out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
   232:         out_x = rearrange(out_x, 'b h n d -> b n (h d)')
   233:         return self.to_out(out_x)
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `pointnet` baseline — editable region  [READ-ONLY — reference implementation]

In `Neural-Solver-Library/models/Custom.py`:

```python
Lines 1–49:
     1: import torch
     2: import torch.nn as nn
     3: import torch_geometric.nn as nng
     4: from layers.Embedding import unified_pos_embedding
     5: from layers.Basic import MLP
     6: 
     7: 
     8: class Model(nn.Module):
     9:     def __init__(self, args):
    10:         super(Model, self).__init__()
    11:         self.__name__ = 'Custom'
    12: 
    13:         self.in_block = MLP(args.n_hidden, args.n_hidden * 2, args.n_hidden * 2, n_layers=0, res=False,
    14:                             act=args.act)
    15:         self.max_block = MLP(args.n_hidden * 2, args.n_hidden * 8, args.n_hidden * 32, n_layers=0, res=False,
    16:                              act=args.act)
    17: 
    18:         self.out_block = MLP(args.n_hidden * (2 + 32), args.n_hidden * 16, args.n_hidden * 4, n_layers=0, res=False,
    19:                              act=args.act)
    20: 
    21:         self.encoder = MLP(args.fun_dim + args.space_dim, args.n_hidden * 2, args.n_hidden, n_layers=0, res=False,
    22:                            act=args.act)
    23:         self.decoder = MLP(args.n_hidden, args.n_hidden * 2, args.out_dim, n_layers=0, res=False, act=args.act)
    24: 
    25:         self.fcfinal = nn.Linear(args.n_hidden * 4, args.n_hidden)
    26: 
    27:     def forward(self, x, fx, T=None, geo=None):
    28:         if geo is None:
    29:             raise ValueError('Please provide edge index for Graph Neural Networks')
    30:         z, batch = torch.cat((x, fx), dim=-1).float().squeeze(0), torch.zeros([x.shape[1]]).cuda().long()
    31: 
    32:         z = self.encoder(z)
    33:         z = self.in_block(z)
    34: 
    35:         global_coef = self.max_block(z)
    36:         global_coef = nng.global_max_pool(global_coef, batch=batch)
    37:         nb_points = torch.zeros(global_coef.shape[0], device=z.device)
    38: 
    39:         for i in range(batch.max() + 1):
    40:             nb_points[i] = (batch == i).sum()
    41:         nb_points = nb_points.long()
    42:         global_coef = torch.repeat_interleave(global_coef, nb_points, dim=0)
    43: 
    44:         z = torch.cat([z, global_coef], dim=1)
    45:         z = self.out_block(z)
    46:         z = self.fcfinal(z)
    47:         z = self.decoder(z)
    48: 
    49:         return z.unsqueeze(0)
    50: 
    51: # =====================================================================
    52: # CONFIG_OVERRIDES: per-method hyperparameter overrides

Lines 59–59:
    56: # Defaults follow the baseline shell scripts (n_hidden=128, slice_num=32),
    57: # matching the GraphSAGE configuration in Neural-Solver-Library/scripts/DesignBench/car/.
    58: # Other paper settings (for reference): PointNet=16, Transolver=256, Graph_UNet=16, GNOT=256.
    59: CONFIG_OVERRIDES = {'n_hidden': 16}
    60: 
    61: 
    62: # =====================================================================
```

### `graphsage` baseline — editable region  [READ-ONLY — reference implementation]

In `Neural-Solver-Library/models/Custom.py`:

```python
Lines 1–60:
     1: import torch
     2: import torch.nn as nn
     3: import torch_geometric.nn as nng
     4: from layers.Basic import MLP
     5: 
     6: 
     7: class Model(nn.Module):
     8:     def __init__(self, args):
     9:         super(Model, self).__init__()
    10:         self.__name__ = 'Custom'
    11: 
    12:         self.nb_hidden_layers = args.n_layers
    13:         self.size_hidden_layers = args.n_hidden
    14:         self.bn_bool = True
    15:         self.activation = nn.ReLU()
    16: 
    17:         self.encoder = MLP(args.fun_dim + args.space_dim, args.n_hidden * 2, args.n_hidden, n_layers=0, res=False,
    18:                            act=args.act)
    19:         self.decoder = MLP(args.n_hidden, args.n_hidden * 2, args.out_dim, n_layers=0, res=False, act=args.act)
    20: 
    21:         self.in_layer = nng.SAGEConv(
    22:             in_channels=args.n_hidden,
    23:             out_channels=self.size_hidden_layers
    24:         )
    25: 
    26:         self.hidden_layers = nn.ModuleList()
    27:         for n in range(self.nb_hidden_layers - 1):
    28:             self.hidden_layers.append(nng.SAGEConv(
    29:                 in_channels=self.size_hidden_layers,
    30:                 out_channels=self.size_hidden_layers
    31:             ))
    32: 
    33:         self.out_layer = nng.SAGEConv(
    34:             in_channels=self.size_hidden_layers,
    35:             out_channels=self.size_hidden_layers
    36:         )
    37: 
    38:         if self.bn_bool:
    39:             self.bn = nn.ModuleList()
    40:             for n in range(self.nb_hidden_layers):
    41:                 self.bn.append(nn.BatchNorm1d(self.size_hidden_layers, track_running_stats=False))
    42: 
    43:     def forward(self, x, fx, T=None, geo=None):
    44:         if geo is None:
    45:             raise ValueError('Please provide edge index for Graph Neural Networks')
    46:         z, edge_index = torch.cat((x, fx), dim=-1).squeeze(0), geo
    47:         z = self.encoder(z)
    48:         z = self.in_layer(z, edge_index)
    49:         if self.bn_bool:
    50:             z = self.bn[0](z)
    51:         z = self.activation(z)
    52: 
    53:         for n in range(self.nb_hidden_layers - 1):
    54:             z = self.hidden_layers[n](z, edge_index)
    55:             if self.bn_bool:
    56:                 z = self.bn[n + 1](z)
    57:             z = self.activation(z)
    58:         z = self.out_layer(z, edge_index)
    59:         z = self.decoder(z)
    60:         return z.unsqueeze(0)
    61: 
    62: # =====================================================================
    63: # CONFIG_OVERRIDES: per-method hyperparameter overrides

Lines 70–70:
    67: # Defaults follow the baseline shell scripts (n_hidden=128, slice_num=32),
    68: # matching the GraphSAGE configuration in Neural-Solver-Library/scripts/DesignBench/car/.
    69: # Other paper settings (for reference): PointNet=16, Transolver=256, Graph_UNet=16, GNOT=256.
    70: CONFIG_OVERRIDES = {'n_hidden': 128}
    71: 
    72: 
    73: # =====================================================================
```

### `graphunet` baseline — editable region  [READ-ONLY — reference implementation]

In `Neural-Solver-Library/models/Custom.py`:

```python
Lines 1–227:
     1: import torch
     2: import torch.nn as nn
     3: import torch_geometric.nn as nng
     4: import random
     5: from layers.Basic import MLP
     6: 
     7: 
     8: def DownSample(id, x, edge_index, pos_x, pool, pool_ratio, r, max_neighbors):
     9:     y = x.clone()
    10:     n = int(x.size(0))
    11: 
    12:     if pool is not None:
    13:         y, _, _, _, id_sampled, _ = pool(y, edge_index)
    14:     else:
    15:         k = int((pool_ratio * torch.tensor(n, dtype=torch.float)).ceil())
    16:         id_sampled = random.sample(range(n), k)
    17:         id_sampled = torch.tensor(id_sampled, dtype=torch.long)
    18:         y = y[id_sampled]
    19: 
    20:     pos_x = pos_x[id_sampled]
    21:     id.append(id_sampled)
    22: 
    23:     edge_index_sampled = nng.radius_graph(x=pos_x.detach(), r=r, loop=True, max_num_neighbors=max_neighbors)
    24: 
    25:     return y, edge_index_sampled
    26: 
    27: 
    28: def UpSample(x, pos_x_up, pos_x_down):
    29:     cluster = nng.nearest(pos_x_up, pos_x_down)
    30:     x_up = x[cluster]
    31: 
    32:     return x_up
    33: 
    34: 
    35: class Model(nn.Module):
    36:     def __init__(self, args, pool='random', scale=5, list_r=[0.05, 0.2, 0.5, 1, 10],
    37:                  pool_ratio=[0.5, 0.5, 0.5, 0.5, 0.5], max_neighbors=64, layer='SAGE', head=2):
    38:         super(Model, self).__init__()
    39:         self.__name__ = 'Custom'
    40: 
    41:         self.L = scale
    42:         self.layer = layer
    43:         self.pool_type = pool
    44:         self.pool_ratio = pool_ratio
    45:         self.list_r = list_r
    46:         self.size_hidden_layers = args.n_hidden
    47:         self.size_hidden_layers_init = args.n_hidden
    48:         self.max_neighbors = max_neighbors
    49:         self.dim_enc = args.n_hidden
    50:         self.bn_bool = True
    51:         self.res = False
    52:         self.head = head
    53:         self.activation = nn.ReLU()
    54: 
    55:         self.encoder = MLP(args.fun_dim, args.n_hidden * 2, args.n_hidden, n_layers=0, res=False,
    56:                            act=args.act)
    57:         self.decoder = MLP(args.n_hidden, args.n_hidden * 2, args.out_dim, n_layers=0, res=False, act=args.act)
    58: 
    59:         self.down_layers = nn.ModuleList()
    60: 
    61:         if self.pool_type != 'random':
    62:             self.pool = nn.ModuleList()
    63:         else:
    64:             self.pool = None
    65: 
    66:         if self.layer == 'SAGE':
    67:             self.down_layers.append(nng.SAGEConv(
    68:                 in_channels=self.dim_enc,
    69:                 out_channels=self.size_hidden_layers
    70:             ))
    71:             bn_in = self.size_hidden_layers
    72: 
    73:         elif self.layer == 'GAT':
    74:             self.down_layers.append(nng.GATConv(
    75:                 in_channels=self.dim_enc,
    76:                 out_channels=self.size_hidden_layers,
    77:                 heads=self.head,
    78:                 add_self_loops=False,
    79:                 concat=True
    80:             ))
    81:             bn_in = self.head * self.size_hidden_layers
    82: 
    83:         if self.bn_bool == True:
    84:             self.bn = nn.ModuleList()
    85:             self.bn.append(nng.BatchNorm(
    86:                 in_channels=bn_in,
    87:                 track_running_stats=False
    88:             ))
    89:         else:
    90:             self.bn = None
    91: 
    92:         for n in range(1, self.L):
    93:             if self.pool_type != 'random':
    94:                 self.pool.append(nng.TopKPooling(
    95:                     in_channels=self.size_hidden_layers,
    96:                     ratio=self.pool_ratio[n - 1],
    97:                     nonlinearity=torch.sigmoid
    98:                 ))
    99: 
   100:             if self.layer == 'SAGE':
   101:                 self.down_layers.append(nng.SAGEConv(
   102:                     in_channels=self.size_hidden_layers,
   103:                     out_channels=2 * self.size_hidden_layers,
   104:                 ))
   105:                 self.size_hidden_layers = 2 * self.size_hidden_layers
   106:                 bn_in = self.size_hidden_layers
   107: 
   108:             elif self.layer == 'GAT':
   109:                 self.down_layers.append(nng.GATConv(
   110:                     in_channels=self.head * self.size_hidden_layers,
   111:                     out_channels=self.size_hidden_layers,
   112:                     heads=2,
   113:                     add_self_loops=False,
   114:                     concat=True
   115:                 ))
   116: 
   117:             if self.bn_bool == True:
   118:                 self.bn.append(nng.BatchNorm(
   119:                     in_channels=bn_in,
   120:                     track_running_stats=False
   121:                 ))
   122: 
   123:         self.up_layers = nn.ModuleList()
   124: 
   125:         if self.layer == 'SAGE':
   126:             self.up_layers.append(nng.SAGEConv(
   127:                 in_channels=3 * self.size_hidden_layers_init,
   128:                 out_channels=self.dim_enc
   129:             ))
   130:             self.size_hidden_layers_init = 2 * self.size_hidden_layers_init
   131: 
   132:         elif self.layer == 'GAT':
   133:             self.up_layers.append(nng.GATConv(
   134:                 in_channels=2 * self.head * self.size_hidden_layers,
   135:                 out_channels=self.dim_enc,
   136:                 heads=2,
   137:                 add_self_loops=False,
   138:                 concat=False
   139:             ))
   140: 
   141:         if self.bn_bool == True:
   142:             self.bn.append(nng.BatchNorm(
   143:                 in_channels=self.dim_enc,
   144:                 track_running_stats=False
   145:             ))
   146: 
   147:         for n in range(1, self.L - 1):
   148:             if self.layer == 'SAGE':
   149:                 self.up_layers.append(nng.SAGEConv(
   150:                     in_channels=3 * self.size_hidden_layers_init,
   151:                     out_channels=self.size_hidden_layers_init,
   152:                 ))
   153:                 bn_in = self.size_hidden_layers_init
   154:                 self.size_hidden_layers_init = 2 * self.size_hidden_layers_init
   155: 
   156:             elif self.layer == 'GAT':
   157:                 self.up_layers.append(nng.GATConv(
   158:                     in_channels=2 * self.head * self.size_hidden_layers,
   159:                     out_channels=self.size_hidden_layers,
   160:                     heads=2,
   161:                     add_self_loops=False,
   162:                     concat=True
   163:                 ))
   164: 
   165:             if self.bn_bool == True:
   166:                 self.bn.append(nng.BatchNorm(
   167:                     in_channels=bn_in,
   168:                     track_running_stats=False
   169:                 ))
   170: 
   171:     def forward(self, x, fx, T=None, geo=None):
   172:         if geo is None:
   173:             raise ValueError('Please provide edge index for Graph Neural Networks')
   174:         x, edge_index = fx.squeeze(0), geo
   175:         id = []
   176:         edge_index_list = [edge_index.clone()]
   177:         pos_x_list = []
   178:         z = self.encoder(x)
   179:         if self.res:
   180:             z_res = z.clone()
   181: 
   182:         z = self.down_layers[0](z, edge_index)
   183: 
   184:         if self.bn_bool == True:
   185:             z = self.bn[0](z)
   186: 
   187:         z = self.activation(z)
   188:         z_list = [z.clone()]
   189:         for n in range(self.L - 1):
   190:             pos_x = x[:, :2] if n == 0 else pos_x[id[n - 1]]
   191:             pos_x_list.append(pos_x.clone())
   192: 
   193:             if self.pool_type != 'random':
   194:                 z, edge_index = DownSample(id, z, edge_index, pos_x, self.pool[n], self.pool_ratio[n], self.list_r[n],
   195:                                            self.max_neighbors)
   196:             else:
   197:                 z, edge_index = DownSample(id, z, edge_index, pos_x, None, self.pool_ratio[n], self.list_r[n],
   198:                                            self.max_neighbors)
   199:             edge_index_list.append(edge_index.clone())
   200: 
   201:             z = self.down_layers[n + 1](z, edge_index)
   202: 
   203:             if self.bn_bool == True:
   204:                 z = self.bn[n + 1](z)
   205: 
   206:             z = self.activation(z)
   207:             z_list.append(z.clone())
   208:         pos_x_list.append(pos_x[id[-1]].clone())
   209: 
   210:         for n in range(self.L - 1, 0, -1):
   211:             z = UpSample(z, pos_x_list[n - 1], pos_x_list[n])
   212:             z = torch.cat([z, z_list[n - 1]], dim=1)
   213:             z = self.up_layers[n - 1](z, edge_index_list[n - 1])
   214: 
   215:             if self.bn_bool == True:
   216:                 z = self.bn[self.L + n - 1](z)
   217: 
   218:             z = self.activation(z) if n != 1 else z
   219: 
   220:         del (z_list, pos_x_list, edge_index_list)
   221: 
   222:         if self.res:
   223:             z = z + z_res
   224: 
   225:         z = self.decoder(z)
   226: 
   227:         return z.unsqueeze(0)
   228: 
   229: # =====================================================================
   230: # CONFIG_OVERRIDES: per-method hyperparameter overrides

Lines 237–237:
   234: # Defaults follow the baseline shell scripts (n_hidden=128, slice_num=32),
   235: # matching the GraphSAGE configuration in Neural-Solver-Library/scripts/DesignBench/car/.
   236: # Other paper settings (for reference): PointNet=16, Transolver=256, Graph_UNet=16, GNOT=256.
   237: CONFIG_OVERRIDES = {'n_hidden': 16}
   238: 
   239: 
   240: # =====================================================================
```

### `transolver` baseline — editable region  [READ-ONLY — reference implementation]

In `Neural-Solver-Library/models/Custom.py`:

```python
Lines 1–136:
     1: import torch
     2: import torch.nn as nn
     3: import numpy as np
     4: from timm.models.layers import trunc_normal_
     5: from layers.Basic import MLP
     6: from layers.Embedding import timestep_embedding, unified_pos_embedding
     7: from layers.Physics_Attention import Physics_Attention_Irregular_Mesh
     8: from layers.Physics_Attention import Physics_Attention_Structured_Mesh_1D
     9: from layers.Physics_Attention import Physics_Attention_Structured_Mesh_2D
    10: from layers.Physics_Attention import Physics_Attention_Structured_Mesh_3D
    11: 
    12: PHYSICS_ATTENTION = {
    13:     'unstructured': Physics_Attention_Irregular_Mesh,
    14:     'structured_1D': Physics_Attention_Structured_Mesh_1D,
    15:     'structured_2D': Physics_Attention_Structured_Mesh_2D,
    16:     'structured_3D': Physics_Attention_Structured_Mesh_3D
    17: }
    18: 
    19: 
    20: class Transolver_block(nn.Module):
    21:     def __init__(
    22:             self,
    23:             num_heads: int,
    24:             hidden_dim: int,
    25:             dropout: float,
    26:             act='gelu',
    27:             mlp_ratio=4,
    28:             last_layer=False,
    29:             out_dim=1,
    30:             slice_num=32,
    31:             geotype='unstructured',
    32:             shapelist=None
    33:     ):
    34:         super().__init__()
    35:         self.last_layer = last_layer
    36:         self.ln_1 = nn.LayerNorm(hidden_dim)
    37: 
    38:         self.Attn = PHYSICS_ATTENTION[geotype](hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
    39:                                                dropout=dropout, slice_num=slice_num, shapelist=shapelist)
    40:         self.ln_2 = nn.LayerNorm(hidden_dim)
    41:         self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
    42:         if self.last_layer:
    43:             self.ln_3 = nn.LayerNorm(hidden_dim)
    44:             self.mlp2 = nn.Linear(hidden_dim, out_dim)
    45: 
    46:     def forward(self, fx):
    47:         fx = self.Attn(self.ln_1(fx)) + fx
    48:         fx = self.mlp(self.ln_2(fx)) + fx
    49:         if self.last_layer:
    50:             return self.mlp2(self.ln_3(fx))
    51:         else:
    52:             return fx
    53: 
    54: 
    55: class Model(nn.Module):
    56:     def __init__(self, args):
    57:         super(Model, self).__init__()
    58:         self.__name__ = 'Custom'
    59:         self.args = args
    60:         if args.unified_pos and args.geotype != 'unstructured':
    61:             self.pos = unified_pos_embedding(args.shapelist, args.ref)
    62:             self.preprocess = MLP(args.fun_dim + args.ref ** len(args.shapelist), args.n_hidden * 2,
    63:                                   args.n_hidden, n_layers=0, res=False, act=args.act)
    64:         else:
    65:             self.preprocess = MLP(args.fun_dim + args.space_dim, args.n_hidden * 2, args.n_hidden,
    66:                                   n_layers=0, res=False, act=args.act)
    67:         if args.time_input:
    68:             self.time_fc = nn.Sequential(nn.Linear(args.n_hidden, args.n_hidden), nn.SiLU(),
    69:                                          nn.Linear(args.n_hidden, args.n_hidden))
    70: 
    71:         self.blocks = nn.ModuleList([Transolver_block(num_heads=args.n_heads, hidden_dim=args.n_hidden,
    72:                                                       dropout=args.dropout,
    73:                                                       act=args.act,
    74:                                                       mlp_ratio=args.mlp_ratio,
    75:                                                       out_dim=args.out_dim,
    76:                                                       slice_num=args.slice_num,
    77:                                                       last_layer=(_ == args.n_layers - 1),
    78:                                                       geotype=args.geotype,
    79:                                                       shapelist=args.shapelist)
    80:                                      for _ in range(args.n_layers)])
    81:         self.placeholder = nn.Parameter((1 / (args.n_hidden)) * torch.rand(args.n_hidden, dtype=torch.float))
    82:         self.initialize_weights()
    83: 
    84:     def initialize_weights(self):
    85:         self.apply(self._init_weights)
    86: 
    87:     def _init_weights(self, m):
    88:         if isinstance(m, nn.Linear):
    89:             trunc_normal_(m.weight, std=0.02)
    90:             if isinstance(m, nn.Linear) and m.bias is not None:
    91:                 nn.init.constant_(m.bias, 0)
    92:         elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
    93:             nn.init.constant_(m.bias, 0)
    94:             nn.init.constant_(m.weight, 1.0)
    95: 
    96:     def structured_geo(self, x, fx, T=None):
    97:         if self.args.unified_pos:
    98:             x = self.pos.repeat(x.shape[0], 1, 1)
    99:         if fx is not None:
   100:             fx = torch.cat((x, fx), -1)
   101:             fx = self.preprocess(fx)
   102:         else:
   103:             fx = self.preprocess(x)
   104:         fx = fx + self.placeholder[None, None, :]
   105: 
   106:         if T is not None:
   107:             Time_emb = timestep_embedding(T, self.args.n_hidden).repeat(1, x.shape[1], 1)
   108:             Time_emb = self.time_fc(Time_emb)
   109:             fx = fx + Time_emb
   110: 
   111:         for block in self.blocks:
   112:             fx = block(fx)
   113:         return fx
   114: 
   115:     def unstructured_geo(self, x, fx, T=None):
   116:         if fx is not None:
   117:             fx = torch.cat((x, fx), -1)
   118:             fx = self.preprocess(fx)
   119:         else:
   120:             fx = self.preprocess(x)
   121:         fx = fx + self.placeholder[None, None, :]
   122: 
   123:         if T is not None:
   124:             Time_emb = timestep_embedding(T, self.args.n_hidden).repeat(1, x.shape[1], 1)
   125:             Time_emb = self.time_fc(Time_emb)
   126:             fx = fx + Time_emb
   127: 
   128:         for block in self.blocks:
   129:             fx = block(fx)
   130:         return fx
   131: 
   132:     def forward(self, x, fx, T=None, geo=None):
   133:         if self.args.geotype == 'unstructured':
   134:             return self.unstructured_geo(x, fx, T)
   135:         else:
   136:             return self.structured_geo(x, fx, T)
   137: 
   138: # =====================================================================
   139: # CONFIG_OVERRIDES: per-method hyperparameter overrides

Lines 146–146:
   143: # Defaults follow the baseline shell scripts (n_hidden=128, slice_num=32),
   144: # matching the GraphSAGE configuration in Neural-Solver-Library/scripts/DesignBench/car/.
   145: # Other paper settings (for reference): PointNet=16, Transolver=256, Graph_UNet=16, GNOT=256.
   146: CONFIG_OVERRIDES = {'n_hidden': 256, 'slice_num': 32}
   147: 
   148: 
   149: # =====================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
