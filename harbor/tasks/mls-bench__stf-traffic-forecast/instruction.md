# MLS-Bench: stf-traffic-forecast

# Spatial-Temporal Traffic Forecasting on Sensor Networks

## Research Question
What modular spatial-temporal forecasting component (architecture or training scheme) generalizes across traffic-sensor networks of different sizes and modalities (speed vs. flow), under a fixed 12-step → 12-step horizon and a common evaluation protocol?

## Background
Spatial-temporal forecasting predicts future values across a network of spatial nodes — for traffic, sensors on highway segments — by jointly modeling temporal patterns at each node and spatial correlations across nodes. The METR-LA / PEMS-BAY benchmarks were introduced together with DCRNN (Li et al., ICLR 2018, "Diffusion Convolutional Recurrent Neural Network", arXiv 1707.01926) and have since become the canonical testbeds for graph- and attention-based spatial-temporal models. Design choices include (a) **spatial modeling**: learnable node embeddings, graph convolutions, spatial attention, learned adjacency; (b) **temporal modeling**: RNNs, temporal convolutions, Transformers; (c) **spatial-temporal fusion**: how the two are combined.

## Objective
Implement a `Custom` `nn.Module` and `CustomConfig` dataclass in `custom_model.py` for the BasicTS framework. The model is trained and evaluated by the fixed BasicTS pipeline on three datasets.

## Model Interface
```python
def forward(self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor) -> torch.Tensor:
    """
    inputs:            [batch_size, input_len=12, num_features]   # num_features = number of spatial nodes
    inputs_timestamps: [batch_size, input_len=12, 2]              # [time-of-day, day-of-week] normalized to [0,1]
    Returns:           [batch_size, output_len=12, num_features]  # next-hour predictions for every node
    """
```
`CustomConfig` extends `basicts.configs.BasicTSModelConfig` with at least `input_len`, `output_len`, `num_features`.

## Datasets and Fixed Protocol
- **METR-LA** — 207 sensors, traffic speed, Los Angeles highway (Li et al., ICLR 2018).
- **PEMS-BAY** — 325 sensors, traffic speed, San Francisco Bay Area (Li et al., ICLR 2018).
- **PEMS04** — 307 sensors, traffic flow, California Caltrans District 4 (commonly used with ASTGCN, AAAI 2019).

All settings use `input_len=12`, `output_len=12` (one hour of 5-min intervals → next hour). Data is Z-score normalized per dataset; metrics are computed after the inverse transform. Missing values (encoded as 0.0) are masked during loss and metric computation.

## Available Modules
You may import components from `basicts.modules`:
- `basicts.modules.mlps` — `MLPLayer`, `ResMLPLayer`
- `basicts.modules.norm` — `RevIN`, `LayerNorm`
- `basicts.modules.embed` — sequence embeddings
- `basicts.modules.transformer` — `Encoder`, `MultiHeadAttention`
- `basicts.modules.activations` — common activations

## Training Hyperparameter Override
The harness uses Adam with `lr=2e-3`, `weight_decay=1e-4`, and `MultiStepLR(milestones=[1, 50, 80], gamma=0.5)` for 100 epochs at `batch_size=64`. If your method needs a different `lr` or `weight_decay`, set them in the `CONFIG_OVERRIDES` dict at the bottom of `custom_model.py`:
```python
CONFIG_OVERRIDES = {'lr': 5e-4, 'weight_decay': 1e-3}
```
Only `lr` and `weight_decay` are forwarded; epochs, batch size, scheduler, and gradient clipping are fixed.

## Metrics
MAE, RMSE, MAPE — all lower is better, computed in original scale after inverse transform with the missing-value mask applied.

## Reference Implementations (read-only)
Six reference models live in `basicts/models/` and serve as context:
- **SOFTS** — Han et al., "SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion", NeurIPS 2024. Inverted architecture (variates as tokens) with a STar Aggregate-Redistribute (STAR) module for O(N) cross-variate fusion instead of self-attention. Source: https://github.com/Secilia-Cxy/SOFTS.
- **DLinear** — Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023 (arXiv 2205.13504). Decomposition into trend (moving-average kernel) + seasonal, each projected by a linear layer. Source: https://github.com/cure-lab/LTSF-Linear.
- **StemGNN** — Cao et al., "Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting", NeurIPS 2020 (arXiv 2103.07719). Graph Fourier transform + DFT for joint spatial-spectral modeling. Source: https://github.com/microsoft/StemGNN.
- **iTransformer** — Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting", ICLR 2024 (arXiv 2310.06625). Treats variates as tokens; attention across variables, FFN within each variate token. Source: https://github.com/thuml/iTransformer.
- **TimesNet** — Wu et al., "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis", ICLR 2023 (arXiv 2210.02186). Reshapes 1D series into 2D tensors at multiple FFT-discovered periods, processed by Inception 2D conv blocks. Source: https://github.com/thuml/Time-Series-Library.
- **TimeMixer** — Wang et al., "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting", ICLR 2024 (arXiv 2405.14616). Multi-scale decomposition with Past-Decomposable-Mixing and Future-Multipredictor-Mixing. Source: https://github.com/kwuking/TimeMixer.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/BasicTS/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `BasicTS/custom_model.py`
- editable lines **1–75**


Other files you may **read** for context (do not modify):
- `BasicTS/src/basicts/modules/mlps.py`
- `BasicTS/src/basicts/modules/embed/__init__.py`


## Readable Context


### `BasicTS/custom_model.py`  [EDITABLE — lines 1–75 only]

```python
     1: import torch
     2: import torch.nn as nn
     3: from dataclasses import dataclass, field
     4: from typing import Optional
     5: 
     6: from basicts.configs import BasicTSModelConfig
     7: 
     8: 
     9: @dataclass
    10: class CustomConfig(BasicTSModelConfig):
    11:     """Configuration for the Custom spatial-temporal forecasting model.
    12: 
    13:     Required fields (set by training script):
    14:         input_len: Length of input historical sequence.
    15:         output_len: Length of output prediction sequence.
    16:         num_features: Number of spatial nodes (sensors).
    17: 
    18:     Optional fields (tunable):
    19:         hidden_size: Hidden dimension size.
    20:         num_layers: Number of model layers.
    21:         dropout: Dropout rate.
    22:     """
    23: 
    24:     input_len: int = field(default=12, metadata={"help": "Input sequence length."})
    25:     output_len: int = field(default=12, metadata={"help": "Output sequence length."})
    26:     num_features: int = field(default=207, metadata={"help": "Number of spatial nodes."})
    27:     hidden_size: int = field(default=64, metadata={"help": "Hidden dimension size."})
    28:     num_layers: int = field(default=2, metadata={"help": "Number of model layers."})
    29:     dropout: float = field(default=0.1, metadata={"help": "Dropout rate."})
    30: 
    31: 
    32: class Custom(nn.Module):
    33:     """
    34:     Custom model for spatial-temporal traffic forecasting.
    35: 
    36:     The model receives traffic measurements from N spatial nodes over T time steps
    37:     and predicts the next T' time steps for all N nodes.
    38: 
    39:     Forward signature: forward(inputs, inputs_timestamps)
    40:     - inputs: [batch_size, input_len, num_features] — historical traffic data
    41:       Each feature dimension corresponds to one spatial node (sensor).
    42:     - inputs_timestamps: [batch_size, input_len, num_timestamps] — temporal features
    43:       Typically contains normalized time-of-day and day-of-week.
    44: 
    45:     Must return: [batch_size, output_len, num_features] — predicted traffic values
    46:     """
    47: 
    48:     def __init__(self, config: CustomConfig):
    49:         super().__init__()
    50:         self.input_len = config.input_len
    51:         self.output_len = config.output_len
    52:         self.num_features = config.num_features
    53:         self.hidden_size = config.hidden_size
    54:         self.num_layers = config.num_layers
    55:         self.dropout = config.dropout
    56:         # TODO: Define your model architecture here
    57: 
    58:     def forward(self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor) -> torch.Tensor:
    59:         """
    60:         Args:
    61:             inputs: [batch_size, input_len, num_features] — historical traffic data
    62:             inputs_timestamps: [batch_size, input_len, num_timestamps] — time features
    63: 
    64:         Returns:
    65:             prediction: [batch_size, output_len, num_features]
    66:         """
    67:         # TODO: Implement your spatial-temporal forecasting model
    68:         # Placeholder: simple linear projection (no spatial modeling)
    69:         batch_size = inputs.shape[0]
    70:         return torch.zeros(batch_size, self.output_len, self.num_features, device=inputs.device)
    71: 
    72: 
    73: # CONFIG_OVERRIDES: override training hyperparameters for your method.
    74: # Allowed keys: lr, weight_decay.
    75: CONFIG_OVERRIDES = {}
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


### `dlinear` baseline — editable region  [READ-ONLY — reference implementation]

In `BasicTS/custom_model.py`:

```python
Lines 1–52:
     1: import torch
     2: import torch.nn as nn
     3: import torch.nn.functional as F
     4: from dataclasses import dataclass, field
     5: from basicts.configs import BasicTSModelConfig
     6: 
     7: 
     8: @dataclass
     9: class CustomConfig(BasicTSModelConfig):
    10:     input_len: int = field(default=12)
    11:     output_len: int = field(default=12)
    12:     num_features: int = field(default=207)
    13:     moving_avg: int = field(default=25)
    14: 
    15: 
    16: class Custom(nn.Module):
    17:     """DLinear: Decomposition-Linear baseline.
    18: 
    19:     Decomposes input into trend (moving average) and seasonal (residual),
    20:     then projects each component independently to the prediction horizon.
    21:     """
    22: 
    23:     def __init__(self, config: CustomConfig):
    24:         super().__init__()
    25:         self.input_len = config.input_len
    26:         self.output_len = config.output_len
    27:         k = config.moving_avg
    28:         self.pad_left = (k - 1) // 2
    29:         self.pad_right = k // 2
    30:         self.avg_pool = nn.AvgPool1d(k, stride=1)
    31:         self.linear_seasonal = nn.Linear(config.input_len, config.output_len)
    32:         self.linear_trend = nn.Linear(config.input_len, config.output_len)
    33: 
    34:     def _decompose(self, x):
    35:         # x: [B, T, N] -> trend via moving average, seasonal = x - trend
    36:         padded = F.pad(x.transpose(1, 2), (self.pad_left, self.pad_right), mode='replicate')
    37:         trend = self.avg_pool(padded).transpose(1, 2)
    38:         seasonal = x - trend
    39:         return seasonal, trend
    40: 
    41:     def forward(self, inputs, inputs_timestamps):
    42:         # inputs: [B, T, N]
    43:         seasonal, trend = self._decompose(inputs)
    44:         # Per-feature linear: [B, N, T] -> [B, N, T']
    45:         seasonal_out = self.linear_seasonal(seasonal.transpose(1, 2))
    46:         trend_out = self.linear_trend(trend.transpose(1, 2))
    47:         prediction = (seasonal_out + trend_out).transpose(1, 2)  # [B, T', N]
    48:         return prediction
    49: 
    50: # CONFIG_OVERRIDES: override training hyperparameters for your method.
    51: # Allowed keys: lr, weight_decay.
    52: CONFIG_OVERRIDES = {}
```

### `stemgnn` baseline — editable region  [READ-ONLY — reference implementation]

In `BasicTS/custom_model.py`:

```python
Lines 1–151:
     1: import torch
     2: import torch.nn as nn
     3: import torch.nn.functional as F
     4: from dataclasses import dataclass, field
     5: from basicts.configs import BasicTSModelConfig
     6: 
     7: 
     8: @dataclass
     9: class CustomConfig(BasicTSModelConfig):
    10:     input_len: int = field(default=12)
    11:     output_len: int = field(default=12)
    12:     num_features: int = field(default=207)
    13:     hidden_size: int = field(default=5)
    14:     num_blocks: int = field(default=2)
    15:     dropout: float = field(default=0.5)
    16: 
    17: 
    18: class GLU(nn.Module):
    19:     def __init__(self, in_dim, out_dim):
    20:         super().__init__()
    21:         self.left = nn.Linear(in_dim, out_dim)
    22:         self.right = nn.Linear(in_dim, out_dim)
    23: 
    24:     def forward(self, x):
    25:         return self.left(x) * torch.sigmoid(self.right(x))
    26: 
    27: 
    28: class StockBlock(nn.Module):
    29:     def __init__(self, input_len, num_features, hidden_size, layer_idx):
    30:         super().__init__()
    31:         self.input_len = input_len
    32:         self.num_features = num_features
    33:         self.hidden_size = hidden_size
    34:         self.layer_idx = layer_idx
    35:         self.output_hidden_size = 4 * hidden_size
    36: 
    37:         self.weight = nn.Parameter(
    38:             torch.Tensor(1, 4, 1, input_len * hidden_size, hidden_size * input_len))
    39:         nn.init.xavier_normal_(self.weight)
    40:         self.forecast = nn.Linear(input_len * hidden_size, input_len * hidden_size)
    41:         self.forecast_result = nn.Linear(input_len * hidden_size, input_len)
    42:         if layer_idx == 0:
    43:             self.backcast = nn.Linear(input_len * hidden_size, input_len)
    44:         self.backcast_short_cut = nn.Linear(input_len, input_len)
    45: 
    46:         self.GLUs = nn.ModuleList()
    47:         for i in range(3):
    48:             in_d = input_len * 4 if i == 0 else input_len * self.output_hidden_size
    49:             self.GLUs.append(GLU(in_d, input_len * self.output_hidden_size))
    50:             self.GLUs.append(GLU(in_d, input_len * self.output_hidden_size))
    51: 
    52:     def spe_seq_cell(self, inputs):
    53:         B, _, _, N, L = inputs.size()
    54:         inputs = inputs.view(B, -1, N, L)
    55:         ffted = torch.fft.fft(inputs, dim=-1)
    56:         real = ffted.real.permute(0, 2, 1, 3).contiguous().reshape(B, N, -1)
    57:         imag = ffted.imag.permute(0, 2, 1, 3).contiguous().reshape(B, N, -1)
    58:         for i in range(3):
    59:             real = self.GLUs[i * 2](real)
    60:             imag = self.GLUs[i * 2 + 1](imag)
    61:         real = real.reshape(B, N, 4, -1).permute(0, 2, 1, 3).contiguous()
    62:         imag = imag.reshape(B, N, 4, -1).permute(0, 2, 1, 3).contiguous()
    63:         return torch.fft.ifft(torch.complex(real, imag), dim=-1).real
    64: 
    65:     def forward(self, x, graph):
    66:         graph = graph.unsqueeze(1)
    67:         x = x.unsqueeze(1)
    68:         gfted = torch.matmul(graph, x)
    69:         gconv_input = self.spe_seq_cell(gfted).unsqueeze(2)
    70:         igfted = torch.matmul(gconv_input, self.weight).sum(dim=1)
    71:         forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1))
    72:         forecast = self.forecast_result(forecast_source)
    73:         if self.layer_idx == 0:
    74:             backcast_short = self.backcast_short_cut(x).squeeze(1)
    75:             backcast_source = torch.sigmoid(self.backcast(igfted) - backcast_short)
    76:         else:
    77:             backcast_source = None
    78:         return forecast, backcast_source
    79: 
    80: 
    81: class Custom(nn.Module):
    82:     """StemGNN: Spectral Temporal Graph Neural Network baseline.
    83: 
    84:     Learns a latent graph via self-attention, applies Chebyshev graph
    85:     convolution, and processes temporal patterns in the spectral domain.
    86:     """
    87: 
    88:     def __init__(self, config: CustomConfig):
    89:         super().__init__()
    90:         N = config.num_features
    91:         L = config.input_len
    92:         self.num_blocks = config.num_blocks
    93: 
    94:         # Latent graph via self-attention
    95:         self.weight_key = nn.Parameter(torch.zeros(N, 1))
    96:         nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
    97:         self.weight_query = nn.Parameter(torch.zeros(N, 1))
    98:         nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
    99:         self.GRU = nn.GRU(L, N)
   100: 
   101:         # Backbone
   102:         self.stock_block = nn.ModuleList([
   103:             StockBlock(L, N, config.hidden_size, i)
   104:             for i in range(config.num_blocks)
   105:         ])
   106: 
   107:         # Output
   108:         self.fc = nn.Sequential(
   109:             nn.Linear(L, L), nn.LeakyReLU(),
   110:             nn.Linear(L, config.output_len))
   111:         self.leakyrelu = nn.LeakyReLU(0.2)
   112:         self.dropout = nn.Dropout(config.dropout)
   113: 
   114:     def _latent_graph(self, x):
   115:         # x: [B, T, N]
   116:         h, _ = self.GRU(x.permute(2, 0, 1))  # [N, B, N]
   117:         h = h.permute(1, 0, 2).contiguous()  # [B, N, N]
   118:         h = h.permute(0, 2, 1).contiguous()  # [B, N, N] transposed for attention
   119:         key = torch.matmul(h, self.weight_key)    # [B, N, 1]
   120:         query = torch.matmul(h, self.weight_query) # [B, N, 1]
   121:         N = h.size(1)
   122:         attn = key.repeat(1, 1, N).view(-1, N * N, 1) + query.repeat(1, N, 1)
   123:         attn = self.leakyrelu(attn.squeeze(2).view(-1, N, N))
   124:         attn = self.dropout(F.softmax(attn, dim=2))
   125:         attn = torch.mean(attn, dim=0)
   126:         attn = 0.5 * (attn + attn.T)
   127:         degree = torch.sum(attn, dim=1)
   128:         D_inv_sqrt = torch.diag(1.0 / (torch.sqrt(degree) + 1e-7))
   129:         L = D_inv_sqrt @ (torch.diag(degree) - attn) @ D_inv_sqrt
   130:         # Chebyshev polynomials (order 3)
   131:         L = L.unsqueeze(0)
   132:         T0 = torch.zeros_like(L)
   133:         T1 = L
   134:         T2 = 2 * torch.matmul(L, T1) - T0
   135:         T3 = 2 * torch.matmul(L, T2) - T1
   136:         return torch.cat([T0, T1, T2, T3], dim=0)
   137: 
   138:     def forward(self, inputs, inputs_timestamps):
   139:         graph = self._latent_graph(inputs)
   140:         x = inputs.unsqueeze(1).transpose(-1, -2)  # [B, 1, N, T]
   141:         results = []
   142:         for i in range(self.num_blocks):
   143:             pred, x = self.stock_block[i](x, graph)
   144:             results.append(pred)
   145:         prediction = sum(results)
   146:         prediction = self.fc(prediction).transpose(1, 2)  # [B, T', N]
   147:         return prediction
   148: 
   149: # CONFIG_OVERRIDES: override training hyperparameters for your method.
   150: # Allowed keys: lr, weight_decay.
   151: CONFIG_OVERRIDES = {'lr': 0.001}
```

### `itransformer` baseline — editable region  [READ-ONLY — reference implementation]

In `BasicTS/custom_model.py`:

```python
Lines 1–121:
     1: import math
     2: import torch
     3: import torch.nn as nn
     4: import torch.nn.functional as F
     5: from dataclasses import dataclass, field
     6: from basicts.configs import BasicTSModelConfig
     7: 
     8: 
     9: @dataclass
    10: class CustomConfig(BasicTSModelConfig):
    11:     input_len: int = field(default=12)
    12:     output_len: int = field(default=12)
    13:     num_features: int = field(default=207)
    14:     hidden_size: int = field(default=512)
    15:     n_heads: int = field(default=8)
    16:     num_layers: int = field(default=3)
    17:     dropout: float = field(default=0.1)
    18: 
    19: 
    20: class RevIN(nn.Module):
    21:     """Reversible Instance Normalization."""
    22:     def __init__(self, eps=1e-6):
    23:         super().__init__()
    24:         self.eps = eps
    25: 
    26:     def forward(self, x, mode):
    27:         if mode == "norm":
    28:             self.mean = x.mean(dim=1, keepdim=True).detach()
    29:             self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
    30:             return (x - self.mean) / self.stdev
    31:         else:  # denorm
    32:             return x * self.stdev + self.mean
    33: 
    34: 
    35: class MultiHeadAttention(nn.Module):
    36:     def __init__(self, hidden_size, n_heads, dropout=0.1):
    37:         super().__init__()
    38:         self.n_heads = n_heads
    39:         self.head_dim = hidden_size // n_heads
    40:         self.q_proj = nn.Linear(hidden_size, hidden_size)
    41:         self.k_proj = nn.Linear(hidden_size, hidden_size)
    42:         self.v_proj = nn.Linear(hidden_size, hidden_size)
    43:         self.out_proj = nn.Linear(hidden_size, hidden_size)
    44:         self.dropout = nn.Dropout(dropout)
    45:         self.scale = math.sqrt(self.head_dim)
    46: 
    47:     def forward(self, x):
    48:         B, N, D = x.shape
    49:         q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
    50:         k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
    51:         v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
    52:         attn = (q @ k.transpose(-2, -1)) / self.scale
    53:         attn = self.dropout(F.softmax(attn, dim=-1))
    54:         out = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
    55:         return self.out_proj(out)
    56: 
    57: 
    58: class TransformerBlock(nn.Module):
    59:     def __init__(self, hidden_size, n_heads, dropout=0.1):
    60:         super().__init__()
    61:         self.attn = MultiHeadAttention(hidden_size, n_heads, dropout)
    62:         self.ffn = nn.Sequential(
    63:             nn.Linear(hidden_size, hidden_size * 4),
    64:             nn.GELU(),
    65:             nn.Dropout(dropout),
    66:             nn.Linear(hidden_size * 4, hidden_size),
    67:         )
    68:         self.norm1 = nn.LayerNorm(hidden_size)
    69:         self.norm2 = nn.LayerNorm(hidden_size)
    70: 
    71:     def forward(self, x):
    72:         x = self.norm1(x + self.attn(x))
    73:         x = self.norm2(x + self.ffn(x))
    74:         return x
    75: 
    76: 
    77: class Custom(nn.Module):
    78:     """iTransformer: Inverted Transformer baseline.
    79: 
    80:     Treats each node's time series as a token (inverted view).
    81:     Self-attention captures cross-variate dependencies.
    82:     """
    83: 
    84:     def __init__(self, config: CustomConfig):
    85:         super().__init__()
    86:         self.num_features = config.num_features
    87:         self.revin = RevIN()
    88: 
    89:         # Embed each node's input_len time series -> hidden_size
    90:         self.embed = nn.Linear(config.input_len, config.hidden_size)
    91:         self.dropout = nn.Dropout(config.dropout)
    92: 
    93:         # Transformer encoder over nodes
    94:         self.layers = nn.ModuleList([
    95:             TransformerBlock(config.hidden_size, config.n_heads, config.dropout)
    96:             for _ in range(config.num_layers)
    97:         ])
    98:         self.norm = nn.LayerNorm(config.hidden_size)
    99: 
   100:         # Project hidden_size -> output_len per node
   101:         self.head = nn.Linear(config.hidden_size, config.output_len)
   102: 
   103:     def forward(self, inputs, inputs_timestamps):
   104:         # inputs: [B, T, N]
   105:         x = self.revin(inputs, "norm")
   106: 
   107:         # Invert: [B, N, T] -> embed -> [B, N, D]
   108:         h = self.dropout(self.embed(x.transpose(1, 2)))
   109: 
   110:         for layer in self.layers:
   111:             h = layer(h)
   112:         h = self.norm(h)
   113: 
   114:         # Project: [B, N, D] -> [B, N, T'] -> [B, T', N]
   115:         pred = self.head(h).transpose(1, 2)[:, :, :self.num_features]
   116:         pred = self.revin(pred, "denorm")
   117:         return pred
   118: 
   119: # CONFIG_OVERRIDES: override training hyperparameters for your method.
   120: # Allowed keys: lr, weight_decay.
   121: CONFIG_OVERRIDES = {'lr': 0.0005}
```

### `timesnet` baseline — editable region  [READ-ONLY — reference implementation]

In `BasicTS/custom_model.py`:

```python
Lines 1–150:
     1: import torch
     2: import torch.nn as nn
     3: import torch.nn.functional as F
     4: import torch.fft
     5: from dataclasses import dataclass, field
     6: from basicts.configs import BasicTSModelConfig
     7: 
     8: 
     9: @dataclass
    10: class CustomConfig(BasicTSModelConfig):
    11:     input_len: int = field(default=12)
    12:     output_len: int = field(default=12)
    13:     num_features: int = field(default=207)
    14:     hidden_size: int = field(default=64)
    15:     num_layers: int = field(default=2)
    16:     num_kernels: int = field(default=3)
    17:     top_k: int = field(default=3)
    18:     dropout: float = field(default=0.1)
    19: 
    20: 
    21: class RevIN(nn.Module):
    22:     def __init__(self, eps=1e-6):
    23:         super().__init__()
    24:         self.eps = eps
    25: 
    26:     def forward(self, x, mode):
    27:         if mode == "norm":
    28:             self.mean = x.mean(dim=1, keepdim=True).detach()
    29:             self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
    30:             return (x - self.mean) / self.stdev
    31:         else:
    32:             return x * self.stdev + self.mean
    33: 
    34: 
    35: class InceptionBlock(nn.Module):
    36:     """Multi-scale 2D convolution with different kernel sizes."""
    37:     def __init__(self, in_channels, out_channels, num_kernels=3):
    38:         super().__init__()
    39:         self.num_kernels = num_kernels
    40:         self.convs = nn.ModuleList([
    41:             nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
    42:             for i in range(num_kernels)
    43:         ])
    44:         self._init_weights()
    45: 
    46:     def _init_weights(self):
    47:         for m in self.modules():
    48:             if isinstance(m, nn.Conv2d):
    49:                 nn.init.kaiming_normal_(m.weight)
    50:                 if m.bias is not None:
    51:                     nn.init.constant_(m.bias, 0)
    52: 
    53:     def forward(self, x):
    54:         return torch.stack([conv(x) for conv in self.convs], dim=-1).mean(-1)
    55: 
    56: 
    57: class TimesBlock(nn.Module):
    58:     def __init__(self, input_len, output_len, hidden_size, num_kernels, top_k):
    59:         super().__init__()
    60:         self.input_len = input_len
    61:         self.output_len = output_len
    62:         self.top_k = top_k
    63:         intermediate = hidden_size * 4
    64:         self.conv = nn.Sequential(
    65:             InceptionBlock(hidden_size, intermediate, num_kernels),
    66:             nn.GELU(),
    67:             InceptionBlock(intermediate, hidden_size, num_kernels),
    68:         )
    69: 
    70:     def forward(self, x):
    71:         B, T, D = x.size()
    72:         # FFT to find dominant periods
    73:         xf = torch.fft.rfft(x, dim=1)
    74:         freq_amp = xf.abs().mean(dim=(0, -1))
    75:         freq_amp[0] = 0  # ignore DC
    76:         _, top_idx = torch.topk(freq_amp, self.top_k)
    77:         periods = T // top_idx.detach().cpu().numpy()
    78:         period_weight = xf.abs().mean(dim=-1)[:, top_idx]
    79: 
    80:         # Process each period
    81:         results = []
    82:         for p in periods:
    83:             if T % p != 0:
    84:                 pad_len = ((T // p) + 1) * p - T
    85:                 out = F.pad(x, (0, 0, 0, pad_len))
    86:             else:
    87:                 pad_len = 0
    88:                 out = x
    89:             out = out.reshape(B, -1, p, D).permute(0, 3, 1, 2)  # [B, D, rows, p]
    90:             out = self.conv(out)
    91:             out = out.permute(0, 2, 3, 1).reshape(B, -1, D)[:, :T, :]
    92:             results.append(out)
    93: 
    94:         results = torch.stack(results, dim=-1)
    95:         weights = F.softmax(period_weight, dim=1).unsqueeze(1).unsqueeze(1).expand_as(results)
    96:         return (results * weights).sum(-1) + x
    97: 
    98: 
    99: class Custom(nn.Module):
   100:     """TimesNet: Temporal 2D-Variation Modeling baseline.
   101: 
   102:     Transforms 1D time series to 2D based on detected periodicity,
   103:     then applies 2D Inception convolution to capture temporal patterns.
   104:     """
   105: 
   106:     def __init__(self, config: CustomConfig):
   107:         super().__init__()
   108:         self.output_len = config.output_len
   109:         self.revin = RevIN()
   110: 
   111:         # Embedding: feature -> hidden
   112:         padding = 1 if torch.__version__ >= "1.5.0" else 2
   113:         self.value_embed = nn.Conv1d(
   114:             config.num_features, config.hidden_size,
   115:             kernel_size=3, padding=padding, padding_mode="circular", bias=False)
   116: 
   117:         # Temporal alignment for forecasting
   118:         total_len = config.input_len + config.output_len
   119:         self.predict_linear = nn.Linear(config.input_len, total_len)
   120: 
   121:         # TimesNet blocks
   122:         self.blocks = nn.ModuleList([
   123:             TimesBlock(config.input_len, config.output_len,
   124:                        config.hidden_size, config.num_kernels, config.top_k)
   125:             for _ in range(config.num_layers)
   126:         ])
   127:         self.layer_norm = nn.LayerNorm(config.hidden_size)
   128: 
   129:         # Output projection
   130:         self.projection = nn.Linear(config.hidden_size, config.num_features)
   131: 
   132:     def forward(self, inputs, inputs_timestamps):
   133:         x = self.revin(inputs, "norm")
   134: 
   135:         # Embed: [B, T, N] -> [B, T, D]
   136:         h = self.value_embed(x.transpose(1, 2)).transpose(1, 2)
   137: 
   138:         # Extend to input_len + output_len
   139:         h = self.predict_linear(h.transpose(1, 2)).transpose(1, 2)
   140: 
   141:         for block in self.blocks:
   142:             h = self.layer_norm(block(h))
   143: 
   144:         pred = self.projection(h[:, -self.output_len:, :])
   145:         pred = self.revin(pred, "denorm")
   146:         return pred
   147: 
   148: # CONFIG_OVERRIDES: override training hyperparameters for your method.
   149: # Allowed keys: lr, weight_decay.
   150: CONFIG_OVERRIDES = {'lr': 0.001}
```

### `softs` baseline — editable region  [READ-ONLY — reference implementation]

In `BasicTS/custom_model.py`:

```python
Lines 1–134:
     1: import math
     2: import torch
     3: import torch.nn as nn
     4: import torch.nn.functional as F
     5: from dataclasses import dataclass, field
     6: from basicts.configs import BasicTSModelConfig
     7: 
     8: 
     9: @dataclass
    10: class CustomConfig(BasicTSModelConfig):
    11:     input_len: int = field(default=12)
    12:     output_len: int = field(default=12)
    13:     num_features: int = field(default=207)
    14:     hidden_size: int = field(default=512)
    15:     core_size: int = field(default=128)
    16:     num_layers: int = field(default=2)
    17:     dropout: float = field(default=0.05)
    18: 
    19: 
    20: class RevIN(nn.Module):
    21:     def __init__(self, eps=1e-6):
    22:         super().__init__()
    23:         self.eps = eps
    24: 
    25:     def forward(self, x, mode):
    26:         if mode == "norm":
    27:             self.mean = x.mean(dim=1, keepdim=True).detach()
    28:             self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
    29:             return (x - self.mean) / self.stdev
    30:         else:
    31:             return x * self.stdev + self.mean
    32: 
    33: 
    34: class MLP(nn.Module):
    35:     def __init__(self, in_dim, mid_dim, out_dim):
    36:         super().__init__()
    37:         self.fc1 = nn.Linear(in_dim, mid_dim)
    38:         self.fc2 = nn.Linear(mid_dim, out_dim)
    39: 
    40:     def forward(self, x):
    41:         return self.fc2(F.gelu(self.fc1(x)))
    42: 
    43: 
    44: class STAR(nn.Module):
    45:     """STar Aggregate-Redistribute module.
    46: 
    47:     Aggregates cross-variate info into a core representation via
    48:     stochastic pooling (training) or weighted mean (inference),
    49:     then redistributes back to each variate.
    50:     """
    51:     def __init__(self, hidden_size, core_size):
    52:         super().__init__()
    53:         self.ffn1 = MLP(hidden_size, hidden_size, core_size)
    54:         self.ffn2 = MLP(hidden_size + core_size, hidden_size, hidden_size)
    55: 
    56:     def forward(self, x):
    57:         B, N, D = x.shape
    58:         combined = self.ffn1(x)  # [B, N, core_size]
    59: 
    60:         if self.training:
    61:             # Stochastic pooling
    62:             ratio = F.softmax(combined, dim=1)  # [B, N, core_size]
    63:             ratio = ratio.transpose(1, 2).reshape(-1, N)
    64:             indices = torch.multinomial(ratio, 1)
    65:             indices = indices.view(B, -1, 1).transpose(1, 2)  # [B, 1, core_size]
    66:             core = torch.gather(combined, 1, indices)  # [B, 1, core_size]
    67:             core = core.repeat(1, N, 1)
    68:         else:
    69:             # Weighted mean
    70:             weight = F.softmax(combined, dim=1)
    71:             core = (combined * weight).sum(dim=1, keepdim=True).repeat(1, N, 1)
    72: 
    73:         return self.ffn2(torch.cat([x, core], dim=-1))
    74: 
    75: 
    76: class SOFTSBlock(nn.Module):
    77:     def __init__(self, hidden_size, core_size, dropout):
    78:         super().__init__()
    79:         self.star = STAR(hidden_size, core_size)
    80:         self.ffn = nn.Sequential(
    81:             nn.Linear(hidden_size, hidden_size * 4),
    82:             nn.GELU(),
    83:             nn.Dropout(dropout),
    84:             nn.Linear(hidden_size * 4, hidden_size),
    85:         )
    86:         self.norm1 = nn.LayerNorm(hidden_size)
    87:         self.norm2 = nn.LayerNorm(hidden_size)
    88: 
    89:     def forward(self, x):
    90:         x = self.norm1(x + self.star(x))
    91:         x = self.norm2(x + self.ffn(x))
    92:         return x
    93: 
    94: 
    95: class Custom(nn.Module):
    96:     """SOFTS: Series-Core Fusion baseline.
    97: 
    98:     Inverted architecture (nodes as tokens), using STAR modules
    99:     instead of self-attention for O(N) cross-variate communication.
   100:     """
   101: 
   102:     def __init__(self, config: CustomConfig):
   103:         super().__init__()
   104:         self.revin = RevIN()
   105: 
   106:         # Sequence embedding: [B, T, N] -> transpose -> [B, N, T] -> [B, N, D]
   107:         self.embed = nn.Linear(config.input_len, config.hidden_size)
   108:         self.embed_drop = nn.Dropout(config.dropout)
   109: 
   110:         self.layers = nn.ModuleList([
   111:             SOFTSBlock(config.hidden_size, config.core_size, config.dropout)
   112:             for _ in range(config.num_layers)
   113:         ])
   114:         self.norm = nn.LayerNorm(config.hidden_size)
   115: 
   116:         # Output: [B, N, D] -> [B, N, T'] -> [B, T', N]
   117:         self.head = nn.Linear(config.hidden_size, config.output_len)
   118: 
   119:     def forward(self, inputs, inputs_timestamps):
   120:         x = self.revin(inputs, "norm")
   121:         N = x.size(-1)
   122: 
   123:         h = self.embed_drop(self.embed(x.transpose(1, 2)))
   124:         for layer in self.layers:
   125:             h = layer(h)
   126:         h = self.norm(h)
   127: 
   128:         pred = self.head(h).transpose(1, 2)[:, :, :N]
   129:         pred = self.revin(pred, "denorm")
   130:         return pred
   131: 
   132: # CONFIG_OVERRIDES: override training hyperparameters for your method.
   133: # Allowed keys: lr, weight_decay.
   134: CONFIG_OVERRIDES = {'lr': 0.0005}
```

### `timemixer` baseline — editable region  [READ-ONLY — reference implementation]

In `BasicTS/custom_model.py`:

```python
Lines 1–254:
     1: import torch
     2: import torch.nn as nn
     3: import torch.nn.functional as F
     4: from dataclasses import dataclass, field
     5: from basicts.configs import BasicTSModelConfig
     6: 
     7: 
     8: @dataclass
     9: class CustomConfig(BasicTSModelConfig):
    10:     input_len: int = field(default=12)
    11:     output_len: int = field(default=12)
    12:     num_features: int = field(default=207)
    13:     hidden_size: int = field(default=64)
    14:     num_layers: int = field(default=2)
    15:     down_sampling_layers: int = field(default=2)
    16:     down_sampling_window: int = field(default=2)
    17:     dropout: float = field(default=0.1)
    18:     moving_avg: int = field(default=5)
    19: 
    20: 
    21: class RevIN(nn.Module):
    22:     def __init__(self, num_features, affine=True):
    23:         super().__init__()
    24:         self.eps = 1e-6
    25:         self.affine = affine
    26:         if affine:
    27:             self.weight = nn.Parameter(torch.ones(num_features))
    28:             self.bias = nn.Parameter(torch.zeros(num_features))
    29: 
    30:     def forward(self, x, mode):
    31:         if mode == "norm":
    32:             self.mean = x.mean(dim=1, keepdim=True).detach()
    33:             self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
    34:             x = (x - self.mean) / self.stdev
    35:             if self.affine:
    36:                 x = x * self.weight + self.bias
    37:             return x
    38:         else:
    39:             if self.affine:
    40:                 x = (x - self.bias) / (self.weight + self.eps * self.eps)
    41:             return x * self.stdev + self.mean
    42: 
    43: 
    44: class MovingAvgDecomp(nn.Module):
    45:     def __init__(self, kernel_size):
    46:         super().__init__()
    47:         self.avg = nn.AvgPool1d(kernel_size, stride=1)
    48:         self.pad_left = (kernel_size - 1) // 2
    49:         self.pad_right = kernel_size // 2
    50: 
    51:     def forward(self, x):
    52:         # x: [B, T, C]
    53:         trend = self.avg(F.pad(x.transpose(1, 2),
    54:                                 (self.pad_left, self.pad_right),
    55:                                 mode='replicate')).transpose(1, 2)
    56:         seasonal = x - trend
    57:         return seasonal, trend
    58: 
    59: 
    60: class MLPMixer(nn.Module):
    61:     '''2-layer MLP for scale mixing on the temporal dimension.'''
    62:     def __init__(self, in_dim, out_dim):
    63:         super().__init__()
    64:         self.fc1 = nn.Linear(in_dim, out_dim)
    65:         self.act = nn.GELU()
    66:         self.fc2 = nn.Linear(out_dim, out_dim)
    67: 
    68:     def forward(self, x):
    69:         return self.fc2(self.act(self.fc1(x)))
    70: 
    71: 
    72: class MultiScaleSeasonMixing(nn.Module):
    73:     '''Bottom-up mixing: finer -> coarser scales via learned MLP.'''
    74:     def __init__(self, input_len, down_sampling_layers, down_sampling_window):
    75:         super().__init__()
    76:         self.down_layers = nn.ModuleList([
    77:             MLPMixer(
    78:                 input_len // (down_sampling_window ** i),
    79:                 input_len // (down_sampling_window ** (i + 1))
    80:             )
    81:             for i in range(down_sampling_layers)
    82:         ])
    83: 
    84:     def forward(self, seasonal_list):
    85:         # seasonal_list[i]: [B, D, L_i] (permuted)
    86:         out_high = seasonal_list[0]
    87:         out_low = seasonal_list[1]
    88:         out_season_list = [out_high.permute(0, 2, 1)]
    89: 
    90:         for i in range(len(seasonal_list) - 1):
    91:             out_low_res = self.down_layers[i](out_high)
    92:             out_low = out_low + out_low_res
    93:             out_high = out_low
    94:             if i + 2 <= len(seasonal_list) - 1:
    95:                 out_low = seasonal_list[i + 2]
    96:             out_season_list.append(out_high.permute(0, 2, 1))
    97: 
    98:         return out_season_list
    99: 
   100: 
   101: class MultiScaleTrendMixing(nn.Module):
   102:     '''Top-down mixing: coarser -> finer scales via learned MLP.'''
   103:     def __init__(self, input_len, down_sampling_layers, down_sampling_window):
   104:         super().__init__()
   105:         self.up_layers = nn.ModuleList([
   106:             MLPMixer(
   107:                 input_len // (down_sampling_window ** (i + 1)),
   108:                 input_len // (down_sampling_window ** i)
   109:             )
   110:             for i in reversed(range(down_sampling_layers))
   111:         ])
   112: 
   113:     def forward(self, trend_list):
   114:         # trend_list[i]: [B, D, L_i] (permuted)
   115:         trend_rev = trend_list.copy()
   116:         trend_rev.reverse()
   117:         out_low = trend_rev[0]
   118:         out_high = trend_rev[1]
   119:         out_trend_list = [out_low.permute(0, 2, 1)]
   120: 
   121:         for i in range(len(trend_rev) - 1):
   122:             out_high_res = self.up_layers[i](out_low)
   123:             out_high = out_high + out_high_res
   124:             out_low = out_high
   125:             if i + 2 <= len(trend_rev) - 1:
   126:                 out_high = trend_rev[i + 2]
   127:             out_trend_list.append(out_low.permute(0, 2, 1))
   128: 
   129:         out_trend_list.reverse()
   130:         return out_trend_list
   131: 
   132: 
   133: class PastDecomposableMixing(nn.Module):
   134:     '''Decompose each scale, mix seasonal bottom-up and trend top-down.'''
   135:     def __init__(self, input_len, hidden_size, down_sampling_layers,
   136:                  down_sampling_window, moving_avg):
   137:         super().__init__()
   138:         self.decomp = MovingAvgDecomp(moving_avg)
   139:         self.season_mixing = MultiScaleSeasonMixing(
   140:             input_len, down_sampling_layers, down_sampling_window)
   141:         self.trend_mixing = MultiScaleTrendMixing(
   142:             input_len, down_sampling_layers, down_sampling_window)
   143:         self.out_cross_layer = nn.Sequential(
   144:             nn.Linear(hidden_size, hidden_size * 4),
   145:             nn.GELU(),
   146:             nn.Linear(hidden_size * 4, hidden_size),
   147:         )
   148: 
   149:     def forward(self, x_list):
   150:         seasonal_list, trend_list = [], []
   151:         for x in x_list:
   152:             seasonal, trend = self.decomp(x)
   153:             seasonal_list.append(seasonal.permute(0, 2, 1))
   154:             trend_list.append(trend.permute(0, 2, 1))
   155: 
   156:         seasonal_list = self.season_mixing(seasonal_list)
   157:         trend_list = self.trend_mixing(trend_list)
   158: 
   159:         out_list = []
   160:         for x, seasonal, trend in zip(x_list, seasonal_list, trend_list):
   161:             out = seasonal + trend
   162:             out = x + self.out_cross_layer(out)
   163:             out_list.append(out)
   164:         return out_list
   165: 
   166: 
   167: class Custom(nn.Module):
   168:     '''TimeMixer: Decomposable Multiscale Mixing baseline.
   169: 
   170:     Channel-independent mode: each variate processed separately.
   171:     Multi-scale decomposition + Past-Decomposable Mixing across scales.
   172:     '''
   173: 
   174:     def __init__(self, config: CustomConfig):
   175:         super().__init__()
   176:         self.num_features = config.num_features
   177:         self.output_len = config.output_len
   178:         self.down_layers = config.down_sampling_layers
   179:         self.down_window = config.down_sampling_window
   180:         D = config.hidden_size
   181: 
   182:         self.down_pool = nn.AvgPool1d(config.down_sampling_window)
   183: 
   184:         # Per-scale RevIN
   185:         self.norm_layers = nn.ModuleList([
   186:             RevIN(config.num_features, affine=True)
   187:             for _ in range(self.down_layers + 1)
   188:         ])
   189: 
   190:         # Embedding (channel-independent: 1 feature -> D)
   191:         padding = 1 if torch.__version__ >= "1.5.0" else 2
   192:         self.embed = nn.Conv1d(1, D, kernel_size=3, padding=padding,
   193:                                padding_mode="circular", bias=False)
   194: 
   195:         # PDM blocks (decomposition happens inside each block)
   196:         self.pdm_blocks = nn.ModuleList([
   197:             PastDecomposableMixing(
   198:                 config.input_len, D, self.down_layers,
   199:                 self.down_window, config.moving_avg)
   200:             for _ in range(config.num_layers)
   201:         ])
   202: 
   203:         # Per-scale prediction heads
   204:         self.predict_layers = nn.ModuleList([
   205:             nn.Linear(config.input_len // (self.down_window ** i), config.output_len)
   206:             for i in range(self.down_layers + 1)
   207:         ])
   208: 
   209:         # Channel-independent projection
   210:         self.projection = nn.Linear(D, 1)
   211: 
   212:     def forward(self, inputs, inputs_timestamps):
   213:         # inputs: [B, T, N]
   214:         B, T, N = inputs.shape
   215: 
   216:         # Multi-scale inputs
   217:         x_list = [inputs]
   218:         sample = inputs.permute(0, 2, 1)  # [B, N, T]
   219:         for _ in range(self.down_layers):
   220:             sample = self.down_pool(sample)
   221:             x_list.append(sample.permute(0, 2, 1))
   222: 
   223:         # Per-scale normalization + channel independence
   224:         for i in range(len(x_list)):
   225:             x_list[i] = self.norm_layers[i](x_list[i], "norm")
   226:             _, Li, _ = x_list[i].shape
   227:             x_list[i] = x_list[i].transpose(1, 2).reshape(-1, Li, 1)  # [B*N, Li, 1]
   228: 
   229:         # Embedding
   230:         h_list = []
   231:         for x in x_list:
   232:             h = self.embed(x.transpose(1, 2)).transpose(1, 2)  # [B*N, Li, D]
   233:             h_list.append(h)
   234: 
   235:         # Past Decomposable Mixing (decomposition inside blocks)
   236:         for block in self.pdm_blocks:
   237:             h_list = block(h_list)
   238: 
   239:         # Per-scale prediction and sum
   240:         pred_list = []
   241:         for i, h in enumerate(h_list):
   242:             # h: [B*N, Li, D] -> predict -> [B*N, T', D] -> project -> [B*N, T', 1]
   243:             p = self.predict_layers[i](h.permute(0, 2, 1)).permute(0, 2, 1)
   244:             p = self.projection(p)  # [B*N, T', 1]
   245:             p = p.reshape(B, N, self.output_len).permute(0, 2, 1)  # [B, T', N]
   246:             pred_list.append(p)
   247: 
   248:         prediction = sum(pred_list)
   249:         prediction = self.norm_layers[0](prediction, "denorm")
   250:         return prediction
   251: 
   252: # CONFIG_OVERRIDES: override training hyperparameters for your method.
   253: # Allowed keys: lr, weight_decay.
   254: CONFIG_OVERRIDES = {'lr': 0.001}
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
