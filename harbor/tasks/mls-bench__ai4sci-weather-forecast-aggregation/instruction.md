# MLS-Bench: ai4sci-weather-forecast-aggregation

# Weather Forecast Variable Aggregation

## Research Question
How should a weather forecasting model aggregate information across heterogeneous meteorological variables for optimal prediction?

## Background
Modern weather forecasting models process many meteorological variables simultaneously (temperature, pressure, wind, humidity at various pressure levels). ClimaX (Nguyen, Brandstetter, Kapoor, Gupta, Grover, "ClimaX: A foundation model for weather and climate", ICML 2023; arXiv:2301.10343) tokenizes each variable independently via per-variable patch embeddings, then aggregates them into a unified spatial representation before feeding into a Vision Transformer backbone. The default aggregation uses a learnable query with cross-attention over variable tokens at each spatial location, but this is just one design choice. Better aggregation strategies could capture inter-variable correlations more effectively. Code: https://github.com/microsoft/ClimaX.

## Task
Modify the `VariableAggregator` class in `custom_forecast.py` to implement a novel variable aggregation mechanism. The module receives per-variable patch embeddings and must produce a single aggregated representation per spatial location.

## Interface
```python
class VariableAggregator(nn.Module):
    def __init__(self, embed_dim, num_heads, num_vars):
        """
        Args:
            embed_dim (int): Embedding dimension D (1024).
            num_heads (int): Number of attention heads (16).
            num_vars (int): Number of input variables V (48).
        """
        ...

    def forward(self, x):
        """
        Args:
            x: [B, V, L, D] — per-variable patch embeddings
                B = batch size
                V = number of meteorological variables (48)
                L = number of spatial patches (512 = 16x32)
                D = embedding dimension (1024)

        Returns:
            [B, L, D] — aggregated representation per spatial location
        """
        ...
```

The input contains 48 variables: 3 surface constants (land-sea mask, orography, latitude), 3 surface fields (2 m temperature, 10 m wind u/v), and 42 pressure-level fields (geopotential, u/v wind, temperature, relative/specific humidity at 50–925 hPa). Each variable has been independently tokenized into L=512 patch embeddings of dimension D=1024.

## Available Components
You have access to standard PyTorch modules (`nn.Linear`, `nn.MultiheadAttention`, `nn.LayerNorm`, etc.) and `torch.nn.functional`. The FIXED section imports `torch`, `torch.nn`, and `torch.nn.functional as F`.

## Fixed Pipeline
ClimaX backbone, per-variable patch tokenization, fine-tuning recipe (initialized from pretrained ClimaX weights), data pipeline, ERA5 reanalysis at 5.625° resolution, optimizer/schedule, and the latitude-weighted RMSE metric are all fixed.

## Evaluation
The model is fine-tuned from pretrained ClimaX weights on ERA5 reanalysis data at 5.625-degree resolution and evaluated on three forecasting targets:
- **z500-3day**: Geopotential height at 500 hPa, 3-day lead time.
- **t850-5day**: Temperature at 850 hPa, 5-day lead time.
- **wind10m-7day**: 10 m wind speed, 7-day lead time.

Metric: Latitude-weighted RMSE (lower is better). The metric accounts for the convergence of meridians at the poles by weighting errors by the cosine of latitude.

## Reference Baselines
- **cross_attention**: ClimaX default aggregation. A learnable query token attends to all V variable tokens at each spatial location via multi-head cross-attention, producing one token per location.
- **mean_pooling**: Simple uniform mean across all V variable tokens at each spatial location. No additional learnable parameters; serves as a parameter-free lower bound.
- **learned_weighted_sum**: Learnable per-variable scalar weights normalized via softmax, then used to compute a weighted sum across variable tokens. More expressive than mean pooling but much simpler than cross-attention.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/ClimaX/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `ClimaX/custom_forecast.py`
- editable lines **310–351**
- editable lines **636–638**


Other files you may **read** for context (do not modify):
- `ClimaX/src/climax/arch.py`
- `ClimaX/src/climax/parallelpatchembed.py`
- `ClimaX/src/climax/utils/metrics.py`


## Readable Context


### `ClimaX/custom_forecast.py`  [EDITABLE — lines 310–351, lines 636–638 only]

```python
     1: """Custom Weather Forecast Variable Aggregation Script
     2: Based on ClimaX (Nguyen et al., 2023), evaluated on ERA5 at 5.625 deg.
     3: 
     4: The EDITABLE section contains the variable aggregation module that combines
     5: per-variable patch embeddings into a unified spatial representation.
     6: Everything else (ViT backbone, data loading, training loop) is FIXED.
     7: """
     8: 
     9: import math
    10: import os
    11: import time
    12: from functools import lru_cache
    13: 
    14: import numpy as np
    15: import torch
    16: import torch.nn as nn
    17: import torch.nn.functional as F
    18: from torch.utils.data import DataLoader, IterableDataset
    19: 
    20: # ============================================================================
    21: # FIXED — Data Loading (ClimaX-style ERA5 npy shards)
    22: # ============================================================================
    23: 
    24: DEFAULT_VARS = [
    25:     "land_sea_mask", "orography", "lattitude",
    26:     "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
    27:     "geopotential_50", "geopotential_250", "geopotential_500",
    28:     "geopotential_600", "geopotential_700", "geopotential_850", "geopotential_925",
    29:     "u_component_of_wind_50", "u_component_of_wind_250", "u_component_of_wind_500",
    30:     "u_component_of_wind_600", "u_component_of_wind_700", "u_component_of_wind_850",
    31:     "u_component_of_wind_925",
    32:     "v_component_of_wind_50", "v_component_of_wind_250", "v_component_of_wind_500",
    33:     "v_component_of_wind_600", "v_component_of_wind_700", "v_component_of_wind_850",
    34:     "v_component_of_wind_925",
    35:     "temperature_50", "temperature_250", "temperature_500",
    36:     "temperature_600", "temperature_700", "temperature_850", "temperature_925",
    37:     "relative_humidity_50", "relative_humidity_250", "relative_humidity_500",
    38:     "relative_humidity_600", "relative_humidity_700", "relative_humidity_850",
    39:     "relative_humidity_925",
    40:     "specific_humidity_50", "specific_humidity_250", "specific_humidity_500",
    41:     "specific_humidity_600", "specific_humidity_700", "specific_humidity_850",
    42:     "specific_humidity_925",
    43: ]
    44: 
    45: import random
    46: 
    47: 
    48: class NpyShardDataset(IterableDataset):
    49:     """Read npz shards of ERA5 data, yield individual time-step pairs."""
    50: 
    51:     def __init__(self, file_list, variables, out_variables, predict_range,
    52:                  hrs_each_step=1, shuffle=False):
    53:         super().__init__()
    54:         self.file_list = [f for f in file_list if "climatology" not in f]
    55:         self.variables = variables
    56:         self.out_variables = out_variables if out_variables else variables
    57:         self.predict_range = predict_range
    58:         self.hrs_each_step = hrs_each_step
    59:         self.shuffle = shuffle
    60: 
    61:     def __iter__(self):
    62:         files = list(self.file_list)
    63:         if self.shuffle:
    64:             random.shuffle(files)
    65:         for path in files:
    66:             data = np.load(path)
    67:             x = np.concatenate([data[k].astype(np.float32) for k in self.variables], axis=1)
    68:             x = torch.from_numpy(x)
    69:             y = np.concatenate([data[k].astype(np.float32) for k in self.out_variables], axis=1)
    70:             y = torch.from_numpy(y)
    71: 
    72:             inputs = x[: -self.predict_range]
    73:             predict_ranges = torch.ones(inputs.shape[0], dtype=torch.long) * self.predict_range
    74:             lead_times = (self.hrs_each_step * predict_ranges / 100.0).to(inputs.dtype)
    75:             output_ids = torch.arange(inputs.shape[0]) + predict_ranges
    76:             outputs = y[output_ids]
    77: 
    78:             for i in range(inputs.shape[0]):
    79:                 yield inputs[i], outputs[i], lead_times[i]
    80: 
    81: 
    82: class ShuffleBuffer(IterableDataset):
    83:     """Buffer-based shuffling for iterable datasets."""
    84: 
    85:     def __init__(self, dataset, buffer_size=5000):
    86:         super().__init__()
    87:         self.dataset = dataset
    88:         self.buffer_size = buffer_size
    89: 
    90:     def __iter__(self):
    91:         buf = []
    92:         for x in self.dataset:
    93:             if len(buf) == self.buffer_size:
    94:                 idx = random.randint(0, self.buffer_size - 1)
    95:                 yield buf[idx]
    96:                 buf[idx] = x
    97:             else:
    98:                 buf.append(x)
    99:         random.shuffle(buf)
   100:         while buf:
   101:             yield buf.pop()
   102: 
   103: 
   104: def collate_fn(batch):
   105:     inp = torch.stack([b[0] for b in batch])
   106:     out = torch.stack([b[1] for b in batch])
   107:     lead = torch.stack([b[2] for b in batch])
   108:     return inp, out, lead
   109: 
   110: 
   111: class Normalize:
   112:     """Channel-wise normalization transform."""
   113: 
   114:     def __init__(self, mean, std):
   115:         self.mean = torch.from_numpy(np.array(mean, dtype=np.float32))
   116:         self.std = torch.from_numpy(np.array(std, dtype=np.float32))
   117: 
   118:     def __call__(self, x):
   119:         # x: [C, H, W]
   120:         m = self.mean.to(x.device).view(-1, 1, 1)
   121:         s = self.std.to(x.device).view(-1, 1, 1)
   122:         return (x - m) / s
   123: 
   124:     def inverse(self, x):
   125:         m = self.mean.to(x.device).view(-1, 1, 1)
   126:         s = self.std.to(x.device).view(-1, 1, 1)
   127:         return x * s + m
   128: 
   129: 
   130: def build_normalize(data_dir, variables):
   131:     mean_dict = dict(np.load(os.path.join(data_dir, "normalize_mean.npz")))
   132:     std_dict = dict(np.load(os.path.join(data_dir, "normalize_std.npz")))
   133:     mean = np.concatenate([mean_dict.get(v, np.array([0.0])) for v in variables])
   134:     std = np.concatenate([std_dict[v] for v in variables])
   135:     return Normalize(mean, std)
   136: 
   137: 
   138: # ============================================================================
   139: # FIXED — Positional Embeddings
   140: # ============================================================================
   141: 
   142: def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w):
   143:     grid_h_arr = np.arange(grid_h, dtype=np.float64)
   144:     grid_w_arr = np.arange(grid_w, dtype=np.float64)
   145:     grid = np.meshgrid(grid_w_arr, grid_h_arr)
   146:     grid = np.stack(grid, axis=0).reshape([2, 1, grid_h, grid_w])
   147:     emb_h = _get_1d_sincos(embed_dim // 2, grid[0])
   148:     emb_w = _get_1d_sincos(embed_dim // 2, grid[1])
   149:     return np.concatenate([emb_h, emb_w], axis=1)
   150: 
   151: 
   152: def _get_1d_sincos(embed_dim, pos):
   153:     omega = np.arange(embed_dim // 2, dtype=np.float64)
   154:     omega /= embed_dim / 2.0
   155:     omega = 1.0 / 10000 ** omega
   156:     pos = pos.reshape(-1)
   157:     out = np.einsum("m,d->md", pos, omega)
   158:     return np.concatenate([np.sin(out), np.cos(out)], axis=1)
   159: 
   160: 
   161: def get_1d_sincos_pos_embed_from_grid(embed_dim, positions):
   162:     omega = np.arange(embed_dim // 2, dtype=np.float64)
   163:     omega /= embed_dim / 2.0
   164:     omega = 1.0 / 10000 ** omega
   165:     pos = np.array(positions, dtype=np.float64).reshape(-1)
   166:     out = np.einsum("m,d->md", pos, omega)
   167:     return np.concatenate([np.sin(out), np.cos(out)], axis=1)
   168: 
   169: 
   170: # ============================================================================
   171: # FIXED — Parallel Patch Embedding (from ClimaX)
   172: # ============================================================================
   173: 
   174: class ParallelVarPatchEmbed(nn.Module):
   175:     """Per-variable patch embedding using grouped convolutions."""
   176: 
   177:     def __init__(self, max_vars, img_size, patch_size, embed_dim):
   178:         super().__init__()
   179:         self.max_vars = max_vars
   180:         self.img_size = img_size
   181:         self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
   182:         self.grid_size = (img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1])
   183:         self.num_patches = self.grid_size[0] * self.grid_size[1]
   184: 
   185:         weights = torch.stack([torch.empty(embed_dim, 1, *self.patch_size) for _ in range(max_vars)])
   186:         self.proj_weights = nn.Parameter(weights)
   187:         biases = torch.stack([torch.empty(embed_dim) for _ in range(max_vars)])
   188:         self.proj_biases = nn.Parameter(biases)
   189:         self.reset_parameters()
   190: 
   191:     def reset_parameters(self):
   192:         for idx in range(self.max_vars):
   193:             nn.init.kaiming_uniform_(self.proj_weights[idx], a=math.sqrt(5))
   194:             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.proj_weights[idx])
   195:             if fan_in != 0:
   196:                 bound = 1 / math.sqrt(fan_in)
   197:                 nn.init.uniform_(self.proj_biases[idx], -bound, bound)
   198: 
   199:     def forward(self, x, var_ids=None):
   200:         B, C, H, W = x.shape
   201:         if var_ids is None:
   202:             var_ids = list(range(self.max_vars))
   203:         weights = self.proj_weights[var_ids].flatten(0, 1)
   204:         biases = self.proj_biases[var_ids].flatten(0, 1)
   205:         groups = len(var_ids)
   206:         proj = F.conv2d(x, weights, biases, groups=groups, stride=self.patch_size)
   207:         proj = proj.reshape(B, groups, -1, *proj.shape[-2:])
   208:         proj = proj.flatten(3).transpose(2, 3)  # B, V, L, D
   209:         return proj
   210: 
   211: 
   212: # ============================================================================
   213: # FIXED — ViT Backbone Components (from timm)
   214: # ============================================================================
   215: 
   216: class Attention(nn.Module):
   217:     """Multi-head self-attention."""
   218: 
   219:     def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
   220:         super().__init__()
   221:         self.num_heads = num_heads
   222:         self.head_dim = dim // num_heads
   223:         self.scale = self.head_dim ** -0.5
   224:         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
   225:         self.attn_drop = nn.Dropout(attn_drop)
   226:         self.proj = nn.Linear(dim, dim)
   227:         self.proj_drop = nn.Dropout(proj_drop)
   228: 
   229:     def forward(self, x):
   230:         B, N, C = x.shape
   231:         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
   232:         q, k, v = qkv.unbind(0)
   233:         attn = (q @ k.transpose(-2, -1)) * self.scale
   234:         attn = attn.softmax(dim=-1)
   235:         attn = self.attn_drop(attn)
   236:         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
   237:         x = self.proj(x)
   238:         x = self.proj_drop(x)
   239:         return x
   240: 
   241: 
   242: class DropPath(nn.Module):
   243:     """Stochastic depth."""
   244: 
   245:     def __init__(self, drop_prob=0.):
   246:         super().__init__()
   247:         self.drop_prob = drop_prob
   248: 
   249:     def forward(self, x):
   250:         if not self.training or self.drop_prob == 0.:
   251:             return x
   252:         keep_prob = 1 - self.drop_prob
   253:         shape = (x.shape[0],) + (1,) * (x.ndim - 1)
   254:         random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
   255:         if keep_prob > 0.:
   256:             random_tensor.div_(keep_prob)
   257:         return x * random_tensor
   258: 
   259: 
   260: class Mlp(nn.Module):
   261:     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
   262:         super().__init__()
   263:         out_features = out_features or in_features
   264:         hidden_features = hidden_features or in_features
   265:         self.fc1 = nn.Linear(in_features, hidden_features)
   266:         self.act = act_layer()
   267:         self.fc2 = nn.Linear(hidden_features, out_features)
   268:         self.drop = nn.Dropout(drop)
   269: 
   270:     def forward(self, x):
   271:         x = self.fc1(x)
   272:         x = self.act(x)
   273:         x = self.drop(x)
   274:         x = self.fc2(x)
   275:         x = self.drop(x)
   276:         return x
   277: 
   278: 
   279: class TransformerBlock(nn.Module):
   280:     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
   281:                  drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
   282:         super().__init__()
   283:         self.norm1 = norm_layer(dim)
   284:         self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
   285:                               attn_drop=attn_drop, proj_drop=drop)
   286:         self.drop_path1 = DropPath(drop_path)
   287:         self.norm2 = norm_layer(dim)
   288:         self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
   289:         self.drop_path2 = DropPath(drop_path)
   290: 
   291:     def forward(self, x):
   292:         x = x + self.drop_path1(self.attn(self.norm1(x)))
   293:         x = x + self.drop_path2(self.mlp(self.norm2(x)))
   294:         return x
   295: 
   296: 
   297: # ============================================================================
   298: # EDITABLE SECTION — Variable Aggregation Module (lines 310 to 351)
   299: # ============================================================================
   300: # This module takes per-variable patch embeddings and aggregates them into a
   301: # single representation per spatial location. The input is x: [B, V, L, D]
   302: # where B=batch, V=num_variables, L=num_patches, D=embed_dim. The output must
   303: # be [B, L, D].
   304: #
   305: # You may define any helper classes/functions within this section. The
   306: # VariableAggregator class MUST implement:
   307: #   __init__(self, embed_dim, num_heads, num_vars)
   308: #   forward(self, x)  where x: [B, V, L, D] -> returns [B, L, D]
   309: 
   310: class VariableAggregator(nn.Module):
   311:     """Aggregates per-variable patch embeddings into a unified representation.
   312: 
   313:     Default: learnable query with single-layer cross-attention (ClimaX default).
   314: 
   315:     Args:
   316:         embed_dim (int): Embedding dimension D.
   317:         num_heads (int): Number of attention heads for cross-attention.
   318:         num_vars (int): Number of input variables V.
   319:     """
   320: 
   321:     def __init__(self, embed_dim, num_heads, num_vars):
   322:         super().__init__()
   323:         self.embed_dim = embed_dim
   324:         self.num_heads = num_heads
   325:         self.num_vars = num_vars
   326:         # Learnable query token for cross-attention aggregation
   327:         self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
   328:         self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
   329: 
   330:     def forward(self, x):
   331:         """Aggregate variable embeddings.
   332: 
   333:         Args:
   334:             x: [B, V, L, D] — per-variable patch embeddings.
   335: 
   336:         Returns:
   337:             [B, L, D] — aggregated representation.
   338:         """
   339:         b, v, l, d = x.shape
   340:         # Reshape to treat each spatial location independently
   341:         x = x.permute(0, 2, 1, 3)   # B, L, V, D
   342:         x = x.reshape(b * l, v, d)  # B*L, V, D
   343: 
   344:         # Cross-attention: query attends to all variable tokens
   345:         query = self.var_query.expand(b * l, -1, -1)  # B*L, 1, D
   346:         out, _ = self.var_agg(query, x, x)             # B*L, 1, D
   347:         out = out.squeeze(1)                            # B*L, D
   348: 
   349:         out = out.reshape(b, l, d)  # B, L, D
   350:         return out
   351: 
   352: 
   353: # ============================================================================
   354: # FIXED — ClimaX Model (uses VariableAggregator from editable section)
   355: # ============================================================================
   356: 
   357: class ClimaXModel(nn.Module):
   358:     """ClimaX weather forecasting model with pluggable variable aggregation."""
   359: 
   360:     def __init__(self, default_vars, img_size=(32, 64), patch_size=2,
   361:                  embed_dim=1024, depth=8, decoder_depth=2, num_heads=16,
   362:                  mlp_ratio=4.0, drop_path=0.1, drop_rate=0.1):
   363:         super().__init__()
   364:         self.img_size = img_size
   365:         self.patch_size = patch_size
   366:         self.default_vars = default_vars
   367:         self.embed_dim = embed_dim
   368:         num_vars = len(default_vars)
   369: 
   370:         # Per-variable patch embedding
   371:         self.token_embeds = ParallelVarPatchEmbed(num_vars, img_size, patch_size, embed_dim)
   372:         self.num_patches = self.token_embeds.num_patches
   373: 
   374:         # Variable embedding
   375:         self.var_embed = nn.Parameter(torch.zeros(1, num_vars, embed_dim), requires_grad=True)
   376:         self.var_map = {var: i for i, var in enumerate(default_vars)}
   377: 
   378:         # Variable aggregation (EDITABLE component)
   379:         self.var_aggregator = VariableAggregator(embed_dim, num_heads, num_vars)
   380: 
   381:         # Positional embedding and lead time embedding
   382:         self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)
   383:         self.lead_time_embed = nn.Linear(1, embed_dim)
   384: 
   385:         # ViT backbone
   386:         self.pos_drop = nn.Dropout(p=drop_rate)
   387:         dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
   388:         self.blocks = nn.ModuleList([
   389:             TransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
   390:                              drop=drop_rate, drop_path=dpr[i])
   391:             for i in range(depth)
   392:         ])
   393:         self.norm = nn.LayerNorm(embed_dim)
   394: 
   395:         # Prediction head
   396:         head_layers = []
   397:         for _ in range(decoder_depth):
   398:             head_layers.append(nn.Linear(embed_dim, embed_dim))
   399:             head_layers.append(nn.GELU())
   400:         head_layers.append(nn.Linear(embed_dim, num_vars * patch_size ** 2))
   401:         self.head = nn.Sequential(*head_layers)
   402: 
   403:         self.initialize_weights()
   404: 
   405:     def initialize_weights(self):
   406:         pos_embed = get_2d_sincos_pos_embed(
   407:             self.pos_embed.shape[-1],
   408:             self.img_size[0] // self.patch_size,
   409:             self.img_size[1] // self.patch_size,
   410:         )
   411:         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
   412: 
   413:         var_embed = get_1d_sincos_pos_embed_from_grid(
   414:             self.var_embed.shape[-1], np.arange(len(self.default_vars))
   415:         )
   416:         self.var_embed.data.copy_(torch.from_numpy(var_embed).float().unsqueeze(0))
   417: 
   418:         for i in range(len(self.token_embeds.proj_weights)):
   419:             w = self.token_embeds.proj_weights[i].data
   420:             nn.init.trunc_normal_(w.view([w.shape[0], -1]), std=0.02)
   421: 
   422:         self.apply(self._init_weights)
   423: 
   424:     def _init_weights(self, m):
   425:         if isinstance(m, nn.Linear):
   426:             nn.init.trunc_normal_(m.weight, std=0.02)
   427:             if m.bias is not None:
   428:                 nn.init.constant_(m.bias, 0)
   429:         elif isinstance(m, nn.LayerNorm):
   430:             nn.init.constant_(m.bias, 0)
   431:             nn.init.constant_(m.weight, 1.0)
   432: 
   433:     @lru_cache(maxsize=None)
   434:     def get_var_ids(self, vars_tuple, device):
   435:         ids = np.array([self.var_map[var] for var in vars_tuple])
   436:         return torch.from_numpy(ids).to(device)
   437: 
   438:     def unpatchify(self, x):
   439:         p = self.patch_size
   440:         c = len(self.default_vars)
   441:         h = self.img_size[0] // p
   442:         w = self.img_size[1] // p
   443:         x = x.reshape(x.shape[0], h, w, p, p, c)
   444:         x = torch.einsum("nhwpqc->nchpwq", x)
   445:         return x.reshape(x.shape[0], c, h * p, w * p)
   446: 
   447:     def forward_encoder(self, x, lead_times, variables):
   448:         if isinstance(variables, list):
   449:             variables = tuple(variables)
   450:         var_ids = self.get_var_ids(variables, x.device)
   451: 
   452:         # Per-variable tokenization
   453:         x = self.token_embeds(x, var_ids)  # B, V, L, D
   454: 
   455:         # Add variable embedding
   456:         var_embed = self.var_embed[:, var_ids, :]
   457:         x = x + var_embed.unsqueeze(2)  # B, V, L, D
   458: 
   459:         # Variable aggregation (EDITABLE)
   460:         x = self.var_aggregator(x)  # B, L, D
   461: 
   462:         # Add positional embedding
   463:         x = x + self.pos_embed
   464: 
   465:         # Add lead time embedding
   466:         lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))  # B, D
   467:         x = x + lead_time_emb.unsqueeze(1)
   468: 
   469:         x = self.pos_drop(x)
   470: 
   471:         for blk in self.blocks:
   472:             x = blk(x)
   473:         x = self.norm(x)
   474:         return x
   475: 
   476:     def forward(self, x, lead_times, variables):
   477:         out = self.forward_encoder(x, lead_times, variables)
   478:         preds = self.head(out)
   479:         preds = self.unpatchify(preds)
   480:         return preds
   481: 
   482: 
   483: # ============================================================================
   484: # FIXED — Metrics (Latitude-weighted RMSE)
   485: # ============================================================================
   486: 
   487: def lat_weighted_mse(pred, y, lat):
   488:     """Latitude-weighted MSE for training loss."""
   489:     error = (pred - y) ** 2
   490:     w_lat = np.cos(np.deg2rad(lat))
   491:     w_lat = w_lat / w_lat.mean()
   492:     w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(
   493:         dtype=error.dtype, device=error.device
   494:     )
   495:     return (error * w_lat.unsqueeze(1)).mean(dim=1).mean()
   496: 
   497: 
   498: def lat_weighted_rmse(pred, y, lat, out_variables):
   499:     """Latitude-weighted RMSE per variable (for evaluation)."""
   500:     error = (pred - y) ** 2

[truncated: showing at most 500 lines / 60000 bytes from ClimaX/custom_forecast.py]
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **z500-3day** — wall-clock budget `08:00:00`, compute share `1.0`
- **t850-5day** — wall-clock budget `08:00:00`, compute share `1.0`
- **wind10m-7day** — wall-clock budget `08:00:00`, compute share `1.0`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.



## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `cross_attention` baseline — editable region  [READ-ONLY — reference implementation]

In `ClimaX/custom_forecast.py`:

```python
Lines 310–346:
   307: #   __init__(self, embed_dim, num_heads, num_vars)
   308: #   forward(self, x)  where x: [B, V, L, D] -> returns [B, L, D]
   309: 
   310: class VariableAggregator(nn.Module):
   311:     """Cross-attention variable aggregation (ClimaX default).
   312: 
   313:     A learnable query token attends to all V variable tokens at each spatial
   314:     location via multi-head cross-attention, producing one token per location.
   315: 
   316:     Args:
   317:         embed_dim (int): Embedding dimension D.
   318:         num_heads (int): Number of attention heads.
   319:         num_vars (int): Number of input variables V.
   320:     """
   321: 
   322:     def __init__(self, embed_dim, num_heads, num_vars):
   323:         super().__init__()
   324:         self.embed_dim = embed_dim
   325:         self.num_heads = num_heads
   326:         self.num_vars = num_vars
   327:         self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
   328:         self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
   329: 
   330:     def forward(self, x):
   331:         """
   332:         Args:
   333:             x: [B, V, L, D] — per-variable patch embeddings.
   334:         Returns:
   335:             [B, L, D] — aggregated representation.
   336:         """
   337:         b, v, l, d = x.shape
   338:         x = x.permute(0, 2, 1, 3)   # B, L, V, D
   339:         x = x.reshape(b * l, v, d)  # B*L, V, D
   340: 
   341:         query = self.var_query.expand(b * l, -1, -1)  # B*L, 1, D
   342:         out, _ = self.var_agg(query, x, x)             # B*L, 1, D
   343:         out = out.squeeze(1)                            # B*L, D
   344: 
   345:         out = out.reshape(b, l, d)  # B, L, D
   346:         return out
   347: 
   348: # ============================================================================
   349: # FIXED — ClimaX Model (uses VariableAggregator from editable section)

Lines 631–633:
   628:     # ========================================================================
   629:     # EDITABLE — CONFIG_OVERRIDES
   630:     # ========================================================================
   631:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   632:     # Allowed keys: learning_rate, weight_decay, warmup_steps, patience, grad_clip.
   633:     CONFIG_OVERRIDES = {}
   634:     # ========================================================================
   635:     # FIXED — Apply overrides and continue training setup
   636:     # ========================================================================
```

### `mean_pooling` baseline — editable region  [READ-ONLY — reference implementation]

In `ClimaX/custom_forecast.py`:

```python
Lines 310–337:
   307: #   __init__(self, embed_dim, num_heads, num_vars)
   308: #   forward(self, x)  where x: [B, V, L, D] -> returns [B, L, D]
   309: 
   310: class VariableAggregator(nn.Module):
   311:     """Mean pooling variable aggregation.
   312: 
   313:     Simply averages all V variable tokens at each spatial location.
   314:     No additional learnable parameters.
   315: 
   316:     Args:
   317:         embed_dim (int): Embedding dimension D.
   318:         num_heads (int): Number of attention heads (unused).
   319:         num_vars (int): Number of input variables V (unused).
   320:     """
   321: 
   322:     def __init__(self, embed_dim, num_heads, num_vars):
   323:         super().__init__()
   324:         self.embed_dim = embed_dim
   325:         self.num_heads = num_heads
   326:         self.num_vars = num_vars
   327: 
   328:     def forward(self, x):
   329:         """
   330:         Args:
   331:             x: [B, V, L, D] — per-variable patch embeddings.
   332:         Returns:
   333:             [B, L, D] — aggregated representation.
   334:         """
   335:         # Average across variable dimension
   336:         out = x.mean(dim=1)  # B, L, D
   337:         return out
   338: 
   339: # ============================================================================
   340: # FIXED — ClimaX Model (uses VariableAggregator from editable section)

Lines 622–624:
   619:     # ========================================================================
   620:     # EDITABLE — CONFIG_OVERRIDES
   621:     # ========================================================================
   622:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   623:     # Allowed keys: learning_rate, weight_decay, warmup_steps, patience, grad_clip.
   624:     CONFIG_OVERRIDES = {}
   625:     # ========================================================================
   626:     # FIXED — Apply overrides and continue training setup
   627:     # ========================================================================
```

### `learned_weighted_sum` baseline — editable region  [READ-ONLY — reference implementation]

In `ClimaX/custom_forecast.py`:

```python
Lines 310–342:
   307: #   __init__(self, embed_dim, num_heads, num_vars)
   308: #   forward(self, x)  where x: [B, V, L, D] -> returns [B, L, D]
   309: 
   310: class VariableAggregator(nn.Module):
   311:     """Learned weighted sum variable aggregation.
   312: 
   313:     Learns a scalar weight per variable, applies softmax normalization,
   314:     then computes a weighted sum across variable tokens.
   315: 
   316:     Args:
   317:         embed_dim (int): Embedding dimension D.
   318:         num_heads (int): Number of attention heads (unused).
   319:         num_vars (int): Number of input variables V.
   320:     """
   321: 
   322:     def __init__(self, embed_dim, num_heads, num_vars):
   323:         super().__init__()
   324:         self.embed_dim = embed_dim
   325:         self.num_heads = num_heads
   326:         self.num_vars = num_vars
   327:         # Learnable weight per variable
   328:         self.var_weights = nn.Parameter(torch.zeros(num_vars), requires_grad=True)
   329: 
   330:     def forward(self, x):
   331:         """
   332:         Args:
   333:             x: [B, V, L, D] — per-variable patch embeddings.
   334:         Returns:
   335:             [B, L, D] — aggregated representation.
   336:         """
   337:         # Softmax-normalized variable weights
   338:         w = F.softmax(self.var_weights, dim=0)  # V
   339:         w = w.view(1, -1, 1, 1)                # 1, V, 1, 1
   340:         # Weighted sum across variables
   341:         out = (x * w).sum(dim=1)  # B, L, D
   342:         return out
   343: 
   344: # ============================================================================
   345: # FIXED — ClimaX Model (uses VariableAggregator from editable section)

Lines 627–629:
   624:     # ========================================================================
   625:     # EDITABLE — CONFIG_OVERRIDES
   626:     # ========================================================================
   627:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   628:     # Allowed keys: learning_rate, weight_decay, warmup_steps, patience, grad_clip.
   629:     CONFIG_OVERRIDES = {}
   630:     # ========================================================================
   631:     # FIXED — Apply overrides and continue training setup
   632:     # ========================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
