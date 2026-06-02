# MLS-Bench: ts-short-term-forecast

# Univariate Short-Term Forecasting on the M4 Competition Dataset

## Research Question
Can one univariate forecasting component (seasonal decomposition, scale normalization, horizon-aware decoding, multi-scale temporal mixing) deliver consistently low SMAPE across the very different seasonal regimes — monthly, quarterly, yearly — of the M4 competition, under a fixed training and evaluation protocol?

## Background
The M4 competition (Makridakis, Spiliotis & Assimakopoulos, "The M4 Competition: 100,000 time series and 61 forecasting methods", *International Journal of Forecasting*, 2018/2020) collected 100,000 short, real-world series across 6 seasonal patterns (Yearly, Quarterly, Monthly, Weekly, Daily, Hourly). It is the standard benchmark for *short, univariate, many-series* forecasting and is dominated by combinations of statistical and ML methods. The Time-Series-Library protocol (Wu et al., ICLR 2023) wraps M4 with per-pattern fixed look-back / horizon settings, SMAPE training loss, and SMAPE / MAPE / OWA scoring.

## Objective
Implement the `Model` class in `models/Custom.py`. Output shape is `[batch, pred_len, c_out]`; for the M4 univariate setting `enc_in == c_out == 1`.

## Model Interface
```python
class Model(nn.Module):
    def __init__(self, configs):
        # configs.task_name == "short_term_forecast"
        # configs.seq_len, configs.pred_len, configs.enc_in (=1), configs.c_out (=1)
        ...

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc:   [batch, seq_len, 1]
        # returns: [batch, pred_len, 1]
        ...

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return out[:, -self.pred_len:, :]
```

## Datasets and Fixed Protocol
Three M4 seasonal patterns. Per Time-Series-Library defaults:
- **Monthly** — `seq_len=104`, `pred_len=18`, `frequency_map=12`.
- **Quarterly** — `seq_len=52`, `pred_len=8`, `frequency_map=4`.
- **Yearly** — `seq_len=42`, `pred_len=6`, `frequency_map=1`.

All settings: `features=M`, `enc_in=1`, `loss=SMAPE`. Train/test splits are the official M4 splits.

## Metrics
SMAPE (primary) and MAPE — lower is better. Computed by the Time-Series-Library M4 evaluator on the official test horizon.

## Reference Implementations (read-only)
Four reference models from `models/`:

- **DLinear** — Zeng et al., AAAI 2023 (arXiv 2205.13504). Trend+seasonal decomposition followed by linear projection. TS-Lib short-term defaults: `moving_avg=25`, Adam `lr=1e-3`, `train_epochs=10`, `batch_size=16`. Source: https://github.com/cure-lab/LTSF-Linear.
- **TimesNet** — Wu et al., ICLR 2023 (arXiv 2210.02186). FFT-based period discovery + 2D Inception conv. TS-Lib short-term defaults: `e_layers=2`, `d_model=32`, `d_ff=32`, `top_k=5`, `num_kernels=6`. Source: https://github.com/thuml/Time-Series-Library.
- **PatchTST** — Nie et al., ICLR 2023 (arXiv 2211.14730). Channel-independent Transformer over patches. TS-Lib short-term defaults: `e_layers=3`, `n_heads=4`, `d_model=128`, `d_ff=256`, `patch_len=16`, `stride=8`. Source: https://github.com/yuqinie98/PatchTST.
- **TimeMixer** — Wang et al., ICLR 2024 (arXiv 2405.14616). MLP-based multiscale decomposition with Past-Decomposable-Mixing and Future-Multipredictor-Mixing. TS-Lib short-term defaults: `e_layers=4`, `d_model=16`, `d_ff=32`, `down_sampling_layers=1`, `down_sampling_window=2`, `down_sampling_method="avg"`. Source: https://github.com/kwuking/TimeMixer.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/Time-Series-Library/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `Time-Series-Library/models/Custom.py`
- editable: **entire file**


Other files you may **read** for context (do not modify):
- `Time-Series-Library/models/DLinear.py`
- `Time-Series-Library/models/TimesNet.py`
- `Time-Series-Library/models/PatchTST.py`
- `Time-Series-Library/layers/AutoCorrelation.py`
- `Time-Series-Library/layers/Autoformer_EncDec.py`
- `Time-Series-Library/layers/Conv_Blocks.py`
- `Time-Series-Library/layers/Crossformer_EncDec.py`
- `Time-Series-Library/layers/Embed.py`
- `Time-Series-Library/layers/FourierCorrelation.py`
- `Time-Series-Library/layers/SelfAttention_Family.py`
- `Time-Series-Library/layers/StandardNorm.py`
- `Time-Series-Library/layers/Transformer_EncDec.py`


## Readable Context


### `Time-Series-Library/models/Custom.py`  [EDITABLE — entire file only]

```python
     1: import torch
     2: import torch.nn as nn
     3: 
     4: 
     5: class Model(nn.Module):
     6:     """
     7:     Custom model for short-term time series forecasting (M4 dataset).
     8: 
     9:     Forward signature: forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)
    10:     - x_enc: [batch, seq_len, enc_in] — input time series (enc_in=1 for M4)
    11:     - x_mark_enc: [batch, seq_len, time_features] — time feature encoding
    12:     - x_dec: [batch, label_len+pred_len, dec_in] — decoder input
    13:     - x_mark_dec: [batch, label_len+pred_len, time_features] — decoder time features
    14: 
    15:     Must return: [batch, pred_len, c_out] for forecasting
    16:     """
    17: 
    18:     def __init__(self, configs):
    19:         super(Model, self).__init__()
    20:         self.task_name = configs.task_name
    21:         self.seq_len = configs.seq_len
    22:         self.pred_len = configs.pred_len
    23:         self.enc_in = configs.enc_in
    24:         self.c_out = configs.c_out
    25:         # TODO: Define your model architecture here
    26: 
    27:     def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
    28:         """
    29:         Forecasting: given input sequence, predict future values.
    30:         Input: x_enc [batch, seq_len, 1]
    31:         Output: [batch, pred_len, 1]
    32:         """
    33:         # TODO: Implement your forecasting logic
    34:         batch_size = x_enc.shape[0]
    35:         return torch.zeros(batch_size, self.pred_len, self.c_out).to(x_enc.device)
    36: 
    37:     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
    38:         if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
    39:             dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
    40:             return dec_out[:, -self.pred_len:, :]
    41:         return None
```

### `Time-Series-Library/models/DLinear.py`  [READ-ONLY — do not edit]

```python
     1: import torch
     2: import torch.nn as nn
     3: import torch.nn.functional as F
     4: from layers.Autoformer_EncDec import series_decomp
     5: 
     6: 
     7: class Model(nn.Module):
     8:     """
     9:     Paper link: https://arxiv.org/pdf/2205.13504.pdf
    10:     """
    11: 
    12:     def __init__(self, configs, individual=False):
    13:         """
    14:         individual: Bool, whether shared model among different variates.
    15:         """
    16:         super(Model, self).__init__()
    17:         self.task_name = configs.task_name
    18:         self.seq_len = configs.seq_len
    19:         if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
    20:             self.pred_len = configs.seq_len
    21:         else:
    22:             self.pred_len = configs.pred_len
    23:         # Series decomposition block from Autoformer
    24:         self.decompsition = series_decomp(configs.moving_avg)
    25:         self.individual = individual
    26:         self.channels = configs.enc_in
    27: 
    28:         if self.individual:
    29:             self.Linear_Seasonal = nn.ModuleList()
    30:             self.Linear_Trend = nn.ModuleList()
    31: 
    32:             for i in range(self.channels):
    33:                 self.Linear_Seasonal.append(
    34:                     nn.Linear(self.seq_len, self.pred_len))
    35:                 self.Linear_Trend.append(
    36:                     nn.Linear(self.seq_len, self.pred_len))
    37: 
    38:                 self.Linear_Seasonal[i].weight = nn.Parameter(
    39:                     (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
    40:                 self.Linear_Trend[i].weight = nn.Parameter(
    41:                     (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
    42:         else:
    43:             self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
    44:             self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
    45: 
    46:             self.Linear_Seasonal.weight = nn.Parameter(
    47:                 (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
    48:             self.Linear_Trend.weight = nn.Parameter(
    49:                 (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
    50: 
    51:         if self.task_name == 'classification':
    52:             self.projection = nn.Linear(
    53:                 configs.enc_in * configs.seq_len, configs.num_class)
    54: 
    55:     def encoder(self, x):
    56:         seasonal_init, trend_init = self.decompsition(x)
    57:         seasonal_init, trend_init = seasonal_init.permute(
    58:             0, 2, 1), trend_init.permute(0, 2, 1)
    59:         if self.individual:
    60:             seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
    61:                                           dtype=seasonal_init.dtype).to(seasonal_init.device)
    62:             trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
    63:                                        dtype=trend_init.dtype).to(trend_init.device)
    64:             for i in range(self.channels):
    65:                 seasonal_output[:, i, :] = self.Linear_Seasonal[i](
    66:                     seasonal_init[:, i, :])
    67:                 trend_output[:, i, :] = self.Linear_Trend[i](
    68:                     trend_init[:, i, :])
    69:         else:
    70:             seasonal_output = self.Linear_Seasonal(seasonal_init)
    71:             trend_output = self.Linear_Trend(trend_init)
    72:         x = seasonal_output + trend_output
    73:         return x.permute(0, 2, 1)
    74: 
    75:     def forecast(self, x_enc):
    76:         # Encoder
    77:         return self.encoder(x_enc)
    78: 
    79:     def imputation(self, x_enc):
    80:         # Encoder
    81:         return self.encoder(x_enc)
    82: 
    83:     def anomaly_detection(self, x_enc):
    84:         # Encoder
    85:         return self.encoder(x_enc)
    86: 
    87:     def classification(self, x_enc):
    88:         # Encoder
    89:         enc_out = self.encoder(x_enc)
    90:         # Output
    91:         # (batch_size, seq_length * d_model)
    92:         output = enc_out.reshape(enc_out.shape[0], -1)
    93:         # (batch_size, num_classes)
    94:         output = self.projection(output)
    95:         return output
    96: 
    97:     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
    98:         if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
    99:             dec_out = self.forecast(x_enc)
   100:             return dec_out[:, -self.pred_len:, :]  # [B, L, D]
   101:         if self.task_name == 'imputation':
   102:             dec_out = self.imputation(x_enc)
   103:             return dec_out  # [B, L, D]
   104:         if self.task_name == 'anomaly_detection':
   105:             dec_out = self.anomaly_detection(x_enc)
   106:             return dec_out  # [B, L, D]
   107:         if self.task_name == 'classification':
   108:             dec_out = self.classification(x_enc)
   109:             return dec_out  # [B, N]
   110:         return None
```

### `Time-Series-Library/models/TimesNet.py`  [READ-ONLY — do not edit]

```python
     1: import torch
     2: import torch.nn as nn
     3: import torch.nn.functional as F
     4: import torch.fft
     5: from layers.Embed import DataEmbedding
     6: from layers.Conv_Blocks import Inception_Block_V1
     7: 
     8: 
     9: def FFT_for_Period(x, k=2):
    10:     # [B, T, C]
    11:     xf = torch.fft.rfft(x, dim=1)
    12:     # find period by amplitudes
    13:     frequency_list = abs(xf).mean(0).mean(-1)
    14:     frequency_list[0] = 0
    15:     _, top_list = torch.topk(frequency_list, k)
    16:     top_list = top_list.detach().cpu().numpy()
    17:     period = x.shape[1] // top_list
    18:     return period, abs(xf).mean(-1)[:, top_list]
    19: 
    20: 
    21: class TimesBlock(nn.Module):
    22:     def __init__(self, configs):
    23:         super(TimesBlock, self).__init__()
    24:         self.seq_len = configs.seq_len
    25:         self.pred_len = configs.pred_len
    26:         self.k = configs.top_k
    27:         # parameter-efficient design
    28:         self.conv = nn.Sequential(
    29:             Inception_Block_V1(configs.d_model, configs.d_ff,
    30:                                num_kernels=configs.num_kernels),
    31:             nn.GELU(),
    32:             Inception_Block_V1(configs.d_ff, configs.d_model,
    33:                                num_kernels=configs.num_kernels)
    34:         )
    35: 
    36:     def forward(self, x):
    37:         B, T, N = x.size()
    38:         period_list, period_weight = FFT_for_Period(x, self.k)
    39: 
    40:         res = []
    41:         for i in range(self.k):
    42:             period = period_list[i]
    43:             # padding
    44:             if (self.seq_len + self.pred_len) % period != 0:
    45:                 length = (
    46:                                  ((self.seq_len + self.pred_len) // period) + 1) * period
    47:                 padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
    48:                 out = torch.cat([x, padding], dim=1)
    49:             else:
    50:                 length = (self.seq_len + self.pred_len)
    51:                 out = x
    52:             # reshape
    53:             out = out.reshape(B, length // period, period,
    54:                               N).permute(0, 3, 1, 2).contiguous()
    55:             # 2D conv: from 1d Variation to 2d Variation
    56:             out = self.conv(out)
    57:             # reshape back
    58:             out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
    59:             res.append(out[:, :(self.seq_len + self.pred_len), :])
    60:         res = torch.stack(res, dim=-1)
    61:         # adaptive aggregation
    62:         period_weight = F.softmax(period_weight, dim=1)
    63:         period_weight = period_weight.unsqueeze(
    64:             1).unsqueeze(1).repeat(1, T, N, 1)
    65:         res = torch.sum(res * period_weight, -1)
    66:         # residual connection
    67:         res = res + x
    68:         return res
    69: 
    70: 
    71: class Model(nn.Module):
    72:     """
    73:     Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    74:     """
    75: 
    76:     def __init__(self, configs):
    77:         super(Model, self).__init__()
    78:         self.configs = configs
    79:         self.task_name = configs.task_name
    80:         self.seq_len = configs.seq_len
    81:         self.label_len = configs.label_len
    82:         self.pred_len = configs.pred_len
    83:         self.model = nn.ModuleList([TimesBlock(configs)
    84:                                     for _ in range(configs.e_layers)])
    85:         self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
    86:                                            configs.dropout)
    87:         self.layer = configs.e_layers
    88:         self.layer_norm = nn.LayerNorm(configs.d_model)
    89:         if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
    90:             self.predict_linear = nn.Linear(
    91:                 self.seq_len, self.pred_len + self.seq_len)
    92:             self.projection = nn.Linear(
    93:                 configs.d_model, configs.c_out, bias=True)
    94:         if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
    95:             self.projection = nn.Linear(
    96:                 configs.d_model, configs.c_out, bias=True)
    97:         if self.task_name == 'classification':
    98:             self.act = F.gelu
    99:             self.dropout = nn.Dropout(configs.dropout)
   100:             self.projection = nn.Linear(
   101:                 configs.d_model * configs.seq_len, configs.num_class)
   102: 
   103:     def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
   104:         # Normalization from Non-stationary Transformer
   105:         means = x_enc.mean(1, keepdim=True).detach()
   106:         x_enc = x_enc.sub(means)
   107:         stdev = torch.sqrt(
   108:             torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
   109:         x_enc = x_enc.div(stdev)
   110: 
   111:         # embedding
   112:         enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
   113:         enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
   114:             0, 2, 1)  # align temporal dimension
   115:         # TimesNet
   116:         for i in range(self.layer):
   117:             enc_out = self.layer_norm(self.model[i](enc_out))
   118:         # project back
   119:         dec_out = self.projection(enc_out)
   120: 
   121:         # De-Normalization from Non-stationary Transformer
   122:         dec_out = dec_out.mul(
   123:                   (stdev[:, 0, :].unsqueeze(1).repeat(
   124:                       1, self.pred_len + self.seq_len, 1)))
   125:         dec_out = dec_out.add(
   126:                   (means[:, 0, :].unsqueeze(1).repeat(
   127:                       1, self.pred_len + self.seq_len, 1)))
   128:         return dec_out
   129: 
   130:     def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
   131:         # Normalization from Non-stationary Transformer
   132:         means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
   133:         means = means.unsqueeze(1).detach()
   134:         x_enc = x_enc.sub(means)
   135:         x_enc = x_enc.masked_fill(mask == 0, 0)
   136:         stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
   137:                            torch.sum(mask == 1, dim=1) + 1e-5)
   138:         stdev = stdev.unsqueeze(1).detach()
   139:         x_enc = x_enc.div(stdev)
   140: 
   141:         # embedding
   142:         enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
   143:         # TimesNet
   144:         for i in range(self.layer):
   145:             enc_out = self.layer_norm(self.model[i](enc_out))
   146:         # project back
   147:         dec_out = self.projection(enc_out)
   148: 
   149:         # De-Normalization from Non-stationary Transformer
   150:         dec_out = dec_out.mul(
   151:                   (stdev[:, 0, :].unsqueeze(1).repeat(
   152:                       1, self.pred_len + self.seq_len, 1)))
   153:         dec_out = dec_out.add(
   154:                   (means[:, 0, :].unsqueeze(1).repeat(
   155:                       1, self.pred_len + self.seq_len, 1)))
   156:         return dec_out
   157: 
   158:     def anomaly_detection(self, x_enc):
   159:         # Normalization from Non-stationary Transformer
   160:         means = x_enc.mean(1, keepdim=True).detach()
   161:         x_enc = x_enc.sub(means)
   162:         stdev = torch.sqrt(
   163:             torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
   164:         x_enc = x_enc.div(stdev)
   165: 
   166:         # embedding
   167:         enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
   168:         # TimesNet
   169:         for i in range(self.layer):
   170:             enc_out = self.layer_norm(self.model[i](enc_out))
   171:         # project back
   172:         dec_out = self.projection(enc_out)
   173: 
   174:         # De-Normalization from Non-stationary Transformer
   175:         dec_out = dec_out.mul(
   176:                   (stdev[:, 0, :].unsqueeze(1).repeat(
   177:                       1, self.pred_len + self.seq_len, 1)))
   178:         dec_out = dec_out.add(
   179:                   (means[:, 0, :].unsqueeze(1).repeat(
   180:                       1, self.pred_len + self.seq_len, 1)))
   181:         return dec_out
   182: 
   183:     def classification(self, x_enc, x_mark_enc):
   184:         # embedding
   185:         enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
   186:         # TimesNet
   187:         for i in range(self.layer):
   188:             enc_out = self.layer_norm(self.model[i](enc_out))
   189: 
   190:         # Output
   191:         # the output transformer encoder/decoder embeddings don't include non-linearity
   192:         output = self.act(enc_out)
   193:         output = self.dropout(output)
   194:         # zero-out padding embeddings
   195:         output = output * x_mark_enc.unsqueeze(-1)
   196:         # (batch_size, seq_length * d_model)
   197:         output = output.reshape(output.shape[0], -1)
   198:         output = self.projection(output)  # (batch_size, num_classes)
   199:         return output
   200: 
   201:     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
   202:         if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
   203:             dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
   204:             return dec_out[:, -self.pred_len:, :]  # [B, L, D]
   205:         if self.task_name == 'imputation':
   206:             dec_out = self.imputation(
   207:                 x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
   208:             return dec_out  # [B, L, D]
   209:         if self.task_name == 'anomaly_detection':
   210:             dec_out = self.anomaly_detection(x_enc)
   211:             return dec_out  # [B, L, D]
   212:         if self.task_name == 'classification':
   213:             dec_out = self.classification(x_enc, x_mark_enc)
   214:             return dec_out  # [B, N]
   215:         return None
```

### `Time-Series-Library/models/PatchTST.py`  [READ-ONLY — do not edit]

```python
     1: import torch
     2: from torch import nn
     3: from layers.Transformer_EncDec import Encoder, EncoderLayer
     4: from layers.SelfAttention_Family import FullAttention, AttentionLayer
     5: from layers.Embed import PatchEmbedding
     6: 
     7: class Transpose(nn.Module):
     8:     def __init__(self, *dims, contiguous=False): 
     9:         super().__init__()
    10:         self.dims, self.contiguous = dims, contiguous
    11:     def forward(self, x):
    12:         if self.contiguous: return x.transpose(*self.dims).contiguous()
    13:         else: return x.transpose(*self.dims)
    14: 
    15: 
    16: class FlattenHead(nn.Module):
    17:     def __init__(self, n_vars, nf, target_window, head_dropout=0):
    18:         super().__init__()
    19:         self.n_vars = n_vars
    20:         self.flatten = nn.Flatten(start_dim=-2)
    21:         self.linear = nn.Linear(nf, target_window)
    22:         self.dropout = nn.Dropout(head_dropout)
    23: 
    24:     def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
    25:         x = self.flatten(x)
    26:         x = self.linear(x)
    27:         x = self.dropout(x)
    28:         return x
    29: 
    30: 
    31: class Model(nn.Module):
    32:     """
    33:     Paper link: https://arxiv.org/pdf/2211.14730.pdf
    34:     """
    35: 
    36:     def __init__(self, configs, patch_len=16, stride=8):
    37:         """
    38:         patch_len: int, patch len for patch_embedding
    39:         stride: int, stride for patch_embedding
    40:         """
    41:         super().__init__()
    42:         self.task_name = configs.task_name
    43:         self.seq_len = configs.seq_len
    44:         self.pred_len = configs.pred_len
    45:         padding = stride
    46: 
    47:         # patching and embedding
    48:         self.patch_embedding = PatchEmbedding(
    49:             configs.d_model, patch_len, stride, padding, configs.dropout)
    50: 
    51:         # Encoder
    52:         self.encoder = Encoder(
    53:             [
    54:                 EncoderLayer(
    55:                     AttentionLayer(
    56:                         FullAttention(False, configs.factor, attention_dropout=configs.dropout,
    57:                                       output_attention=False), configs.d_model, configs.n_heads),
    58:                     configs.d_model,
    59:                     configs.d_ff,
    60:                     dropout=configs.dropout,
    61:                     activation=configs.activation
    62:                 ) for l in range(configs.e_layers)
    63:             ],
    64:             norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
    65:         )
    66: 
    67:         # Prediction Head
    68:         self.head_nf = configs.d_model * \
    69:                        int((configs.seq_len - patch_len) / stride + 2)
    70:         if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
    71:             self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
    72:                                     head_dropout=configs.dropout)
    73:         elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
    74:             self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
    75:                                     head_dropout=configs.dropout)
    76:         elif self.task_name == 'classification':
    77:             self.flatten = nn.Flatten(start_dim=-2)
    78:             self.dropout = nn.Dropout(configs.dropout)
    79:             self.projection = nn.Linear(
    80:                 self.head_nf * configs.enc_in, configs.num_class)
    81: 
    82:     def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
    83:         # Normalization from Non-stationary Transformer
    84:         means = x_enc.mean(1, keepdim=True).detach()
    85:         x_enc = x_enc - means
    86:         stdev = torch.sqrt(
    87:             torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
    88:         x_enc /= stdev
    89: 
    90:         # do patching and embedding
    91:         x_enc = x_enc.permute(0, 2, 1)
    92:         # u: [bs * nvars x patch_num x d_model]
    93:         enc_out, n_vars = self.patch_embedding(x_enc)
    94: 
    95:         # Encoder
    96:         # z: [bs * nvars x patch_num x d_model]
    97:         enc_out, attns = self.encoder(enc_out)
    98:         # z: [bs x nvars x patch_num x d_model]
    99:         enc_out = torch.reshape(
   100:             enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
   101:         # z: [bs x nvars x d_model x patch_num]
   102:         enc_out = enc_out.permute(0, 1, 3, 2)
   103: 
   104:         # Decoder
   105:         dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
   106:         dec_out = dec_out.permute(0, 2, 1)
   107: 
   108:         # De-Normalization from Non-stationary Transformer
   109:         dec_out = dec_out * \
   110:                   (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
   111:         dec_out = dec_out + \
   112:                   (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
   113:         return dec_out
   114: 
   115:     def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
   116:         # Normalization from Non-stationary Transformer
   117:         means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
   118:         means = means.unsqueeze(1).detach()
   119:         x_enc = x_enc - means
   120:         x_enc = x_enc.masked_fill(mask == 0, 0)
   121:         stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
   122:                            torch.sum(mask == 1, dim=1) + 1e-5)
   123:         stdev = stdev.unsqueeze(1).detach()
   124:         x_enc /= stdev
   125: 
   126:         # do patching and embedding
   127:         x_enc = x_enc.permute(0, 2, 1)
   128:         # u: [bs * nvars x patch_num x d_model]
   129:         enc_out, n_vars = self.patch_embedding(x_enc)
   130: 
   131:         # Encoder
   132:         # z: [bs * nvars x patch_num x d_model]
   133:         enc_out, attns = self.encoder(enc_out)
   134:         # z: [bs x nvars x patch_num x d_model]
   135:         enc_out = torch.reshape(
   136:             enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
   137:         # z: [bs x nvars x d_model x patch_num]
   138:         enc_out = enc_out.permute(0, 1, 3, 2)
   139: 
   140:         # Decoder
   141:         dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
   142:         dec_out = dec_out.permute(0, 2, 1)
   143: 
   144:         # De-Normalization from Non-stationary Transformer
   145:         dec_out = dec_out * \
   146:                   (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
   147:         dec_out = dec_out + \
   148:                   (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
   149:         return dec_out
   150: 
   151:     def anomaly_detection(self, x_enc):
   152:         # Normalization from Non-stationary Transformer
   153:         means = x_enc.mean(1, keepdim=True).detach()
   154:         x_enc = x_enc - means
   155:         stdev = torch.sqrt(
   156:             torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
   157:         x_enc /= stdev
   158: 
   159:         # do patching and embedding
   160:         x_enc = x_enc.permute(0, 2, 1)
   161:         # u: [bs * nvars x patch_num x d_model]
   162:         enc_out, n_vars = self.patch_embedding(x_enc)
   163: 
   164:         # Encoder
   165:         # z: [bs * nvars x patch_num x d_model]
   166:         enc_out, attns = self.encoder(enc_out)
   167:         # z: [bs x nvars x patch_num x d_model]
   168:         enc_out = torch.reshape(
   169:             enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
   170:         # z: [bs x nvars x d_model x patch_num]
   171:         enc_out = enc_out.permute(0, 1, 3, 2)
   172: 
   173:         # Decoder
   174:         dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
   175:         dec_out = dec_out.permute(0, 2, 1)
   176: 
   177:         # De-Normalization from Non-stationary Transformer
   178:         dec_out = dec_out * \
   179:                   (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
   180:         dec_out = dec_out + \
   181:                   (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
   182:         return dec_out
   183: 
   184:     def classification(self, x_enc, x_mark_enc):
   185:         # Normalization from Non-stationary Transformer
   186:         means = x_enc.mean(1, keepdim=True).detach()
   187:         x_enc = x_enc - means
   188:         stdev = torch.sqrt(
   189:             torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
   190:         x_enc /= stdev
   191: 
   192:         # do patching and embedding
   193:         x_enc = x_enc.permute(0, 2, 1)
   194:         # u: [bs * nvars x patch_num x d_model]
   195:         enc_out, n_vars = self.patch_embedding(x_enc)
   196: 
   197:         # Encoder
   198:         # z: [bs * nvars x patch_num x d_model]
   199:         enc_out, attns = self.encoder(enc_out)
   200:         # z: [bs x nvars x patch_num x d_model]
   201:         enc_out = torch.reshape(
   202:             enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
   203:         # z: [bs x nvars x d_model x patch_num]
   204:         enc_out = enc_out.permute(0, 1, 3, 2)
   205: 
   206:         # Decoder
   207:         output = self.flatten(enc_out)
   208:         output = self.dropout(output)
   209:         output = output.reshape(output.shape[0], -1)
   210:         output = self.projection(output)  # (batch_size, num_classes)
   211:         return output
   212: 
   213:     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
   214:         if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
   215:             dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
   216:             return dec_out[:, -self.pred_len:, :]  # [B, L, D]
   217:         if self.task_name == 'imputation':
   218:             dec_out = self.imputation(
   219:                 x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
   220:             return dec_out  # [B, L, D]
   221:         if self.task_name == 'anomaly_detection':
   222:             dec_out = self.anomaly_detection(x_enc)
   223:             return dec_out  # [B, L, D]
   224:         if self.task_name == 'classification':
   225:             dec_out = self.classification(x_enc, x_mark_enc)
   226:             return dec_out  # [B, N]
   227:         return None
```

### `Time-Series-Library/layers/AutoCorrelation.py`  [READ-ONLY — do not edit]

```python
     1: import torch
     2: import torch.nn as nn
     3: import torch.nn.functional as F
     4: import matplotlib.pyplot as plt
     5: import numpy as np
     6: import math
     7: from math import sqrt
     8: import os
     9: 
    10: 
    11: class AutoCorrelation(nn.Module):
    12:     """
    13:     AutoCorrelation Mechanism with the following two phases:
    14:     (1) period-based dependencies discovery
    15:     (2) time delay aggregation
    16:     This block can replace the self-attention family mechanism seamlessly.
    17:     """
    18: 
    19:     def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
    20:         super(AutoCorrelation, self).__init__()
    21:         self.factor = factor
    22:         self.scale = scale
    23:         self.mask_flag = mask_flag
    24:         self.output_attention = output_attention
    25:         self.dropout = nn.Dropout(attention_dropout)
    26: 
    27:     def time_delay_agg_training(self, values, corr):
    28:         """
    29:         SpeedUp version of Autocorrelation (a batch-normalization style design)
    30:         This is for the training phase.
    31:         """
    32:         head = values.shape[1]
    33:         channel = values.shape[2]
    34:         length = values.shape[3]
    35:         # find top k
    36:         top_k = int(self.factor * math.log(length))
    37:         mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
    38:         index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
    39:         weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
    40:         # update corr
    41:         tmp_corr = torch.softmax(weights, dim=-1)
    42:         # aggregation
    43:         tmp_values = values
    44:         delays_agg = torch.zeros_like(values).float()
    45:         for i in range(top_k):
    46:             pattern = torch.roll(tmp_values, -int(index[i]), -1)
    47:             delays_agg = delays_agg + pattern * \
    48:                          (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
    49:         return delays_agg
    50: 
    51:     def time_delay_agg_inference(self, values, corr):
    52:         """
    53:         SpeedUp version of Autocorrelation (a batch-normalization style design)
    54:         This is for the inference phase.
    55:         """
    56:         batch = values.shape[0]
    57:         head = values.shape[1]
    58:         channel = values.shape[2]
    59:         length = values.shape[3]
    60:         # index init
    61:         init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).to(values.device)
    62:         # find top k
    63:         top_k = int(self.factor * math.log(length))
    64:         mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
    65:         weights, delay = torch.topk(mean_value, top_k, dim=-1)
    66:         # update corr
    67:         tmp_corr = torch.softmax(weights, dim=-1)
    68:         # aggregation
    69:         tmp_values = values.repeat(1, 1, 1, 2)
    70:         delays_agg = torch.zeros_like(values).float()
    71:         for i in range(top_k):
    72:             tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
    73:             pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
    74:             delays_agg = delays_agg + pattern * \
    75:                          (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
    76:         return delays_agg
    77: 
    78:     def time_delay_agg_full(self, values, corr):
    79:         """
    80:         Standard version of Autocorrelation
    81:         """
    82:         batch = values.shape[0]
    83:         head = values.shape[1]
    84:         channel = values.shape[2]
    85:         length = values.shape[3]
    86:         # index init
    87:         init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).to(values.device)
    88:         # find top k
    89:         top_k = int(self.factor * math.log(length))
    90:         weights, delay = torch.topk(corr, top_k, dim=-1)
    91:         # update corr
    92:         tmp_corr = torch.softmax(weights, dim=-1)
    93:         # aggregation
    94:         tmp_values = values.repeat(1, 1, 1, 2)
    95:         delays_agg = torch.zeros_like(values).float()
    96:         for i in range(top_k):
    97:             tmp_delay = init_index + delay[..., i].unsqueeze(-1)
    98:             pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
    99:             delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
   100:         return delays_agg
   101: 
   102:     def forward(self, queries, keys, values, attn_mask):
   103:         B, L, H, E = queries.shape
   104:         _, S, _, D = values.shape
   105:         if L > S:
   106:             zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
   107:             values = torch.cat([values, zeros], dim=1)
   108:             keys = torch.cat([keys, zeros], dim=1)
   109:         else:
   110:             values = values[:, :L, :, :]
   111:             keys = keys[:, :L, :, :]
   112: 
   113:         # period-based dependencies
   114:         q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
   115:         k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
   116:         res = q_fft * torch.conj(k_fft)
   117:         corr = torch.fft.irfft(res, dim=-1)
   118: 
   119:         # time delay agg
   120:         if self.training:
   121:             V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
   122:         else:
   123:             V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
   124: 
   125:         if self.output_attention:
   126:             return (V.contiguous(), corr.permute(0, 3, 1, 2))
   127:         else:
   128:             return (V.contiguous(), None)
   129: 
   130: 
   131: class AutoCorrelationLayer(nn.Module):
   132:     def __init__(self, correlation, d_model, n_heads, d_keys=None,
   133:                  d_values=None):
   134:         super(AutoCorrelationLayer, self).__init__()
   135: 
   136:         d_keys = d_keys or (d_model // n_heads)
   137:         d_values = d_values or (d_model // n_heads)
   138: 
   139:         self.inner_correlation = correlation
   140:         self.query_projection = nn.Linear(d_model, d_keys * n_heads)
   141:         self.key_projection = nn.Linear(d_model, d_keys * n_heads)
   142:         self.value_projection = nn.Linear(d_model, d_values * n_heads)
   143:         self.out_projection = nn.Linear(d_values * n_heads, d_model)
   144:         self.n_heads = n_heads
   145: 
   146:     def forward(self, queries, keys, values, attn_mask):
   147:         B, L, _ = queries.shape
   148:         _, S, _ = keys.shape
   149:         H = self.n_heads
   150: 
   151:         queries = self.query_projection(queries).view(B, L, H, -1)
   152:         keys = self.key_projection(keys).view(B, S, H, -1)
   153:         values = self.value_projection(values).view(B, S, H, -1)
   154: 
   155:         out, attn = self.inner_correlation(
   156:             queries,
   157:             keys,
   158:             values,
   159:             attn_mask
   160:         )
   161:         out = out.view(B, L, -1)
   162: 
   163:         return self.out_projection(out), attn
```

### `Time-Series-Library/layers/Autoformer_EncDec.py`  [READ-ONLY — do not edit]

```python
     1: import torch
     2: import torch.nn as nn
     3: import torch.nn.functional as F
     4: 
     5: 
     6: class my_Layernorm(nn.Module):
     7:     """
     8:     Special designed layernorm for the seasonal part
     9:     """
    10: 
    11:     def __init__(self, channels):
    12:         super(my_Layernorm, self).__init__()
    13:         self.layernorm = nn.LayerNorm(channels)
    14: 
    15:     def forward(self, x):
    16:         x_hat = self.layernorm(x)
    17:         bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
    18:         return x_hat - bias
    19: 
    20: 
    21: class moving_avg(nn.Module):
    22:     """
    23:     Moving average block to highlight the trend of time series
    24:     """
    25: 
    26:     def __init__(self, kernel_size, stride):
    27:         super(moving_avg, self).__init__()
    28:         self.kernel_size = kernel_size
    29:         self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    30: 
    31:     def forward(self, x):
    32:         # padding on the both ends of time series
    33:         front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
    34:         end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
    35:         x = torch.cat([front, x, end], dim=1)
    36:         x = self.avg(x.permute(0, 2, 1))
    37:         x = x.permute(0, 2, 1)
    38:         return x
    39: 
    40: 
    41: class series_decomp(nn.Module):
    42:     """
    43:     Series decomposition block
    44:     """
    45: 
    46:     def __init__(self, kernel_size):
    47:         super(series_decomp, self).__init__()
    48:         self.moving_avg = moving_avg(kernel_size, stride=1)
    49: 
    50:     def forward(self, x):
    51:         moving_mean = self.moving_avg(x)
    52:         res = x - moving_mean
    53:         return res, moving_mean
    54: 
    55: 
    56: class series_decomp_multi(nn.Module):
    57:     """
    58:     Multiple Series decomposition block from FEDformer
    59:     """
    60: 
    61:     def __init__(self, kernel_size):
    62:         super(series_decomp_multi, self).__init__()
    63:         self.kernel_size = kernel_size
    64:         self.series_decomp = [series_decomp(kernel) for kernel in kernel_size]
    65: 
    66:     def forward(self, x):
    67:         moving_mean = []
    68:         res = []
    69:         for func in self.series_decomp:
    70:             sea, moving_avg = func(x)
    71:             moving_mean.append(moving_avg)
    72:             res.append(sea)
    73: 
    74:         sea = sum(res) / len(res)
    75:         moving_mean = sum(moving_mean) / len(moving_mean)
    76:         return sea, moving_mean
    77: 
    78: 
    79: class EncoderLayer(nn.Module):
    80:     """
    81:     Autoformer encoder layer with the progressive decomposition architecture
    82:     """
    83: 
    84:     def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
    85:         super(EncoderLayer, self).__init__()
    86:         d_ff = d_ff or 4 * d_model
    87:         self.attention = attention
    88:         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
    89:         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
    90:         self.decomp1 = series_decomp(moving_avg)
    91:         self.decomp2 = series_decomp(moving_avg)
    92:         self.dropout = nn.Dropout(dropout)
    93:         self.activation = F.relu if activation == "relu" else F.gelu
    94: 
    95:     def forward(self, x, attn_mask=None):
    96:         new_x, attn = self.attention(
    97:             x, x, x,
    98:             attn_mask=attn_mask
    99:         )
   100:         x = x + self.dropout(new_x)
   101:         x, _ = self.decomp1(x)
   102:         y = x
   103:         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
   104:         y = self.dropout(self.conv2(y).transpose(-1, 1))
   105:         res, _ = self.decomp2(x + y)
   106:         return res, attn
   107: 
   108: 
   109: class Encoder(nn.Module):
   110:     """
   111:     Autoformer encoder
   112:     """
   113: 
   114:     def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
   115:         super(Encoder, self).__init__()
   116:         self.attn_layers = nn.ModuleList(attn_layers)
   117:         self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
   118:         self.norm = norm_layer
   119: 
   120:     def forward(self, x, attn_mask=None):
   121:         attns = []
   122:         if self.conv_layers is not None:
   123:             for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
   124:                 x, attn = attn_layer(x, attn_mask=attn_mask)
   125:                 x = conv_layer(x)
   126:                 attns.append(attn)
   127:             x, attn = self.attn_layers[-1](x)
   128:             attns.append(attn)
   129:         else:
   130:             for attn_layer in self.attn_layers:
   131:                 x, attn = attn_layer(x, attn_mask=attn_mask)
   132:                 attns.append(attn)
   133: 
   134:         if self.norm is not None:
   135:             x = self.norm(x)
   136: 
   137:         return x, attns
   138: 
   139: 
   140: class DecoderLayer(nn.Module):
   141:     """
   142:     Autoformer decoder layer with the progressive decomposition architecture
   143:     """
   144: 
   145:     def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
   146:                  moving_avg=25, dropout=0.1, activation="relu"):
   147:         super(DecoderLayer, self).__init__()
   148:         d_ff = d_ff or 4 * d_model
   149:         self.self_attention = self_attention
   150:         self.cross_attention = cross_attention
   151:         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
   152:         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
   153:         self.decomp1 = series_decomp(moving_avg)
   154:         self.decomp2 = series_decomp(moving_avg)
   155:         self.decomp3 = series_decomp(moving_avg)
   156:         self.dropout = nn.Dropout(dropout)
   157:         self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
   158:                                     padding_mode='circular', bias=False)
   159:         self.activation = F.relu if activation == "relu" else F.gelu
   160: 
   161:     def forward(self, x, cross, x_mask=None, cross_mask=None):
   162:         x = x + self.dropout(self.self_attention(
   163:             x, x, x,
   164:             attn_mask=x_mask
   165:         )[0])
   166:         x, trend1 = self.decomp1(x)
   167:         x = x + self.dropout(self.cross_attention(
   168:             x, cross, cross,
   169:             attn_mask=cross_mask
   170:         )[0])
   171:         x, trend2 = self.decomp2(x)
   172:         y = x
   173:         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
   174:         y = self.dropout(self.conv2(y).transpose(-1, 1))
   175:         x, trend3 = self.decomp3(x + y)
   176: 
   177:         residual_trend = trend1 + trend2 + trend3
   178:         residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
   179:         return x, residual_trend
   180: 
   181: 
   182: class Decoder(nn.Module):
   183:     """
   184:     Autoformer encoder
   185:     """
   186: 
   187:     def __init__(self, layers, norm_layer=None, projection=None):
   188:         super(Decoder, self).__init__()
   189:         self.layers = nn.ModuleList(layers)
   190:         self.norm = norm_layer
   191:         self.projection = projection
   192: 
   193:     def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
   194:         for layer in self.layers:
   195:             x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
   196:             trend = trend + residual_trend
   197: 
   198:         if self.norm is not None:
   199:             x = self.norm(x)
   200: 
   201:         if self.projection is not None:
   202:             x = self.projection(x)
   203:         return x, trend
```

### `Time-Series-Library/layers/Conv_Blocks.py`  [READ-ONLY — do not edit]

```python
     1: import torch
     2: import torch.nn as nn
     3: 
     4: 
     5: class Inception_Block_V1(nn.Module):
     6:     def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
     7:         super(Inception_Block_V1, self).__init__()
     8:         self.in_channels = in_channels
     9:         self.out_channels = out_channels
    10:         self.num_kernels = num_kernels
    11:         kernels = []
    12:         for i in range(self.num_kernels):
    13:             kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
    14:         self.kernels = nn.ModuleList(kernels)
    15:         if init_weight:
    16:             self._initialize_weights()
    17: 
    18:     def _initialize_weights(self):
    19:         for m in self.modules():
    20:             if isinstance(m, nn.Conv2d):
    21:                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    22:                 if m.bias is not None:
    23:                     nn.init.constant_(m.bias, 0)
    24: 
    25:     def forward(self, x):
    26:         res_list = []
    27:         for i in range(self.num_kernels):
    28:             res_list.append(self.kernels[i](x))
    29:         res = torch.stack(res_list, dim=-1).mean(-1)
    30:         return res
    31: 
    32: 
    33: class Inception_Block_V2(nn.Module):
    34:     def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
    35:         super(Inception_Block_V2, self).__init__()
    36:         self.in_channels = in_channels
    37:         self.out_channels = out_channels
    38:         self.num_kernels = num_kernels
    39:         kernels = []
    40:         for i in range(self.num_kernels // 2):
    41:             kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
    42:             kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
    43:         kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
    44:         self.kernels = nn.ModuleList(kernels)
    45:         if init_weight:
    46:             self._initialize_weights()
    47: 
    48:     def _initialize_weights(self):
    49:         for m in self.modules():
    50:             if isinstance(m, nn.Conv2d):
    51:                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    52:                 if m.bias is not None:
    53:                     nn.init.constant_(m.bias, 0)
    54: 
    55:     def forward(self, x):
    56:         res_list = []
    57:         for i in range(self.num_kernels // 2 * 2 + 1):
    58:             res_list.append(self.kernels[i](x))
    59:         res = torch.stack(res_list, dim=-1).mean(-1)
    60:         return res
```

### `Time-Series-Library/layers/Crossformer_EncDec.py`  [READ-ONLY — do not edit]

```python
     1: import torch
     2: import torch.nn as nn
     3: from einops import rearrange, repeat
     4: from layers.SelfAttention_Family import TwoStageAttentionLayer
     5: 
     6: 
     7: class SegMerging(nn.Module):
     8:     def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
     9:         super().__init__()
    10:         self.d_model = d_model
    11:         self.win_size = win_size
    12:         self.linear_trans = nn.Linear(win_size * d_model, d_model)
    13:         self.norm = norm_layer(win_size * d_model)
    14: 
    15:     def forward(self, x):
    16:         batch_size, ts_d, seg_num, d_model = x.shape
    17:         pad_num = seg_num % self.win_size
    18:         if pad_num != 0:
    19:             pad_num = self.win_size - pad_num
    20:             x = torch.cat((x, x[:, :, -pad_num:, :]), dim=-2)
    21: 
    22:         seg_to_merge = []
    23:         for i in range(self.win_size):
    24:             seg_to_merge.append(x[:, :, i::self.win_size, :])
    25:         x = torch.cat(seg_to_merge, -1)
    26: 
    27:         x = self.norm(x)
    28:         x = self.linear_trans(x)
    29: 
    30:         return x
    31: 
    32: 
    33: class scale_block(nn.Module):
    34:     def __init__(self, configs, win_size, d_model, n_heads, d_ff, depth, dropout, \
    35:                  seg_num=10, factor=10):
    36:         super(scale_block, self).__init__()
    37: 
    38:         if win_size > 1:
    39:             self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
    40:         else:
    41:             self.merge_layer = None
    42: 
    43:         self.encode_layers = nn.ModuleList()
    44: 
    45:         for i in range(depth):
    46:             self.encode_layers.append(TwoStageAttentionLayer(configs, seg_num, factor, d_model, n_heads, \
    47:                                                              d_ff, dropout))
    48: 
    49:     def forward(self, x, attn_mask=None, tau=None, delta=None):
    50:         _, ts_dim, _, _ = x.shape
    51: 
    52:         if self.merge_layer is not None:
    53:             x = self.merge_layer(x)
    54: 
    55:         for layer in self.encode_layers:
    56:             x = layer(x)
    57: 
    58:         return x, None
    59: 
    60: 
    61: class Encoder(nn.Module):
    62:     def __init__(self, attn_layers):
    63:         super(Encoder, self).__init__()
    64:         self.encode_blocks = nn.ModuleList(attn_layers)
    65: 
    66:     def forward(self, x):
    67:         encode_x = []
    68:         encode_x.append(x)
    69: 
    70:         for block in self.encode_blocks:
    71:             x, attns = block(x)
    72:             encode_x.append(x)
    73: 
    74:         return encode_x, None
    75: 
    76: 
    77: class DecoderLayer(nn.Module):
    78:     def __init__(self, self_attention, cross_attention, seg_len, d_model, d_ff=None, dropout=0.1):
    79:         super(DecoderLayer, self).__init__()
    80:         self.self_attention = self_attention
    81:         self.cross_attention = cross_attention
    82:         self.norm1 = nn.LayerNorm(d_model)
    83:         self.norm2 = nn.LayerNorm(d_model)
    84:         self.dropout = nn.Dropout(dropout)
    85:         self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model),
    86:                                   nn.GELU(),
    87:                                   nn.Linear(d_model, d_model))
    88:         self.linear_pred = nn.Linear(d_model, seg_len)
    89: 
    90:     def forward(self, x, cross):
    91:         batch = x.shape[0]
    92:         x = self.self_attention(x)
    93:         x = rearrange(x, 'b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model')
    94: 
    95:         cross = rearrange(cross, 'b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model')
    96:         tmp, attn = self.cross_attention(x, cross, cross, None, None, None,)
    97:         x = x + self.dropout(tmp)
    98:         y = x = self.norm1(x)
    99:         y = self.MLP1(y)
   100:         dec_output = self.norm2(x + y)
   101: 
   102:         dec_output = rearrange(dec_output, '(b ts_d) seg_dec_num d_model -> b ts_d seg_dec_num d_model', b=batch)
   103:         layer_predict = self.linear_pred(dec_output)
   104:         layer_predict = rearrange(layer_predict, 'b out_d seg_num seg_len -> b (out_d seg_num) seg_len')
   105: 
   106:         return dec_output, layer_predict
   107: 
   108: 
   109: class Decoder(nn.Module):
   110:     def __init__(self, layers):
   111:         super(Decoder, self).__init__()
   112:         self.decode_layers = nn.ModuleList(layers)
   113: 
   114: 
   115:     def forward(self, x, cross):
   116:         final_predict = None
   117:         i = 0
   118: 
   119:         ts_d = x.shape[1]
   120:         for layer in self.decode_layers:
   121:             cross_enc = cross[i]
   122:             x, layer_predict = layer(x, cross_enc)
   123:             if final_predict is None:
   124:                 final_predict = layer_predict
   125:             else:
   126:                 final_predict = final_predict + layer_predict
   127:             i += 1
   128: 
   129:         final_predict = rearrange(final_predict, 'b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d', out_d=ts_d)
   130: 
   131:         return final_predict
```

### `Time-Series-Library/layers/Embed.py`  [READ-ONLY — do not edit]

```python
     1: import torch
     2: import torch.nn as nn
     3: import torch.nn.functional as F
     4: from torch.nn.utils import weight_norm
     5: import math
     6: 
     7: 
     8: class PositionalEmbedding(nn.Module):
     9:     def __init__(self, d_model, max_len=5000):
    10:         super(PositionalEmbedding, self).__init__()
    11:         # Compute the positional encodings once in log space.
    12:         pe = torch.zeros(max_len, d_model).float()
    13:         pe.require_grad = False
    14: 
    15:         position = torch.arange(0, max_len).float().unsqueeze(1)
    16:         div_term = (torch.arange(0, d_model, 2).float()
    17:                     * -(math.log(10000.0) / d_model)).exp()
    18: 
    19:         pe[:, 0::2] = torch.sin(position * div_term)
    20:         pe[:, 1::2] = torch.cos(position * div_term)
    21: 
    22:         pe = pe.unsqueeze(0)
    23:         self.register_buffer('pe', pe)
    24: 
    25:     def forward(self, x):
    26:         return self.pe[:, :x.size(1)]
    27: 
    28: 
    29: class TokenEmbedding(nn.Module):
    30:     def __init__(self, c_in, d_model):
    31:         super(TokenEmbedding, self).__init__()
    32:         padding = 1 if torch.__version__ >= '1.5.0' else 2
    33:         self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
    34:                                    kernel_size=3, padding=padding, padding_mode='circular', bias=False)
    35:         for m in self.modules():
    36:             if isinstance(m, nn.Conv1d):
    37:                 nn.init.kaiming_normal_(
    38:                     m.weight, mode='fan_in', nonlinearity='leaky_relu')
    39: 
    40:     def forward(self, x):
    41:         x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
    42:         return x
    43: 
    44: 
    45: class FixedEmbedding(nn.Module):
    46:     def __init__(self, c_in, d_model):
    47:         super(FixedEmbedding, self).__init__()
    48: 
    49:         w = torch.zeros(c_in, d_model).float()
    50:         w.require_grad = False
    51: 
    52:         position = torch.arange(0, c_in).float().unsqueeze(1)
    53:         div_term = (torch.arange(0, d_model, 2).float()
    54:                     * -(math.log(10000.0) / d_model)).exp()
    55: 
    56:         w[:, 0::2] = torch.sin(position * div_term)
    57:         w[:, 1::2] = torch.cos(position * div_term)
    58: 
    59:         self.emb = nn.Embedding(c_in, d_model)
    60:         self.emb.weight = nn.Parameter(w, requires_grad=False)
    61: 
    62:     def forward(self, x):
    63:         return self.emb(x).detach()
    64: 
    65: 
    66: class TemporalEmbedding(nn.Module):
    67:     def __init__(self, d_model, embed_type='fixed', freq='h'):
    68:         super(TemporalEmbedding, self).__init__()
    69: 
    70:         minute_size = 4
    71:         hour_size = 24
    72:         weekday_size = 7
    73:         day_size = 32
    74:         month_size = 13
    75: 
    76:         Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
    77:         if freq == 't':
    78:             self.minute_embed = Embed(minute_size, d_model)
    79:         self.hour_embed = Embed(hour_size, d_model)
    80:         self.weekday_embed = Embed(weekday_size, d_model)
    81:         self.day_embed = Embed(day_size, d_model)
    82:         self.month_embed = Embed(month_size, d_model)
    83: 
    84:     def forward(self, x):
    85:         x = x.long()
    86:         minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
    87:             self, 'minute_embed') else 0.
    88:         hour_x = self.hour_embed(x[:, :, 3])
    89:         weekday_x = self.weekday_embed(x[:, :, 2])
    90:         day_x = self.day_embed(x[:, :, 1])
    91:         month_x = self.month_embed(x[:, :, 0])
    92: 
    93:         return hour_x + weekday_x + day_x + month_x + minute_x
    94: 
    95: 
    96: class TimeFeatureEmbedding(nn.Module):
    97:     def __init__(self, d_model, embed_type='timeF', freq='h'):
    98:         super(TimeFeatureEmbedding, self).__init__()
    99: 
   100:         freq_map = {'h': 4, 't': 5, 's': 6,
   101:                     'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
   102:         d_inp = freq_map[freq]
   103:         self.embed = nn.Linear(d_inp, d_model, bias=False)
   104: 
   105:     def forward(self, x):
   106:         return self.embed(x)
   107: 
   108: 
   109: class DataEmbedding(nn.Module):
   110:     def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
   111:         super(DataEmbedding, self).__init__()
   112: 
   113:         self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
   114:         self.position_embedding = PositionalEmbedding(d_model=d_model)
   115:         self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
   116:                                                     freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
   117:             d_model=d_model, embed_type=embed_type, freq=freq)
   118:         self.dropout = nn.Dropout(p=dropout)
   119: 
   120:     def forward(self, x, x_mark):
   121:         if x_mark is None:
   122:             x = self.value_embedding(x) + self.position_embedding(x)
   123:         else:
   124:             x = self.value_embedding(
   125:                 x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
   126:         return self.dropout(x)
   127: 
   128: 
   129: class DataEmbedding_inverted(nn.Module):
   130:     def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
   131:         super(DataEmbedding_inverted, self).__init__()
   132:         self.value_embedding = nn.Linear(c_in, d_model)
   133:         self.dropout = nn.Dropout(p=dropout)
   134: 
   135:     def forward(self, x, x_mark):
   136:         x = x.permute(0, 2, 1)
   137:         # x: [Batch Variate Time]
   138:         if x_mark is None:
   139:             x = self.value_embedding(x)
   140:         else:
   141:             x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
   142:         # x: [Batch Variate d_model]
   143:         return self.dropout(x)
   144: 
   145: 
   146: class DataEmbedding_wo_pos(nn.Module):
   147:     def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
   148:         super(DataEmbedding_wo_pos, self).__init__()
   149: 
   150:         self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
   151:         self.position_embedding = PositionalEmbedding(d_model=d_model)
   152:         self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
   153:                                                     freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
   154:             d_model=d_model, embed_type=embed_type, freq=freq)
   155:         self.dropout = nn.Dropout(p=dropout)
   156: 
   157:     def forward(self, x, x_mark):
   158:         if x_mark is None:
   159:             x = self.value_embedding(x)
   160:         else:
   161:             x = self.value_embedding(x) + self.temporal_embedding(x_mark)
   162:         return self.dropout(x)
   163: 
   164: 
   165: class PatchEmbedding(nn.Module):
   166:     def __init__(self, d_model, patch_len, stride, padding, dropout):
   167:         super(PatchEmbedding, self).__init__()
   168:         # Patching
   169:         self.patch_len = patch_len
   170:         self.stride = stride
   171:         self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
   172: 
   173:         # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
   174:         self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
   175: 
   176:         # Positional embedding
   177:         self.position_embedding = PositionalEmbedding(d_model)
   178: 
   179:         # Residual dropout
   180:         self.dropout = nn.Dropout(dropout)
   181: 
   182:     def forward(self, x):
   183:         # do patching
   184:         n_vars = x.shape[1]
   185:         x = self.padding_patch_layer(x)
   186:         x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
   187:         x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
   188:         # Input encoding
   189:         x = self.value_embedding(x) + self.position_embedding(x)
   190:         return self.dropout(x), n_vars
```

### `Time-Series-Library/layers/FourierCorrelation.py`  [READ-ONLY — do not edit]

```python
     1: # coding=utf-8
     2: # author=maziqing
     3: # email=maziqing.mzq@alibaba-inc.com
     4: 
     5: import numpy as np
     6: import torch
     7: import torch.nn as nn
     8: 
     9: 
    10: def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    11:     """
    12:     get modes on frequency domain:
    13:     'random' means sampling randomly;
    14:     'else' means sampling the lowest modes;
    15:     """
    16:     modes = min(modes, seq_len // 2)
    17:     if mode_select_method == 'random':
    18:         index = list(range(0, seq_len // 2))
    19:         np.random.shuffle(index)
    20:         index = index[:modes]
    21:     else:
    22:         index = list(range(0, modes))
    23:     index.sort()
    24:     return index
    25: 
    26: 
    27: # ########## fourier layer #############
    28: class FourierBlock(nn.Module):
    29:     def __init__(self, in_channels, out_channels, n_heads, seq_len, modes=0, mode_select_method='random'):
    30:         super(FourierBlock, self).__init__()
    31:         print('fourier enhanced block used!')
    32:         """
    33:         1D Fourier block. It performs representation learning on frequency domain, 
    34:         it does FFT, linear transform, and Inverse FFT.    
    35:         """
    36:         # get modes on frequency domain
    37:         self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
    38:         print('modes={}, index={}'.format(modes, self.index))
    39: 
    40:         self.n_heads = n_heads
    41:         self.scale = (1 / (in_channels * out_channels))
    42:         self.weights1 = nn.Parameter(
    43:             self.scale * torch.rand(self.n_heads, in_channels // self.n_heads, out_channels // self.n_heads,
    44:                                     len(self.index), dtype=torch.float))
    45:         self.weights2 = nn.Parameter(
    46:             self.scale * torch.rand(self.n_heads, in_channels // self.n_heads, out_channels // self.n_heads,
    47:                                     len(self.index), dtype=torch.float))
    48: 
    49:     # Complex multiplication
    50:     def compl_mul1d(self, order, x, weights):
    51:         x_flag = True
    52:         w_flag = True
    53:         if not torch.is_complex(x):
    54:             x_flag = False
    55:             x = torch.complex(x, torch.zeros_like(x).to(x.device))
    56:         if not torch.is_complex(weights):
    57:             w_flag = False
    58:             weights = torch.complex(weights, torch.zeros_like(weights).to(weights.device))
    59:         if x_flag or w_flag:
    60:             return torch.complex(torch.einsum(order, x.real, weights.real) - torch.einsum(order, x.imag, weights.imag),
    61:                                  torch.einsum(order, x.real, weights.imag) + torch.einsum(order, x.imag, weights.real))
    62:         else:
    63:             return torch.einsum(order, x.real, weights.real)
    64: 
    65:     def forward(self, q, k, v, mask):
    66:         # size = [B, L, H, E]
    67:         B, L, H, E = q.shape
    68:         x = q.permute(0, 2, 3, 1)
    69:         # Compute Fourier coefficients
    70:         x_ft = torch.fft.rfft(x, dim=-1)
    71:         # Perform Fourier neural operations
    72:         out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
    73:         for wi, i in enumerate(self.index):
    74:             if i >= x_ft.shape[3] or wi >= out_ft.shape[3]:
    75:                 continue
    76:             out_ft[:, :, :, wi] = self.compl_mul1d("bhi,hio->bho", x_ft[:, :, :, i],
    77:                                                    torch.complex(self.weights1, self.weights2)[:, :, :, wi])
    78:         # Return to time domain
    79:         x = torch.fft.irfft(out_ft, n=x.size(-1))
    80:         return (x, None)
    81: 
    82: # ########## Fourier Cross Former ####################
    83: class FourierCrossAttention(nn.Module):
    84:     def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=64, mode_select_method='random',
    85:                  activation='tanh', policy=0, num_heads=8):
    86:         super(FourierCrossAttention, self).__init__()
    87:         print(' fourier enhanced cross attention used!')
    88:         """
    89:         1D Fourier Cross Attention layer. It does FFT, linear transform, attention mechanism and Inverse FFT.    
    90:         """
    91:         self.activation = activation
    92:         self.in_channels = in_channels
    93:         self.out_channels = out_channels
    94:         # get modes for queries and keys (& values) on frequency domain
    95:         self.index_q = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)
    96:         self.index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)
    97: 
    98:         print('modes_q={}, index_q={}'.format(len(self.index_q), self.index_q))
    99:         print('modes_kv={}, index_kv={}'.format(len(self.index_kv), self.index_kv))
   100: 
   101:         self.scale = (1 / (in_channels * out_channels))
   102:         self.weights1 = nn.Parameter(
   103:             self.scale * torch.rand(num_heads, in_channels // num_heads, out_channels // num_heads, len(self.index_q), dtype=torch.float))
   104:         self.weights2 = nn.Parameter(
   105:             self.scale * torch.rand(num_heads, in_channels // num_heads, out_channels // num_heads, len(self.index_q), dtype=torch.float))
   106: 
   107:     # Complex multiplication
   108:     def compl_mul1d(self, order, x, weights):
   109:         x_flag = True
   110:         w_flag = True
   111:         if not torch.is_complex(x):
   112:             x_flag = False
   113:             x = torch.complex(x, torch.zeros_like(x).to(x.device))
   114:         if not torch.is_complex(weights):
   115:             w_flag = False
   116:             weights = torch.complex(weights, torch.zeros_like(weights).to(weights.device))
   117:         if x_flag or w_flag:
   118:             return torch.complex(torch.einsum(order, x.real, weights.real) - torch.einsum(order, x.imag, weights.imag),
   119:                                  torch.einsum(order, x.real, weights.imag) + torch.einsum(order, x.imag, weights.real))
   120:         else:
   121:             return torch.einsum(order, x.real, weights.real)
   122: 
   123:     def forward(self, q, k, v, mask):
   124:         # size = [B, L, H, E]
   125:         B, L, H, E = q.shape
   126:         xq = q.permute(0, 2, 3, 1)  # size = [B, H, E, L]
   127:         xk = k.permute(0, 2, 3, 1)
   128:         xv = v.permute(0, 2, 3, 1)
   129: 
   130:         # Compute Fourier coefficients
   131:         xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
   132:         xq_ft = torch.fft.rfft(xq, dim=-1)
   133:         for i, j in enumerate(self.index_q):
   134:             if j >= xq_ft.shape[3]:
   135:                 continue
   136:             xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
   137:         xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xq.device, dtype=torch.cfloat)
   138:         xk_ft = torch.fft.rfft(xk, dim=-1)
   139:         for i, j in enumerate(self.index_kv):
   140:             if j >= xk_ft.shape[3]:
   141:                 continue
   142:             xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]
   143: 
   144:         # perform attention mechanism on frequency domain
   145:         xqk_ft = (self.compl_mul1d("bhex,bhey->bhxy", xq_ft_, xk_ft_))
   146:         if self.activation == 'tanh':
   147:             xqk_ft = torch.complex(xqk_ft.real.tanh(), xqk_ft.imag.tanh())
   148:         elif self.activation == 'softmax':
   149:             xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
   150:             xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
   151:         else:
   152:             raise Exception('{} actiation function is not implemented'.format(self.activation))
   153:         xqkv_ft = self.compl_mul1d("bhxy,bhey->bhex", xqk_ft, xk_ft_)
   154:         xqkvw = self.compl_mul1d("bhex,heox->bhox", xqkv_ft, torch.complex(self.weights1, self.weights2))
   155:         out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
   156:         for i, j in enumerate(self.index_q):
   157:             if i >= xqkvw.shape[3] or j >= out_ft.shape[3]:
   158:                 continue
   159:             out_ft[:, :, :, j] = xqkvw[:, :, :, i]
   160:         # Return to time domain
   161:         out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1))
   162:         return (out, None)
```

### `Time-Series-Library/layers/SelfAttention_Family.py`  [READ-ONLY — do not edit]

```python
     1: import torch
     2: import torch.nn as nn
     3: import numpy as np
     4: from math import sqrt
     5: from utils.masking import TriangularCausalMask, ProbMask
     6: from reformer_pytorch import LSHSelfAttention
     7: from einops import rearrange, repeat
     8: 
     9: 
    10: class DSAttention(nn.Module):
    11:     '''De-stationary Attention'''
    12: 
    13:     def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
    14:         super(DSAttention, self).__init__()
    15:         self.scale = scale
    16:         self.mask_flag = mask_flag
    17:         self.output_attention = output_attention
    18:         self.dropout = nn.Dropout(attention_dropout)
    19: 
    20:     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
    21:         B, L, H, E = queries.shape
    22:         _, S, _, D = values.shape
    23:         scale = self.scale or 1. / sqrt(E)
    24: 
    25:         tau = 1.0 if tau is None else tau.unsqueeze(
    26:             1).unsqueeze(1)  # B x 1 x 1 x 1
    27:         delta = 0.0 if delta is None else delta.unsqueeze(
    28:             1).unsqueeze(1)  # B x 1 x 1 x S
    29: 
    30:         # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
    31:         scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta
    32: 
    33:         if self.mask_flag:
    34:             if attn_mask is None:
    35:                 attn_mask = TriangularCausalMask(B, L, device=queries.device)
    36: 
    37:             scores.masked_fill_(attn_mask.mask, -np.inf)
    38: 
    39:         A = self.dropout(torch.softmax(scale * scores, dim=-1))
    40:         V = torch.einsum("bhls,bshd->blhd", A, values)
    41: 
    42:         if self.output_attention:
    43:             return V.contiguous(), A
    44:         else:
    45:             return V.contiguous(), None
    46: 
    47: 
    48: class FullAttention(nn.Module):
    49:     def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
    50:         super(FullAttention, self).__init__()
    51:         self.scale = scale
    52:         self.mask_flag = mask_flag
    53:         self.output_attention = output_attention
    54:         self.dropout = nn.Dropout(attention_dropout)
    55: 
    56:     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
    57:         B, L, H, E = queries.shape
    58:         _, S, _, D = values.shape
    59:         scale = self.scale or 1. / sqrt(E)
    60: 
    61:         scores = torch.einsum("blhe,bshe->bhls", queries, keys)
    62: 
    63:         if self.mask_flag:
    64:             if attn_mask is None:
    65:                 attn_mask = TriangularCausalMask(B, L, device=queries.device)
    66: 
    67:             scores.masked_fill_(attn_mask.mask, -np.inf)
    68: 
    69:         A = self.dropout(torch.softmax(scale * scores, dim=-1))
    70:         V = torch.einsum("bhls,bshd->blhd", A, values)
    71: 
    72:         if self.output_attention:
    73:             return V.contiguous(), A
    74:         else:
    75:             return V.contiguous(), None
    76: 
    77: 
    78: class ProbAttention(nn.Module):
    79:     def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
    80:         super(ProbAttention, self).__init__()
    81:         self.factor = factor
    82:         self.scale = scale
    83:         self.mask_flag = mask_flag
    84:         self.output_attention = output_attention
    85:         self.dropout = nn.Dropout(attention_dropout)
    86: 
    87:     def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
    88:         # Q [B, H, L, D]
    89:         B, H, L_K, E = K.shape
    90:         _, _, L_Q, _ = Q.shape
    91: 
    92:         # calculate the sampled Q_K
    93:         K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
    94:         # real U = U_part(factor*ln(L_k))*L_q
    95:         index_sample = torch.randint(L_K, (L_Q, sample_k))
    96:         K_sample = K_expand[:, :, torch.arange(
    97:             L_Q).unsqueeze(1), index_sample, :]
    98:         Q_K_sample = torch.matmul(
    99:             Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()
   100: 
   101:         # find the Top_k query with sparisty measurement
   102:         M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
   103:         M_top = M.topk(n_top, sorted=False)[1]
   104: 
   105:         # use the reduced Q to calculate Q_K
   106:         Q_reduce = Q[torch.arange(B)[:, None, None],
   107:                    torch.arange(H)[None, :, None],
   108:                    M_top, :]  # factor*ln(L_q)
   109:         Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k
   110: 
   111:         return Q_K, M_top
   112: 
   113:     def _get_initial_context(self, V, L_Q):
   114:         B, H, L_V, D = V.shape
   115:         if not self.mask_flag:
   116:             # V_sum = V.sum(dim=-2)
   117:             V_sum = V.mean(dim=-2)
   118:             contex = V_sum.unsqueeze(-2).expand(B, H,
   119:                                                 L_Q, V_sum.shape[-1]).clone()
   120:         else:  # use mask
   121:             # requires that L_Q == L_V, i.e. for self-attention only
   122:             assert (L_Q == L_V)
   123:             contex = V.cumsum(dim=-2)
   124:         return contex
   125: 
   126:     def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
   127:         B, H, L_V, D = V.shape
   128: 
   129:         if self.mask_flag:
   130:             attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
   131:             scores.masked_fill_(attn_mask.mask, -np.inf)
   132: 
   133:         attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)
   134: 
   135:         context_in[torch.arange(B)[:, None, None],
   136:         torch.arange(H)[None, :, None],
   137:         index, :] = torch.matmul(attn, V).type_as(context_in)
   138:         if self.output_attention:
   139:             attns = (torch.ones([B, H, L_V, L_V]) /
   140:                      L_V).type_as(attn).to(attn.device)
   141:             attns[torch.arange(B)[:, None, None], torch.arange(H)[
   142:                                                   None, :, None], index, :] = attn
   143:             return context_in, attns
   144:         else:
   145:             return context_in, None
   146: 
   147:     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
   148:         B, L_Q, H, D = queries.shape
   149:         _, L_K, _, _ = keys.shape
   150: 
   151:         queries = queries.transpose(2, 1)
   152:         keys = keys.transpose(2, 1)
   153:         values = values.transpose(2, 1)
   154: 
   155:         U_part = self.factor * \
   156:                  np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
   157:         u = self.factor * \
   158:             np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)
   159: 
   160:         U_part = U_part if U_part < L_K else L_K
   161:         u = u if u < L_Q else L_Q
   162: 
   163:         scores_top, index = self._prob_QK(
   164:             queries, keys, sample_k=U_part, n_top=u)
   165: 
   166:         # add scale factor
   167:         scale = self.scale or 1. / sqrt(D)
   168:         if scale is not None:
   169:             scores_top = scores_top * scale
   170:         # get the context
   171:         context = self._get_initial_context(values, L_Q)
   172:         # update the context with selected top_k queries
   173:         context, attn = self._update_context(
   174:             context, values, scores_top, index, L_Q, attn_mask)
   175: 
   176:         return context.contiguous(), attn
   177: 
   178: 
   179: class AttentionLayer(nn.Module):
   180:     def __init__(self, attention, d_model, n_heads, d_keys=None,
   181:                  d_values=None):
   182:         super(AttentionLayer, self).__init__()
   183: 
   184:         d_keys = d_keys or (d_model // n_heads)
   185:         d_values = d_values or (d_model // n_heads)
   186: 
   187:         self.inner_attention = attention
   188:         self.query_projection = nn.Linear(d_model, d_keys * n_heads)
   189:         self.key_projection = nn.Linear(d_model, d_keys * n_heads)
   190:         self.value_projection = nn.Linear(d_model, d_values * n_heads)
   191:         self.out_projection = nn.Linear(d_values * n_heads, d_model)
   192:         self.n_heads = n_heads
   193: 
   194:     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
   195:         B, L, _ = queries.shape
   196:         _, S, _ = keys.shape
   197:         H = self.n_heads
   198: 
   199:         queries = self.query_projection(queries).view(B, L, H, -1)
   200:         keys = self.key_projection(keys).view(B, S, H, -1)
   201:         values = self.value_projection(values).view(B, S, H, -1)
   202: 
   203:         out, attn = self.inner_attention(
   204:             queries,
   205:             keys,
   206:             values,
   207:             attn_mask,
   208:             tau=tau,
   209:             delta=delta
   210:         )
   211:         out = out.view(B, L, -1)
   212: 
   213:         return self.out_projection(out), attn
   214: 
   215: 
   216: class ReformerLayer(nn.Module):
   217:     def __init__(self, attention, d_model, n_heads, d_keys=None,
   218:                  d_values=None, causal=False, bucket_size=4, n_hashes=4):
   219:         super().__init__()
   220:         self.bucket_size = bucket_size
   221:         self.attn = LSHSelfAttention(
   222:             dim=d_model,
   223:             heads=n_heads,
   224:             bucket_size=bucket_size,
   225:             n_hashes=n_hashes,
   226:             causal=causal
   227:         )
   228: 
   229:     def fit_length(self, queries):
   230:         # inside reformer: assert N % (bucket_size * 2) == 0
   231:         B, N, C = queries.shape
   232:         if N % (self.bucket_size * 2) == 0:
   233:             return queries
   234:         else:
   235:             # fill the time series
   236:             fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
   237:             return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)
   238: 
   239:     def forward(self, queries, keys, values, attn_mask, tau, delta):
   240:         # in Reformer: defalut queries=keys
   241:         B, N, C = queries.shape
   242:         queries = self.attn(self.fit_length(queries))[:, :N, :]
   243:         return queries, None
   244: 
   245: 
   246: class TwoStageAttentionLayer(nn.Module):
   247:     '''
   248:     The Two Stage Attention (TSA) Layer
   249:     input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
   250:     '''
   251: 
   252:     def __init__(self, configs,
   253:                  seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
   254:         super(TwoStageAttentionLayer, self).__init__()
   255:         d_ff = d_ff or 4 * d_model
   256:         self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
   257:                                                            output_attention=False), d_model, n_heads)
   258:         self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
   259:                                                        output_attention=False), d_model, n_heads)
   260:         self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
   261:                                                          output_attention=False), d_model, n_heads)
   262:         self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))
   263: 
   264:         self.dropout = nn.Dropout(dropout)
   265: 
   266:         self.norm1 = nn.LayerNorm(d_model)
   267:         self.norm2 = nn.LayerNorm(d_model)
   268:         self.norm3 = nn.LayerNorm(d_model)
   269:         self.norm4 = nn.LayerNorm(d_model)
   270: 
   271:         self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
   272:                                   nn.GELU(),
   273:                                   nn.Linear(d_ff, d_model))
   274:         self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
   275:                                   nn.GELU(),
   276:                                   nn.Linear(d_ff, d_model))
   277: 
   278:     def forward(self, x, attn_mask=None, tau=None, delta=None):
   279:         # Cross Time Stage: Directly apply MSA to each dimension
   280:         batch = x.shape[0]
   281:         time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
   282:         time_enc, attn = self.time_attention(
   283:             time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
   284:         )
   285:         dim_in = time_in + self.dropout(time_enc)
   286:         dim_in = self.norm1(dim_in)
   287:         dim_in = dim_in + self.dropout(self.MLP1(dim_in))
   288:         dim_in = self.norm2(dim_in)
   289: 
   290:         # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
   291:         dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
   292:         batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
   293:         dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
   294:         dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
   295:         dim_enc = dim_send + self.dropout(dim_receive)
   296:         dim_enc = self.norm3(dim_enc)
   297:         dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
   298:         dim_enc = self.norm4(dim_enc)
   299: 
   300:         final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)
   301: 
   302:         return final_out
```

### `Time-Series-Library/layers/StandardNorm.py`  [READ-ONLY — do not edit]

```python
     1: import torch
     2: import torch.nn as nn
     3: 
     4: 
     5: class Normalize(nn.Module):
     6:     def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
     7:         """
     8:         :param num_features: the number of features or channels
     9:         :param eps: a value added for numerical stability
    10:         :param affine: if True, RevIN has learnable affine parameters
    11:         """
    12:         super(Normalize, self).__init__()
    13:         self.num_features = num_features
    14:         self.eps = eps
    15:         self.affine = affine
    16:         self.subtract_last = subtract_last
    17:         self.non_norm = non_norm
    18:         if self.affine:
    19:             self._init_params()
    20: 
    21:     def forward(self, x, mode: str):
    22:         if mode == 'norm':
    23:             self._get_statistics(x)
    24:             x = self._normalize(x)
    25:         elif mode == 'denorm':
    26:             x = self._denormalize(x)
    27:         else:
    28:             raise NotImplementedError
    29:         return x
    30: 
    31:     def _init_params(self):
    32:         # initialize RevIN params: (C,)
    33:         self.affine_weight = nn.Parameter(torch.ones(self.num_features))
    34:         self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
    35: 
    36:     def _get_statistics(self, x):
    37:         dim2reduce = tuple(range(1, x.ndim - 1))
    38:         if self.subtract_last:
    39:             self.last = x[:, -1, :].unsqueeze(1)
    40:         else:
    41:             self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
    42:         self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
    43: 
    44:     def _normalize(self, x):
    45:         if self.non_norm:
    46:             return x
    47:         if self.subtract_last:
    48:             x = x - self.last
    49:         else:
    50:             x = x - self.mean
    51:         x = x / self.stdev
    52:         if self.affine:
    53:             x = x * self.affine_weight
    54:             x = x + self.affine_bias
    55:         return x
    56: 
    57:     def _denormalize(self, x):
    58:         if self.non_norm:
    59:             return x
    60:         if self.affine:
    61:             x = x - self.affine_bias
    62:             x = x / (self.affine_weight + self.eps * self.eps)
    63:         x = x * self.stdev
    64:         if self.subtract_last:
    65:             x = x + self.last
    66:         else:
    67:             x = x + self.mean
    68:         return x
```

### `Time-Series-Library/layers/Transformer_EncDec.py`  [READ-ONLY — do not edit]

```python
     1: import torch
     2: import torch.nn as nn
     3: import torch.nn.functional as F
     4: 
     5: 
     6: class ConvLayer(nn.Module):
     7:     def __init__(self, c_in):
     8:         super(ConvLayer, self).__init__()
     9:         self.downConv = nn.Conv1d(in_channels=c_in,
    10:                                   out_channels=c_in,
    11:                                   kernel_size=3,
    12:                                   padding=2,
    13:                                   padding_mode='circular')
    14:         self.norm = nn.BatchNorm1d(c_in)
    15:         self.activation = nn.ELU()
    16:         self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
    17: 
    18:     def forward(self, x):
    19:         x = self.downConv(x.permute(0, 2, 1))
    20:         x = self.norm(x)
    21:         x = self.activation(x)
    22:         x = self.maxPool(x)
    23:         x = x.transpose(1, 2)
    24:         return x
    25: 
    26: 
    27: class EncoderLayer(nn.Module):
    28:     def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
    29:         super(EncoderLayer, self).__init__()
    30:         d_ff = d_ff or 4 * d_model
    31:         self.attention = attention
    32:         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
    33:         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
    34:         self.norm1 = nn.LayerNorm(d_model)
    35:         self.norm2 = nn.LayerNorm(d_model)
    36:         self.dropout = nn.Dropout(dropout)
    37:         self.activation = F.relu if activation == "relu" else F.gelu
    38: 
    39:     def forward(self, x, attn_mask=None, tau=None, delta=None):
    40:         new_x, attn = self.attention(
    41:             x, x, x,
    42:             attn_mask=attn_mask,
    43:             tau=tau, delta=delta
    44:         )
    45:         x = x + self.dropout(new_x)
    46: 
    47:         y = x = self.norm1(x)
    48:         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
    49:         y = self.dropout(self.conv2(y).transpose(-1, 1))
    50: 
    51:         return self.norm2(x + y), attn
    52: 
    53: 
    54: class Encoder(nn.Module):
    55:     def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
    56:         super(Encoder, self).__init__()
    57:         self.attn_layers = nn.ModuleList(attn_layers)
    58:         self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
    59:         self.norm = norm_layer
    60: 
    61:     def forward(self, x, attn_mask=None, tau=None, delta=None):
    62:         # x [B, L, D]
    63:         attns = []
    64:         if self.conv_layers is not None:
    65:             for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
    66:                 delta = delta if i == 0 else None
    67:                 x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
    68:                 x = conv_layer(x)
    69:                 attns.append(attn)
    70:             x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
    71:             attns.append(attn)
    72:         else:
    73:             for attn_layer in self.attn_layers:
    74:                 x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
    75:                 attns.append(attn)
    76: 
    77:         if self.norm is not None:
    78:             x = self.norm(x)
    79: 
    80:         return x, attns
    81: 
    82: 
    83: class DecoderLayer(nn.Module):
    84:     def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
    85:                  dropout=0.1, activation="relu"):
    86:         super(DecoderLayer, self).__init__()
    87:         d_ff = d_ff or 4 * d_model
    88:         self.self_attention = self_attention
    89:         self.cross_attention = cross_attention
    90:         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
    91:         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
    92:         self.norm1 = nn.LayerNorm(d_model)
    93:         self.norm2 = nn.LayerNorm(d_model)
    94:         self.norm3 = nn.LayerNorm(d_model)
    95:         self.dropout = nn.Dropout(dropout)
    96:         self.activation = F.relu if activation == "relu" else F.gelu
    97: 
    98:     def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
    99:         x = x + self.dropout(self.self_attention(
   100:             x, x, x,
   101:             attn_mask=x_mask,
   102:             tau=tau, delta=None
   103:         )[0])
   104:         x = self.norm1(x)
   105: 
   106:         x = x + self.dropout(self.cross_attention(
   107:             x, cross, cross,
   108:             attn_mask=cross_mask,
   109:             tau=tau, delta=delta
   110:         )[0])
   111: 
   112:         y = x = self.norm2(x)
   113:         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
   114:         y = self.dropout(self.conv2(y).transpose(-1, 1))
   115: 
   116:         return self.norm3(x + y)
   117: 
   118: 
   119: class Decoder(nn.Module):
   120:     def __init__(self, layers, norm_layer=None, projection=None):
   121:         super(Decoder, self).__init__()
   122:         self.layers = nn.ModuleList(layers)
   123:         self.norm = norm_layer
   124:         self.projection = projection
   125: 
   126:     def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
   127:         for layer in self.layers:
   128:             x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
   129: 
   130:         if self.norm is not None:
   131:             x = self.norm(x)
   132: 
   133:         if self.projection is not None:
   134:             x = self.projection(x)
   135:         return x
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **m4_monthly** — wall-clock budget `00:59:00`, compute share `0.33`
- **m4_quarterly** — wall-clock budget `00:59:00`, compute share `0.33`
- **m4_yearly** — wall-clock budget `00:59:00`, compute share `0.33`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.

## Parameter Budget

This task enforces a parameter-count cap. Your edits will be rejected if
the resulting model exceeds **1.05×** the strongest
baseline's parameter count. The check runs automatically inside the eval
scripts — you don't need to invoke it.



## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
