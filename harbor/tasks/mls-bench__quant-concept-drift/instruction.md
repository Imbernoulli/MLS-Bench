# MLS-Bench: quant-concept-drift

# Concept Drift Adaptation in Stock Prediction on CSI300

## Research Question
Can a stock-return predictor be made robust to *temporal* distribution shift (concept drift) — that is, to changes over time in the joint distribution of features and returns — while still using the standard CSI300 universe, Alpha360 features, and the fixed qlib backtest?

## Background
Financial markets are non-stationary: regime changes, macro shocks, and microstructure shifts mean that a model trained on one period often degrades on later periods even with the same features and label definition. This is the "concept drift" or *temporal covariate shift* problem. Two main families of approaches address it: (i) sequence models with explicit multi-pattern routing (e.g., TRA, KDD 2021), which learn distinct sub-predictors and a router that assigns time slices to predictors; and (ii) domain-adaptation-style training that aligns feature distributions across time windows (e.g., AdaRNN, CIKM 2021).

This task isolates *temporal drift adaptation* on a single universe (CSI300), evaluated under three different temporal regimes — it is **not** about cross-universe transfer.

## Objective
Implement a `CustomModel` in `custom_model.py` that exposes the qlib `Model` interface (`fit(dataset)`, `predict(dataset, segment="test")`). The class is wired into `workflow_config.yaml`, where the dataset adapter / processor block is editable so methods like TRA can request a different dataset view (e.g., `TSDatasetH`, custom processors). Instruments, date ranges, train/valid/test splits, and the backtest are fixed by the workflow.

## Fixed Pipeline
- **Universe**: CSI300 (instruments fixed by the workflow YAML).
- **Features**: Alpha360.
- **Label**: `Ref($close, -2) / Ref($close, -1) - 1`.
- **Temporal regimes** (three different fixed splits, all CSI300):
  - `csi300` — long-horizon split with a 2017–2020 test window.
  - `csi300_shifted` — shifted split with a 2016–2018 test window.
  - `csi300_recent` — most recent 2019–2020 test regime.
- **Backtest**: TopkDropout, top 50 / drop 5.

## Model Interface
```python
class CustomModel(qlib.model.base.Model):
    def fit(self, dataset): ...
    def predict(self, dataset, segment="test") -> pd.Series: ...
```
`predict` returns a `pd.Series` indexed by `(datetime, instrument)`.

## Evaluation Metrics
Per regime:
- Signal: IC, ICIR, Rank IC, Rank ICIR (higher is better).
- Portfolio: annualized return, information ratio (higher is better); max drawdown (closer to zero is better).

Computed by qlib's `SignalAnalysisRecord` and `PortAnaRecord`.

## Reference Implementations (read-only)
Three reference models ship with qlib's `examples/benchmarks/`:

- **TRA** — Lin et al., "Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport", KDD 2021 (arXiv 2106.12950). A lightweight router on top of a base predictor (LSTM/Transformer) that dispatches samples to a small set of independent predictors via an optimal-transport-based assignment loss. qlib defaults: backbone LSTM with `d_feat=6`, `hidden_size=64`, `num_layers=2`, `dropout=0.0`; TRA `num_states=3`, `tau=1.0`, λ from the paper. Source: https://github.com/microsoft/qlib (`examples/benchmarks/TRA`).
- **AdaRNN** — Du et al., "AdaRNN: Adaptive Learning and Forecasting of Time Series", CIKM 2021 (arXiv 2108.04443). Splits the training window into segments by Temporal Distribution Characterization, then aligns segment representations with Temporal Distribution Matching. qlib defaults: `d_feat=6`, `hidden_size=64`, `num_layers=2`, `dropout=0.0`, `n_epochs=200`. Source: https://github.com/jindongwang/transferlearning (`code/deep/adarnn`).
- **LightGBM** — Ke et al., NeurIPS 2017. Standard non-adaptive reference; qlib CSI300 defaults as in the other quant tasks.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/qlib/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `qlib/custom_model.py`
- editable lines **16–103**
- `qlib/workflow_config.yaml`
- editable lines **13–26**
- editable lines **32–45**


Other files you may **read** for context (do not modify):
- `qlib/qlib/model/base.py`


## Readable Context


### `qlib/custom_model.py`  [EDITABLE — lines 16–103 only]

```python
     1: # Custom stock prediction model for MLS-Bench (concept drift adaptation)
     2: #
     3: # EDITABLE section: CustomModel class with fit() and predict() methods.
     4: # FIXED sections: imports below.
     5: import numpy as np
     6: import pandas as pd
     7: import torch
     8: import torch.nn as nn
     9: import torch.nn.functional as F
    10: from qlib.model.base import Model
    11: from qlib.data.dataset import DatasetH
    12: from qlib.data.dataset.handler import DataHandlerLP
    13: 
    14: DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    15: 
    16: # =====================================================================
    17: # EDITABLE: CustomModel — implement your stock prediction model here
    18: # =====================================================================
    19: class CustomModel(Model):
    20:     """Custom stock prediction model for concept drift adaptation.
    21: 
    22:     You must implement:
    23:         fit(dataset)    — train the model on the training data
    24:         predict(dataset, segment="test") — return predictions as pd.Series
    25: 
    26:     The dataset is a qlib DatasetH with Alpha158 features (158 engineered
    27:     features per stock per day). Alpha158 computes rolling statistics over
    28:     multiple windows (5, 10, 20, 30, 60 days) from raw OHLCV data:
    29:       - Rolling mean, std, and max/min of returns and volume
    30:       - Momentum indicators (ROC at various horizons)
    31:       - K-line ratios (KLEN, KLOW, KSFT, etc.)
    32:       - Rolling correlation/covariance between price and volume (CORR, CORD)
    33:       - Volatility measures (VSTD, WVMA, residual-based RESI, RSQR)
    34:     Features are pre-normalized (RobustZScoreNorm) and NaN-filled.
    35: 
    36:     Getting data from the dataset:
    37:         df_train = dataset.prepare("train", col_set=["feature", "label"],
    38:                                     data_key=DataHandlerLP.DK_L)
    39:         features = df_train["feature"]   # DataFrame: (n_samples, 158)
    40:         labels = df_train["label"]       # DataFrame: (n_samples, 1)
    41: 
    42:     The label is: Ref($close, -2) / Ref($close, -1) - 1
    43:     (i.e., the return from T+1 to T+2, predicted at time T)
    44: 
    45:     predict() must return a pd.Series indexed by (datetime, instrument)
    46:     matching the target segment's index.
    47: 
    48:     Available imports: torch, torch.nn, numpy, pandas, lightgbm, sklearn, scipy
    49:     All network definitions and training logic go in this class.
    50:     """
    51: 
    52:     def __init__(self):
    53:         super().__init__()
    54:         self.fitted = False
    55:         # --- Default: Ridge regression baseline ---
    56:         from sklearn.linear_model import Ridge
    57: 
    58:         self.model = Ridge(alpha=1.0)
    59: 
    60:     def fit(self, dataset: DatasetH):
    61:         """Train the model.
    62: 
    63:         Args:
    64:             dataset: DatasetH with "train" and "valid" segments.
    65:         """
    66:         df_train = dataset.prepare(
    67:             "train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
    68:         )
    69:         features = df_train["feature"].values
    70:         labels = df_train["label"].values.ravel()
    71: 
    72:         # Remove NaN rows
    73:         mask = ~(np.isnan(features).any(axis=1) | np.isnan(labels))
    74:         features = features[mask]
    75:         labels = labels[mask]
    76: 
    77:         self.model.fit(features, labels)
    78:         self.fitted = True
    79: 
    80:     def predict(self, dataset: DatasetH, segment="test"):
    81:         """Generate predictions.
    82: 
    83:         Args:
    84:             dataset: DatasetH with the target segment.
    85:             segment: Which segment to predict on (default: "test").
    86: 
    87:         Returns:
    88:             pd.Series of predictions, indexed by (datetime, instrument).
    89:         """
    90:         if not self.fitted:
    91:             raise ValueError("Model is not fitted yet!")
    92: 
    93:         df_test = dataset.prepare(
    94:             segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_I
    95:         )
    96:         features = df_test["feature"]
    97:         index = features.index
    98: 
    99:         features_np = features.values
   100:         features_np = np.nan_to_num(features_np, nan=0.0)
   101: 
   102:         preds = self.model.predict(features_np)
   103:         return pd.Series(preds, index=index, name="score")
```

### `qlib/workflow_config.yaml`  [EDITABLE — lines 13–26, lines 32–45 only]

```yaml
     1: # Qlib workflow configuration for CSI300 concept drift adaptation benchmark.
     2: # Used by run_workflow.py — default Alpha158/CSI300/DatasetH pipeline.
     3: # Alpha158: 158 engineered features per stock per day.
     4: 
     5: qlib_init:
     6:   provider_uri: "~/.qlib/qlib_data/cn_data"
     7:   region: cn
     8: 
     9: sys:
    10:   rel_path:
    11:     - "."           # So custom_model.py is importable via module_path
    12: 
    13: task:
    14:   model:
    15:     class: CustomModel
    16:     module_path: custom_model
    17:     kwargs: {}
    18: 
    19:   dataset:
    20:     class: DatasetH
    21:     module_path: qlib.data.dataset
    22:     kwargs:
    23:       handler:
    24:         class: Alpha158
    25:         module_path: qlib.contrib.data.handler
    26:         kwargs:
    27:           start_time: "2008-01-01"
    28:           end_time: "2020-08-01"
    29:           fit_start_time: "2008-01-01"
    30:           fit_end_time: "2014-12-31"
    31:           instruments: csi300
    32:           infer_processors:
    33:             - class: RobustZScoreNorm
    34:               kwargs:
    35:                 fields_group: feature
    36:                 clip_outlier: true
    37:             - class: Fillna
    38:               kwargs:
    39:                 fields_group: feature
    40:           learn_processors:
    41:             - class: DropnaLabel
    42:             - class: CSRankNorm
    43:               kwargs:
    44:                 fields_group: label
    45:           label: ["Ref($close, -2) / Ref($close, -1) - 1"]
    46:       segments:
    47:         train: ["2008-01-01", "2014-12-31"]
    48:         valid: ["2015-01-01", "2016-12-31"]
    49:         test: ["2017-01-01", "2020-08-01"]
    50: 
    51:   record:
    52:     - class: SignalRecord
    53:       module_path: qlib.workflow.record_temp
    54:       kwargs:
    55:         model: "<MODEL>"
    56:         dataset: "<DATASET>"
    57:     - class: SigAnaRecord
    58:       module_path: qlib.workflow.record_temp
    59:       kwargs:
    60:         ana_long_short: false
    61:         ann_scaler: 252
    62:     - class: PortAnaRecord
    63:       module_path: qlib.workflow.record_temp
    64:       kwargs:
    65:         config: &port_analysis_config
    66:           strategy:
    67:             class: TopkDropoutStrategy
    68:             module_path: qlib.contrib.strategy
    69:             kwargs:
    70:               signal: "<PRED>"
    71:               topk: 50
    72:               n_drop: 5
    73:           backtest:
    74:             start_time: "2017-01-01"
    75:             end_time: "2020-08-01"
    76:             account: 100000000
    77:             benchmark: SH000300
    78:             exchange_kwargs:
    79:               limit_threshold: 0.095
    80:               deal_price: close
    81:               open_cost: 0.0005
    82:               close_cost: 0.0015
    83:               min_cost: 5
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


### `tra` baseline — editable region  [READ-ONLY — reference implementation]

In `qlib/custom_model.py`:

```python
Lines 16–824:
    13: 
    14: DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    15: 
    16: # =====================================================================
    17: # EDITABLE: CustomModel — implement your stock prediction model here
    18: # =====================================================================
    19: import io
    20: import os
    21: import copy
    22: import math
    23: import json
    24: import torch.optim as optim
    25: import matplotlib
    26: matplotlib.use("Agg")
    27: import matplotlib.pyplot as plt
    28: 
    29: try:
    30:     from torch.utils.tensorboard import SummaryWriter
    31: except ImportError:
    32:     SummaryWriter = None
    33: 
    34: from tqdm import tqdm
    35: 
    36: from qlib.constant import EPS
    37: from qlib.log import get_module_logger
    38: 
    39: device = "cuda" if torch.cuda.is_available() else "cpu"
    40: 
    41: 
    42: class RNN(nn.Module):
    43:     """RNN Model — verbatim from qlib/contrib/model/pytorch_tra.py.
    44: 
    45:     Args:
    46:         input_size (int): input size (# features)
    47:         hidden_size (int): hidden size
    48:         num_layers (int): number of hidden layers
    49:         rnn_arch (str): rnn architecture
    50:         use_attn (bool): whether use attention layer.
    51:             we use concat attention as https://github.com/fulifeng/Adv-ALSTM/
    52:         dropout (float): dropout rate
    53:     """
    54: 
    55:     def __init__(
    56:         self,
    57:         input_size=16,
    58:         hidden_size=64,
    59:         num_layers=2,
    60:         rnn_arch="GRU",
    61:         use_attn=True,
    62:         dropout=0.0,
    63:         **kwargs,
    64:     ):
    65:         super().__init__()
    66: 
    67:         self.input_size = input_size
    68:         self.hidden_size = hidden_size
    69:         self.num_layers = num_layers
    70:         self.rnn_arch = rnn_arch
    71:         self.use_attn = use_attn
    72: 
    73:         if hidden_size < input_size:
    74:             # compression
    75:             self.input_proj = nn.Linear(input_size, hidden_size)
    76:         else:
    77:             self.input_proj = None
    78: 
    79:         self.rnn = getattr(nn, rnn_arch)(
    80:             input_size=min(input_size, hidden_size),
    81:             hidden_size=hidden_size,
    82:             num_layers=num_layers,
    83:             batch_first=True,
    84:             dropout=dropout,
    85:         )
    86: 
    87:         if self.use_attn:
    88:             self.W = nn.Linear(hidden_size, hidden_size)
    89:             self.u = nn.Linear(hidden_size, 1, bias=False)
    90:             self.output_size = hidden_size * 2
    91:         else:
    92:             self.output_size = hidden_size
    93: 
    94:     def forward(self, x):
    95:         if self.input_proj is not None:
    96:             x = self.input_proj(x)
    97: 
    98:         rnn_out, last_out = self.rnn(x)
    99:         if self.rnn_arch == "LSTM":
   100:             last_out = last_out[0]
   101:         last_out = last_out.mean(dim=0)
   102: 
   103:         if self.use_attn:
   104:             laten = self.W(rnn_out).tanh()
   105:             scores = self.u(laten).softmax(dim=1)
   106:             att_out = (rnn_out * scores).sum(dim=1)
   107:             last_out = torch.cat([last_out, att_out], dim=1)
   108: 
   109:         return last_out
   110: 
   111: 
   112: class PositionalEncoding(nn.Module):
   113:     # reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
   114:     def __init__(self, d_model, dropout=0.1, max_len=5000):
   115:         super(PositionalEncoding, self).__init__()
   116:         self.dropout = nn.Dropout(p=dropout)
   117: 
   118:         pe = torch.zeros(max_len, d_model)
   119:         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
   120:         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
   121:         pe[:, 0::2] = torch.sin(position * div_term)
   122:         pe[:, 1::2] = torch.cos(position * div_term)
   123:         pe = pe.unsqueeze(0).transpose(0, 1)
   124:         self.register_buffer("pe", pe)
   125: 
   126:     def forward(self, x):
   127:         x = x + self.pe[: x.size(0), :]
   128:         return self.dropout(x)
   129: 
   130: 
   131: class Transformer(nn.Module):
   132:     """Transformer Model — verbatim from qlib/contrib/model/pytorch_tra.py.
   133: 
   134:     Args:
   135:         input_size (int): input size (# features)
   136:         hidden_size (int): hidden size
   137:         num_layers (int): number of transformer layers
   138:         num_heads (int): number of heads in transformer
   139:         dropout (float): dropout rate
   140:     """
   141: 
   142:     def __init__(
   143:         self,
   144:         input_size=16,
   145:         hidden_size=64,
   146:         num_layers=2,
   147:         num_heads=2,
   148:         dropout=0.0,
   149:         **kwargs,
   150:     ):
   151:         super().__init__()
   152: 
   153:         self.input_size = input_size
   154:         self.hidden_size = hidden_size
   155:         self.num_layers = num_layers
   156:         self.num_heads = num_heads
   157: 
   158:         self.input_proj = nn.Linear(input_size, hidden_size)
   159: 
   160:         self.pe = PositionalEncoding(input_size, dropout)
   161:         layer = nn.TransformerEncoderLayer(
   162:             nhead=num_heads, dropout=dropout, d_model=hidden_size, dim_feedforward=hidden_size * 4
   163:         )
   164:         self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
   165: 
   166:         self.output_size = hidden_size
   167: 
   168:     def forward(self, x):
   169:         x = x.permute(1, 0, 2).contiguous()  # the first dim need to be time
   170:         x = self.pe(x)
   171: 
   172:         x = self.input_proj(x)
   173:         out = self.encoder(x)
   174: 
   175:         return out[-1]
   176: 
   177: 
   178: class TRA(nn.Module):
   179:     """Temporal Routing Adaptor (TRA) — verbatim from qlib/contrib/model/pytorch_tra.py.
   180: 
   181:     TRA takes historical prediction errors & latent representation as inputs,
   182:     then routes the input sample to a specific predictor for training & inference.
   183: 
   184:     Args:
   185:         input_size (int): input size (RNN/Transformer's hidden size)
   186:         num_states (int): number of latent states (i.e., trading patterns)
   187:             If `num_states=1`, then TRA falls back to traditional methods
   188:         hidden_size (int): hidden size of the router
   189:         tau (float): gumbel softmax temperature
   190:         src_info (str): information for the router
   191:     """
   192: 
   193:     def __init__(
   194:         self,
   195:         input_size,
   196:         num_states=1,
   197:         hidden_size=8,
   198:         rnn_arch="GRU",
   199:         num_layers=1,
   200:         dropout=0.0,
   201:         tau=1.0,
   202:         src_info="LR_TPE",
   203:     ):
   204:         super().__init__()
   205: 
   206:         assert src_info in ["LR", "TPE", "LR_TPE"], "invalid `src_info`"
   207: 
   208:         self.num_states = num_states
   209:         self.tau = tau
   210:         self.rnn_arch = rnn_arch
   211:         self.src_info = src_info
   212: 
   213:         self.predictors = nn.Linear(input_size, num_states)
   214: 
   215:         if self.num_states > 1:
   216:             if "TPE" in src_info:
   217:                 self.router = getattr(nn, rnn_arch)(
   218:                     input_size=num_states,
   219:                     hidden_size=hidden_size,
   220:                     num_layers=num_layers,
   221:                     batch_first=True,
   222:                     dropout=dropout,
   223:                 )
   224:                 self.fc = nn.Linear(hidden_size + input_size if "LR" in src_info else hidden_size, num_states)
   225:             else:
   226:                 self.fc = nn.Linear(input_size, num_states)
   227: 
   228:     def reset_parameters(self):
   229:         for child in self.children():
   230:             child.reset_parameters()
   231: 
   232:     def forward(self, hidden, hist_loss):
   233:         preds = self.predictors(hidden)
   234: 
   235:         if self.num_states == 1:  # no need for router when having only one prediction
   236:             return preds, None, None
   237: 
   238:         if "TPE" in self.src_info:
   239:             out = self.router(hist_loss)[1]  # TPE
   240:             if self.rnn_arch == "LSTM":
   241:                 out = out[0]
   242:             out = out.mean(dim=0)
   243:             if "LR" in self.src_info:
   244:                 out = torch.cat([hidden, out], dim=-1)  # LR_TPE
   245:         else:
   246:             out = hidden  # LR
   247: 
   248:         out = self.fc(out)
   249: 
   250:         choice = F.gumbel_softmax(out, dim=-1, tau=self.tau, hard=True)
   251:         prob = torch.softmax(out / self.tau, dim=-1)
   252: 
   253:         return preds, choice, prob
   254: 
   255: 
   256: def evaluate(pred):
   257:     pred = pred.rank(pct=True)  # transform into percentiles
   258:     score = pred.score
   259:     label = pred.label
   260:     diff = score - label
   261:     MSE = (diff**2).mean()
   262:     MAE = (diff.abs()).mean()
   263:     IC = score.corr(label, method="spearman")
   264:     return {"MSE": MSE, "MAE": MAE, "IC": IC}
   265: 
   266: 
   267: def shoot_infs(inp_tensor):
   268:     """Replaces inf by maximum of tensor"""
   269:     mask_inf = torch.isinf(inp_tensor)
   270:     ind_inf = torch.nonzero(mask_inf, as_tuple=False)
   271:     if len(ind_inf) > 0:
   272:         for ind in ind_inf:
   273:             if len(ind) == 2:
   274:                 inp_tensor[ind[0], ind[1]] = 0
   275:             elif len(ind) == 1:
   276:                 inp_tensor[ind[0]] = 0
   277:         m = torch.max(inp_tensor)
   278:         for ind in ind_inf:
   279:             if len(ind) == 2:
   280:                 inp_tensor[ind[0], ind[1]] = m
   281:             elif len(ind) == 1:
   282:                 inp_tensor[ind[0]] = m
   283:     return inp_tensor
   284: 
   285: 
   286: def sinkhorn(Q, n_iters=3, epsilon=0.1):
   287:     # epsilon should be adjusted according to logits value's scale
   288:     with torch.no_grad():
   289:         Q = torch.exp(Q / epsilon)
   290:         Q = shoot_infs(Q)
   291:         for i in range(n_iters):
   292:             Q /= Q.sum(dim=0, keepdim=True)
   293:             Q /= Q.sum(dim=1, keepdim=True)
   294:     return Q
   295: 
   296: 
   297: def loss_fn(pred, label):
   298:     mask = ~torch.isnan(label)
   299:     if len(pred.shape) == 2:
   300:         label = label[:, None]
   301:     return (pred[mask] - label[mask]).pow(2).mean(dim=0)
   302: 
   303: 
   304: def minmax_norm(x):
   305:     xmin = x.min(dim=-1, keepdim=True).values
   306:     xmax = x.max(dim=-1, keepdim=True).values
   307:     mask = (xmin == xmax).squeeze()
   308:     x = (x - xmin) / (xmax - xmin + EPS)
   309:     x[mask] = 1
   310:     return x
   311: 
   312: 
   313: def transport_sample(all_preds, label, choice, prob, hist_loss, count, transport_method, alpha, training=False):
   314:     """
   315:     sample-wise transport — verbatim from qlib/contrib/model/pytorch_tra.py.
   316:     """
   317:     assert all_preds.shape == choice.shape
   318:     assert len(all_preds) == len(label)
   319:     assert transport_method in ["oracle", "router"]
   320: 
   321:     all_loss = torch.zeros_like(all_preds)
   322:     mask = ~torch.isnan(label)
   323:     all_loss[mask] = (all_preds[mask] - label[mask, None]).pow(2)  # [sample x states]
   324: 
   325:     L = minmax_norm(all_loss.detach())
   326:     Lh = L * alpha + minmax_norm(hist_loss) * (1 - alpha)  # add hist loss for transport
   327:     Lh = minmax_norm(Lh)
   328:     P = sinkhorn(-Lh)
   329:     del Lh
   330: 
   331:     if transport_method == "router":
   332:         if training:
   333:             pred = (all_preds * choice).sum(dim=1)  # gumbel softmax
   334:         else:
   335:             pred = all_preds[range(len(all_preds)), prob.argmax(dim=-1)]  # argmax
   336:     else:
   337:         pred = (all_preds * P).sum(dim=1)
   338: 
   339:     if transport_method == "router":
   340:         loss = loss_fn(pred, label)
   341:     else:
   342:         loss = (all_loss * P).sum(dim=1).mean()
   343: 
   344:     return loss, pred, L, P
   345: 
   346: 
   347: def transport_daily(all_preds, label, choice, prob, hist_loss, count, transport_method, alpha, training=False):
   348:     """
   349:     daily transport — verbatim from qlib/contrib/model/pytorch_tra.py.
   350:     """
   351:     assert len(prob) == len(count)
   352:     assert len(all_preds) == sum(count)
   353:     assert transport_method in ["oracle", "router"]
   354: 
   355:     all_loss = []  # loss of all predictions
   356:     start = 0
   357:     for i, cnt in enumerate(count):
   358:         slc = slice(start, start + cnt)  # samples from the i-th day
   359:         start += cnt
   360:         tloss = loss_fn(all_preds[slc], label[slc])  # loss of the i-th day
   361:         all_loss.append(tloss)
   362:     all_loss = torch.stack(all_loss, dim=0)  # [days x states]
   363: 
   364:     L = minmax_norm(all_loss.detach())
   365:     Lh = L * alpha + minmax_norm(hist_loss) * (1 - alpha)  # add hist loss for transport
   366:     Lh = minmax_norm(Lh)
   367:     P = sinkhorn(-Lh)
   368:     del Lh
   369: 
   370:     pred = []
   371:     start = 0
   372:     for i, cnt in enumerate(count):
   373:         slc = slice(start, start + cnt)  # samples from the i-th day
   374:         start += cnt
   375:         if transport_method == "router":
   376:             if training:
   377:                 tpred = all_preds[slc] @ choice[i]  # gumbel softmax
   378:             else:
   379:                 tpred = all_preds[slc][:, prob[i].argmax(dim=-1)]  # argmax
   380:         else:
   381:             tpred = all_preds[slc] @ P[i]
   382:         pred.append(tpred)
   383:     pred = torch.cat(pred, dim=0)  # [samples]
   384: 
   385:     if transport_method == "router":
   386:         loss = loss_fn(pred, label)
   387:     else:
   388:         loss = (all_loss * P).sum(dim=1).mean()
   389: 
   390:     return loss, pred, L, P
   391: 
   392: 
   393: def load_state_dict_unsafe(model, state_dict):
   394:     """
   395:     Load state dict to provided model while ignore exceptions.
   396:     """
   397: 
   398:     missing_keys = []
   399:     unexpected_keys = []
   400:     error_msgs = []
   401: 
   402:     # copy state_dict so _load_from_state_dict can modify it
   403:     metadata = getattr(state_dict, "_metadata", None)
   404:     state_dict = state_dict.copy()
   405:     if metadata is not None:
   406:         state_dict._metadata = metadata
   407: 
   408:     def load(module, prefix=""):
   409:         local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
   410:         module._load_from_state_dict(
   411:             state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
   412:         )
   413:         for name, child in module._modules.items():
   414:             if child is not None:
   415:                 load(child, prefix + name + ".")
   416: 
   417:     load(model)
   418:     load = None  # break load->load reference cycle
   419: 
   420:     return {"unexpected_keys": unexpected_keys, "missing_keys": missing_keys, "error_msgs": error_msgs}
   421: 
   422: 
   423: def plot(P):
   424:     assert isinstance(P, pd.DataFrame)
   425: 
   426:     fig, axes = plt.subplots(1, 2, figsize=(10, 4))
   427:     P.plot.area(ax=axes[0], xlabel="")
   428:     P.idxmax(axis=1).value_counts().sort_index().plot.bar(ax=axes[1], xlabel="")
   429:     plt.tight_layout()
   430: 
   431:     with io.BytesIO() as buf:
   432:         plt.savefig(buf, format="png")
   433:         buf.seek(0)
   434:         img = plt.imread(buf)
   435:         plt.close()
   436: 
   437:     return np.uint8(img * 255)
   438: 
   439: 
   440: class CustomModel(Model):
   441:     """TRA Model — faithful to qlib's official TRAModel (pytorch_tra.py).
   442: 
   443:     Hyperparameters from official benchmark:
   444:     examples/benchmarks/TRA/workflow_config_tra_Alpha158.yaml
   445: 
   446:     The workflow provides MTSDatasetH with Alpha158+FilterCol (20 features).
   447:     fit()/predict() use dataset.prepare() to get MTSDataSampler loaders.
   448:     """
   449: 
   450:     def __init__(self):
   451:         super().__init__()
   452:         self.logger = get_module_logger("TRA")
   453: 
   454:         # Official benchmark hyperparameters
   455:         self.model_config = {
   456:             "input_size": 20,
   457:             "hidden_size": 64,
   458:             "num_layers": 2,
   459:             "rnn_arch": "LSTM",
   460:             "use_attn": True,
   461:             "dropout": 0.0,
   462:         }
   463:         self.tra_config = {
   464:             "num_states": 3,
   465:             "rnn_arch": "LSTM",
   466:             "hidden_size": 32,
   467:             "num_layers": 1,
   468:             "dropout": 0.0,
   469:             "tau": 1.0,
   470:             "src_info": "LR_TPE",
   471:         }
   472:         self.model_type = "RNN"
   473:         self.lr = 1e-3
   474:         self.n_epochs = 100
   475:         self.early_stop = 20
   476:         self.update_freq = 1
   477:         self.max_steps_per_epoch = None
   478:         self.lamb = 1.0
   479:         self.rho = 0.99
   480:         self.alpha = 0.5
   481:         self.seed = int(os.environ.get("SEED", "42"))
   482:         self.logdir = None
   483:         self.eval_train = False
   484:         self.eval_test = True
   485:         self.pretrain = True
   486:         self.init_state = None
   487:         self.reset_router = False
   488:         self.freeze_model = False
   489:         self.freeze_predictors = False
   490:         self.transport_method = "router"
   491:         self.use_daily_transport = False  # memory_mode=sample
   492:         self.transport_fn = transport_sample
   493: 
   494:         self._writer = None
   495: 
   496:         if self.seed is not None:
   497:             np.random.seed(self.seed)
   498:             torch.manual_seed(self.seed)
   499: 
   500:         self._init_model()
   501: 
   502:     def _init_model(self):
   503:         self.logger.info("init TRAModel...")
   504: 
   505:         self.model = eval(self.model_type)(**self.model_config).to(device)
   506:         print(self.model)
   507: 
   508:         self.tra = TRA(self.model.output_size, **self.tra_config).to(device)
   509:         print(self.tra)
   510: 
   511:         if self.init_state:
   512:             self.logger.warning(f"load state dict from `init_state`")
   513:             state_dict = torch.load(self.init_state, map_location="cpu")
   514:             self.model.load_state_dict(state_dict["model"])
   515:             res = load_state_dict_unsafe(self.tra, state_dict["tra"])
   516:             self.logger.warning(str(res))
   517: 
   518:         if self.reset_router:
   519:             self.logger.warning(f"reset TRA.router parameters")
   520:             self.tra.fc.reset_parameters()
   521:             self.tra.router.reset_parameters()
   522: 
   523:         if self.freeze_model:
   524:             self.logger.warning(f"freeze model parameters")
   525:             for param in self.model.parameters():
   526:                 param.requires_grad_(False)
   527: 
   528:         if self.freeze_predictors:
   529:             self.logger.warning(f"freeze TRA.predictors parameters")
   530:             for param in self.tra.predictors.parameters():
   531:                 param.requires_grad_(False)
   532: 
   533:         self.logger.info("# model params: %d" % sum(p.numel() for p in self.model.parameters() if p.requires_grad))
   534:         self.logger.info("# tra params: %d" % sum(p.numel() for p in self.tra.parameters() if p.requires_grad))
   535: 
   536:         self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.tra.parameters()), lr=self.lr)
   537: 
   538:         self.fitted = False
   539:         self.global_step = -1
   540: 
   541:     def train_epoch(self, epoch, data_set, is_pretrain=False):
   542:         self.model.train()
   543:         self.tra.train()
   544:         data_set.train()
   545:         self.optimizer.zero_grad()
   546: 
   547:         P_all = []
   548:         prob_all = []
   549:         choice_all = []
   550:         max_steps = len(data_set)
   551:         if self.max_steps_per_epoch is not None:
   552:             if epoch == 0 and self.max_steps_per_epoch < max_steps:
   553:                 self.logger.info(f"max steps updated from {max_steps} to {self.max_steps_per_epoch}")
   554:             max_steps = min(self.max_steps_per_epoch, max_steps)
   555: 
   556:         cur_step = 0
   557:         total_loss = 0
   558:         total_count = 0
   559:         for batch in tqdm(data_set, total=max_steps):
   560:             cur_step += 1
   561:             if cur_step > max_steps:
   562:                 break
   563: 
   564:             if not is_pretrain:
   565:                 self.global_step += 1
   566: 
   567:             data, state, label, count = batch["data"], batch["state"], batch["label"], batch["daily_count"]
   568:             index = batch["daily_index"] if self.use_daily_transport else batch["index"]
   569: 
   570:             with torch.set_grad_enabled(not self.freeze_model):
   571:                 hidden = self.model(data)
   572: 
   573:             all_preds, choice, prob = self.tra(hidden, state)
   574: 
   575:             if is_pretrain or self.transport_method != "none":
   576:                 # NOTE: use oracle transport for pre-training
   577:                 loss, pred, L, P = self.transport_fn(
   578:                     all_preds,
   579:                     label,
   580:                     choice,
   581:                     prob,
   582:                     state.mean(dim=1),
   583:                     count,
   584:                     self.transport_method if not is_pretrain else "oracle",
   585:                     self.alpha,
   586:                     training=True,
   587:                 )
   588:                 data_set.assign_data(index, L)  # save loss to memory
   589:                 if self.use_daily_transport:  # only save for daily transport
   590:                     P_all.append(pd.DataFrame(P.detach().cpu().numpy(), index=index))
   591:                     prob_all.append(pd.DataFrame(prob.detach().cpu().numpy(), index=index))
   592:                     choice_all.append(pd.DataFrame(choice.detach().cpu().numpy(), index=index))
   593:                 decay = self.rho ** (self.global_step // 100)  # decay every 100 steps
   594:                 lamb = 0 if is_pretrain else self.lamb * decay
   595:                 reg = prob.log().mul(P).sum(dim=1).mean()  # train router to predict TO assignment
   596:                 if self._writer is not None and not is_pretrain:
   597:                     self._writer.add_scalar("training/router_loss", -reg.item(), self.global_step)
   598:                     self._writer.add_scalar("training/reg_loss", loss.item(), self.global_step)
   599:                     self._writer.add_scalar("training/lamb", lamb, self.global_step)
   600:                     if not self.use_daily_transport:
   601:                         P_mean = P.mean(axis=0).detach()
   602:                         self._writer.add_scalar("training/P", P_mean.max() / P_mean.min(), self.global_step)
   603:                 loss = loss - lamb * reg
   604:             else:
   605:                 pred = all_preds.mean(dim=1)
   606:                 loss = loss_fn(pred, label)
   607: 
   608:             (loss / self.update_freq).backward()
   609:             if cur_step % self.update_freq == 0:
   610:                 self.optimizer.step()
   611:                 self.optimizer.zero_grad()
   612: 
   613:             if self._writer is not None and not is_pretrain:
   614:                 self._writer.add_scalar("training/total_loss", loss.item(), self.global_step)
   615: 
   616:             total_loss += loss.item()
   617:             total_count += 1
   618: 
   619:         if self.use_daily_transport and len(P_all) > 0:
   620:             P_all = pd.concat(P_all, axis=0)
   621:             prob_all = pd.concat(prob_all, axis=0)
   622:             choice_all = pd.concat(choice_all, axis=0)
   623:             P_all.index = data_set.restore_daily_index(P_all.index)
   624:             prob_all.index = P_all.index
   625:             choice_all.index = P_all.index
   626:             if not is_pretrain:
   627:                 self._writer.add_image("P", plot(P_all), epoch, dataformats="HWC")
   628:                 self._writer.add_image("prob", plot(prob_all), epoch, dataformats="HWC")
   629:                 self._writer.add_image("choice", plot(choice_all), epoch, dataformats="HWC")
   630: 
   631:         total_loss /= total_count
   632: 
   633:         if self._writer is not None and not is_pretrain:
   634:             self._writer.add_scalar("training/loss", total_loss, epoch)
   635: 
   636:         return total_loss
   637: 
   638:     def test_epoch(self, epoch, data_set, return_pred=False, prefix="test", is_pretrain=False):
   639:         self.model.eval()
   640:         self.tra.eval()
   641:         data_set.eval()
   642: 
   643:         preds = []
   644:         probs = []
   645:         P_all = []
   646:         metrics = []
   647:         for batch in tqdm(data_set):
   648:             data, state, label, count = batch["data"], batch["state"], batch["label"], batch["daily_count"]
   649:             index = batch["daily_index"] if self.use_daily_transport else batch["index"]
   650: 
   651:             with torch.no_grad():
   652:                 hidden = self.model(data)
   653:                 all_preds, choice, prob = self.tra(hidden, state)
   654: 
   655:             if is_pretrain or self.transport_method != "none":
   656:                 loss, pred, L, P = self.transport_fn(
   657:                     all_preds,
   658:                     label,
   659:                     choice,
   660:                     prob,
   661:                     state.mean(dim=1),
   662:                     count,
   663:                     self.transport_method if not is_pretrain else "oracle",
   664:                     self.alpha,
   665:                     training=False,
   666:                 )
   667:                 data_set.assign_data(index, L)  # save loss to memory
   668:                 if P is not None and return_pred:
   669:                     P_all.append(pd.DataFrame(P.cpu().numpy(), index=index))
   670:             else:
   671:                 pred = all_preds.mean(dim=1)
   672: 
   673:             X = np.c_[pred.cpu().numpy(), label.cpu().numpy(), all_preds.cpu().numpy()]
   674:             columns = ["score", "label"] + ["score_%d" % d for d in range(all_preds.shape[1])]
   675:             pred = pd.DataFrame(X, index=batch["index"], columns=columns)
   676: 
   677:             metrics.append(evaluate(pred))
   678: 
   679:             if return_pred:
   680:                 preds.append(pred)
   681:                 if prob is not None:
   682:                     columns = ["prob_%d" % d for d in range(all_preds.shape[1])]
   683:                     probs.append(pd.DataFrame(prob.cpu().numpy(), index=index, columns=columns))
   684: 
   685:         metrics = pd.DataFrame(metrics)
   686:         metrics = {
   687:             "MSE": metrics.MSE.mean(),
   688:             "MAE": metrics.MAE.mean(),
   689:             "IC": metrics.IC.mean(),
   690:             "ICIR": metrics.IC.mean() / metrics.IC.std(),
   691:         }
   692: 
   693:         if self._writer is not None and epoch >= 0 and not is_pretrain:
   694:             for key, value in metrics.items():
   695:                 self._writer.add_scalar(prefix + "/" + key, value, epoch)
   696: 
   697:         if return_pred:
   698:             preds = pd.concat(preds, axis=0)
   699:             preds.index = data_set.restore_index(preds.index)
   700:             preds.index = preds.index.swaplevel()
   701:             preds.sort_index(inplace=True)
   702: 
   703:             if probs:
   704:                 probs = pd.concat(probs, axis=0)
   705:                 if self.use_daily_transport:
   706:                     probs.index = data_set.restore_daily_index(probs.index)
   707:                 else:
   708:                     probs.index = data_set.restore_index(probs.index)
   709:                     probs.index = probs.index.swaplevel()
   710:                     probs.sort_index(inplace=True)
   711: 
   712:             if len(P_all):
   713:                 P_all = pd.concat(P_all, axis=0)
   714:                 if self.use_daily_transport:
   715:                     P_all.index = data_set.restore_daily_index(P_all.index)
   716:                 else:
   717:                     P_all.index = data_set.restore_index(P_all.index)
   718:                     P_all.index = P_all.index.swaplevel()
   719:                     P_all.sort_index(inplace=True)
   720: 
   721:         return metrics, preds, probs, P_all
   722: 
   723:     def _fit(self, train_set, valid_set, test_set, evals_result, is_pretrain=True):
   724:         best_score = -1
   725:         best_epoch = 0
   726:         stop_rounds = 0
   727:         best_params = {
   728:             "model": copy.deepcopy(self.model.state_dict()),
   729:             "tra": copy.deepcopy(self.tra.state_dict()),
   730:         }
   731:         # train
   732:         if not is_pretrain and self.transport_method != "none":
   733:             self.logger.info("init memory...")
   734:             self.test_epoch(-1, train_set)
   735: 
   736:         for epoch in range(self.n_epochs):
   737:             self.logger.info("Epoch %d:", epoch)
   738: 
   739:             self.logger.info("training...")
   740:             self.train_epoch(epoch, train_set, is_pretrain=is_pretrain)
   741: 
   742:             self.logger.info("evaluating...")
   743:             # NOTE: during evaluating, the whole memory will be refreshed
   744:             if not is_pretrain and (self.transport_method == "router" or self.eval_train):
   745:                 train_set.clear_memory()  # NOTE: clear the shared memory
   746:                 train_metrics = self.test_epoch(epoch, train_set, is_pretrain=is_pretrain, prefix="train")[0]
   747:                 evals_result["train"].append(train_metrics)
   748:                 self.logger.info("train metrics: %s" % train_metrics)
   749: 
   750:             valid_metrics = self.test_epoch(epoch, valid_set, is_pretrain=is_pretrain, prefix="valid")[0]
   751:             evals_result["valid"].append(valid_metrics)
   752:             self.logger.info("valid metrics: %s" % valid_metrics)
   753: 
   754:             if self.eval_test:
   755:                 test_metrics = self.test_epoch(epoch, test_set, is_pretrain=is_pretrain, prefix="test")[0]
   756:                 evals_result["test"].append(test_metrics)
   757:                 self.logger.info("test metrics: %s" % test_metrics)
   758: 
   759:             if valid_metrics["IC"] > best_score:
   760:                 best_score = valid_metrics["IC"]
   761:                 stop_rounds = 0
   762:                 best_epoch = epoch
   763:                 best_params = {
   764:                     "model": copy.deepcopy(self.model.state_dict()),
   765:                     "tra": copy.deepcopy(self.tra.state_dict()),
   766:                 }
   767:                 if self.logdir is not None:
   768:                     torch.save(best_params, self.logdir + "/model.bin")
   769:             else:
   770:                 stop_rounds += 1
   771:                 if stop_rounds >= self.early_stop:
   772:                     self.logger.info("early stop @ %s" % epoch)
   773:                     break
   774: 
   775:         self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
   776:         self.model.load_state_dict(best_params["model"])
   777:         self.tra.load_state_dict(best_params["tra"])
   778: 
   779:         return best_score
   780: 
   781:     def fit(self, dataset, evals_result=dict()):
   782:         # MTSDatasetH is provided by the workflow (Alpha158+FilterCol, 20 features).
   783:         train_set, valid_set, test_set = dataset.prepare(["train", "valid", "test"])
   784: 
   785:         self.fitted = True
   786:         self.global_step = -1
   787: 
   788:         evals_result["train"] = []
   789:         evals_result["valid"] = []
   790:         evals_result["test"] = []
   791: 
   792:         if self.pretrain:
   793:             self.logger.info("pretraining...")
   794:             self.optimizer = optim.Adam(
   795:                 list(self.model.parameters()) + list(self.tra.predictors.parameters()), lr=self.lr
   796:             )
   797:             self._fit(train_set, valid_set, test_set, evals_result, is_pretrain=True)
   798: 
   799:             # reset optimizer
   800:             self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.tra.parameters()), lr=self.lr)
   801: 
   802:         self.logger.info("training...")
   803:         best_score = self._fit(train_set, valid_set, test_set, evals_result, is_pretrain=False)
   804: 
   805:         self.logger.info("inference")
   806:         train_metrics, train_preds, train_probs, train_P = self.test_epoch(-1, train_set, return_pred=True)
   807:         self.logger.info("train metrics: %s" % train_metrics)
   808: 
   809:         valid_metrics, valid_preds, valid_probs, valid_P = self.test_epoch(-1, valid_set, return_pred=True)
   810:         self.logger.info("valid metrics: %s" % valid_metrics)
   811: 
   812:         test_metrics, test_preds, test_probs, test_P = self.test_epoch(-1, test_set, return_pred=True)
   813:         self.logger.info("test metrics: %s" % test_metrics)
   814: 
   815:     def predict(self, dataset, segment="test"):
   816:         if not self.fitted:
   817:             raise ValueError("model is not fitted yet!")
   818: 
   819:         test_set = dataset.prepare(segment)
   820: 
   821:         metrics, preds, _, _ = self.test_epoch(-1, test_set, return_pred=True)
   822:         self.logger.info("test metrics: %s" % metrics)
   823: 
   824:         return preds
```

### `adarnn` baseline — editable region  [READ-ONLY — reference implementation]

In `qlib/custom_model.py`:

```python
Lines 16–729:
    13: 
    14: DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    15: 
    16: # =====================================================================
    17: # EDITABLE: CustomModel — implement your stock prediction model here
    18: # =====================================================================
    19: import os
    20: import copy
    21: from typing import Text, Union
    22: from torch.utils.data import Dataset, DataLoader
    23: import torch.optim as optim
    24: from torch.autograd import Function
    25: from qlib.log import get_module_logger
    26: from qlib.utils import get_or_create_path
    27: 
    28: 
    29: class data_loader(Dataset):
    30:     """Data loader for Alpha158 flat features.
    31: 
    32:     Alpha158 provides 158 features per sample (flat vector).
    33:     Unsqueezes to [N, 1, 158] for GRU input (single timestep).
    34:     """
    35:     def __init__(self, df):
    36:         self.df_feature = df["feature"]
    37:         self.df_label_reg = df["label"]
    38:         self.df_index = df.index
    39:         self.df_feature = torch.tensor(
    40:             self.df_feature.values, dtype=torch.float32
    41:         ).unsqueeze(1)  # [N, 1, 158]
    42:         self.df_label_reg = torch.tensor(self.df_label_reg.values.reshape(-1), dtype=torch.float32)
    43: 
    44:     def __getitem__(self, index):
    45:         sample, label_reg = self.df_feature[index], self.df_label_reg[index]
    46:         return sample, label_reg
    47: 
    48:     def __len__(self):
    49:         return len(self.df_feature)
    50: 
    51: 
    52: def get_stock_loader(df, batch_size, shuffle=True):
    53:     train_loader = DataLoader(data_loader(df), batch_size=batch_size, shuffle=shuffle)
    54:     return train_loader
    55: 
    56: 
    57: def get_index(num_domain=2):
    58:     index = []
    59:     for i in range(num_domain):
    60:         for j in range(i + 1, num_domain + 1):
    61:             index.append((i, j))
    62:     return index
    63: 
    64: 
    65: def cosine(source, target):
    66:     source, target = source.mean(), target.mean()
    67:     cos = nn.CosineSimilarity(dim=0)
    68:     loss = cos(source, target)
    69:     return loss.mean()
    70: 
    71: 
    72: class ReverseLayerF(Function):
    73:     @staticmethod
    74:     def forward(ctx, x, alpha):
    75:         ctx.alpha = alpha
    76:         return x.view_as(x)
    77: 
    78:     @staticmethod
    79:     def backward(ctx, grad_output):
    80:         output = grad_output.neg() * ctx.alpha
    81:         return output, None
    82: 
    83: 
    84: class Discriminator(nn.Module):
    85:     def __init__(self, input_dim=256, hidden_dim=256):
    86:         super(Discriminator, self).__init__()
    87:         self.input_dim = input_dim
    88:         self.hidden_dim = hidden_dim
    89:         self.dis1 = nn.Linear(input_dim, hidden_dim)
    90:         self.dis2 = nn.Linear(hidden_dim, 1)
    91: 
    92:     def forward(self, x):
    93:         x = F.relu(self.dis1(x))
    94:         x = self.dis2(x)
    95:         x = torch.sigmoid(x)
    96:         return x
    97: 
    98: 
    99: def adv(source, target, device, input_dim=256, hidden_dim=512):
   100:     domain_loss = nn.BCELoss()
   101:     adv_net = Discriminator(input_dim, hidden_dim).to(device)
   102:     domain_src = torch.ones(len(source)).to(device)
   103:     domain_tar = torch.zeros(len(target)).to(device)
   104:     domain_src, domain_tar = domain_src.view(domain_src.shape[0], 1), domain_tar.view(domain_tar.shape[0], 1)
   105:     reverse_src = ReverseLayerF.apply(source, 1)
   106:     reverse_tar = ReverseLayerF.apply(target, 1)
   107:     pred_src = adv_net(reverse_src)
   108:     pred_tar = adv_net(reverse_tar)
   109:     loss_s, loss_t = domain_loss(pred_src, domain_src), domain_loss(pred_tar, domain_tar)
   110:     loss = loss_s + loss_t
   111:     return loss
   112: 
   113: 
   114: def CORAL(source, target, device):
   115:     d = source.size(1)
   116:     ns, nt = source.size(0), target.size(0)
   117: 
   118:     # source covariance
   119:     tmp_s = torch.ones((1, ns)).to(device) @ source
   120:     cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)
   121: 
   122:     # target covariance
   123:     tmp_t = torch.ones((1, nt)).to(device) @ target
   124:     ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)
   125: 
   126:     # frobenius norm
   127:     loss = (cs - ct).pow(2).sum()
   128:     loss = loss / (4 * d * d)
   129: 
   130:     return loss
   131: 
   132: 
   133: class MMD_loss(nn.Module):
   134:     def __init__(self, kernel_type="linear", kernel_mul=2.0, kernel_num=5):
   135:         super(MMD_loss, self).__init__()
   136:         self.kernel_num = kernel_num
   137:         self.kernel_mul = kernel_mul
   138:         self.fix_sigma = None
   139:         self.kernel_type = kernel_type
   140: 
   141:     @staticmethod
   142:     def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
   143:         n_samples = int(source.size()[0]) + int(target.size()[0])
   144:         total = torch.cat([source, target], dim=0)
   145:         total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
   146:         total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
   147:         L2_distance = ((total0 - total1) ** 2).sum(2)
   148:         if fix_sigma:
   149:             bandwidth = fix_sigma
   150:         else:
   151:             bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
   152:         bandwidth /= kernel_mul ** (kernel_num // 2)
   153:         bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
   154:         kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
   155:         return sum(kernel_val)
   156: 
   157:     @staticmethod
   158:     def linear_mmd(X, Y):
   159:         delta = X.mean(axis=0) - Y.mean(axis=0)
   160:         loss = delta.dot(delta.T)
   161:         return loss
   162: 
   163:     def forward(self, source, target):
   164:         if self.kernel_type == "linear":
   165:             return self.linear_mmd(source, target)
   166:         elif self.kernel_type == "rbf":
   167:             batch_size = int(source.size()[0])
   168:             kernels = self.guassian_kernel(
   169:                 source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma
   170:             )
   171:             with torch.no_grad():
   172:                 XX = torch.mean(kernels[:batch_size, :batch_size])
   173:                 YY = torch.mean(kernels[batch_size:, batch_size:])
   174:                 XY = torch.mean(kernels[:batch_size, batch_size:])
   175:                 YX = torch.mean(kernels[batch_size:, :batch_size])
   176:                 loss = torch.mean(XX + YY - XY - YX)
   177:             return loss
   178: 
   179: 
   180: class Mine_estimator(nn.Module):
   181:     def __init__(self, input_dim=2048, hidden_dim=512):
   182:         super(Mine_estimator, self).__init__()
   183:         self.mine_model = Mine(input_dim, hidden_dim)
   184: 
   185:     def forward(self, X, Y):
   186:         Y_shffle = Y[torch.randperm(len(Y))]
   187:         loss_joint = self.mine_model(X, Y)
   188:         loss_marginal = self.mine_model(X, Y_shffle)
   189:         ret = torch.mean(loss_joint) - torch.log(torch.mean(torch.exp(loss_marginal)))
   190:         loss = -ret
   191:         return loss
   192: 
   193: 
   194: class Mine(nn.Module):
   195:     def __init__(self, input_dim=2048, hidden_dim=512):
   196:         super(Mine, self).__init__()
   197:         self.fc1_x = nn.Linear(input_dim, hidden_dim)
   198:         self.fc1_y = nn.Linear(input_dim, hidden_dim)
   199:         self.fc2 = nn.Linear(hidden_dim, 1)
   200: 
   201:     def forward(self, x, y):
   202:         h1 = F.leaky_relu(self.fc1_x(x) + self.fc1_y(y))
   203:         h2 = self.fc2(h1)
   204:         return h2
   205: 
   206: 
   207: def pairwise_dist(X, Y):
   208:     n, d = X.shape
   209:     m, _ = Y.shape
   210:     assert d == Y.shape[1]
   211:     a = X.unsqueeze(1).expand(n, m, d)
   212:     b = Y.unsqueeze(0).expand(n, m, d)
   213:     return torch.pow(a - b, 2).sum(2)
   214: 
   215: 
   216: def kl_div(source, target):
   217:     if len(source) < len(target):
   218:         target = target[: len(source)]
   219:     elif len(source) > len(target):
   220:         source = source[: len(target)]
   221:     criterion = nn.KLDivLoss(reduction="batchmean")
   222:     loss = criterion(source.log(), target)
   223:     return loss
   224: 
   225: 
   226: def js(source, target):
   227:     if len(source) < len(target):
   228:         target = target[: len(source)]
   229:     elif len(source) > len(target):
   230:         source = source[: len(target)]
   231:     M = 0.5 * (source + target)
   232:     loss_1, loss_2 = kl_div(source, M), kl_div(target, M)
   233:     return 0.5 * (loss_1 + loss_2)
   234: 
   235: 
   236: class TransferLoss:
   237:     def __init__(self, loss_type="cosine", input_dim=512, GPU=0):
   238:         """
   239:         Supported loss_type: mmd(mmd_lin), mmd_rbf, coral, cosine, kl, js, mine, adv
   240:         """
   241:         self.loss_type = loss_type
   242:         self.input_dim = input_dim
   243:         self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
   244: 
   245:     def compute(self, X, Y):
   246:         """Compute adaptation loss"""
   247:         loss = None
   248:         if self.loss_type in ("mmd_lin", "mmd"):
   249:             mmdloss = MMD_loss(kernel_type="linear")
   250:             loss = mmdloss(X, Y)
   251:         elif self.loss_type == "coral":
   252:             loss = CORAL(X, Y, self.device)
   253:         elif self.loss_type in ("cosine", "cos"):
   254:             loss = 1 - cosine(X, Y)
   255:         elif self.loss_type == "kl":
   256:             loss = kl_div(X, Y)
   257:         elif self.loss_type == "js":
   258:             loss = js(X, Y)
   259:         elif self.loss_type == "mine":
   260:             mine_model = Mine_estimator(input_dim=self.input_dim, hidden_dim=60).to(self.device)
   261:             loss = mine_model(X, Y)
   262:         elif self.loss_type == "adv":
   263:             loss = adv(X, Y, self.device, input_dim=self.input_dim, hidden_dim=32)
   264:         elif self.loss_type == "mmd_rbf":
   265:             mmdloss = MMD_loss(kernel_type="rbf")
   266:             loss = mmdloss(X, Y)
   267:         elif self.loss_type == "pairwise":
   268:             pair_mat = pairwise_dist(X, Y)
   269:             loss = torch.norm(pair_mat)
   270: 
   271:         return loss
   272: 
   273: 
   274: class AdaRNN(nn.Module):
   275:     """AdaRNN network — verbatim from qlib/contrib/model/pytorch_adarnn.py.
   276: 
   277:     model_type:  'Boosting', 'AdaRNN'
   278:     """
   279: 
   280:     def __init__(
   281:         self,
   282:         use_bottleneck=False,
   283:         bottleneck_width=256,
   284:         n_input=128,
   285:         n_hiddens=[64, 64],
   286:         n_output=6,
   287:         dropout=0.0,
   288:         len_seq=9,
   289:         model_type="AdaRNN",
   290:         trans_loss="mmd",
   291:         GPU=0,
   292:     ):
   293:         super(AdaRNN, self).__init__()
   294:         self.use_bottleneck = use_bottleneck
   295:         self.n_input = n_input
   296:         self.num_layers = len(n_hiddens)
   297:         self.hiddens = n_hiddens
   298:         self.n_output = n_output
   299:         self.model_type = model_type
   300:         self.trans_loss = trans_loss
   301:         self.len_seq = len_seq
   302:         self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
   303:         in_size = self.n_input
   304: 
   305:         features = nn.ModuleList()
   306:         for hidden in n_hiddens:
   307:             rnn = nn.GRU(input_size=in_size, num_layers=1, hidden_size=hidden, batch_first=True, dropout=dropout)
   308:             features.append(rnn)
   309:             in_size = hidden
   310:         self.features = nn.Sequential(*features)
   311: 
   312:         if use_bottleneck is True:  # finance
   313:             self.bottleneck = nn.Sequential(
   314:                 nn.Linear(n_hiddens[-1], bottleneck_width),
   315:                 nn.Linear(bottleneck_width, bottleneck_width),
   316:                 nn.BatchNorm1d(bottleneck_width),
   317:                 nn.ReLU(),
   318:                 nn.Dropout(),
   319:             )
   320:             self.bottleneck[0].weight.data.normal_(0, 0.005)
   321:             self.bottleneck[0].bias.data.fill_(0.1)
   322:             self.bottleneck[1].weight.data.normal_(0, 0.005)
   323:             self.bottleneck[1].bias.data.fill_(0.1)
   324:             self.fc = nn.Linear(bottleneck_width, n_output)
   325:             torch.nn.init.xavier_normal_(self.fc.weight)
   326:         else:
   327:             self.fc_out = nn.Linear(n_hiddens[-1], self.n_output)
   328: 
   329:         if self.model_type == "AdaRNN":
   330:             gate = nn.ModuleList()
   331:             for i in range(len(n_hiddens)):
   332:                 gate_weight = nn.Linear(len_seq * self.hiddens[i] * 2, len_seq)
   333:                 gate.append(gate_weight)
   334:             self.gate = gate
   335: 
   336:             bnlst = nn.ModuleList()
   337:             for i in range(len(n_hiddens)):
   338:                 bnlst.append(nn.BatchNorm1d(len_seq))
   339:             self.bn_lst = bnlst
   340:             self.softmax = torch.nn.Softmax(dim=0)
   341:             self.init_layers()
   342: 
   343:     def init_layers(self):
   344:         for i in range(len(self.hiddens)):
   345:             self.gate[i].weight.data.normal_(0, 0.05)
   346:             self.gate[i].bias.data.fill_(0.0)
   347: 
   348:     def forward_pre_train(self, x, len_win=0):
   349:         out = self.gru_features(x)
   350:         fea = out[0]  # [2N,L,H]
   351:         if self.use_bottleneck is True:
   352:             fea_bottleneck = self.bottleneck(fea[:, -1, :])
   353:             fc_out = self.fc(fea_bottleneck).squeeze()
   354:         else:
   355:             fc_out = self.fc_out(fea[:, -1, :]).squeeze()  # [N,]
   356: 
   357:         out_list_all, out_weight_list = out[1], out[2]
   358:         out_list_s, out_list_t = self.get_features(out_list_all)
   359:         loss_transfer = torch.zeros((1,)).to(self.device)
   360:         for i, n in enumerate(out_list_s):
   361:             criterion_transder = TransferLoss(loss_type=self.trans_loss, input_dim=n.shape[2])
   362:             h_start = 0
   363:             for j in range(h_start, self.len_seq, 1):
   364:                 i_start = j - len_win if j - len_win >= 0 else 0
   365:                 i_end = j + len_win if j + len_win < self.len_seq else self.len_seq - 1
   366:                 for k in range(i_start, i_end + 1):
   367:                     weight = (
   368:                         out_weight_list[i][j].item()
   369:                         if self.model_type == "AdaRNN"
   370:                         else 1 / (self.len_seq - h_start) * (2 * len_win + 1)
   371:                     )
   372:                     loss_transfer = loss_transfer + weight * criterion_transder.compute(
   373:                         n[:, j, :], out_list_t[i][:, k, :]
   374:                     )
   375:         return fc_out, loss_transfer, out_weight_list
   376: 
   377:     def gru_features(self, x, predict=False):
   378:         x_input = x
   379:         out = None
   380:         out_lis = []
   381:         out_weight_list = [] if (self.model_type == "AdaRNN") else None
   382:         for i in range(self.num_layers):
   383:             out, _ = self.features[i](x_input.float())
   384:             x_input = out
   385:             out_lis.append(out)
   386:             if self.model_type == "AdaRNN" and predict is False:
   387:                 out_gate = self.process_gate_weight(x_input, i)
   388:                 out_weight_list.append(out_gate)
   389:         return out, out_lis, out_weight_list
   390: 
   391:     def process_gate_weight(self, out, index):
   392:         x_s = out[0 : int(out.shape[0] // 2)]
   393:         x_t = out[out.shape[0] // 2 : out.shape[0]]
   394:         x_all = torch.cat((x_s, x_t), 2)
   395:         x_all = x_all.view(x_all.shape[0], -1)
   396:         weight = torch.sigmoid(self.bn_lst[index](self.gate[index](x_all.float())))
   397:         weight = torch.mean(weight, dim=0)
   398:         res = self.softmax(weight)
   399:         return res
   400: 
   401:     @staticmethod
   402:     def get_features(output_list):
   403:         fea_list_src, fea_list_tar = [], []
   404:         for fea in output_list:
   405:             fea_list_src.append(fea[0 : fea.size(0) // 2])
   406:             fea_list_tar.append(fea[fea.size(0) // 2 :])
   407:         return fea_list_src, fea_list_tar
   408: 
   409:     # For Boosting-based
   410:     def forward_Boosting(self, x, weight_mat=None):
   411:         out = self.gru_features(x)
   412:         fea = out[0]
   413:         if self.use_bottleneck:
   414:             fea_bottleneck = self.bottleneck(fea[:, -1, :])
   415:             fc_out = self.fc(fea_bottleneck).squeeze()
   416:         else:
   417:             fc_out = self.fc_out(fea[:, -1, :]).squeeze()
   418: 
   419:         out_list_all = out[1]
   420:         out_list_s, out_list_t = self.get_features(out_list_all)
   421:         loss_transfer = torch.zeros((1,)).to(self.device)
   422:         if weight_mat is None:
   423:             weight = (1.0 / self.len_seq * torch.ones(self.num_layers, self.len_seq)).to(self.device)
   424:         else:
   425:             weight = weight_mat
   426:         dist_mat = torch.zeros(self.num_layers, self.len_seq).to(self.device)
   427:         for i, n in enumerate(out_list_s):
   428:             criterion_transder = TransferLoss(loss_type=self.trans_loss, input_dim=n.shape[2])
   429:             for j in range(self.len_seq):
   430:                 loss_trans = criterion_transder.compute(n[:, j, :], out_list_t[i][:, j, :])
   431:                 loss_transfer = loss_transfer + weight[i, j] * loss_trans
   432:                 dist_mat[i, j] = loss_trans
   433:         return fc_out, loss_transfer, dist_mat, weight
   434: 
   435:     # For Boosting-based
   436:     def update_weight_Boosting(self, weight_mat, dist_old, dist_new):
   437:         epsilon = 1e-5
   438:         dist_old = dist_old.detach()
   439:         dist_new = dist_new.detach()
   440:         ind = dist_new > dist_old + epsilon
   441:         weight_mat[ind] = weight_mat[ind] * (1 + torch.sigmoid(dist_new[ind] - dist_old[ind]))
   442:         weight_norm = torch.norm(weight_mat, dim=1, p=1)
   443:         weight_mat = weight_mat / weight_norm.t().unsqueeze(1).repeat(1, self.len_seq)
   444:         return weight_mat
   445: 
   446:     def predict(self, x):
   447:         out = self.gru_features(x, predict=True)
   448:         fea = out[0]
   449:         if self.use_bottleneck is True:
   450:             fea_bottleneck = self.bottleneck(fea[:, -1, :])
   451:             fc_out = self.fc(fea_bottleneck).squeeze(-1)
   452:         else:
   453:             fc_out = self.fc_out(fea[:, -1, :]).squeeze(-1)
   454:         return fc_out
   455: 
   456: 
   457: class CustomModel(Model):
   458:     """ADARNN model — faithful to qlib's official ADARNN (pytorch_adarnn.py).
   459: 
   460:     Adapted for Alpha158 features (158 flat features per sample).
   461:     Data is unsqueezed to [N, 1, 158] for GRU input (single timestep).
   462:     """
   463: 
   464:     def __init__(self):
   465:         super().__init__()
   466:         self.logger = get_module_logger("ADARNN")
   467:         self.logger.info("ADARNN pytorch version...")
   468: 
   469:         # Hyperparameters adapted for Alpha158 flat features
   470:         self.d_feat = 158
   471:         self.hidden_size = 64
   472:         self.num_layers = 2
   473:         self.dropout = 0.0
   474:         self.n_epochs = 200
   475:         self.pre_epoch = 40
   476:         self.dw = 0.5
   477:         self.loss_type = "cosine"
   478:         self.len_seq = 1
   479:         self.len_win = 0
   480:         self.lr = 1e-3
   481:         self.metric = "loss"
   482:         self.batch_size = 800
   483:         self.early_stop = 20
   484:         self.loss = "mse"
   485:         self.optimizer_name = "adam"
   486:         self.n_splits = 2
   487:         self.seed = None
   488:         self.device = torch.device(
   489:             "cuda:0" if torch.cuda.is_available() else "cpu"
   490:         )
   491: 
   492:         self.logger.info(
   493:             "ADARNN parameters setting:"
   494:             "\nd_feat : {}"
   495:             "\nhidden_size : {}"
   496:             "\nnum_layers : {}"
   497:             "\ndropout : {}"
   498:             "\nn_epochs : {}"
   499:             "\nlr : {}"
   500:             "\nmetric : {}"
   501:             "\nbatch_size : {}"
   502:             "\nearly_stop : {}"
   503:             "\noptimizer : {}"
   504:             "\nloss_type : {}"
   505:             "\nuse_GPU : {}".format(
   506:                 self.d_feat,
   507:                 self.hidden_size,
   508:                 self.num_layers,
   509:                 self.dropout,
   510:                 self.n_epochs,
   511:                 self.lr,
   512:                 self.metric,
   513:                 self.batch_size,
   514:                 self.early_stop,
   515:                 self.optimizer_name,
   516:                 self.loss,
   517:                 self.use_gpu,
   518:             )
   519:         )
   520: 
   521:         if self.seed is not None:
   522:             np.random.seed(self.seed)
   523:             torch.manual_seed(self.seed)
   524: 
   525:         n_hiddens = [self.hidden_size for _ in range(self.num_layers)]
   526:         self.model = AdaRNN(
   527:             use_bottleneck=False,
   528:             bottleneck_width=64,
   529:             n_input=self.d_feat,
   530:             n_hiddens=n_hiddens,
   531:             n_output=1,
   532:             dropout=self.dropout,
   533:             model_type="AdaRNN",
   534:             len_seq=self.len_seq,
   535:             trans_loss=self.loss_type,
   536:         )
   537:         self.logger.info("model:\n{:}".format(self.model))
   538: 
   539:         self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
   540: 
   541:         self.fitted = False
   542:         self.model.to(self.device)
   543: 
   544:     @property
   545:     def use_gpu(self):
   546:         return self.device != torch.device("cpu")
   547: 
   548:     def train_AdaRNN(self, train_loader_list, epoch, dist_old=None, weight_mat=None):
   549:         self.model.train()
   550:         criterion = nn.MSELoss()
   551:         dist_mat = torch.zeros(self.num_layers, self.len_seq).to(self.device)
   552:         out_weight_list = None
   553:         for data_all in zip(*train_loader_list):
   554:             self.train_optimizer.zero_grad()
   555:             list_feat = []
   556:             list_label = []
   557:             for data in data_all:
   558:                 feature, label_reg = data[0].to(self.device).float(), data[1].to(self.device).float()
   559:                 list_feat.append(feature)
   560:                 list_label.append(label_reg)
   561:             flag = False
   562:             index = get_index(len(data_all) - 1)
   563:             for temp_index in index:
   564:                 s1 = temp_index[0]
   565:                 s2 = temp_index[1]
   566:                 if list_feat[s1].shape[0] != list_feat[s2].shape[0]:
   567:                     flag = True
   568:                     break
   569:             if flag:
   570:                 continue
   571: 
   572:             total_loss = torch.zeros(1).to(self.device)
   573:             for i, n in enumerate(index):
   574:                 feature_s = list_feat[n[0]]
   575:                 feature_t = list_feat[n[1]]
   576:                 label_reg_s = list_label[n[0]]
   577:                 label_reg_t = list_label[n[1]]
   578:                 feature_all = torch.cat((feature_s, feature_t), 0)
   579: 
   580:                 if epoch < self.pre_epoch:
   581:                     pred_all, loss_transfer, out_weight_list = self.model.forward_pre_train(
   582:                         feature_all, len_win=self.len_win
   583:                     )
   584:                 else:
   585:                     pred_all, loss_transfer, dist, weight_mat = self.model.forward_Boosting(feature_all, weight_mat)
   586:                     dist_mat = dist_mat + dist
   587:                 pred_s = pred_all[0 : feature_s.size(0)]
   588:                 pred_t = pred_all[feature_s.size(0) :]
   589: 
   590:                 loss_s = criterion(pred_s, label_reg_s)
   591:                 loss_t = criterion(pred_t, label_reg_t)
   592: 
   593:                 total_loss = total_loss + loss_s + loss_t + self.dw * loss_transfer
   594:             self.train_optimizer.zero_grad()
   595:             total_loss.backward()
   596:             torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
   597:             self.train_optimizer.step()
   598:         if epoch >= self.pre_epoch:
   599:             if epoch > self.pre_epoch:
   600:                 weight_mat = self.model.update_weight_Boosting(weight_mat, dist_old, dist_mat)
   601:             return weight_mat, dist_mat
   602:         else:
   603:             weight_mat = self.transform_type(out_weight_list)
   604:             return weight_mat, None
   605: 
   606:     @staticmethod
   607:     def calc_all_metrics(pred):
   608:         """pred is a pandas dataframe that has two attributes: score (pred) and label (real)"""
   609:         res = {}
   610:         ic = pred.groupby(level="datetime", group_keys=False).apply(lambda x: x.label.corr(x.score))
   611:         rank_ic = pred.groupby(level="datetime", group_keys=False).apply(
   612:             lambda x: x.label.corr(x.score, method="spearman")
   613:         )
   614:         res["ic"] = ic.mean()
   615:         res["icir"] = ic.mean() / ic.std()
   616:         res["ric"] = rank_ic.mean()
   617:         res["ricir"] = rank_ic.mean() / rank_ic.std()
   618:         res["mse"] = -(pred["label"] - pred["score"]).mean()
   619:         res["loss"] = res["mse"]
   620:         return res
   621: 
   622:     def test_epoch(self, df):
   623:         self.model.eval()
   624:         preds = self.infer(df["feature"])
   625:         label = df["label"].squeeze()
   626:         preds = pd.DataFrame({"label": label, "score": preds}, index=df.index)
   627:         metrics = self.calc_all_metrics(preds)
   628:         return metrics
   629: 
   630:     def log_metrics(self, mode, metrics):
   631:         metrics_str = ["{}/{}: {:.6f}".format(k, mode, v) for k, v in metrics.items()]
   632:         metrics_str = ", ".join(metrics_str)
   633:         self.logger.info(metrics_str)
   634: 
   635:     def fit(
   636:         self,
   637:         dataset: DatasetH,
   638:         evals_result=dict(),
   639:         save_path=None,
   640:     ):
   641:         df_train, df_valid = dataset.prepare(
   642:             ["train", "valid"],
   643:             col_set=["feature", "label"],
   644:             data_key=DataHandlerLP.DK_L,
   645:         )
   646:         days = df_train.index.get_level_values(level=0).unique()
   647:         train_splits = np.array_split(days, self.n_splits)
   648:         train_splits = [df_train[s[0] : s[-1]] for s in train_splits]
   649:         train_loader_list = [get_stock_loader(df, self.batch_size) for df in train_splits]
   650: 
   651:         save_path = get_or_create_path(save_path)
   652:         stop_steps = 0
   653:         evals_result["train"] = []
   654:         evals_result["valid"] = []
   655: 
   656:         # train
   657:         self.logger.info("training...")
   658:         self.fitted = True
   659:         best_score = -np.inf
   660:         best_epoch = 0
   661:         weight_mat, dist_mat = None, None
   662: 
   663:         for step in range(self.n_epochs):
   664:             self.logger.info("Epoch%d:", step)
   665:             self.logger.info("training...")
   666:             weight_mat, dist_mat = self.train_AdaRNN(train_loader_list, step, dist_mat, weight_mat)
   667:             self.logger.info("evaluating...")
   668:             train_metrics = self.test_epoch(df_train)
   669:             valid_metrics = self.test_epoch(df_valid)
   670:             self.log_metrics("train: ", train_metrics)
   671:             self.log_metrics("valid: ", valid_metrics)
   672: 
   673:             valid_score = valid_metrics[self.metric]
   674:             train_score = train_metrics[self.metric]
   675:             evals_result["train"].append(train_score)
   676:             evals_result["valid"].append(valid_score)
   677:             if valid_score > best_score:
   678:                 best_score = valid_score
   679:                 stop_steps = 0
   680:                 best_epoch = step
   681:                 best_param = copy.deepcopy(self.model.state_dict())
   682:             else:
   683:                 stop_steps += 1
   684:                 if stop_steps >= self.early_stop:
   685:                     self.logger.info("early stop")
   686:                     break
   687: 
   688:         self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
   689:         self.model.load_state_dict(best_param)
   690:         torch.save(best_param, save_path)
   691: 
   692:         if self.use_gpu:
   693:             torch.cuda.empty_cache()
   694:         return best_score
   695: 
   696:     def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
   697:         if not self.fitted:
   698:             raise ValueError("model is not fitted yet!")
   699:         x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
   700:         return self.infer(x_test)
   701: 
   702:     def infer(self, x_test):
   703:         index = x_test.index
   704:         self.model.eval()
   705:         x_values = x_test.values
   706:         sample_num = x_values.shape[0]
   707:         preds = []
   708: 
   709:         for begin in range(sample_num)[:: self.batch_size]:
   710:             if sample_num - begin < self.batch_size:
   711:                 end = sample_num
   712:             else:
   713:                 end = begin + self.batch_size
   714: 
   715:             x_batch = torch.from_numpy(x_values[begin:end]).float().unsqueeze(1).to(self.device)  # [B, 1, 158]
   716: 
   717:             with torch.no_grad():
   718:                 pred = self.model.predict(x_batch).detach().cpu().numpy()
   719: 
   720:             preds.append(pred)
   721: 
   722:         return pd.Series(np.concatenate(preds), index=index)
   723: 
   724:     def transform_type(self, init_weight):
   725:         weight = torch.ones(self.num_layers, self.len_seq).to(self.device)
   726:         for i in range(self.num_layers):
   727:             for j in range(self.len_seq):
   728:                 weight[i, j] = init_weight[i][j].item()
   729:         return weight
```

### `lgbm` baseline — editable region  [READ-ONLY — reference implementation]

In `qlib/custom_model.py`:

```python
Lines 16–102:
    13: 
    14: DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    15: 
    16: # =====================================================================
    17: # EDITABLE: CustomModel — implement your stock prediction model here
    18: # =====================================================================
    19: class CustomModel(Model):
    20:     """LightGBM model — faithful to qlib's official LGBModel (gbdt.py).
    21: 
    22:     Hyperparameters from official benchmark:
    23:     examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
    24:     """
    25: 
    26:     def __init__(self):
    27:         super().__init__()
    28:         # Official benchmark kwargs (passed to lgb.train via self.params)
    29:         self.params = {
    30:             "objective": "mse",
    31:             "colsample_bytree": 0.8879,
    32:             "learning_rate": 0.2,
    33:             "subsample": 0.8789,
    34:             "lambda_l1": 205.6999,
    35:             "lambda_l2": 580.9768,
    36:             "max_depth": 8,
    37:             "num_leaves": 210,
    38:             "num_threads": 20,
    39:             "verbosity": -1,
    40:         }
    41:         self.early_stopping_rounds = 50
    42:         self.num_boost_round = 1000
    43:         self.model = None
    44: 
    45:     def _prepare_data(self, dataset):
    46:         """Prepare LightGBM datasets — matches LGBModel._prepare_data()."""
    47:         import lightgbm as lgb
    48: 
    49:         ds_l = []
    50:         for key in ["train", "valid"]:
    51:             if key in dataset.segments:
    52:                 df = dataset.prepare(
    53:                     key, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
    54:                 )
    55:                 if df.empty:
    56:                     raise ValueError(
    57:                         "Empty data from dataset, please check your dataset config."
    58:                     )
    59:                 x, y = df["feature"], df["label"]
    60:                 # Lightgbm need 1D array as its label
    61:                 if y.values.ndim == 2 and y.values.shape[1] == 1:
    62:                     y = np.squeeze(y.values)
    63:                 else:
    64:                     raise ValueError(
    65:                         "LightGBM doesn't support multi-label training"
    66:                     )
    67:                 ds_l.append(
    68:                     (lgb.Dataset(x.values, label=y, free_raw_data=False), key)
    69:                 )
    70:         return ds_l
    71: 
    72:     def fit(self, dataset: DatasetH):
    73:         import lightgbm as lgb
    74: 
    75:         ds_l = self._prepare_data(dataset)
    76:         ds, names = list(zip(*ds_l))
    77:         early_stopping_callback = lgb.early_stopping(
    78:             self.early_stopping_rounds
    79:         )
    80:         verbose_eval_callback = lgb.log_evaluation(period=20)
    81:         evals_result = {}
    82:         evals_result_callback = lgb.record_evaluation(evals_result)
    83:         self.model = lgb.train(
    84:             self.params,
    85:             ds[0],  # training dataset
    86:             num_boost_round=self.num_boost_round,
    87:             valid_sets=ds,
    88:             valid_names=names,
    89:             callbacks=[
    90:                 early_stopping_callback,
    91:                 verbose_eval_callback,
    92:                 evals_result_callback,
    93:             ],
    94:         )
    95: 
    96:     def predict(self, dataset: DatasetH, segment="test"):
    97:         if self.model is None:
    98:             raise ValueError("model is not fitted yet!")
    99:         x_test = dataset.prepare(
   100:             segment, col_set="feature", data_key=DataHandlerLP.DK_I
   101:         )
   102:         return pd.Series(self.model.predict(x_test.values), index=x_test.index)
```

### `tra` baseline — editable region  [READ-ONLY — reference implementation]

In `qlib/workflow_config.yaml`:

```python
Lines 13–26:
    10:   rel_path:
    11:     - "."           # So custom_model.py is importable via module_path
    12: 
    13: task:
    14:   model:
    15:     class: CustomModel
    16:     module_path: custom_model
    17:     kwargs: {}
    18: 
    19:   dataset:
    20:     class: MTSDatasetH
    21:     module_path: qlib.contrib.data.dataset
    22:     kwargs:
    23:       handler:
    24:         class: Alpha158
    25:         module_path: qlib.contrib.data.handler
    26:         kwargs:
    27:           start_time: "2008-01-01"
    28:           end_time: "2020-08-01"
    29:           fit_start_time: "2008-01-01"

Lines 32–73:
    29:           fit_start_time: "2008-01-01"
    30:           fit_end_time: "2014-12-31"
    31:           instruments: csi300
    32:           infer_processors:
    33:             - class: FilterCol
    34:               kwargs:
    35:                 fields_group: feature
    36:                 col_list:
    37:                   - RESI5
    38:                   - WVMA5
    39:                   - RSQR5
    40:                   - KLEN
    41:                   - RSQR10
    42:                   - CORR5
    43:                   - CORD5
    44:                   - CORR10
    45:                   - ROC60
    46:                   - RESI10
    47:                   - VSTD5
    48:                   - RSQR60
    49:                   - CORR60
    50:                   - WVMA60
    51:                   - STD5
    52:                   - RSQR20
    53:                   - CORD60
    54:                   - CORD10
    55:                   - CORR20
    56:                   - KLOW
    57:             - class: RobustZScoreNorm
    58:               kwargs:
    59:                 fields_group: feature
    60:                 clip_outlier: true
    61:             - class: Fillna
    62:               kwargs:
    63:                 fields_group: feature
    64:           learn_processors:
    65:             - class: CSRankNorm
    66:               kwargs:
    67:                 fields_group: label
    68:           label: ["Ref($close, -2) / Ref($close, -1) - 1"]
    69:       seq_len: 60
    70:       num_states: 3
    71:       batch_size: 1024
    72:       memory_mode: "sample"
    73:       drop_last: true
    74:       segments:
    75:         train: ["2008-01-01", "2014-12-31"]
    76:         valid: ["2015-01-01", "2016-12-31"]
```

### `lgbm` baseline — editable region  [READ-ONLY — reference implementation]

In `qlib/workflow_config.yaml`:

```python
Lines 13–26:
    10:   rel_path:
    11:     - "."           # So custom_model.py is importable via module_path
    12: 
    13: task:
    14:   model:
    15:     class: CustomModel
    16:     module_path: custom_model
    17:     kwargs: {}
    18: 
    19:   dataset:
    20:     class: DatasetH
    21:     module_path: qlib.data.dataset
    22:     kwargs:
    23:       handler:
    24:         class: Alpha158
    25:         module_path: qlib.contrib.data.handler
    26:         kwargs:
    27:           start_time: "2008-01-01"
    28:           end_time: "2020-08-01"
    29:           fit_start_time: "2008-01-01"

Lines 32–38:
    29:           fit_start_time: "2008-01-01"
    30:           fit_end_time: "2014-12-31"
    31:           instruments: csi300
    32:           infer_processors: []
    33:           learn_processors:
    34:             - class: DropnaLabel
    35:             - class: CSRankNorm
    36:               kwargs:
    37:                 fields_group: label
    38:           label: ["Ref($close, -2) / Ref($close, -1) - 1"]
    39:       segments:
    40:         train: ["2008-01-01", "2014-12-31"]
    41:         valid: ["2015-01-01", "2016-12-31"]
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
