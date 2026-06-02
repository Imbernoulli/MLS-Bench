# MLS-Bench: quant-graph-stock

# Graph-Based Multi-Stock Prediction

## Research Question
Can a relation-aware predictor exploit cross-stock structure (sector / concept membership, learned relations, attention across instruments) to deliver consistently better next-day return rankings than instrument-independent models, while keeping the data, labels, splits, and backtest fixed?

## Background
Stocks are not independent: prices co-move within sectors, react jointly to macro shocks, and share information through institutional flows and news. A line of work models this with graph neural networks or concept-aware aggregation over a stock-relation graph (e.g., GATs, HIST, RSR). The task here is to design such a relation-aware component on the standard `qlib` benchmarking pipeline with Alpha360 features, where the agent has access to a stock-concept membership graph in addition to per-stock features.

## Objective
Implement a `CustomModel` in `custom_model.py` that exposes the qlib model interface (`fit(dataset)` and `predict(dataset, segment="test")`). The class is wired into `workflow_config.yaml`, where the dataset adapter / preprocessor block is editable so the model can pull in graph-structured inputs (e.g., concept membership matrices) — but instruments, date ranges, train/valid/test splits, label, and the backtest configuration are fixed.

## Fixed Pipeline
The data, label, instrument universes, train/valid/test splits, and backtest configuration are fixed by the harness and not editable. A stock-concept membership graph is exposed through the dataset handler as an auxiliary input. Each stock-day feature vector has 360 features, which sequence models reshape to `[N, 60, 6]`.

## Model Interface
```python
class CustomModel(qlib.model.base.Model):
    def fit(self, dataset): ...
    def predict(self, dataset, segment="test") -> pd.Series: ...
```
`predict` returns a `pd.Series` indexed by `(datetime, instrument)` matching the requested segment's index.

## Reference Implementations (read-only)
Three reference models ship with qlib and are available as read-only context.

- **HIST** — Xu et al., "HIST: A Graph-based Framework for Stock Trend Forecasting via Mining Concept-Oriented Shared Information", arXiv 2110.13716 (2021). Uses a predefined-concept module and a hidden-concept module. qlib defaults: `d_feat=6`, `hidden_size=128`, `num_layers=2`, `dropout=0.7`, `n_epochs=200`, Adam `lr=2e-4`, `K=3` (top-k stock-to-concept assignments). Code: https://github.com/Wentao-Xu/HIST.
- **GATs** — Veličković et al.'s graph attention networks (ICLR 2018, arXiv 1710.10903) applied to the stock-relation graph. qlib defaults: `d_feat=6`, `hidden_size=64`, `num_layers=2`, `dropout=0.7`, `n_epochs=200`, Adam `lr=1e-4`.
- **LightGBM** — Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", NeurIPS 2017. qlib defaults: `loss=mse`, `learning_rate=0.0421`, `num_leaves=210`, `feature_fraction=0.879`, `bagging_fraction=0.856`, `bagging_freq=5`, `lambda_l1=205.7`, `lambda_l2=580.9`. Included as the standard non-graph reference.

Code: https://github.com/microsoft/qlib.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/qlib/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `qlib/custom_model.py`
- editable lines **58–156**
- `qlib/workflow_config.yaml`
- editable lines **14–26**
- editable lines **32–45**


Other files you may **read** for context (do not modify):
- `qlib/qlib/model/base.py`


## Readable Context


### `qlib/custom_model.py`  [EDITABLE — lines 58–156 only]

```python
     1: # Custom graph-based stock prediction model for MLS-Bench
     2: #
     3: # EDITABLE section: CustomModel class with fit() and predict() methods.
     4: # FIXED sections: imports and stock-concept graph loading below.
     5: import os
     6: import numpy as np
     7: import pandas as pd
     8: import torch
     9: import torch.nn as nn
    10: import torch.nn.functional as F
    11: from qlib.model.base import Model
    12: from qlib.data.dataset import DatasetH
    13: from qlib.data.dataset.handler import DataHandlerLP
    14: 
    15: DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    16: 
    17: # =====================================================================
    18: # FIXED: Stock-concept graph data loading utilities
    19: # =====================================================================
    20: # Paths to pre-downloaded graph data
    21: STOCK2CONCEPT_PATH = os.path.expanduser("~/.qlib/qlib_data/qlib_csi300_stock2concept.npy")
    22: STOCK_INDEX_PATH = os.path.expanduser("~/.qlib/qlib_data/qlib_csi300_stock_index.npy")
    23: 
    24: # Load the stock-concept mapping matrix and stock index
    25: # stock2concept_matrix: shape (num_stocks, num_concepts), binary membership
    26: # stock_index_dict: dict mapping instrument name -> integer index
    27: _stock2concept_matrix = np.load(STOCK2CONCEPT_PATH)
    28: _stock_index_dict = np.load(STOCK_INDEX_PATH, allow_pickle=True).item()
    29: 
    30: 
    31: def get_stock_index(instruments, default_index=733):
    32:     """Map instrument names to integer indices for stock2concept lookup.
    33: 
    34:     Args:
    35:         instruments: array-like of instrument name strings
    36:         default_index: fallback index for unknown instruments (733 = padding)
    37: 
    38:     Returns:
    39:         np.ndarray of integer indices
    40:     """
    41:     indices = np.array([_stock_index_dict.get(inst, default_index)
    42:                         for inst in instruments])
    43:     return indices.astype(int)
    44: 
    45: 
    46: def get_concept_matrix(stock_indices):
    47:     """Get the concept membership matrix for given stock indices.
    48: 
    49:     Args:
    50:         stock_indices: np.ndarray of integer stock indices
    51: 
    52:     Returns:
    53:         np.ndarray of shape (len(stock_indices), num_concepts), float32
    54:     """
    55:     return _stock2concept_matrix[stock_indices].astype(np.float32)
    56: 
    57: 
    58: # =====================================================================
    59: # EDITABLE: CustomModel — implement your stock prediction model here
    60: # =====================================================================
    61: class CustomModel(Model):
    62:     """Custom graph-based stock prediction model.
    63: 
    64:     You must implement:
    65:         fit(dataset)    — train the model on the training data
    66:         predict(dataset, segment="test") — return predictions as pd.Series
    67: 
    68:     The dataset is a qlib DatasetH with Alpha360 features (6 base features x 60
    69:     days = 360 features per stock per day). Segments: "train", "valid", "test".
    70: 
    71:     Getting data from the dataset:
    72:         df_train = dataset.prepare("train", col_set=["feature", "label"],
    73:                                     data_key=DataHandlerLP.DK_L)
    74:         features = df_train["feature"]   # DataFrame: (n_samples, 360)
    75:         labels = df_train["label"]       # DataFrame: (n_samples, 1)
    76: 
    77:     Stock-concept graph data (loaded above):
    78:         - _stock2concept_matrix: (num_stocks, num_concepts) binary matrix
    79:         - _stock_index_dict: maps instrument name -> stock index
    80:         - get_stock_index(instruments): maps instrument names to indices
    81:         - get_concept_matrix(stock_indices): returns concept membership matrix
    82: 
    83:     Usage in training (daily batches for graph-based models):
    84:         daily_count = df.groupby(level=0).size().values
    85:         daily_index = np.roll(np.cumsum(daily_count), 1)
    86:         daily_index[0] = 0
    87:         for idx, count in zip(daily_index, daily_count):
    88:             batch = slice(idx, idx + count)
    89:             feature = features.values[batch]
    90:             instruments = features.index.get_level_values("instrument")[batch]
    91:             stock_idx = get_stock_index(instruments)
    92:             concept_mat = get_concept_matrix(stock_idx)
    93:             # concept_mat shape: (batch_stocks, num_concepts)
    94: 
    95:     The label is: Ref($close, -2) / Ref($close, -1) - 1
    96:     (i.e., the return from T+1 to T+2, predicted at time T)
    97: 
    98:     predict() must return a pd.Series indexed by (datetime, instrument)
    99:     matching the target segment's index.
   100: 
   101:     Available imports: torch, torch.nn, numpy, pandas, lightgbm, sklearn, scipy
   102:     All network definitions and training logic go in this class.
   103:     """
   104: 
   105:     def __init__(self):
   106:         super().__init__()
   107:         self.fitted = False
   108:         # --- Default: Ridge regression baseline (ignores graph) ---
   109:         from sklearn.linear_model import Ridge
   110: 
   111:         self.model = Ridge(alpha=1.0)
   112: 
   113:     def fit(self, dataset: DatasetH):
   114:         """Train the model.
   115: 
   116:         Args:
   117:             dataset: DatasetH with "train" and "valid" segments.
   118:         """
   119:         df_train = dataset.prepare(
   120:             "train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
   121:         )
   122:         features = df_train["feature"].values
   123:         labels = df_train["label"].values.ravel()
   124: 
   125:         # Remove NaN rows
   126:         mask = ~(np.isnan(features).any(axis=1) | np.isnan(labels))
   127:         features = features[mask]
   128:         labels = labels[mask]
   129: 
   130:         self.model.fit(features, labels)
   131:         self.fitted = True
   132: 
   133:     def predict(self, dataset: DatasetH, segment="test"):
   134:         """Generate predictions.
   135: 
   136:         Args:
   137:             dataset: DatasetH with the target segment.
   138:             segment: Which segment to predict on (default: "test").
   139: 
   140:         Returns:
   141:             pd.Series of predictions, indexed by (datetime, instrument).
   142:         """
   143:         if not self.fitted:
   144:             raise ValueError("Model is not fitted yet!")
   145: 
   146:         df_test = dataset.prepare(
   147:             segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_I
   148:         )
   149:         features = df_test["feature"]
   150:         index = features.index
   151: 
   152:         features_np = features.values
   153:         features_np = np.nan_to_num(features_np, nan=0.0)
   154: 
   155:         preds = self.model.predict(features_np)
   156:         return pd.Series(preds, index=index, name="score")
```

### `qlib/workflow_config.yaml`  [EDITABLE — lines 14–26, lines 32–45 only]

```yaml
     1: # Qlib workflow configuration for CSI300 graph-based stock prediction benchmark.
     2: # Used by run_workflow.py — matches Alpha360/CSI300 official benchmark settings.
     3: # Alpha360: 6 base features x 60 days = 360 features, flattened as DatasetH.
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
    24:         class: Alpha360
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

This task enforces a parameter-count cap. The check runs automatically inside
the training script — you don't need to invoke it separately.

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `hist` baseline — editable region  [READ-ONLY — reference implementation]

In `qlib/custom_model.py`:

```python
Lines 58–483:
    55:     return _stock2concept_matrix[stock_indices].astype(np.float32)
    56: 
    57: 
    58: # =====================================================================
    59: # EDITABLE: CustomModel -- implement your stock prediction model here
    60: # =====================================================================
    61: import copy
    62: import torch.optim as optim
    63: 
    64: 
    65: class LSTMModel(nn.Module):
    66:     """LSTM backbone -- verbatim from qlib/contrib/model/pytorch_lstm.py."""
    67: 
    68:     def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
    69:         super().__init__()
    70:         self.rnn = nn.LSTM(
    71:             input_size=d_feat,
    72:             hidden_size=hidden_size,
    73:             num_layers=num_layers,
    74:             batch_first=True,
    75:             dropout=dropout,
    76:         )
    77:         self.fc_out = nn.Linear(hidden_size, 1)
    78:         self.d_feat = d_feat
    79: 
    80:     def forward(self, x):
    81:         # x: [N, F*T]
    82:         x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
    83:         x = x.permute(0, 2, 1)  # [N, T, F]
    84:         out, _ = self.rnn(x)
    85:         return self.fc_out(out[:, -1, :]).squeeze()
    86: 
    87: 
    88: class GRUModel(nn.Module):
    89:     """GRU backbone -- verbatim from qlib/contrib/model/pytorch_gru.py."""
    90: 
    91:     def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
    92:         super().__init__()
    93:         self.rnn = nn.GRU(
    94:             input_size=d_feat,
    95:             hidden_size=hidden_size,
    96:             num_layers=num_layers,
    97:             batch_first=True,
    98:             dropout=dropout,
    99:         )
   100:         self.fc_out = nn.Linear(hidden_size, 1)
   101:         self.d_feat = d_feat
   102: 
   103:     def forward(self, x):
   104:         # x: [N, F*T]
   105:         x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
   106:         x = x.permute(0, 2, 1)  # [N, T, F]
   107:         out, _ = self.rnn(x)
   108:         return self.fc_out(out[:, -1, :]).squeeze()
   109: 
   110: 
   111: class HISTModel(nn.Module):
   112:     """HIST network -- verbatim from qlib/contrib/model/pytorch_hist.py."""
   113: 
   114:     def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"):
   115:         super().__init__()
   116: 
   117:         self.d_feat = d_feat
   118:         self.hidden_size = hidden_size
   119: 
   120:         if base_model == "GRU":
   121:             self.rnn = nn.GRU(
   122:                 input_size=d_feat,
   123:                 hidden_size=hidden_size,
   124:                 num_layers=num_layers,
   125:                 batch_first=True,
   126:                 dropout=dropout,
   127:             )
   128:         elif base_model == "LSTM":
   129:             self.rnn = nn.LSTM(
   130:                 input_size=d_feat,
   131:                 hidden_size=hidden_size,
   132:                 num_layers=num_layers,
   133:                 batch_first=True,
   134:                 dropout=dropout,
   135:             )
   136:         else:
   137:             raise ValueError("unknown base model name `%s`" % base_model)
   138: 
   139:         self.fc_es = nn.Linear(hidden_size, hidden_size)
   140:         torch.nn.init.xavier_uniform_(self.fc_es.weight)
   141:         self.fc_is = nn.Linear(hidden_size, hidden_size)
   142:         torch.nn.init.xavier_uniform_(self.fc_is.weight)
   143: 
   144:         self.fc_es_middle = nn.Linear(hidden_size, hidden_size)
   145:         torch.nn.init.xavier_uniform_(self.fc_es_middle.weight)
   146:         self.fc_is_middle = nn.Linear(hidden_size, hidden_size)
   147:         torch.nn.init.xavier_uniform_(self.fc_is_middle.weight)
   148: 
   149:         self.fc_es_fore = nn.Linear(hidden_size, hidden_size)
   150:         torch.nn.init.xavier_uniform_(self.fc_es_fore.weight)
   151:         self.fc_is_fore = nn.Linear(hidden_size, hidden_size)
   152:         torch.nn.init.xavier_uniform_(self.fc_is_fore.weight)
   153:         self.fc_indi_fore = nn.Linear(hidden_size, hidden_size)
   154:         torch.nn.init.xavier_uniform_(self.fc_indi_fore.weight)
   155: 
   156:         self.fc_es_back = nn.Linear(hidden_size, hidden_size)
   157:         torch.nn.init.xavier_uniform_(self.fc_es_back.weight)
   158:         self.fc_is_back = nn.Linear(hidden_size, hidden_size)
   159:         torch.nn.init.xavier_uniform_(self.fc_is_back.weight)
   160:         self.fc_indi = nn.Linear(hidden_size, hidden_size)
   161:         torch.nn.init.xavier_uniform_(self.fc_indi.weight)
   162: 
   163:         self.leaky_relu = nn.LeakyReLU()
   164:         self.softmax_s2t = torch.nn.Softmax(dim=0)
   165:         self.softmax_t2s = torch.nn.Softmax(dim=1)
   166: 
   167:         self.fc_out_es = nn.Linear(hidden_size, 1)
   168:         self.fc_out_is = nn.Linear(hidden_size, 1)
   169:         self.fc_out_indi = nn.Linear(hidden_size, 1)
   170:         self.fc_out = nn.Linear(hidden_size, 1)
   171: 
   172:     def cal_cos_similarity(self, x, y):  # the 2nd dimension of x and y are the same
   173:         xy = x.mm(torch.t(y))
   174:         x_norm = torch.sqrt(torch.sum(x * x, dim=1)).reshape(-1, 1)
   175:         y_norm = torch.sqrt(torch.sum(y * y, dim=1)).reshape(-1, 1)
   176:         cos_similarity = xy / (x_norm.mm(torch.t(y_norm)) + 1e-6)
   177:         return cos_similarity
   178: 
   179:     def forward(self, x, concept_matrix):
   180:         device = torch.device(torch.get_device(x))
   181: 
   182:         x_hidden = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
   183:         x_hidden = x_hidden.permute(0, 2, 1)  # [N, T, F]
   184:         x_hidden, _ = self.rnn(x_hidden)
   185:         x_hidden = x_hidden[:, -1, :]
   186: 
   187:         # Predefined Concept Module
   188: 
   189:         stock_to_concept = concept_matrix
   190: 
   191:         stock_to_concept_sum = torch.sum(stock_to_concept, 0).reshape(1, -1).repeat(stock_to_concept.shape[0], 1)
   192:         stock_to_concept_sum = stock_to_concept_sum.mul(concept_matrix)
   193: 
   194:         stock_to_concept_sum = stock_to_concept_sum + (
   195:             torch.ones(stock_to_concept.shape[0], stock_to_concept.shape[1]).to(device)
   196:         )
   197:         stock_to_concept = stock_to_concept / stock_to_concept_sum
   198:         hidden = torch.t(stock_to_concept).mm(x_hidden)
   199: 
   200:         hidden = hidden[hidden.sum(1) != 0]
   201: 
   202:         concept_to_stock = self.cal_cos_similarity(x_hidden, hidden)
   203:         concept_to_stock = self.softmax_t2s(concept_to_stock)
   204: 
   205:         e_shared_info = concept_to_stock.mm(hidden)
   206:         e_shared_info = self.fc_es(e_shared_info)
   207: 
   208:         e_shared_back = self.fc_es_back(e_shared_info)
   209:         output_es = self.fc_es_fore(e_shared_info)
   210:         output_es = self.leaky_relu(output_es)
   211: 
   212:         # Hidden Concept Module
   213:         i_shared_info = x_hidden - e_shared_back
   214:         hidden = i_shared_info
   215:         i_stock_to_concept = self.cal_cos_similarity(i_shared_info, hidden)
   216:         dim = i_stock_to_concept.shape[0]
   217:         diag = i_stock_to_concept.diagonal(0)
   218:         i_stock_to_concept = i_stock_to_concept * (torch.ones(dim, dim) - torch.eye(dim)).to(device)
   219:         row = torch.linspace(0, dim - 1, dim).to(device).long()
   220:         column = i_stock_to_concept.max(1)[1].long()
   221:         value = i_stock_to_concept.max(1)[0]
   222:         i_stock_to_concept[row, column] = 10
   223:         i_stock_to_concept[i_stock_to_concept != 10] = 0
   224:         i_stock_to_concept[row, column] = value
   225:         i_stock_to_concept = i_stock_to_concept + torch.diag_embed((i_stock_to_concept.sum(0) != 0).float() * diag)
   226:         hidden = torch.t(i_shared_info).mm(i_stock_to_concept).t()
   227:         hidden = hidden[hidden.sum(1) != 0]
   228: 
   229:         i_concept_to_stock = self.cal_cos_similarity(i_shared_info, hidden)
   230:         i_concept_to_stock = self.softmax_t2s(i_concept_to_stock)
   231:         i_shared_info = i_concept_to_stock.mm(hidden)
   232:         i_shared_info = self.fc_is(i_shared_info)
   233: 
   234:         i_shared_back = self.fc_is_back(i_shared_info)
   235:         output_is = self.fc_is_fore(i_shared_info)
   236:         output_is = self.leaky_relu(output_is)
   237: 
   238:         # Individual Information Module
   239:         individual_info = x_hidden - e_shared_back - i_shared_back
   240:         output_indi = individual_info
   241:         output_indi = self.fc_indi(output_indi)
   242:         output_indi = self.leaky_relu(output_indi)
   243: 
   244:         # Stock Trend Prediction
   245:         all_info = output_es + output_is + output_indi
   246:         pred_all = self.fc_out(all_info).squeeze()
   247: 
   248:         return pred_all
   249: 
   250: 
   251: class CustomModel(Model):
   252:     """HIST model -- faithful to qlib's official HIST (pytorch_hist.py).
   253: 
   254:     Hyperparameters from official benchmark:
   255:     examples/benchmarks/HIST/workflow_config_hist_Alpha360.yaml
   256:     """
   257: 
   258:     def __init__(self):
   259:         super().__init__()
   260:         # Official benchmark hyperparameters
   261:         self.d_feat = 6
   262:         self.hidden_size = 64
   263:         self.num_layers = 2
   264:         self.dropout = 0.0
   265:         self.n_epochs = 200
   266:         self.lr = 1e-4
   267:         self.metric = "ic"
   268:         self.early_stop = 20
   269:         self.loss = "mse"
   270:         self.base_model = "LSTM"
   271:         self.model_path = "examples/benchmarks/LSTM/model_lstm_csi300.pkl"
   272:         self.stock2concept = os.path.expanduser("~/.qlib/qlib_data/qlib_csi300_stock2concept.npy")
   273:         self.stock_index = os.path.expanduser("~/.qlib/qlib_data/qlib_csi300_stock_index.npy")
   274:         self.optimizer_name = "adam"
   275:         self.device = torch.device(
   276:             "cuda:0" if torch.cuda.is_available() else "cpu"
   277:         )
   278: 
   279:         self.HIST_model = HISTModel(
   280:             d_feat=self.d_feat,
   281:             hidden_size=self.hidden_size,
   282:             num_layers=self.num_layers,
   283:             dropout=self.dropout,
   284:             base_model=self.base_model,
   285:         )
   286:         self.train_optimizer = optim.Adam(
   287:             self.HIST_model.parameters(), lr=self.lr
   288:         )
   289:         self.fitted = False
   290:         self.HIST_model.to(self.device)
   291: 
   292:     @property
   293:     def use_gpu(self):
   294:         return self.device != torch.device("cpu")
   295: 
   296:     def mse(self, pred, label):
   297:         loss = (pred - label) ** 2
   298:         return torch.mean(loss)
   299: 
   300:     def loss_fn(self, pred, label):
   301:         mask = ~torch.isnan(label)
   302:         if self.loss == "mse":
   303:             return self.mse(pred[mask], label[mask])
   304:         raise ValueError("unknown loss `%s`" % self.loss)
   305: 
   306:     def metric_fn(self, pred, label):
   307:         mask = torch.isfinite(label)
   308:         if self.metric == "ic":
   309:             x = pred[mask]
   310:             y = label[mask]
   311:             vx = x - torch.mean(x)
   312:             vy = y - torch.mean(y)
   313:             return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)))
   314:         if self.metric in ("", "loss"):
   315:             return -self.loss_fn(pred[mask], label[mask])
   316:         raise ValueError("unknown metric `%s`" % self.metric)
   317: 
   318:     def get_daily_inter(self, df, shuffle=False):
   319:         # organize the train data into daily batches
   320:         daily_count = df.groupby(level=0, group_keys=False).size().values
   321:         daily_index = np.roll(np.cumsum(daily_count), 1)
   322:         daily_index[0] = 0
   323:         if shuffle:
   324:             # shuffle data
   325:             daily_shuffle = list(zip(daily_index, daily_count))
   326:             np.random.shuffle(daily_shuffle)
   327:             daily_index, daily_count = zip(*daily_shuffle)
   328:         return daily_index, daily_count
   329: 
   330:     def train_epoch(self, x_train, y_train, stock_index):
   331:         stock2concept_matrix = np.load(self.stock2concept)
   332:         x_train_values = x_train.values
   333:         y_train_values = np.squeeze(y_train.values)
   334:         stock_index = stock_index.values
   335:         stock_index[np.isnan(stock_index)] = 733
   336:         self.HIST_model.train()
   337: 
   338:         # organize the train data into daily batches
   339:         daily_index, daily_count = self.get_daily_inter(x_train, shuffle=True)
   340: 
   341:         for idx, count in zip(daily_index, daily_count):
   342:             batch = slice(idx, idx + count)
   343:             feature = torch.from_numpy(x_train_values[batch]).float().to(self.device)
   344:             concept_matrix = torch.from_numpy(stock2concept_matrix[stock_index[batch]]).float().to(self.device)
   345:             label = torch.from_numpy(y_train_values[batch]).float().to(self.device)
   346:             pred = self.HIST_model(feature, concept_matrix)
   347:             loss = self.loss_fn(pred, label)
   348: 
   349:             self.train_optimizer.zero_grad()
   350:             loss.backward()
   351:             torch.nn.utils.clip_grad_value_(self.HIST_model.parameters(), 3.0)
   352:             self.train_optimizer.step()
   353: 
   354:     def test_epoch(self, data_x, data_y, stock_index):
   355:         # prepare training data
   356:         stock2concept_matrix = np.load(self.stock2concept)
   357:         x_values = data_x.values
   358:         y_values = np.squeeze(data_y.values)
   359:         stock_index = stock_index.values
   360:         stock_index[np.isnan(stock_index)] = 733
   361:         self.HIST_model.eval()
   362: 
   363:         scores = []
   364:         losses = []
   365: 
   366:         # organize the test data into daily batches
   367:         daily_index, daily_count = self.get_daily_inter(data_x, shuffle=False)
   368: 
   369:         for idx, count in zip(daily_index, daily_count):
   370:             batch = slice(idx, idx + count)
   371:             feature = torch.from_numpy(x_values[batch]).float().to(self.device)
   372:             concept_matrix = torch.from_numpy(stock2concept_matrix[stock_index[batch]]).float().to(self.device)
   373:             label = torch.from_numpy(y_values[batch]).float().to(self.device)
   374:             with torch.no_grad():
   375:                 pred = self.HIST_model(feature, concept_matrix)
   376:                 loss = self.loss_fn(pred, label)
   377:                 losses.append(loss.item())
   378: 
   379:                 score = self.metric_fn(pred, label)
   380:                 scores.append(score.item())
   381: 
   382:         return np.mean(losses), np.mean(scores)
   383: 
   384:     def fit(self, dataset: DatasetH):
   385:         df_train, df_valid, df_test = dataset.prepare(
   386:             ["train", "valid", "test"],
   387:             col_set=["feature", "label"],
   388:             data_key=DataHandlerLP.DK_L,
   389:         )
   390:         if df_train.empty or df_valid.empty:
   391:             raise ValueError("Empty data from dataset, please check your dataset config.")
   392: 
   393:         stock_index_map = np.load(self.stock_index, allow_pickle=True).item()
   394:         df_train["stock_index"] = 733
   395:         df_train["stock_index"] = df_train.index.get_level_values("instrument").map(stock_index_map)
   396:         df_valid["stock_index"] = 733
   397:         df_valid["stock_index"] = df_valid.index.get_level_values("instrument").map(stock_index_map)
   398: 
   399:         x_train, y_train, stock_index_train = df_train["feature"], df_train["label"], df_train["stock_index"]
   400:         x_valid, y_valid, stock_index_valid = df_valid["feature"], df_valid["label"], df_valid["stock_index"]
   401: 
   402:         stop_steps = 0
   403:         best_score = -np.inf
   404:         best_epoch = 0
   405:         best_param = None
   406: 
   407:         # optionally load pretrained base_model
   408:         if self.base_model == "LSTM":
   409:             pretrained_model = LSTMModel(d_feat=self.d_feat, hidden_size=self.hidden_size, num_layers=self.num_layers)
   410:         elif self.base_model == "GRU":
   411:             pretrained_model = GRUModel(d_feat=self.d_feat, hidden_size=self.hidden_size, num_layers=self.num_layers)
   412:         else:
   413:             raise ValueError("unknown base model name `%s`" % self.base_model)
   414: 
   415:         if self.model_path:
   416:             pretrained_model.load_state_dict(torch.load(self.model_path))
   417: 
   418:         model_dict = self.HIST_model.state_dict()
   419:         pretrained_dict = {
   420:             k: v for k, v in pretrained_model.state_dict().items() if k in model_dict
   421:         }
   422:         model_dict.update(pretrained_dict)
   423:         self.HIST_model.load_state_dict(model_dict)
   424: 
   425:         # train
   426:         self.fitted = True
   427: 
   428:         for step in range(self.n_epochs):
   429:             self.train_epoch(x_train, y_train, stock_index_train)
   430:             train_loss, train_score = self.test_epoch(x_train, y_train, stock_index_train)
   431:             val_loss, val_score = self.test_epoch(x_valid, y_valid, stock_index_valid)
   432:             print("Epoch%d: train %.6f, valid %.6f" % (step, train_score, val_score))
   433: 
   434:             if val_score > best_score:
   435:                 best_score = val_score
   436:                 stop_steps = 0
   437:                 best_epoch = step
   438:                 best_param = copy.deepcopy(self.HIST_model.state_dict())
   439:             else:
   440:                 stop_steps += 1
   441:                 if stop_steps >= self.early_stop:
   442:                     print("early stop")
   443:                     break
   444: 
   445:         print("best score: %.6lf @ %d" % (best_score, best_epoch))
   446:         self.HIST_model.load_state_dict(best_param)
   447: 
   448:         if self.use_gpu:
   449:             torch.cuda.empty_cache()
   450: 
   451:     def predict(self, dataset: DatasetH, segment="test"):
   452:         if not self.fitted:
   453:             raise ValueError("model is not fitted yet!")
   454: 
   455:         stock2concept_matrix = np.load(self.stock2concept)
   456:         stock_index_map = np.load(self.stock_index, allow_pickle=True).item()
   457:         df_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
   458:         df_test["stock_index"] = 733
   459:         df_test["stock_index"] = df_test.index.get_level_values("instrument").map(stock_index_map)
   460:         stock_index_test = df_test["stock_index"].values
   461:         stock_index_test[np.isnan(stock_index_test)] = 733
   462:         stock_index_test = stock_index_test.astype("int")
   463:         df_test = df_test.drop(["stock_index"], axis=1)
   464:         index = df_test.index
   465: 
   466:         self.HIST_model.eval()
   467:         x_values = df_test.values
   468:         preds = []
   469: 
   470:         # organize the data into daily batches
   471:         daily_index, daily_count = self.get_daily_inter(df_test, shuffle=False)
   472: 
   473:         for idx, count in zip(daily_index, daily_count):
   474:             batch = slice(idx, idx + count)
   475:             x_batch = torch.from_numpy(x_values[batch]).float().to(self.device)
   476:             concept_matrix = torch.from_numpy(stock2concept_matrix[stock_index_test[batch]]).float().to(self.device)
   477: 
   478:             with torch.no_grad():
   479:                 pred = self.HIST_model(x_batch, concept_matrix).detach().cpu().numpy()
   480: 
   481:             preds.append(pred)
   482: 
   483:         return pd.Series(np.concatenate(preds), index=index)
```

### `gats` baseline — editable region  [READ-ONLY — reference implementation]

In `qlib/custom_model.py`:

```python
Lines 58–378:
    55:     return _stock2concept_matrix[stock_indices].astype(np.float32)
    56: 
    57: 
    58: # =====================================================================
    59: # EDITABLE: CustomModel -- implement your stock prediction model here
    60: # =====================================================================
    61: import copy
    62: import torch.optim as optim
    63: 
    64: 
    65: class LSTMModel(nn.Module):
    66:     """LSTM backbone -- verbatim from qlib/contrib/model/pytorch_lstm.py."""
    67: 
    68:     def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
    69:         super().__init__()
    70:         self.rnn = nn.LSTM(
    71:             input_size=d_feat,
    72:             hidden_size=hidden_size,
    73:             num_layers=num_layers,
    74:             batch_first=True,
    75:             dropout=dropout,
    76:         )
    77:         self.fc_out = nn.Linear(hidden_size, 1)
    78:         self.d_feat = d_feat
    79: 
    80:     def forward(self, x):
    81:         # x: [N, F*T]
    82:         x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
    83:         x = x.permute(0, 2, 1)  # [N, T, F]
    84:         out, _ = self.rnn(x)
    85:         return self.fc_out(out[:, -1, :]).squeeze()
    86: 
    87: 
    88: class GRUModel(nn.Module):
    89:     """GRU backbone -- verbatim from qlib/contrib/model/pytorch_gru.py."""
    90: 
    91:     def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
    92:         super().__init__()
    93:         self.rnn = nn.GRU(
    94:             input_size=d_feat,
    95:             hidden_size=hidden_size,
    96:             num_layers=num_layers,
    97:             batch_first=True,
    98:             dropout=dropout,
    99:         )
   100:         self.fc_out = nn.Linear(hidden_size, 1)
   101:         self.d_feat = d_feat
   102: 
   103:     def forward(self, x):
   104:         # x: [N, F*T]
   105:         x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
   106:         x = x.permute(0, 2, 1)  # [N, T, F]
   107:         out, _ = self.rnn(x)
   108:         return self.fc_out(out[:, -1, :]).squeeze()
   109: 
   110: 
   111: class GATModel(nn.Module):
   112:     """GAT network -- verbatim from qlib/contrib/model/pytorch_gats.py."""
   113: 
   114:     def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"):
   115:         super().__init__()
   116: 
   117:         if base_model == "GRU":
   118:             self.rnn = nn.GRU(
   119:                 input_size=d_feat,
   120:                 hidden_size=hidden_size,
   121:                 num_layers=num_layers,
   122:                 batch_first=True,
   123:                 dropout=dropout,
   124:             )
   125:         elif base_model == "LSTM":
   126:             self.rnn = nn.LSTM(
   127:                 input_size=d_feat,
   128:                 hidden_size=hidden_size,
   129:                 num_layers=num_layers,
   130:                 batch_first=True,
   131:                 dropout=dropout,
   132:             )
   133:         else:
   134:             raise ValueError("unknown base model name `%s`" % base_model)
   135: 
   136:         self.hidden_size = hidden_size
   137:         self.d_feat = d_feat
   138:         self.transformation = nn.Linear(self.hidden_size, self.hidden_size)
   139:         self.a = nn.Parameter(torch.randn(self.hidden_size * 2, 1))
   140:         self.a.requires_grad = True
   141:         self.fc = nn.Linear(self.hidden_size, self.hidden_size)
   142:         self.fc_out = nn.Linear(hidden_size, 1)
   143:         self.leaky_relu = nn.LeakyReLU()
   144:         self.softmax = nn.Softmax(dim=1)
   145: 
   146:     def cal_attention(self, x, y):
   147:         x = self.transformation(x)
   148:         y = self.transformation(y)
   149: 
   150:         sample_num = x.shape[0]
   151:         dim = x.shape[1]
   152:         e_x = x.expand(sample_num, sample_num, dim)
   153:         e_y = torch.transpose(e_x, 0, 1)
   154:         attention_in = torch.cat((e_x, e_y), 2).view(-1, dim * 2)
   155:         self.a_t = torch.t(self.a)
   156:         attention_out = self.a_t.mm(torch.t(attention_in)).view(sample_num, sample_num)
   157:         attention_out = self.leaky_relu(attention_out)
   158:         att_weight = self.softmax(attention_out)
   159:         return att_weight
   160: 
   161:     def forward(self, x):
   162:         # x: [N, F*T] -- DatasetH provides flattened features
   163:         x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
   164:         x = x.permute(0, 2, 1)  # [N, T, F]
   165:         out, _ = self.rnn(x)
   166:         hidden = out[:, -1, :]
   167:         att_weight = self.cal_attention(hidden, hidden)
   168:         hidden = att_weight.mm(hidden) + hidden
   169:         hidden = self.fc(hidden)
   170:         hidden = self.leaky_relu(hidden)
   171:         return self.fc_out(hidden).squeeze()
   172: 
   173: 
   174: class CustomModel(Model):
   175:     """GATs model -- faithful to qlib's official GATs (pytorch_gats.py).
   176: 
   177:     Hyperparameters from official benchmark:
   178:     examples/benchmarks/GATs/workflow_config_gats_Alpha360.yaml
   179: 
   180:     Uses DatasetH with Alpha360 (d_feat=6, 360 features reshaped to 60x6).
   181:     Daily batching via get_daily_inter() for graph attention across stocks.
   182:     """
   183: 
   184:     def __init__(self):
   185:         super().__init__()
   186:         # Official benchmark hyperparameters (Alpha360)
   187:         self.d_feat = 6
   188:         self.hidden_size = 64
   189:         self.num_layers = 2
   190:         self.dropout = 0.7
   191:         self.n_epochs = 200
   192:         self.lr = 1e-4
   193:         self.metric = "loss"
   194:         self.early_stop = 20
   195:         self.loss = "mse"
   196:         self.base_model = "LSTM"
   197:         self.model_path = "examples/benchmarks/LSTM/model_lstm_csi300.pkl"
   198:         self.device = torch.device(
   199:             "cuda:0" if torch.cuda.is_available() else "cpu"
   200:         )
   201: 
   202:         self.GAT_model = GATModel(
   203:             d_feat=self.d_feat,
   204:             hidden_size=self.hidden_size,
   205:             num_layers=self.num_layers,
   206:             dropout=self.dropout,
   207:             base_model=self.base_model,
   208:         )
   209:         self.train_optimizer = optim.Adam(
   210:             self.GAT_model.parameters(), lr=self.lr
   211:         )
   212:         self.fitted = False
   213:         self.GAT_model.to(self.device)
   214: 
   215:     @property
   216:     def use_gpu(self):
   217:         return self.device != torch.device("cpu")
   218: 
   219:     def mse(self, pred, label):
   220:         loss = (pred - label) ** 2
   221:         return torch.mean(loss)
   222: 
   223:     def loss_fn(self, pred, label):
   224:         mask = ~torch.isnan(label)
   225:         if self.loss == "mse":
   226:             return self.mse(pred[mask], label[mask])
   227:         raise ValueError("unknown loss `%s`" % self.loss)
   228: 
   229:     def metric_fn(self, pred, label):
   230:         mask = torch.isfinite(label)
   231:         if self.metric in ("", "loss"):
   232:             return -self.loss_fn(pred[mask], label[mask])
   233:         raise ValueError("unknown metric `%s`" % self.metric)
   234: 
   235:     def get_daily_inter(self, df, shuffle=False):
   236:         # organize the train data into daily batches
   237:         daily_count = df.groupby(level=0, group_keys=False).size().values
   238:         daily_index = np.roll(np.cumsum(daily_count), 1)
   239:         daily_index[0] = 0
   240:         if shuffle:
   241:             # shuffle data
   242:             daily_shuffle = list(zip(daily_index, daily_count))
   243:             np.random.shuffle(daily_shuffle)
   244:             daily_index, daily_count = zip(*daily_shuffle)
   245:         return daily_index, daily_count
   246: 
   247:     def train_epoch(self, x_train, y_train):
   248:         x_train_values = x_train.values
   249:         y_train_values = np.squeeze(y_train.values)
   250:         self.GAT_model.train()
   251: 
   252:         # organize the train data into daily batches
   253:         daily_index, daily_count = self.get_daily_inter(x_train, shuffle=True)
   254: 
   255:         for idx, count in zip(daily_index, daily_count):
   256:             batch = slice(idx, idx + count)
   257:             feature = torch.from_numpy(x_train_values[batch]).float().to(self.device)
   258:             label = torch.from_numpy(y_train_values[batch]).float().to(self.device)
   259: 
   260:             pred = self.GAT_model(feature)
   261:             loss = self.loss_fn(pred, label)
   262: 
   263:             self.train_optimizer.zero_grad()
   264:             loss.backward()
   265:             torch.nn.utils.clip_grad_value_(self.GAT_model.parameters(), 3.0)
   266:             self.train_optimizer.step()
   267: 
   268:     def test_epoch(self, data_x, data_y):
   269:         # prepare training data
   270:         x_values = data_x.values
   271:         y_values = np.squeeze(data_y.values)
   272:         self.GAT_model.eval()
   273: 
   274:         scores = []
   275:         losses = []
   276: 
   277:         # organize the test data into daily batches
   278:         daily_index, daily_count = self.get_daily_inter(data_x, shuffle=False)
   279: 
   280:         for idx, count in zip(daily_index, daily_count):
   281:             batch = slice(idx, idx + count)
   282:             feature = torch.from_numpy(x_values[batch]).float().to(self.device)
   283:             label = torch.from_numpy(y_values[batch]).float().to(self.device)
   284: 
   285:             with torch.no_grad():
   286:                 pred = self.GAT_model(feature)
   287:                 loss = self.loss_fn(pred, label)
   288:                 losses.append(loss.item())
   289: 
   290:                 score = self.metric_fn(pred, label)
   291:                 scores.append(score.item())
   292: 
   293:         return np.mean(losses), np.mean(scores)
   294: 
   295:     def fit(self, dataset: DatasetH):
   296:         df_train, df_valid = dataset.prepare(
   297:             ["train", "valid"],
   298:             col_set=["feature", "label"],
   299:             data_key=DataHandlerLP.DK_L,
   300:         )
   301:         if df_train.empty or df_valid.empty:
   302:             raise ValueError("Empty data from dataset, please check your dataset config.")
   303: 
   304:         x_train, y_train = df_train["feature"], df_train["label"]
   305:         x_valid, y_valid = df_valid["feature"], df_valid["label"]
   306: 
   307:         stop_steps = 0
   308:         best_score = -np.inf
   309:         best_epoch = 0
   310:         best_param = None
   311: 
   312:         # optionally load pretrained base_model
   313:         if self.base_model == "LSTM":
   314:             pretrained_model = LSTMModel(d_feat=self.d_feat, hidden_size=self.hidden_size, num_layers=self.num_layers)
   315:         elif self.base_model == "GRU":
   316:             pretrained_model = GRUModel(d_feat=self.d_feat, hidden_size=self.hidden_size, num_layers=self.num_layers)
   317:         else:
   318:             raise ValueError("unknown base model name `%s`" % self.base_model)
   319: 
   320:         if self.model_path:
   321:             pretrained_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
   322: 
   323:         model_dict = self.GAT_model.state_dict()
   324:         pretrained_dict = {
   325:             k: v for k, v in pretrained_model.state_dict().items() if k in model_dict
   326:         }
   327:         model_dict.update(pretrained_dict)
   328:         self.GAT_model.load_state_dict(model_dict)
   329: 
   330:         # train
   331:         self.fitted = True
   332: 
   333:         for step in range(self.n_epochs):
   334:             self.train_epoch(x_train, y_train)
   335:             train_loss, train_score = self.test_epoch(x_train, y_train)
   336:             val_loss, val_score = self.test_epoch(x_valid, y_valid)
   337:             print("Epoch%d: train %.6f, valid %.6f" % (step, train_score, val_score))
   338: 
   339:             if val_score > best_score:
   340:                 best_score = val_score
   341:                 stop_steps = 0
   342:                 best_epoch = step
   343:                 best_param = copy.deepcopy(self.GAT_model.state_dict())
   344:             else:
   345:                 stop_steps += 1
   346:                 if stop_steps >= self.early_stop:
   347:                     print("early stop")
   348:                     break
   349: 
   350:         print("best score: %.6lf @ %d" % (best_score, best_epoch))
   351:         self.GAT_model.load_state_dict(best_param)
   352: 
   353:         if self.use_gpu:
   354:             torch.cuda.empty_cache()
   355: 
   356:     def predict(self, dataset: DatasetH, segment="test"):
   357:         if not self.fitted:
   358:             raise ValueError("model is not fitted yet!")
   359: 
   360:         x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
   361:         index = x_test.index
   362:         self.GAT_model.eval()
   363:         x_values = x_test.values
   364:         preds = []
   365: 
   366:         # organize the data into daily batches
   367:         daily_index, daily_count = self.get_daily_inter(x_test, shuffle=False)
   368: 
   369:         for idx, count in zip(daily_index, daily_count):
   370:             batch = slice(idx, idx + count)
   371:             x_batch = torch.from_numpy(x_values[batch]).float().to(self.device)
   372: 
   373:             with torch.no_grad():
   374:                 pred = self.GAT_model(x_batch).detach().cpu().numpy()
   375: 
   376:             preds.append(pred)
   377: 
   378:         return pd.Series(np.concatenate(preds), index=index)
```

### `lgbm` baseline — editable region  [READ-ONLY — reference implementation]

In `qlib/custom_model.py`:

```python
Lines 58–144:
    55:     return _stock2concept_matrix[stock_indices].astype(np.float32)
    56: 
    57: 
    58: # =====================================================================
    59: # EDITABLE: CustomModel -- implement your stock prediction model here
    60: # =====================================================================
    61: class CustomModel(Model):
    62:     """LightGBM model -- faithful to qlib's official LGBModel (gbdt.py).
    63: 
    64:     Hyperparameters from official benchmark:
    65:     examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha360.yaml
    66:     """
    67: 
    68:     def __init__(self):
    69:         super().__init__()
    70:         # Official benchmark kwargs (passed to lgb.train via self.params)
    71:         self.params = {
    72:             "objective": "mse",
    73:             "colsample_bytree": 0.8879,
    74:             "learning_rate": 0.0421,
    75:             "subsample": 0.8789,
    76:             "lambda_l1": 205.6999,
    77:             "lambda_l2": 580.9768,
    78:             "max_depth": 8,
    79:             "num_leaves": 210,
    80:             "num_threads": 20,
    81:             "verbosity": -1,
    82:         }
    83:         self.early_stopping_rounds = 50
    84:         self.num_boost_round = 1000
    85:         self.model = None
    86: 
    87:     def _prepare_data(self, dataset):
    88:         """Prepare LightGBM datasets -- matches LGBModel._prepare_data()."""
    89:         import lightgbm as lgb
    90: 
    91:         ds_l = []
    92:         for key in ["train", "valid"]:
    93:             if key in dataset.segments:
    94:                 df = dataset.prepare(
    95:                     key, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
    96:                 )
    97:                 if df.empty:
    98:                     raise ValueError(
    99:                         "Empty data from dataset, please check your dataset config."
   100:                     )
   101:                 x, y = df["feature"], df["label"]
   102:                 # Lightgbm need 1D array as its label
   103:                 if y.values.ndim == 2 and y.values.shape[1] == 1:
   104:                     y = np.squeeze(y.values)
   105:                 else:
   106:                     raise ValueError(
   107:                         "LightGBM doesn't support multi-label training"
   108:                     )
   109:                 ds_l.append(
   110:                     (lgb.Dataset(x.values, label=y, free_raw_data=False), key)
   111:                 )
   112:         return ds_l
   113: 
   114:     def fit(self, dataset: DatasetH):
   115:         import lightgbm as lgb
   116: 
   117:         ds_l = self._prepare_data(dataset)
   118:         ds, names = list(zip(*ds_l))
   119:         early_stopping_callback = lgb.early_stopping(
   120:             self.early_stopping_rounds
   121:         )
   122:         verbose_eval_callback = lgb.log_evaluation(period=20)
   123:         evals_result = {}
   124:         evals_result_callback = lgb.record_evaluation(evals_result)
   125:         self.model = lgb.train(
   126:             self.params,
   127:             ds[0],  # training dataset
   128:             num_boost_round=self.num_boost_round,
   129:             valid_sets=ds,
   130:             valid_names=names,
   131:             callbacks=[
   132:                 early_stopping_callback,
   133:                 verbose_eval_callback,
   134:                 evals_result_callback,
   135:             ],
   136:         )
   137: 
   138:     def predict(self, dataset: DatasetH, segment="test"):
   139:         if self.model is None:
   140:             raise ValueError("model is not fitted yet!")
   141:         x_test = dataset.prepare(
   142:             segment, col_set="feature", data_key=DataHandlerLP.DK_I
   143:         )
   144:         return pd.Series(self.model.predict(x_test.values), index=x_test.index)
```

### `lgbm` baseline — editable region  [READ-ONLY — reference implementation]

In `qlib/workflow_config.yaml`:

```python
Lines 14–26:
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
    24:         class: Alpha360
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
