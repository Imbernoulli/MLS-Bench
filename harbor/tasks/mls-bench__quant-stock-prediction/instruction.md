# MLS-Bench: quant-stock-prediction

# Quantitative Stock Prediction on Chinese Equity Universes

## Research Question
Can a single, reusable predictive component deliver consistently strong cross-sectional return signals across different Chinese equity universes and time periods, when the input features, label, train/valid/test splits, and downstream backtest are held fixed?

## Background
Quantitative stock prediction in Microsoft `qlib` formulates daily forecasting as a cross-sectional regression: at each trading day, predict the next-day excess return for every stock in the universe, then rank stocks by the predicted score and feed the ranking into a fixed portfolio-construction routine. The challenge is twofold: market data is noisy and non-stationary; and a model must rank well across heterogeneous instruments, not just minimize per-stock loss. A wide variety of methods have been tried — gradient boosted trees on engineered factors (LightGBM), pure sequence models (LSTM, Transformer), and graph-aware extensions — but none dominates universally, motivating principled studies of model-component contributions under a common protocol.

## Objective
Implement a `CustomModel` in `custom_model.py` that exposes the standard qlib model interface (`fit(dataset)` and `predict(dataset, segment="test")`). The class is wired into the qlib `workflow_config.yaml`, which controls the dataset adapter / preprocessor block but keeps the universe, label, and date splits fixed. You may change the dataset class (e.g., to `TSDatasetH`) or processors if your model needs a different input view.

## Fixed Pipeline
The training and evaluation pipeline (universes, label, train/valid/test splits,
and downstream backtest) is fixed by the harness and not editable.

The input feature view your model receives is Alpha360: 360 features per
stock-day (6 base ratios over 60 days of history). For temporal models, reshape
with `x.reshape(N, 6, 60).permute(0, 2, 1) -> [N, 60, 6]` to get 60 time steps of
6 features each.

## Model Interface
```python
from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

class CustomModel(Model):
    def fit(self, dataset: DatasetH): ...
    def predict(self, dataset: DatasetH, segment: str = "test") -> pd.Series: ...
```
`predict` must return a `pd.Series` indexed by `(datetime, instrument)` matching the requested segment's index. Available imports inside the class: `torch`, `numpy`, `pandas`, `lightgbm`, `sklearn`, `scipy`.

## Reference Implementations (read-only)
Three reference models ship with qlib's `examples/benchmarks/` and are available as read-only context. Defaults are taken from each method's qlib example config.

- **LightGBM** — Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", NeurIPS 2017. qlib defaults: `loss=mse`, `learning_rate=0.0421`, `num_leaves=210`, `feature_fraction=0.879`, `bagging_fraction=0.856`, `bagging_freq=5`, `lambda_l1=205.7`, `lambda_l2=580.9`.
- **LSTM** — qlib's RNN baseline with `d_feat=6`, `hidden_size=64`, `num_layers=2`, `dropout=0.0`, Adam `lr=1e-3`, `n_epochs=200`, early-stopping patience 20.
- **Transformer** — qlib defaults `d_feat=6`, `d_model=64`, `nhead=2`, `num_layers=2`, `dropout=0.5`, Adam `lr=1e-4`.

Code: https://github.com/microsoft/qlib.


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
- editable lines **13–25**
- editable lines **31–44**


Other files you may **read** for context (do not modify):
- `qlib/qlib/model/base.py`


## Readable Context


### `qlib/custom_model.py`  [EDITABLE — lines 16–103 only]

```python
     1: # Custom stock prediction model for MLS-Bench
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
    20:     """Custom stock prediction model.
    21: 
    22:     You must implement:
    23:         fit(dataset)    — train the model on the training data
    24:         predict(dataset, segment="test") — return predictions as pd.Series
    25: 
    26:     The dataset is a qlib DatasetH with Alpha360 features (360 features per
    27:     stock per day). The 360 features come from 6 base features
    28:     (open/close/high/low/volume/vwap ratios) x 60 days of history.
    29: 
    30:     For temporal models, features can be reshaped:
    31:         x.reshape(N, 6, 60).permute(0, 2, 1) -> [N, 60, 6]
    32:     giving 60 time steps of 6 features each.
    33: 
    34:     Segments: "train", "valid", "test".
    35: 
    36:     Getting data from the dataset:
    37:         df_train = dataset.prepare("train", col_set=["feature", "label"],
    38:                                     data_key=DataHandlerLP.DK_L)
    39:         features = df_train["feature"]   # DataFrame: (n_samples, 360)
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

### `qlib/workflow_config.yaml`  [EDITABLE — lines 13–25, lines 31–44 only]

```yaml
     1: # Qlib workflow configuration for CSI300 stock prediction benchmark.
     2: # Used by run_workflow.py — matches Alpha360/CSI300 official benchmark settings.
     3: 
     4: qlib_init:
     5:   provider_uri: "~/.qlib/qlib_data/cn_data"
     6:   region: cn
     7: 
     8: sys:
     9:   rel_path:
    10:     - "."           # So custom_model.py is importable via module_path
    11: 
    12: task:
    13:   model:
    14:     class: CustomModel
    15:     module_path: custom_model
    16:     kwargs: {}
    17: 
    18:   dataset:
    19:     class: DatasetH
    20:     module_path: qlib.data.dataset
    21:     kwargs:
    22:       handler:
    23:         class: Alpha360
    24:         module_path: qlib.contrib.data.handler
    25:         kwargs:
    26:           start_time: "2008-01-01"
    27:           end_time: "2020-08-01"
    28:           fit_start_time: "2008-01-01"
    29:           fit_end_time: "2014-12-31"
    30:           instruments: csi300
    31:           infer_processors:
    32:             - class: RobustZScoreNorm
    33:               kwargs:
    34:                 fields_group: feature
    35:                 clip_outlier: true
    36:             - class: Fillna
    37:               kwargs:
    38:                 fields_group: feature
    39:           learn_processors:
    40:             - class: DropnaLabel
    41:             - class: CSRankNorm
    42:               kwargs:
    43:                 fields_group: label
    44:           label: ["Ref($close, -2) / Ref($close, -1) - 1"]
    45:       segments:
    46:         train: ["2008-01-01", "2014-12-31"]
    47:         valid: ["2015-01-01", "2016-12-31"]
    48:         test: ["2017-01-01", "2020-08-01"]
    49: 
    50:   record:
    51:     - class: SignalRecord
    52:       module_path: qlib.workflow.record_temp
    53:       kwargs:
    54:         model: "<MODEL>"
    55:         dataset: "<DATASET>"
    56:     - class: SigAnaRecord
    57:       module_path: qlib.workflow.record_temp
    58:       kwargs:
    59:         ana_long_short: false
    60:         ann_scaler: 252
    61:     - class: PortAnaRecord
    62:       module_path: qlib.workflow.record_temp
    63:       kwargs:
    64:         config: &port_analysis_config
    65:           strategy:
    66:             class: TopkDropoutStrategy
    67:             module_path: qlib.contrib.strategy
    68:             kwargs:
    69:               signal: "<PRED>"
    70:               topk: 50
    71:               n_drop: 5
    72:           backtest:
    73:             start_time: "2017-01-01"
    74:             end_time: "2020-08-01"
    75:             account: 100000000
    76:             benchmark: SH000300
    77:             exchange_kwargs:
    78:               limit_threshold: 0.095
    79:               deal_price: close
    80:               open_cost: 0.0005
    81:               close_cost: 0.0015
    82:               min_cost: 5
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


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
    22:     Hyperparameters from official Alpha360 benchmark:
    23:     examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha360.yaml
    24:     """
    25: 
    26:     def __init__(self):
    27:         super().__init__()
    28:         # Official Alpha360 benchmark kwargs (passed to lgb.train via self.params)
    29:         self.params = {
    30:             "objective": "mse",
    31:             "colsample_bytree": 0.8879,
    32:             "learning_rate": 0.0421,
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

### `lstm` baseline — editable region  [READ-ONLY — reference implementation]

In `qlib/custom_model.py`:

```python
Lines 16–243:
    13: 
    14: DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    15: 
    16: # =====================================================================
    17: # EDITABLE: CustomModel — implement your stock prediction model here
    18: # =====================================================================
    19: import copy
    20: import torch.optim as optim
    21: 
    22: 
    23: class LSTMModel(nn.Module):
    24:     """LSTM network — verbatim from qlib/contrib/model/pytorch_lstm.py."""
    25: 
    26:     def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
    27:         super().__init__()
    28:         self.rnn = nn.LSTM(
    29:             input_size=d_feat,
    30:             hidden_size=hidden_size,
    31:             num_layers=num_layers,
    32:             batch_first=True,
    33:             dropout=dropout,
    34:         )
    35:         self.fc_out = nn.Linear(hidden_size, 1)
    36:         self.d_feat = d_feat
    37: 
    38:     def forward(self, x):
    39:         # x: [N, F*T] — Alpha360 gives 360 flat features
    40:         x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
    41:         x = x.permute(0, 2, 1)  # [N, T, F]
    42:         out, _ = self.rnn(x)
    43:         return self.fc_out(out[:, -1, :]).squeeze()
    44: 
    45: 
    46: class CustomModel(Model):
    47:     """LSTM model — faithful to qlib's official LSTM (pytorch_lstm.py).
    48: 
    49:     Uses DatasetH with Alpha360 features. The LSTMModel reshapes the flat
    50:     360-dim feature vector internally: [N, 360] -> [N, 6, 60] -> [N, 60, 6].
    51: 
    52:     Hyperparameters from official benchmark:
    53:     examples/benchmarks/LSTM/workflow_config_lstm_Alpha360.yaml
    54:     """
    55: 
    56:     def __init__(self):
    57:         super().__init__()
    58:         # Official Alpha360 benchmark hyperparameters
    59:         self.d_feat = 6
    60:         self.hidden_size = 64
    61:         self.num_layers = 2
    62:         self.dropout = 0.0
    63:         self.n_epochs = 200
    64:         self.lr = 0.001
    65:         self.metric = "loss"
    66:         self.batch_size = 800
    67:         self.early_stop = 20
    68:         self.loss = "mse"
    69:         self.device = torch.device(
    70:             "cuda:0" if torch.cuda.is_available() else "cpu"
    71:         )
    72: 
    73:         self.lstm_model = LSTMModel(
    74:             d_feat=self.d_feat,
    75:             hidden_size=self.hidden_size,
    76:             num_layers=self.num_layers,
    77:             dropout=self.dropout,
    78:         ).to(self.device)
    79:         self.train_optimizer = optim.Adam(
    80:             self.lstm_model.parameters(), lr=self.lr
    81:         )
    82:         self.fitted = False
    83: 
    84:     @property
    85:     def use_gpu(self):
    86:         return self.device != torch.device("cpu")
    87: 
    88:     def mse(self, pred, label):
    89:         loss = (pred - label) ** 2
    90:         return torch.mean(loss)
    91: 
    92:     def loss_fn(self, pred, label):
    93:         mask = ~torch.isnan(label)
    94:         if self.loss == "mse":
    95:             return self.mse(pred[mask], label[mask])
    96:         raise ValueError("unknown loss `%s`" % self.loss)
    97: 
    98:     def metric_fn(self, pred, label):
    99:         mask = torch.isfinite(label)
   100:         if self.metric in ("", "loss"):
   101:             return -self.loss_fn(pred[mask], label[mask])
   102:         raise ValueError("unknown metric `%s`" % self.metric)
   103: 
   104:     def train_epoch(self, x_train, y_train):
   105:         x_train_values = x_train.values
   106:         y_train_values = np.squeeze(y_train.values)
   107: 
   108:         self.lstm_model.train()
   109: 
   110:         indices = np.arange(len(x_train_values))
   111:         np.random.shuffle(indices)
   112: 
   113:         for i in range(len(indices))[:: self.batch_size]:
   114:             if len(indices) - i < self.batch_size:
   115:                 break
   116: 
   117:             feature = (
   118:                 torch.from_numpy(x_train_values[indices[i : i + self.batch_size]])
   119:                 .float()
   120:                 .to(self.device)
   121:             )
   122:             label = (
   123:                 torch.from_numpy(y_train_values[indices[i : i + self.batch_size]])
   124:                 .float()
   125:                 .to(self.device)
   126:             )
   127: 
   128:             pred = self.lstm_model(feature)
   129:             loss = self.loss_fn(pred, label)
   130: 
   131:             self.train_optimizer.zero_grad()
   132:             loss.backward()
   133:             torch.nn.utils.clip_grad_value_(self.lstm_model.parameters(), 3.0)
   134:             self.train_optimizer.step()
   135: 
   136:     def test_epoch(self, data_x, data_y):
   137:         x_values = data_x.values
   138:         y_values = np.squeeze(data_y.values)
   139: 
   140:         self.lstm_model.eval()
   141: 
   142:         scores = []
   143:         losses = []
   144: 
   145:         indices = np.arange(len(x_values))
   146: 
   147:         for i in range(len(indices))[:: self.batch_size]:
   148:             if len(indices) - i < self.batch_size:
   149:                 break
   150: 
   151:             feature = (
   152:                 torch.from_numpy(x_values[indices[i : i + self.batch_size]])
   153:                 .float()
   154:                 .to(self.device)
   155:             )
   156:             label = (
   157:                 torch.from_numpy(y_values[indices[i : i + self.batch_size]])
   158:                 .float()
   159:                 .to(self.device)
   160:             )
   161: 
   162:             pred = self.lstm_model(feature)
   163:             loss = self.loss_fn(pred, label)
   164:             losses.append(loss.item())
   165: 
   166:             score = self.metric_fn(pred, label)
   167:             scores.append(score.item())
   168: 
   169:         return np.mean(losses), np.mean(scores)
   170: 
   171:     def fit(self, dataset: DatasetH):
   172:         df_train, df_valid, df_test = dataset.prepare(
   173:             ["train", "valid", "test"],
   174:             col_set=["feature", "label"],
   175:             data_key=DataHandlerLP.DK_L,
   176:         )
   177:         if df_train.empty or df_valid.empty:
   178:             raise ValueError(
   179:                 "Empty data from dataset, please check your dataset config."
   180:             )
   181: 
   182:         x_train, y_train = df_train["feature"], df_train["label"]
   183:         x_valid, y_valid = df_valid["feature"], df_valid["label"]
   184: 
   185:         stop_steps = 0
   186:         best_score = -np.inf
   187:         best_epoch = 0
   188:         best_param = None
   189: 
   190:         self.fitted = True
   191: 
   192:         for step in range(self.n_epochs):
   193:             self.train_epoch(x_train, y_train)
   194:             train_loss, train_score = self.test_epoch(x_train, y_train)
   195:             val_loss, val_score = self.test_epoch(x_valid, y_valid)
   196:             print(
   197:                 "Epoch%d: train %.6f, valid %.6f"
   198:                 % (step, train_score, val_score)
   199:             )
   200: 
   201:             if val_score > best_score:
   202:                 best_score = val_score
   203:                 stop_steps = 0
   204:                 best_epoch = step
   205:                 best_param = copy.deepcopy(self.lstm_model.state_dict())
   206:             else:
   207:                 stop_steps += 1
   208:                 if stop_steps >= self.early_stop:
   209:                     print("early stop")
   210:                     break
   211: 
   212:         print("best score: %.6lf @ %d" % (best_score, best_epoch))
   213:         self.lstm_model.load_state_dict(best_param)
   214: 
   215:         if self.use_gpu:
   216:             torch.cuda.empty_cache()
   217: 
   218:     def predict(self, dataset: DatasetH, segment="test"):
   219:         if not self.fitted:
   220:             raise ValueError("model is not fitted yet!")
   221: 
   222:         x_test = dataset.prepare(
   223:             segment, col_set="feature", data_key=DataHandlerLP.DK_I
   224:         )
   225:         index = x_test.index
   226:         self.lstm_model.eval()
   227:         x_values = x_test.values
   228:         sample_num = x_values.shape[0]
   229:         preds = []
   230: 
   231:         for begin in range(sample_num)[:: self.batch_size]:
   232:             if sample_num - begin < self.batch_size:
   233:                 end = sample_num
   234:             else:
   235:                 end = begin + self.batch_size
   236:             x_batch = (
   237:                 torch.from_numpy(x_values[begin:end]).float().to(self.device)
   238:             )
   239:             with torch.no_grad():
   240:                 pred = self.lstm_model(x_batch).detach().cpu().numpy()
   241:             preds.append(pred)
   242: 
   243:         return pd.Series(np.concatenate(preds), index=index)
```

### `transformer` baseline — editable region  [READ-ONLY — reference implementation]

In `qlib/custom_model.py`:

```python
Lines 16–300:
    13: 
    14: DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    15: 
    16: # =====================================================================
    17: # EDITABLE: CustomModel — implement your stock prediction model here
    18: # =====================================================================
    19: import copy
    20: import math
    21: import os
    22: import torch.optim as optim
    23: 
    24: 
    25: class PositionalEncoding(nn.Module):
    26:     """Positional encoding — verbatim from qlib/contrib/model/pytorch_transformer.py."""
    27: 
    28:     def __init__(self, d_model, max_len=1000):
    29:         super(PositionalEncoding, self).__init__()
    30:         pe = torch.zeros(max_len, d_model)
    31:         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    32:         div_term = torch.exp(
    33:             torch.arange(0, d_model, 2).float()
    34:             * (-math.log(10000.0) / d_model)
    35:         )
    36:         pe[:, 0::2] = torch.sin(position * div_term)
    37:         pe[:, 1::2] = torch.cos(position * div_term)
    38:         pe = pe.unsqueeze(0).transpose(0, 1)
    39:         self.register_buffer("pe", pe)
    40: 
    41:     def forward(self, x):
    42:         # [T, N, F]
    43:         return x + self.pe[: x.size(0), :]
    44: 
    45: 
    46: class Transformer(nn.Module):
    47:     """Transformer network — verbatim from qlib/contrib/model/pytorch_transformer.py.
    48: 
    49:     Reshapes flat Alpha360 features internally:
    50:     [N, F*T] -> [N, d_feat, T] -> [N, T, d_feat]
    51:     """
    52: 
    53:     def __init__(
    54:         self, d_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.5, device=None
    55:     ):
    56:         super(Transformer, self).__init__()
    57:         self.feature_layer = nn.Linear(d_feat, d_model)
    58:         self.pos_encoder = PositionalEncoding(d_model)
    59:         self.encoder_layer = nn.TransformerEncoderLayer(
    60:             d_model=d_model, nhead=nhead, dropout=dropout
    61:         )
    62:         self.transformer_encoder = nn.TransformerEncoder(
    63:             self.encoder_layer, num_layers=num_layers
    64:         )
    65:         self.decoder_layer = nn.Linear(d_model, 1)
    66:         self.device = device
    67:         self.d_feat = d_feat
    68: 
    69:     def forward(self, src):
    70:         # src [N, F*T] --> [N, T, F]
    71:         src = src.reshape(len(src), self.d_feat, -1).permute(0, 2, 1)
    72:         src = self.feature_layer(src)
    73: 
    74:         # src [N, T, F] --> [T, N, F]
    75:         src = src.transpose(1, 0)  # not batch first
    76: 
    77:         mask = None
    78: 
    79:         src = self.pos_encoder(src)
    80:         output = self.transformer_encoder(src, mask)
    81: 
    82:         # [T, N, F] --> [N, T*F]
    83:         output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])
    84: 
    85:         return output.squeeze()
    86: 
    87: 
    88: class CustomModel(Model):
    89:     """Transformer model — faithful to qlib's official TransformerModel
    90:     (pytorch_transformer.py).
    91: 
    92:     Uses DatasetH with Alpha360 features. The Transformer reshapes the flat
    93:     360-dim feature vector internally: [N, 360] -> [N, 6, 60] -> [N, 60, 6].
    94: 
    95:     Hyperparameters from official benchmark:
    96:     examples/benchmarks/Transformer/workflow_config_transformer_Alpha360.yaml
    97:     """
    98: 
    99:     def __init__(self):
   100:         super().__init__()
   101:         # Official Alpha360 benchmark hyperparameters
   102:         self.d_feat = 6
   103:         self.d_model = 64
   104:         self.nhead = 2
   105:         self.num_layers = 2
   106:         self.dropout = 0
   107:         self.n_epochs = 100
   108:         self.lr = 0.0001
   109:         self.metric = ""
   110:         self.batch_size = 2048
   111:         self.early_stop = 5
   112:         self.loss = "mse"
   113:         self.reg = 1e-3
   114:         self.seed = int(os.environ.get("SEED", "42"))
   115:         self.device = torch.device(
   116:             "cuda:0" if torch.cuda.is_available() else "cpu"
   117:         )
   118: 
   119:         if self.seed is not None:
   120:             np.random.seed(self.seed)
   121:             torch.manual_seed(self.seed)
   122: 
   123:         self.model = Transformer(
   124:             self.d_feat,
   125:             self.d_model,
   126:             self.nhead,
   127:             self.num_layers,
   128:             self.dropout,
   129:             self.device,
   130:         )
   131:         self.train_optimizer = optim.Adam(
   132:             self.model.parameters(), lr=self.lr, weight_decay=self.reg
   133:         )
   134:         self.fitted = False
   135:         self.model.to(self.device)
   136: 
   137:     @property
   138:     def use_gpu(self):
   139:         return self.device != torch.device("cpu")
   140: 
   141:     def mse(self, pred, label):
   142:         loss = (pred.float() - label.float()) ** 2
   143:         return torch.mean(loss)
   144: 
   145:     def loss_fn(self, pred, label):
   146:         mask = ~torch.isnan(label)
   147:         if self.loss == "mse":
   148:             return self.mse(pred[mask], label[mask])
   149:         raise ValueError("unknown loss `%s`" % self.loss)
   150: 
   151:     def metric_fn(self, pred, label):
   152:         mask = torch.isfinite(label)
   153:         if self.metric in ("", "loss"):
   154:             return -self.loss_fn(pred[mask], label[mask])
   155:         raise ValueError("unknown metric `%s`" % self.metric)
   156: 
   157:     def train_epoch(self, x_train, y_train):
   158:         x_train_values = x_train.values
   159:         y_train_values = np.squeeze(y_train.values)
   160: 
   161:         self.model.train()
   162: 
   163:         indices = np.arange(len(x_train_values))
   164:         np.random.shuffle(indices)
   165: 
   166:         for i in range(len(indices))[:: self.batch_size]:
   167:             if len(indices) - i < self.batch_size:
   168:                 break
   169: 
   170:             feature = (
   171:                 torch.from_numpy(x_train_values[indices[i : i + self.batch_size]])
   172:                 .float()
   173:                 .to(self.device)
   174:             )
   175:             label = (
   176:                 torch.from_numpy(y_train_values[indices[i : i + self.batch_size]])
   177:                 .float()
   178:                 .to(self.device)
   179:             )
   180: 
   181:             pred = self.model(feature)
   182:             loss = self.loss_fn(pred, label)
   183: 
   184:             self.train_optimizer.zero_grad()
   185:             loss.backward()
   186:             torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
   187:             self.train_optimizer.step()
   188: 
   189:     def test_epoch(self, data_x, data_y):
   190:         x_values = data_x.values
   191:         y_values = np.squeeze(data_y.values)
   192: 
   193:         self.model.eval()
   194: 
   195:         scores = []
   196:         losses = []
   197: 
   198:         indices = np.arange(len(x_values))
   199: 
   200:         for i in range(len(indices))[:: self.batch_size]:
   201:             if len(indices) - i < self.batch_size:
   202:                 break
   203: 
   204:             feature = (
   205:                 torch.from_numpy(x_values[indices[i : i + self.batch_size]])
   206:                 .float()
   207:                 .to(self.device)
   208:             )
   209:             label = (
   210:                 torch.from_numpy(y_values[indices[i : i + self.batch_size]])
   211:                 .float()
   212:                 .to(self.device)
   213:             )
   214: 
   215:             with torch.no_grad():
   216:                 pred = self.model(feature)
   217:                 loss = self.loss_fn(pred, label)
   218:                 losses.append(loss.item())
   219: 
   220:                 score = self.metric_fn(pred, label)
   221:                 scores.append(score.item())
   222: 
   223:         return np.mean(losses), np.mean(scores)
   224: 
   225:     def fit(self, dataset: DatasetH):
   226:         df_train, df_valid, df_test = dataset.prepare(
   227:             ["train", "valid", "test"],
   228:             col_set=["feature", "label"],
   229:             data_key=DataHandlerLP.DK_L,
   230:         )
   231:         if df_train.empty or df_valid.empty:
   232:             raise ValueError(
   233:                 "Empty data from dataset, please check your dataset config."
   234:             )
   235: 
   236:         x_train, y_train = df_train["feature"], df_train["label"]
   237:         x_valid, y_valid = df_valid["feature"], df_valid["label"]
   238: 
   239:         stop_steps = 0
   240:         best_score = -np.inf
   241:         best_epoch = 0
   242:         best_param = None
   243: 
   244:         self.fitted = True
   245: 
   246:         for step in range(self.n_epochs):
   247:             self.train_epoch(x_train, y_train)
   248:             train_loss, train_score = self.test_epoch(x_train, y_train)
   249:             val_loss, val_score = self.test_epoch(x_valid, y_valid)
   250:             print(
   251:                 "Epoch%d: train %.6f, valid %.6f"
   252:                 % (step, train_score, val_score)
   253:             )
   254: 
   255:             if val_score > best_score:
   256:                 best_score = val_score
   257:                 stop_steps = 0
   258:                 best_epoch = step
   259:                 best_param = copy.deepcopy(self.model.state_dict())
   260:             else:
   261:                 stop_steps += 1
   262:                 if stop_steps >= self.early_stop:
   263:                     print("early stop")
   264:                     break
   265: 
   266:         print("best score: %.6lf @ %d" % (best_score, best_epoch))
   267:         self.model.load_state_dict(best_param)
   268: 
   269:         if self.use_gpu:
   270:             torch.cuda.empty_cache()
   271: 
   272:     def predict(self, dataset: DatasetH, segment="test"):
   273:         if not self.fitted:
   274:             raise ValueError("model is not fitted yet!")
   275: 
   276:         x_test = dataset.prepare(
   277:             segment, col_set="feature", data_key=DataHandlerLP.DK_I
   278:         )
   279:         index = x_test.index
   280:         self.model.eval()
   281:         x_values = x_test.values
   282:         sample_num = x_values.shape[0]
   283:         preds = []
   284: 
   285:         for begin in range(sample_num)[:: self.batch_size]:
   286:             if sample_num - begin < self.batch_size:
   287:                 end = sample_num
   288:             else:
   289:                 end = begin + self.batch_size
   290: 
   291:             x_batch = (
   292:                 torch.from_numpy(x_values[begin:end]).float().to(self.device)
   293:             )
   294: 
   295:             with torch.no_grad():
   296:                 pred = self.model(x_batch).detach().cpu().numpy()
   297: 
   298:             preds.append(pred)
   299: 
   300:         return pd.Series(np.concatenate(preds), index=index)
```

### `lgbm` baseline — editable region  [READ-ONLY — reference implementation]

In `qlib/workflow_config.yaml`:

```python
Lines 13–25:
    10:     - "."           # So custom_model.py is importable via module_path
    11: 
    12: task:
    13:   model:
    14:     class: CustomModel
    15:     module_path: custom_model
    16:     kwargs: {}
    17: 
    18:   dataset:
    19:     class: DatasetH
    20:     module_path: qlib.data.dataset
    21:     kwargs:
    22:       handler:
    23:         class: Alpha360
    24:         module_path: qlib.contrib.data.handler
    25:         kwargs:
    26:           start_time: "2008-01-01"
    27:           end_time: "2020-08-01"
    28:           fit_start_time: "2008-01-01"

Lines 31–37:
    28:           fit_start_time: "2008-01-01"
    29:           fit_end_time: "2014-12-31"
    30:           instruments: csi300
    31:           infer_processors: []
    32:           learn_processors:
    33:             - class: DropnaLabel
    34:             - class: CSRankNorm
    35:               kwargs:
    36:                 fields_group: label
    37:           label: ["Ref($close, -2) / Ref($close, -1) - 1"]
    38:       segments:
    39:         train: ["2008-01-01", "2014-12-31"]
    40:         valid: ["2015-01-01", "2016-12-31"]
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
