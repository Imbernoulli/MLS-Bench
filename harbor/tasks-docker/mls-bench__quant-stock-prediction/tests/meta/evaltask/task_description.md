# Quantitative Stock Prediction on Chinese Equity Universes

## Research Question
Can a single, reusable predictive component deliver consistently strong cross-sectional return signals across different Chinese equity universes and time periods, when the input features, label, train/valid/test splits, and downstream backtest are held fixed?

## Background
Quantitative stock prediction in Microsoft `qlib` formulates daily forecasting as a cross-sectional regression: at each trading day, predict the next-day excess return for every stock in the universe, then rank stocks by the predicted score and feed the ranking into a fixed portfolio-construction routine. The challenge is twofold: market data is noisy and non-stationary; and a model must rank well across heterogeneous instruments, not just minimize per-stock loss. A wide variety of methods have been tried — gradient boosted trees on engineered factors (LightGBM), pure sequence models (LSTM, Transformer), and graph-aware extensions — but none dominates universally, motivating principled studies of model-component contributions under a common protocol.

## Objective
Implement a `CustomModel` in `custom_model.py` that exposes the standard qlib model interface (`fit(dataset)` and `predict(dataset, segment="test")`). The class is wired into the qlib `workflow_config.yaml`, which controls the dataset adapter / preprocessor block but keeps the universe, label, and date splits fixed. You may change the dataset class (e.g., to `TSDatasetH`) or processors if your model needs a different input view.

## Fixed Pipeline
- **Features**: Alpha360 (360 features per stock-day = 6 base ratios over 60 days of history). For temporal models, reshape with `x.reshape(N, 6, 60).permute(0, 2, 1) -> [N, 60, 6]`.
- **Label**: `Ref($close, -2) / Ref($close, -1) - 1` (return from T+1 to T+2, predicted at T).
- **Universes / splits**: `csi300`, `csi100`, `csi300_recent` — instruments and date ranges fixed by the workflow YAML.
- **Backtest**: TopkDropout, top 50 / drop 5, executed by the qlib workflow runner.

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

## Evaluation Metrics
Reported per universe:
- **Signal quality**: IC, ICIR, Rank IC, Rank ICIR — higher is better.
- **Portfolio**: annualized return, information ratio — higher is better; max drawdown — closer to zero is better.

All metrics are produced by qlib's standard `SignalAnalysisRecord` and `PortAnaRecord`.

## Reference Implementations (read-only)
Three reference models ship with qlib's `examples/benchmarks/` and are available as read-only context. Defaults are taken from each method's qlib CSI300 example config.

- **LightGBM** — Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", NeurIPS 2017. qlib defaults: `loss=mse`, `learning_rate=0.0421`, `num_leaves=210`, `feature_fraction=0.879`, `bagging_fraction=0.856`, `bagging_freq=5`, `lambda_l1=205.7`, `lambda_l2=580.9`.
- **LSTM** — qlib's RNN baseline with `d_feat=6`, `hidden_size=64`, `num_layers=2`, `dropout=0.0`, Adam `lr=1e-3`, `n_epochs=200`, early-stopping patience 20.
- **Transformer** — qlib defaults `d_feat=6`, `d_model=64`, `nhead=2`, `num_layers=2`, `dropout=0.5`, Adam `lr=1e-4`.

Code: https://github.com/microsoft/qlib.
