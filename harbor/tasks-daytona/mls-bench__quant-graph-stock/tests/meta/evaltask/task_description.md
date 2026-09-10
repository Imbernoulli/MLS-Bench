# Graph-Based Multi-Stock Prediction on Chinese Equity Universes

## Research Question
Can a relation-aware predictor exploit cross-stock structure (sector / concept membership, learned relations, attention across instruments) to deliver consistently better next-day return rankings than instrument-independent models, while keeping the data, labels, splits, and backtest fixed?

## Background
Stocks are not independent: prices co-move within sectors, react jointly to macro shocks, and share information through institutional flows and news. A line of work models this with graph neural networks or concept-aware aggregation over a stock-relation graph (e.g., GATs, HIST, RSR). The task here is to design such a relation-aware component on the standard `qlib` benchmarking pipeline with Alpha360 features, where the agent has access to a stock-concept membership graph in addition to per-stock features.

## Objective
Implement a `CustomModel` in `custom_model.py` that exposes the qlib model interface (`fit(dataset)` and `predict(dataset, segment="test")`). The class is wired into `workflow_config.yaml`, where the dataset adapter / preprocessor block is editable so the model can pull in graph-structured inputs (e.g., concept membership matrices) — but instruments, date ranges, train/valid/test splits, label, and the backtest configuration are fixed.

## Fixed Pipeline
- **Features**: Alpha360 (360 features per stock-day, reshape to `[N, 60, 6]` for sequence models).
- **Auxiliary input**: stock-concept membership graph used by HIST and similar baselines, exposed through the dataset handler.
- **Label**: `Ref($close, -2) / Ref($close, -1) - 1`.
- **Universes / splits**: `csi300`, `csi100`, `csi300_recent` — fixed.
- **Backtest**: TopkDropout, top 50 / drop 5.

## Model Interface
```python
class CustomModel(qlib.model.base.Model):
    def fit(self, dataset): ...
    def predict(self, dataset, segment="test") -> pd.Series: ...
```
`predict` returns a `pd.Series` indexed by `(datetime, instrument)` matching the requested segment's index.

## Evaluation Metrics
Per universe:
- Signal: IC, ICIR, Rank IC, Rank ICIR (higher is better).
- Portfolio: annualized return, information ratio (higher is better); max drawdown (closer to zero is better).
Computed by qlib's `SignalAnalysisRecord` and `PortAnaRecord`.

## Reference Implementations (read-only)
Three reference models ship with qlib and are available as read-only context.

- **HIST** — Xu et al., "HIST: A Graph-based Framework for Stock Trend Forecasting via Mining Concept-Oriented Shared Information", arXiv 2110.13716 (2021). Uses a predefined-concept module and a hidden-concept module. qlib defaults: `d_feat=6`, `hidden_size=128`, `num_layers=2`, `dropout=0.7`, `n_epochs=200`, Adam `lr=2e-4`, `K=3` (top-k stock-to-concept assignments). Code: https://github.com/Wentao-Xu/HIST.
- **GATs** — Veličković et al.'s graph attention networks (ICLR 2018, arXiv 1710.10903) applied to the stock-relation graph. qlib defaults: `d_feat=6`, `hidden_size=64`, `num_layers=2`, `dropout=0.7`, `n_epochs=200`, Adam `lr=1e-4`.
- **LightGBM** — Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", NeurIPS 2017. qlib defaults: `loss=mse`, `learning_rate=0.0421`, `num_leaves=210`, `feature_fraction=0.879`, `bagging_fraction=0.856`, `bagging_freq=5`, `lambda_l1=205.7`, `lambda_l2=580.9`. Included as the standard non-graph reference.

Code: https://github.com/microsoft/qlib.
