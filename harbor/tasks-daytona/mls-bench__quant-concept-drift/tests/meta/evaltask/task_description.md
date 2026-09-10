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
