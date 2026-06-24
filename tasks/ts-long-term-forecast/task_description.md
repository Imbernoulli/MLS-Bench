# Multivariate Long-Term Time Series Forecasting

## Research Question
What forecasting component (sequence modeling, decomposition, cross-variable attention, normalization) generalizes across heterogeneous multivariate datasets at a fixed 96-step look-back / 96-step horizon, under the Time-Series-Library training and evaluation pipeline?

## Background
Long-term multivariate forecasting predicts the next horizon for all channels of a multivariate series given a fixed look-back window. Recent work has explored very different inductive biases: pure linear projections after trend/seasonal decomposition (DLinear); channel-independent Transformer over patches (PatchTST); inverted attention treating variates as tokens (iTransformer); MLP-based multi-scale decomposition (TimeMixer); and explicit endogenous/exogenous separation (TimeXer). The Time-Series-Library protocol (Wu et al., ICLR 2023) standardizes splits, normalization, and metric computation so that architectural contributions can be compared head-to-head.

## Objective
Implement the `Model` class in `Time-Series-Library/models/Custom.py`. Output shape is `[batch, pred_len, c_out]` covering all target channels.

## Model Interface
```python
class Model(nn.Module):
    def __init__(self, configs):
        # configs.task_name == "long_term_forecast"
        # configs.seq_len, configs.pred_len, configs.enc_in, configs.c_out
        ...

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc:        [batch, seq_len,           enc_in]
        # x_mark_enc:   [batch, seq_len,           time_feat]
        # x_dec:        [batch, label_len+pred_len, dec_in]
        # x_mark_dec:   [batch, label_len+pred_len, time_feat]
        # returns:      [batch, pred_len, c_out]
        ...

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return out[:, -self.pred_len:, :]
```

## Datasets and Fixed Protocol
- **ETTh1** — 7 variables, hourly Electricity Transformer Temperature (Zhou et al., AAAI 2021).
- **Weather** — 21 variables, weather observations.
- **ECL** — 321 variables, hourly client electricity consumption.

All settings: `features=M` (multivariate input → multivariate output), `seq_len=96`, `label_len=48`, `pred_len=96`. Standardization, splits, and evaluation are fixed by the Time-Series-Library data pipeline.

## Metrics
MSE and MAE on all channels — lower is better.

## Reference Implementations (read-only)
Five reference models from `models/`:

- **DLinear** — Zeng et al., AAAI 2023 (arXiv 2205.13504). Trend+seasonal linear projections, channel-independent. TS-Lib defaults: `moving_avg=25`, Adam `lr=1e-4`, `train_epochs=10`, `batch_size=32`. Source: https://github.com/cure-lab/LTSF-Linear.
- **PatchTST** — Nie et al., ICLR 2023 (arXiv 2211.14730). Channel-independent Transformer over input patches. TS-Lib defaults: `e_layers=3`, `n_heads=4`, `d_model=128`, `d_ff=256`, `patch_len=16`, `stride=8`. Source: https://github.com/yuqinie98/PatchTST.
- **iTransformer** — Liu et al., ICLR 2024 (arXiv 2310.06625). Attention across variates; FFN within each variate token. TS-Lib defaults: `e_layers=2`, `d_model=512`, `d_ff=512`, `n_heads=8`. Source: https://github.com/thuml/iTransformer.
- **TimeMixer** — Wang et al., ICLR 2024 (arXiv 2405.14616). MLP-based decomposable multiscale mixing (Past-Decomposable-Mixing + Future-Multipredictor-Mixing). TS-Lib defaults: `e_layers=2`, `d_model=16`, `d_ff=32`, `down_sampling_layers=3`, `down_sampling_window=2`, `down_sampling_method="avg"`. Source: https://github.com/kwuking/TimeMixer.
- **TimeXer** — Wang et al., NeurIPS 2024 (arXiv 2402.19072). Patch-wise self-attention on the endogenous series + variate-wise cross-attention with exogenous variables. TS-Lib defaults: `e_layers=1`, `d_model=512`, `d_ff=512`, `patch_len=16`. Source: https://github.com/thuml/TimeXer.
