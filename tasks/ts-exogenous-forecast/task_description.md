# Forecasting with Exogenous Variables (features=MS)

## Research Question
What architectural component best fuses *exogenous* covariates (additional observed channels) with the *endogenous* target series to improve target-channel forecasting, while preserving the fixed look-back window, horizon, and Time-Series-Library evaluation pipeline?

## Background
Many real-world forecasting tasks present a designated target variable accompanied by side-information channels (weather covariates for energy load, exogenous prices, related sensors). The Time-Series-Library `features=MS` mode formalizes this: all variables are fed into the model, but only the last channel (the target) is scored. Recent methods such as TimeXer (Wang et al., NeurIPS 2024) explicitly separate endogenous (patch-level) and exogenous (variate-level) representations and fuse them via cross-attention; iTransformer (Liu et al., ICLR 2024) treats every channel as a token and uses attention across channels; PatchTST (Nie et al., ICLR 2023) takes the channel-independent route and ignores cross-channel structure entirely. The contribution space here is the exogenous-fusion component itself.

## Objective
Implement the `Model` class in `Time-Series-Library/models/Custom.py`. Output shape is `[batch, pred_len, c_out]` where `c_out == enc_in`; the harness slices `outputs[:, :, -1:]` so only the final (target) channel is scored.

## Model Interface
```python
class Model(nn.Module):
    def __init__(self, configs):
        # configs.task_name == "long_term_forecast"
        # configs.seq_len, configs.pred_len, configs.enc_in, configs.c_out
        ...

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc:        [batch, seq_len,           enc_in]   — all variables
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
- **ETTh1** — 7 → 1, hourly Electricity Transformer Temperature (Zhou et al., AAAI 2021).
- **Weather** — 21 → 1, 21 weather observations.
- **ECL** — 321 → 1, hourly electricity consumption of 321 clients.

All settings: `features=MS`, `seq_len=96`, `label_len=48`, `pred_len=96`. Standardization, splits, and the target column are fixed by the Time-Series-Library data loaders.

## Metrics
MSE and MAE on the target variable only (the harness extracts `outputs[:, :, -1:]` before scoring). Lower is better.

## Reference Implementations (read-only)
Four reference models from `models/`:

- **DLinear** — Zeng et al., AAAI 2023 (arXiv 2205.13504). Trend+seasonal linear projections, channel-independent. TS-Lib defaults: `moving_avg=25`, Adam `lr=1e-4`, `train_epochs=10`, `batch_size=32`. Source: https://github.com/cure-lab/LTSF-Linear.
- **PatchTST** — Nie et al., ICLR 2023 (arXiv 2211.14730). Channel-independent Transformer over input patches. TS-Lib defaults: `e_layers=3`, `n_heads=4`, `d_model=128`, `d_ff=256`, `patch_len=16`, `stride=8`. Source: https://github.com/yuqinie98/PatchTST.
- **iTransformer** — Liu et al., ICLR 2024 (arXiv 2310.06625). Attention across variates; FFN within each variate token. TS-Lib defaults: `e_layers=2`, `d_model=512`, `d_ff=512`, `n_heads=8`. Source: https://github.com/thuml/iTransformer.
- **TimeXer** — Wang et al., NeurIPS 2024 (arXiv 2402.19072). Patch-wise self-attention on the endogenous series + variate-wise cross-attention with exogenous variables, bridged by a learnable global token. TS-Lib defaults: `e_layers=1`, `d_model=512`, `d_ff=512`, `patch_len=16`. Source: https://github.com/thuml/TimeXer.
