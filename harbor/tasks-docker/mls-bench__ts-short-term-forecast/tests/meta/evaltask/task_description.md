# Univariate Short-Term Forecasting on the M4 Competition Dataset

## Research Question
Can one univariate forecasting component (seasonal decomposition, scale normalization, horizon-aware decoding, multi-scale temporal mixing) deliver consistently low SMAPE across the very different seasonal regimes — monthly, quarterly, yearly — of the M4 competition, under a fixed training and evaluation protocol?

## Background
The M4 competition (Makridakis, Spiliotis & Assimakopoulos, "The M4 Competition: 100,000 time series and 61 forecasting methods", *International Journal of Forecasting*, 2018/2020) collected 100,000 short, real-world series across 6 seasonal patterns (Yearly, Quarterly, Monthly, Weekly, Daily, Hourly). It is the standard benchmark for *short, univariate, many-series* forecasting and is dominated by combinations of statistical and ML methods. The Time-Series-Library protocol (Wu et al., ICLR 2023) wraps M4 with per-pattern fixed look-back / horizon settings, SMAPE training loss, and SMAPE / MAPE / OWA scoring.

## Objective
Implement the `Model` class in `Time-Series-Library/models/Custom.py`. Output shape is `[batch, pred_len, c_out]`; for the M4 univariate setting `enc_in == c_out == 1`.

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
