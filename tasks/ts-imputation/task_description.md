# Multivariate Time Series Imputation under Random Masking

## Research Question
What modular imputation component (mask-aware temporal modeling, cross-channel dependency learning, denoising objective, normalization) best recovers missing entries from temporal *and* cross-variable context, evaluated under a fixed 25% random-mask protocol on heterogeneous multivariate datasets?

## Background
Time series imputation under random masking is a canonical "fill-in-the-blanks" benchmark for temporal models. The Time-Series-Library protocol (Wu et al., ICLR 2023) masks a fixed fraction of observed entries uniformly at random, hands the model both the masked sequence and a binary observation mask, and scores reconstruction error only at masked positions. This isolates the model's ability to infer missing values from surrounding temporal context and from correlated channels.

## Objective
Implement the `Model` class in `Time-Series-Library/models/Custom.py`. Given a partially-masked input window and a binary observation mask, return a fully reconstructed sequence; only positions where `mask == 0` count toward the metric.

## Model Interface
```python
class Model(nn.Module):
    def __init__(self, configs):
        # configs.task_name == "imputation"
        # configs.seq_len, configs.enc_in
        ...

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # x_enc:      [batch, seq_len, enc_in]   — masked entries set to 0
        # x_mark_enc: [batch, seq_len, time_feat]
        # mask:       [batch, seq_len, enc_in]   — 1 = observed, 0 = masked
        # returns:    [batch, seq_len, enc_in]
        ...

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
```

## Datasets and Fixed Protocol
- **ETTh1** — 7 variables, hourly Electricity Transformer Temperature.
- **Weather** — 21 variables, weather observations.
- **ECL** — 321 variables, hourly client electricity consumption.

All settings: `seq_len=96`, `mask_rate=0.25` random binary mask per (timestep, channel). Standardization, splits, and mask sampling are fixed by the Time-Series-Library imputation pipeline.

## Metrics
MSE and MAE on masked entries only — lower is better.

## Reference Implementations (read-only)
Three reference models from `models/`:

- **DLinear** — Zeng et al., AAAI 2023 (arXiv 2205.13504). Trend+seasonal linear projections on the masked sequence (zeros where masked). TS-Lib imputation defaults: `moving_avg=25`, Adam `lr=1e-3`, `train_epochs=10`, `batch_size=16`. Source: https://github.com/cure-lab/LTSF-Linear.
- **TimesNet** — Wu et al., ICLR 2023 (arXiv 2210.02186). FFT-based period discovery + 2D conv. TS-Lib imputation defaults: `e_layers=2`, `d_model=64`, `d_ff=64`, `top_k=3`, `num_kernels=6`. Source: https://github.com/thuml/Time-Series-Library.
- **PatchTST** — Nie et al., ICLR 2023 (arXiv 2211.14730). Channel-independent Transformer over patches. TS-Lib imputation defaults: `e_layers=3`, `n_heads=4`, `d_model=128`, `d_ff=256`, `patch_len=16`, `stride=8`. Source: https://github.com/yuqinie98/PatchTST.
