# Unsupervised Time Series Anomaly Detection via Reconstruction

## Research Question
What modular reconstruction-based component (sequence representation, training objective, temporal dependency model, normalization, or robust training scheme) yields consistently better unsupervised anomaly detection across heterogeneous multivariate time series — server metrics, spacecraft telemetry, and satellite telemetry?

## Background
Reconstruction-based anomaly detection trains a model to reproduce normal input windows and flags points where the reconstruction error exceeds a threshold. The Time-Series-Library protocol (Wu et al., ICLR 2023) standardizes this setup: model outputs an MSE per timestep, the per-dataset `anomaly_ratio` percentile of train/test scores defines the threshold, and predictions are converted to point-wise binary labels for F1 / precision / recall scoring against ground-truth anomaly intervals.

## Objective
Implement the `Model` class in `Time-Series-Library/models/Custom.py`. Given a window `x_enc` of shape `[batch, seq_len, enc_in]`, return a reconstruction of the same shape. The framework computes the score, threshold, and metrics.

## Model Interface
```python
class Model(nn.Module):
    def __init__(self, configs):
        # configs.task_name == "anomaly_detection"
        # configs.seq_len, configs.enc_in, configs.c_out  (c_out == enc_in for reconstruction)
        ...

    def anomaly_detection(self, x_enc):
        # x_enc: [batch, seq_len, enc_in]  -> [batch, seq_len, c_out]
        ...

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
```
`x_mark_enc`, `x_dec`, `x_mark_dec` are unused for anomaly detection.

## Datasets and Fixed Protocol
- **PSM** — Pooled Server Metrics from eBay; 25 channels of IT-system signals. Released with Abdulaal et al., "Practical Approach to Asynchronous Multivariate Time Series Anomaly Detection and Localization", KDD 2021. Source: https://github.com/eBay/RANSynCoders.
- **MSL** — Mars Science Laboratory rover telemetry; 55 channels. Released with Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding", KDD 2018 (arXiv 1802.04431). Source: https://github.com/khundman/telemanom.
- **SMAP** — Soil Moisture Active Passive satellite telemetry; 25 channels. Same release as MSL.

All runs use `seq_len=100`, `anomaly_ratio=1` (percent of points predicted anomalous; the pipeline picks the threshold so that exactly this percentile is exceeded). Data is Z-score normalized per dataset.

## Metrics
F1 / precision / recall computed via the standard point-adjusted protocol used by Time-Series-Library — higher is better. F1 is the primary metric.

## Reference Implementations (read-only)
Three reference models from `models/` are available as context. Each uses the Time-Series-Library defaults for anomaly detection:

- **DLinear** — Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023 (arXiv 2205.13504). Trend-seasonal decomposition with two linear heads. TS-Lib defaults: `e_layers=3`, `d_model=128`, `d_ff=128`, Adam `lr=1e-4`, `train_epochs=10`, `batch_size=128`. Source: https://github.com/cure-lab/LTSF-Linear.
- **TimesNet** — Wu et al., "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis", ICLR 2023 (arXiv 2210.02186). FFT-based period discovery, reshape to 2D, Inception 2D conv. TS-Lib defaults: `e_layers=2`, `d_model=64`, `d_ff=64`, `top_k=3`, `num_kernels=6`. Source: https://github.com/thuml/Time-Series-Library.
- **PatchTST** — Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers", ICLR 2023 (arXiv 2211.14730). Channel-independent Transformer over patches of the input. TS-Lib defaults: `e_layers=3`, `n_heads=4`, `d_model=128`, `d_ff=256`, `patch_len=16`, `stride=8`. Source: https://github.com/yuqinie98/PatchTST.
