# Multivariate Time Series Classification on UEA Datasets

## Research Question
Can a single classification component (temporal encoder + channel-interaction + padding-aware pooling) generalize across heterogeneous multivariate time series — spectral chemistry signals, MEG brain recordings, accelerometer handwriting traces — when training and evaluation are held to a fixed protocol?

## Background
The UEA Multivariate Time Series Classification archive (Bagnall et al., "The UEA multivariate time series classification archive, 2018", arXiv 1811.00075) is the standard benchmark for multivariate TSC. Datasets vary widely in sequence length, channel count, number of classes, sampling rate, and noise characteristics, making it a stress test for "general-purpose" temporal encoders. The Time-Series-Library protocol (Wu et al., ICLR 2023) standardizes train/test splits, padding to a common per-dataset length, RAdam optimization, cross-entropy loss, and accuracy reporting.

## Objective
Implement the `Model` class in `Time-Series-Library/models/Custom.py`. Given a padded input window plus a binary padding mask, return class logits.

## Model Interface
```python
class Model(nn.Module):
    def __init__(self, configs):
        # configs.task_name == "classification"
        # configs.seq_len, configs.enc_in, configs.num_class — set dynamically per dataset
        ...

    def classification(self, x_enc, x_mark_enc):
        # x_enc:        [batch, seq_len, enc_in]   — padded window
        # x_mark_enc:   [batch, seq_len]           — 1 for valid timesteps, 0 for padding
        # returns:      [batch, num_class]         — pre-softmax logits
        ...

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
```

## Datasets and Fixed Protocol
- **EthanolConcentration** — spectral data classification (4 classes).
- **FaceDetection** — MEG brain-imaging binary classification.
- **Handwriting** — accelerometer-based character recognition (26 classes).

All from the UEA archive. Train/test splits are dataset-provided. Optimization: RAdam, CrossEntropyLoss, early-stopping `patience=10`, framework defaults for batch size and `train_epochs`.

## Metric
Test accuracy — higher is better.

## Reference Implementations (read-only)
Three reference models from `models/`:

- **DLinear** — Zeng et al., AAAI 2023 (arXiv 2205.13504). For classification, the trend+seasonal linear projections feed a global pooling + linear classifier head. TS-Lib classification defaults: `e_layers=2`, `d_model=128`, RAdam `lr=1e-3`, `batch_size=16`. Source: https://github.com/cure-lab/LTSF-Linear.
- **TimesNet** — Wu et al., ICLR 2023 (arXiv 2210.02186). TS-Lib classification defaults: `e_layers=2`, `d_model=64`, `d_ff=64`, `top_k=3`, `num_kernels=6`, RAdam `lr=1e-3`. Source: https://github.com/thuml/Time-Series-Library.
- **PatchTST** — Nie et al., ICLR 2023 (arXiv 2211.14730). TS-Lib classification defaults: `e_layers=3`, `n_heads=4`, `d_model=128`, `d_ff=256`, `patch_len=16`, `stride=8`. Source: https://github.com/yuqinie98/PatchTST.
