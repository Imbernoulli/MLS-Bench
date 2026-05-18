# Spatial-Temporal Traffic Forecasting on Sensor Networks

## Research Question
What modular spatial-temporal forecasting component (architecture or training scheme) generalizes across traffic-sensor networks of different sizes and modalities (speed vs. flow), under a fixed 12-step → 12-step horizon and a common evaluation protocol?

## Background
Spatial-temporal forecasting predicts future values across a network of spatial nodes — for traffic, sensors on highway segments — by jointly modeling temporal patterns at each node and spatial correlations across nodes. The METR-LA / PEMS-BAY benchmarks were introduced together with DCRNN (Li et al., ICLR 2018, "Diffusion Convolutional Recurrent Neural Network", arXiv 1707.01926) and have since become the canonical testbeds for graph- and attention-based spatial-temporal models. Design choices include (a) **spatial modeling**: learnable node embeddings, graph convolutions, spatial attention, learned adjacency; (b) **temporal modeling**: RNNs, temporal convolutions, Transformers; (c) **spatial-temporal fusion**: how the two are combined.

## Objective
Implement a `Custom` `nn.Module` and `CustomConfig` dataclass in `custom_model.py` for the BasicTS framework. The model is trained and evaluated by the fixed BasicTS pipeline on three datasets.

## Model Interface
```python
def forward(self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor) -> torch.Tensor:
    """
    inputs:            [batch_size, input_len=12, num_features]   # num_features = number of spatial nodes
    inputs_timestamps: [batch_size, input_len=12, 2]              # [time-of-day, day-of-week] normalized to [0,1]
    Returns:           [batch_size, output_len=12, num_features]  # next-hour predictions for every node
    """
```
`CustomConfig` extends `basicts.configs.BasicTSModelConfig` with at least `input_len`, `output_len`, `num_features`.

## Datasets and Fixed Protocol
- **METR-LA** — 207 sensors, traffic speed, Los Angeles highway (Li et al., ICLR 2018).
- **PEMS-BAY** — 325 sensors, traffic speed, San Francisco Bay Area (Li et al., ICLR 2018).
- **PEMS04** — 307 sensors, traffic flow, California Caltrans District 4 (commonly used with ASTGCN, AAAI 2019).

All settings use `input_len=12`, `output_len=12` (one hour of 5-min intervals → next hour). Data is Z-score normalized per dataset; metrics are computed after the inverse transform. Missing values (encoded as 0.0) are masked during loss and metric computation.

## Available Modules
You may import components from `basicts.modules`:
- `basicts.modules.mlps` — `MLPLayer`, `ResMLPLayer`
- `basicts.modules.norm` — `RevIN`, `LayerNorm`
- `basicts.modules.embed` — sequence embeddings
- `basicts.modules.transformer` — `Encoder`, `MultiHeadAttention`
- `basicts.modules.activations` — common activations

## Training Hyperparameter Override
The harness uses Adam with `lr=2e-3`, `weight_decay=1e-4`, and `MultiStepLR(milestones=[1, 50, 80], gamma=0.5)` for 100 epochs at `batch_size=64`. If your method needs a different `lr` or `weight_decay`, set them in the `CONFIG_OVERRIDES` dict at the bottom of `custom_model.py`:
```python
CONFIG_OVERRIDES = {'lr': 5e-4, 'weight_decay': 1e-3}
```
Only `lr` and `weight_decay` are forwarded; epochs, batch size, scheduler, and gradient clipping are fixed.

## Metrics
MAE, RMSE, MAPE — all lower is better, computed in original scale after inverse transform with the missing-value mask applied.

## Reference Implementations (read-only)
Six reference models live in `basicts/models/` and serve as context:
- **SOFTS** — Han et al., "SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion", NeurIPS 2024. Inverted architecture (variates as tokens) with a STar Aggregate-Redistribute (STAR) module for O(N) cross-variate fusion instead of self-attention. Source: https://github.com/Secilia-Cxy/SOFTS.
- **DLinear** — Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023 (arXiv 2205.13504). Decomposition into trend (moving-average kernel) + seasonal, each projected by a linear layer. Source: https://github.com/cure-lab/LTSF-Linear.
- **StemGNN** — Cao et al., "Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting", NeurIPS 2020 (arXiv 2103.07719). Graph Fourier transform + DFT for joint spatial-spectral modeling. Source: https://github.com/microsoft/StemGNN.
- **iTransformer** — Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting", ICLR 2024 (arXiv 2310.06625). Treats variates as tokens; attention across variables, FFN within each variate token. Source: https://github.com/thuml/iTransformer.
- **TimesNet** — Wu et al., "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis", ICLR 2023 (arXiv 2210.02186). Reshapes 1D series into 2D tensors at multiple FFT-discovered periods, processed by Inception 2D conv blocks. Source: https://github.com/thuml/Time-Series-Library.
- **TimeMixer** — Wang et al., "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting", ICLR 2024 (arXiv 2405.14616). Multi-scale decomposition with Past-Decomposable-Mixing and Future-Multipredictor-Mixing. Source: https://github.com/kwuking/TimeMixer.
