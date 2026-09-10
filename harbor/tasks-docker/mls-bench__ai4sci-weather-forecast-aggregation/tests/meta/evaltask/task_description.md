# Weather Forecast Variable Aggregation

## Research Question
How should a weather forecasting model aggregate information across heterogeneous meteorological variables for optimal prediction?

## Background
Modern weather forecasting models process many meteorological variables simultaneously (temperature, pressure, wind, humidity at various pressure levels). ClimaX (Nguyen, Brandstetter, Kapoor, Gupta, Grover, "ClimaX: A foundation model for weather and climate", ICML 2023; arXiv:2301.10343) tokenizes each variable independently via per-variable patch embeddings, then aggregates them into a unified spatial representation before feeding into a Vision Transformer backbone. The default aggregation uses a learnable query with cross-attention over variable tokens at each spatial location, but this is just one design choice. Better aggregation strategies could capture inter-variable correlations more effectively. Code: https://github.com/microsoft/ClimaX.

## Task
Modify the `VariableAggregator` class in `custom_forecast.py` to implement a novel variable aggregation mechanism. The module receives per-variable patch embeddings and must produce a single aggregated representation per spatial location.

## Interface
```python
class VariableAggregator(nn.Module):
    def __init__(self, embed_dim, num_heads, num_vars):
        """
        Args:
            embed_dim (int): Embedding dimension D (1024).
            num_heads (int): Number of attention heads (16).
            num_vars (int): Number of input variables V (48).
        """
        ...

    def forward(self, x):
        """
        Args:
            x: [B, V, L, D] — per-variable patch embeddings
                B = batch size
                V = number of meteorological variables (48)
                L = number of spatial patches (512 = 16x32)
                D = embedding dimension (1024)

        Returns:
            [B, L, D] — aggregated representation per spatial location
        """
        ...
```

The input contains 48 variables: 3 surface constants (land-sea mask, orography, latitude), 3 surface fields (2 m temperature, 10 m wind u/v), and 42 pressure-level fields (geopotential, u/v wind, temperature, relative/specific humidity at 50–925 hPa). Each variable has been independently tokenized into L=512 patch embeddings of dimension D=1024.

## Available Components
You have access to standard PyTorch modules (`nn.Linear`, `nn.MultiheadAttention`, `nn.LayerNorm`, etc.) and `torch.nn.functional`. The FIXED section imports `torch`, `torch.nn`, and `torch.nn.functional as F`.

## Fixed Pipeline
ClimaX backbone, per-variable patch tokenization, fine-tuning recipe (initialized from pretrained ClimaX weights), data pipeline, ERA5 reanalysis at 5.625° resolution, optimizer/schedule, and the latitude-weighted RMSE metric are all fixed.

## Evaluation
The model is fine-tuned from pretrained ClimaX weights on ERA5 reanalysis data at 5.625-degree resolution and evaluated on three forecasting targets:
- **z500-3day**: Geopotential height at 500 hPa, 3-day lead time.
- **t850-5day**: Temperature at 850 hPa, 5-day lead time.
- **wind10m-7day**: 10 m wind speed, 7-day lead time.

Metric: Latitude-weighted RMSE (lower is better). The metric accounts for the convergence of meridians at the poles by weighting errors by the cosine of latitude.

## Reference Baselines
- **cross_attention**: ClimaX default aggregation. A learnable query token attends to all V variable tokens at each spatial location via multi-head cross-attention, producing one token per location.
- **mean_pooling**: Simple uniform mean across all V variable tokens at each spatial location. No additional learnable parameters; serves as a parameter-free lower bound.
- **learned_weighted_sum**: Learnable per-variable scalar weights normalized via softmax, then used to compute a weighted sum across variable tokens. More expressive than mean pooling but much simpler than cross-attention.
