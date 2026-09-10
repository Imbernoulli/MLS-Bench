"""DLinear baseline — rigorous codebase edit ops.

DLinear (AAAI'23): Decomposition-Linear. Decomposes input into trend
and seasonal via moving average, then applies separate linear layers.

Reference: basicts/models/DLinear/arch/dlinear_arch.py
"""

_FILE = "BasicTS/custom_model.py"

_CONTENT = """\
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from basicts.configs import BasicTSModelConfig


@dataclass
class CustomConfig(BasicTSModelConfig):
    input_len: int = field(default=12)
    output_len: int = field(default=12)
    num_features: int = field(default=207)
    moving_avg: int = field(default=25)


class Custom(nn.Module):
    \"\"\"DLinear: Decomposition-Linear baseline.

    Decomposes input into trend (moving average) and seasonal (residual),
    then projects each component independently to the prediction horizon.
    \"\"\"

    def __init__(self, config: CustomConfig):
        super().__init__()
        self.input_len = config.input_len
        self.output_len = config.output_len
        k = config.moving_avg
        self.pad_left = (k - 1) // 2
        self.pad_right = k // 2
        self.avg_pool = nn.AvgPool1d(k, stride=1)
        self.linear_seasonal = nn.Linear(config.input_len, config.output_len)
        self.linear_trend = nn.Linear(config.input_len, config.output_len)

    def _decompose(self, x):
        # x: [B, T, N] -> trend via moving average, seasonal = x - trend
        padded = F.pad(x.transpose(1, 2), (self.pad_left, self.pad_right), mode='replicate')
        trend = self.avg_pool(padded).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend

    def forward(self, inputs, inputs_timestamps):
        # inputs: [B, T, N]
        seasonal, trend = self._decompose(inputs)
        # Per-feature linear: [B, N, T] -> [B, N, T']
        seasonal_out = self.linear_seasonal(seasonal.transpose(1, 2))
        trend_out = self.linear_trend(trend.transpose(1, 2))
        prediction = (seasonal_out + trend_out).transpose(1, 2)  # [B, T', N]
        return prediction
"""

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 1, "end_line": 71, "content": _CONTENT},
]
