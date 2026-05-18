"""TimeMixer baseline — rigorous codebase edit ops.

TimeMixer (ICLR'24): Decomposable multiscale mixing. Down-samples input
at multiple scales, decomposes into seasonal/trend, and mixes with
Past-Decomposable Mixing blocks across scales.

Reference: basicts/models/TimeMixer/arch/timemixer_arch.py
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
    hidden_size: int = field(default=64)
    num_layers: int = field(default=2)
    down_sampling_layers: int = field(default=2)
    down_sampling_window: int = field(default=2)
    dropout: float = field(default=0.1)
    moving_avg: int = field(default=5)


class RevIN(nn.Module):
    def __init__(self, num_features, affine=True):
        super().__init__()
        self.eps = 1e-6
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.weight + self.bias
            return x
        else:
            if self.affine:
                x = (x - self.bias) / (self.weight + self.eps * self.eps)
            return x * self.stdev + self.mean


class MovingAvgDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.avg = nn.AvgPool1d(kernel_size, stride=1)
        self.pad_left = (kernel_size - 1) // 2
        self.pad_right = kernel_size // 2

    def forward(self, x):
        # x: [B, T, C]
        trend = self.avg(F.pad(x.transpose(1, 2),
                                (self.pad_left, self.pad_right),
                                mode='replicate')).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


class MLPMixer(nn.Module):
    '''2-layer MLP for scale mixing on the temporal dimension.'''
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MultiScaleSeasonMixing(nn.Module):
    '''Bottom-up mixing: finer -> coarser scales via learned MLP.'''
    def __init__(self, input_len, down_sampling_layers, down_sampling_window):
        super().__init__()
        self.down_layers = nn.ModuleList([
            MLPMixer(
                input_len // (down_sampling_window ** i),
                input_len // (down_sampling_window ** (i + 1))
            )
            for i in range(down_sampling_layers)
        ])

    def forward(self, seasonal_list):
        # seasonal_list[i]: [B, D, L_i] (permuted)
        out_high = seasonal_list[0]
        out_low = seasonal_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(seasonal_list) - 1):
            out_low_res = self.down_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(seasonal_list) - 1:
                out_low = seasonal_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    '''Top-down mixing: coarser -> finer scales via learned MLP.'''
    def __init__(self, input_len, down_sampling_layers, down_sampling_window):
        super().__init__()
        self.up_layers = nn.ModuleList([
            MLPMixer(
                input_len // (down_sampling_window ** (i + 1)),
                input_len // (down_sampling_window ** i)
            )
            for i in reversed(range(down_sampling_layers))
        ])

    def forward(self, trend_list):
        # trend_list[i]: [B, D, L_i] (permuted)
        trend_rev = trend_list.copy()
        trend_rev.reverse()
        out_low = trend_rev[0]
        out_high = trend_rev[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_rev) - 1):
            out_high_res = self.up_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_rev) - 1:
                out_high = trend_rev[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    '''Decompose each scale, mix seasonal bottom-up and trend top-down.'''
    def __init__(self, input_len, hidden_size, down_sampling_layers,
                 down_sampling_window, moving_avg):
        super().__init__()
        self.decomp = MovingAvgDecomp(moving_avg)
        self.season_mixing = MultiScaleSeasonMixing(
            input_len, down_sampling_layers, down_sampling_window)
        self.trend_mixing = MultiScaleTrendMixing(
            input_len, down_sampling_layers, down_sampling_window)
        self.out_cross_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x_list):
        seasonal_list, trend_list = [], []
        for x in x_list:
            seasonal, trend = self.decomp(x)
            seasonal_list.append(seasonal.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        seasonal_list = self.season_mixing(seasonal_list)
        trend_list = self.trend_mixing(trend_list)

        out_list = []
        for x, seasonal, trend in zip(x_list, seasonal_list, trend_list):
            out = seasonal + trend
            out = x + self.out_cross_layer(out)
            out_list.append(out)
        return out_list


class Custom(nn.Module):
    '''TimeMixer: Decomposable Multiscale Mixing baseline.

    Channel-independent mode: each variate processed separately.
    Multi-scale decomposition + Past-Decomposable Mixing across scales.
    '''

    def __init__(self, config: CustomConfig):
        super().__init__()
        self.num_features = config.num_features
        self.output_len = config.output_len
        self.down_layers = config.down_sampling_layers
        self.down_window = config.down_sampling_window
        D = config.hidden_size

        self.down_pool = nn.AvgPool1d(config.down_sampling_window)

        # Per-scale RevIN
        self.norm_layers = nn.ModuleList([
            RevIN(config.num_features, affine=True)
            for _ in range(self.down_layers + 1)
        ])

        # Embedding (channel-independent: 1 feature -> D)
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.embed = nn.Conv1d(1, D, kernel_size=3, padding=padding,
                               padding_mode="circular", bias=False)

        # PDM blocks (decomposition happens inside each block)
        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(
                config.input_len, D, self.down_layers,
                self.down_window, config.moving_avg)
            for _ in range(config.num_layers)
        ])

        # Per-scale prediction heads
        self.predict_layers = nn.ModuleList([
            nn.Linear(config.input_len // (self.down_window ** i), config.output_len)
            for i in range(self.down_layers + 1)
        ])

        # Channel-independent projection
        self.projection = nn.Linear(D, 1)

    def forward(self, inputs, inputs_timestamps):
        # inputs: [B, T, N]
        B, T, N = inputs.shape

        # Multi-scale inputs
        x_list = [inputs]
        sample = inputs.permute(0, 2, 1)  # [B, N, T]
        for _ in range(self.down_layers):
            sample = self.down_pool(sample)
            x_list.append(sample.permute(0, 2, 1))

        # Per-scale normalization + channel independence
        for i in range(len(x_list)):
            x_list[i] = self.norm_layers[i](x_list[i], "norm")
            _, Li, _ = x_list[i].shape
            x_list[i] = x_list[i].transpose(1, 2).reshape(-1, Li, 1)  # [B*N, Li, 1]

        # Embedding
        h_list = []
        for x in x_list:
            h = self.embed(x.transpose(1, 2)).transpose(1, 2)  # [B*N, Li, D]
            h_list.append(h)

        # Past Decomposable Mixing (decomposition inside blocks)
        for block in self.pdm_blocks:
            h_list = block(h_list)

        # Per-scale prediction and sum
        pred_list = []
        for i, h in enumerate(h_list):
            # h: [B*N, Li, D] -> predict -> [B*N, T', D] -> project -> [B*N, T', 1]
            p = self.predict_layers[i](h.permute(0, 2, 1)).permute(0, 2, 1)
            p = self.projection(p)  # [B*N, T', 1]
            p = p.reshape(B, N, self.output_len).permute(0, 2, 1)  # [B, T', N]
            pred_list.append(p)

        prediction = sum(pred_list)
        prediction = self.norm_layers[0](prediction, "denorm")
        return prediction
"""

_CONFIG_OVERRIDES = """\
# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: lr, weight_decay.
CONFIG_OVERRIDES = {'lr': 0.001}
"""

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 73, "end_line": 75, "content": _CONFIG_OVERRIDES},
    {"op": "replace", "file": _FILE, "start_line": 1, "end_line": 71, "content": _CONTENT},
]
