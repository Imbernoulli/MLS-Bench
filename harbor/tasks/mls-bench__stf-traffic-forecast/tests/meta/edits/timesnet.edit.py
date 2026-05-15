"""TimesNet baseline — rigorous codebase edit ops.

TimesNet (ICLR'23): Temporal 2D-Variation Modeling. Detects dominant
periods via FFT, reshapes 1D time series into 2D tensors, and applies
2D convolution (Inception blocks) to capture intra/inter-period patterns.

Reference: basicts/models/TimesNet/arch/timesnet_arch.py
"""

_FILE = "BasicTS/custom_model.py"

_CONTENT = """\
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from dataclasses import dataclass, field
from basicts.configs import BasicTSModelConfig


@dataclass
class CustomConfig(BasicTSModelConfig):
    input_len: int = field(default=12)
    output_len: int = field(default=12)
    num_features: int = field(default=207)
    hidden_size: int = field(default=64)
    num_layers: int = field(default=2)
    num_kernels: int = field(default=3)
    top_k: int = field(default=3)
    dropout: float = field(default=0.1)


class RevIN(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, mode):
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            return (x - self.mean) / self.stdev
        else:
            return x * self.stdev + self.mean


class InceptionBlock(nn.Module):
    \"\"\"Multi-scale 2D convolution with different kernel sizes.\"\"\"
    def __init__(self, in_channels, out_channels, num_kernels=3):
        super().__init__()
        self.num_kernels = num_kernels
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
            for i in range(num_kernels)
        ])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return torch.stack([conv(x) for conv in self.convs], dim=-1).mean(-1)


class TimesBlock(nn.Module):
    def __init__(self, input_len, output_len, hidden_size, num_kernels, top_k):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.top_k = top_k
        intermediate = hidden_size * 4
        self.conv = nn.Sequential(
            InceptionBlock(hidden_size, intermediate, num_kernels),
            nn.GELU(),
            InceptionBlock(intermediate, hidden_size, num_kernels),
        )

    def forward(self, x):
        B, T, D = x.size()
        # FFT to find dominant periods
        xf = torch.fft.rfft(x, dim=1)
        freq_amp = xf.abs().mean(dim=(0, -1))
        freq_amp[0] = 0  # ignore DC
        _, top_idx = torch.topk(freq_amp, self.top_k)
        periods = T // top_idx.detach().cpu().numpy()
        period_weight = xf.abs().mean(dim=-1)[:, top_idx]

        # Process each period
        results = []
        for p in periods:
            if T % p != 0:
                pad_len = ((T // p) + 1) * p - T
                out = F.pad(x, (0, 0, 0, pad_len))
            else:
                pad_len = 0
                out = x
            out = out.reshape(B, -1, p, D).permute(0, 3, 1, 2)  # [B, D, rows, p]
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, D)[:, :T, :]
            results.append(out)

        results = torch.stack(results, dim=-1)
        weights = F.softmax(period_weight, dim=1).unsqueeze(1).unsqueeze(1).expand_as(results)
        return (results * weights).sum(-1) + x


class Custom(nn.Module):
    \"\"\"TimesNet: Temporal 2D-Variation Modeling baseline.

    Transforms 1D time series to 2D based on detected periodicity,
    then applies 2D Inception convolution to capture temporal patterns.
    \"\"\"

    def __init__(self, config: CustomConfig):
        super().__init__()
        self.output_len = config.output_len
        self.revin = RevIN()

        # Embedding: feature -> hidden
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.value_embed = nn.Conv1d(
            config.num_features, config.hidden_size,
            kernel_size=3, padding=padding, padding_mode="circular", bias=False)

        # Temporal alignment for forecasting
        total_len = config.input_len + config.output_len
        self.predict_linear = nn.Linear(config.input_len, total_len)

        # TimesNet blocks
        self.blocks = nn.ModuleList([
            TimesBlock(config.input_len, config.output_len,
                       config.hidden_size, config.num_kernels, config.top_k)
            for _ in range(config.num_layers)
        ])
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        # Output projection
        self.projection = nn.Linear(config.hidden_size, config.num_features)

    def forward(self, inputs, inputs_timestamps):
        x = self.revin(inputs, "norm")

        # Embed: [B, T, N] -> [B, T, D]
        h = self.value_embed(x.transpose(1, 2)).transpose(1, 2)

        # Extend to input_len + output_len
        h = self.predict_linear(h.transpose(1, 2)).transpose(1, 2)

        for block in self.blocks:
            h = self.layer_norm(block(h))

        pred = self.projection(h[:, -self.output_len:, :])
        pred = self.revin(pred, "denorm")
        return pred
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
