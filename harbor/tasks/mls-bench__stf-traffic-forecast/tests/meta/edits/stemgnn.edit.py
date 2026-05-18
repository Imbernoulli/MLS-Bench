"""StemGNN baseline — rigorous codebase edit ops.

StemGNN (NeurIPS'20): Spectral Temporal Graph Neural Network.
Learns latent graph via attention, then applies spectral graph
convolution + spectral temporal processing via FFT.

Reference: basicts/models/StemGNN/arch/stemgnn_arch.py
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
    hidden_size: int = field(default=5)
    num_blocks: int = field(default=2)
    dropout: float = field(default=0.5)


class GLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.left = nn.Linear(in_dim, out_dim)
        self.right = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.left(x) * torch.sigmoid(self.right(x))


class StockBlock(nn.Module):
    def __init__(self, input_len, num_features, hidden_size, layer_idx):
        super().__init__()
        self.input_len = input_len
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.output_hidden_size = 4 * hidden_size

        self.weight = nn.Parameter(
            torch.Tensor(1, 4, 1, input_len * hidden_size, hidden_size * input_len))
        nn.init.xavier_normal_(self.weight)
        self.forecast = nn.Linear(input_len * hidden_size, input_len * hidden_size)
        self.forecast_result = nn.Linear(input_len * hidden_size, input_len)
        if layer_idx == 0:
            self.backcast = nn.Linear(input_len * hidden_size, input_len)
        self.backcast_short_cut = nn.Linear(input_len, input_len)

        self.GLUs = nn.ModuleList()
        for i in range(3):
            in_d = input_len * 4 if i == 0 else input_len * self.output_hidden_size
            self.GLUs.append(GLU(in_d, input_len * self.output_hidden_size))
            self.GLUs.append(GLU(in_d, input_len * self.output_hidden_size))

    def spe_seq_cell(self, inputs):
        B, _, _, N, L = inputs.size()
        inputs = inputs.view(B, -1, N, L)
        ffted = torch.fft.fft(inputs, dim=-1)
        real = ffted.real.permute(0, 2, 1, 3).contiguous().reshape(B, N, -1)
        imag = ffted.imag.permute(0, 2, 1, 3).contiguous().reshape(B, N, -1)
        for i in range(3):
            real = self.GLUs[i * 2](real)
            imag = self.GLUs[i * 2 + 1](imag)
        real = real.reshape(B, N, 4, -1).permute(0, 2, 1, 3).contiguous()
        imag = imag.reshape(B, N, 4, -1).permute(0, 2, 1, 3).contiguous()
        return torch.fft.ifft(torch.complex(real, imag), dim=-1).real

    def forward(self, x, graph):
        graph = graph.unsqueeze(1)
        x = x.unsqueeze(1)
        gfted = torch.matmul(graph, x)
        gconv_input = self.spe_seq_cell(gfted).unsqueeze(2)
        igfted = torch.matmul(gconv_input, self.weight).sum(dim=1)
        forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1))
        forecast = self.forecast_result(forecast_source)
        if self.layer_idx == 0:
            backcast_short = self.backcast_short_cut(x).squeeze(1)
            backcast_source = torch.sigmoid(self.backcast(igfted) - backcast_short)
        else:
            backcast_source = None
        return forecast, backcast_source


class Custom(nn.Module):
    \"\"\"StemGNN: Spectral Temporal Graph Neural Network baseline.

    Learns a latent graph via self-attention, applies Chebyshev graph
    convolution, and processes temporal patterns in the spectral domain.
    \"\"\"

    def __init__(self, config: CustomConfig):
        super().__init__()
        N = config.num_features
        L = config.input_len
        self.num_blocks = config.num_blocks

        # Latent graph via self-attention
        self.weight_key = nn.Parameter(torch.zeros(N, 1))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(N, 1))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        self.GRU = nn.GRU(L, N)

        # Backbone
        self.stock_block = nn.ModuleList([
            StockBlock(L, N, config.hidden_size, i)
            for i in range(config.num_blocks)
        ])

        # Output
        self.fc = nn.Sequential(
            nn.Linear(L, L), nn.LeakyReLU(),
            nn.Linear(L, config.output_len))
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(config.dropout)

    def _latent_graph(self, x):
        # x: [B, T, N]
        h, _ = self.GRU(x.permute(2, 0, 1))  # [N, B, N]
        h = h.permute(1, 0, 2).contiguous()  # [B, N, N]
        h = h.permute(0, 2, 1).contiguous()  # [B, N, N] transposed for attention
        key = torch.matmul(h, self.weight_key)    # [B, N, 1]
        query = torch.matmul(h, self.weight_query) # [B, N, 1]
        N = h.size(1)
        attn = key.repeat(1, 1, N).view(-1, N * N, 1) + query.repeat(1, N, 1)
        attn = self.leakyrelu(attn.squeeze(2).view(-1, N, N))
        attn = self.dropout(F.softmax(attn, dim=2))
        attn = torch.mean(attn, dim=0)
        attn = 0.5 * (attn + attn.T)
        degree = torch.sum(attn, dim=1)
        D_inv_sqrt = torch.diag(1.0 / (torch.sqrt(degree) + 1e-7))
        L = D_inv_sqrt @ (torch.diag(degree) - attn) @ D_inv_sqrt
        # Chebyshev polynomials (order 3)
        L = L.unsqueeze(0)
        T0 = torch.zeros_like(L)
        T1 = L
        T2 = 2 * torch.matmul(L, T1) - T0
        T3 = 2 * torch.matmul(L, T2) - T1
        return torch.cat([T0, T1, T2, T3], dim=0)

    def forward(self, inputs, inputs_timestamps):
        graph = self._latent_graph(inputs)
        x = inputs.unsqueeze(1).transpose(-1, -2)  # [B, 1, N, T]
        results = []
        for i in range(self.num_blocks):
            pred, x = self.stock_block[i](x, graph)
            results.append(pred)
        prediction = sum(results)
        prediction = self.fc(prediction).transpose(1, 2)  # [B, T', N]
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
