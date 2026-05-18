"""Custom Weather Forecast Variable Aggregation Script
Based on ClimaX (Nguyen et al., 2023), evaluated on ERA5 at 5.625 deg.

The EDITABLE section contains the variable aggregation module that combines
per-variable patch embeddings into a unified spatial representation.
Everything else (ViT backbone, data loading, training loop) is FIXED.
"""

import math
import os
import time
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

# ============================================================================
# FIXED — Data Loading (ClimaX-style ERA5 npy shards)
# ============================================================================

DEFAULT_VARS = [
    "land_sea_mask", "orography", "lattitude",
    "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
    "geopotential_50", "geopotential_250", "geopotential_500",
    "geopotential_600", "geopotential_700", "geopotential_850", "geopotential_925",
    "u_component_of_wind_50", "u_component_of_wind_250", "u_component_of_wind_500",
    "u_component_of_wind_600", "u_component_of_wind_700", "u_component_of_wind_850",
    "u_component_of_wind_925",
    "v_component_of_wind_50", "v_component_of_wind_250", "v_component_of_wind_500",
    "v_component_of_wind_600", "v_component_of_wind_700", "v_component_of_wind_850",
    "v_component_of_wind_925",
    "temperature_50", "temperature_250", "temperature_500",
    "temperature_600", "temperature_700", "temperature_850", "temperature_925",
    "relative_humidity_50", "relative_humidity_250", "relative_humidity_500",
    "relative_humidity_600", "relative_humidity_700", "relative_humidity_850",
    "relative_humidity_925",
    "specific_humidity_50", "specific_humidity_250", "specific_humidity_500",
    "specific_humidity_600", "specific_humidity_700", "specific_humidity_850",
    "specific_humidity_925",
]

import random


class NpyShardDataset(IterableDataset):
    """Read npz shards of ERA5 data, yield individual time-step pairs."""

    def __init__(self, file_list, variables, out_variables, predict_range,
                 hrs_each_step=1, shuffle=False):
        super().__init__()
        self.file_list = [f for f in file_list if "climatology" not in f]
        self.variables = variables
        self.out_variables = out_variables if out_variables else variables
        self.predict_range = predict_range
        self.hrs_each_step = hrs_each_step
        self.shuffle = shuffle

    def __iter__(self):
        files = list(self.file_list)
        if self.shuffle:
            random.shuffle(files)
        for path in files:
            data = np.load(path)
            x = np.concatenate([data[k].astype(np.float32) for k in self.variables], axis=1)
            x = torch.from_numpy(x)
            y = np.concatenate([data[k].astype(np.float32) for k in self.out_variables], axis=1)
            y = torch.from_numpy(y)

            inputs = x[: -self.predict_range]
            predict_ranges = torch.ones(inputs.shape[0], dtype=torch.long) * self.predict_range
            lead_times = (self.hrs_each_step * predict_ranges / 100.0).to(inputs.dtype)
            output_ids = torch.arange(inputs.shape[0]) + predict_ranges
            outputs = y[output_ids]

            for i in range(inputs.shape[0]):
                yield inputs[i], outputs[i], lead_times[i]


class ShuffleBuffer(IterableDataset):
    """Buffer-based shuffling for iterable datasets."""

    def __init__(self, dataset, buffer_size=5000):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        buf = []
        for x in self.dataset:
            if len(buf) == self.buffer_size:
                idx = random.randint(0, self.buffer_size - 1)
                yield buf[idx]
                buf[idx] = x
            else:
                buf.append(x)
        random.shuffle(buf)
        while buf:
            yield buf.pop()


def collate_fn(batch):
    inp = torch.stack([b[0] for b in batch])
    out = torch.stack([b[1] for b in batch])
    lead = torch.stack([b[2] for b in batch])
    return inp, out, lead


class Normalize:
    """Channel-wise normalization transform."""

    def __init__(self, mean, std):
        self.mean = torch.from_numpy(np.array(mean, dtype=np.float32))
        self.std = torch.from_numpy(np.array(std, dtype=np.float32))

    def __call__(self, x):
        # x: [C, H, W]
        m = self.mean.to(x.device).view(-1, 1, 1)
        s = self.std.to(x.device).view(-1, 1, 1)
        return (x - m) / s

    def inverse(self, x):
        m = self.mean.to(x.device).view(-1, 1, 1)
        s = self.std.to(x.device).view(-1, 1, 1)
        return x * s + m


def build_normalize(data_dir, variables):
    mean_dict = dict(np.load(os.path.join(data_dir, "normalize_mean.npz")))
    std_dict = dict(np.load(os.path.join(data_dir, "normalize_std.npz")))
    mean = np.concatenate([mean_dict.get(v, np.array([0.0])) for v in variables])
    std = np.concatenate([std_dict[v] for v in variables])
    return Normalize(mean, std)


# ============================================================================
# FIXED — Positional Embeddings
# ============================================================================

def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w):
    grid_h_arr = np.arange(grid_h, dtype=np.float64)
    grid_w_arr = np.arange(grid_w, dtype=np.float64)
    grid = np.meshgrid(grid_w_arr, grid_h_arr)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_h, grid_w])
    emb_h = _get_1d_sincos(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, positions):
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = np.array(positions, dtype=np.float64).reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# ============================================================================
# FIXED — Parallel Patch Embedding (from ClimaX)
# ============================================================================

class ParallelVarPatchEmbed(nn.Module):
    """Per-variable patch embedding using grouped convolutions."""

    def __init__(self, max_vars, img_size, patch_size, embed_dim):
        super().__init__()
        self.max_vars = max_vars
        self.img_size = img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.grid_size = (img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        weights = torch.stack([torch.empty(embed_dim, 1, *self.patch_size) for _ in range(max_vars)])
        self.proj_weights = nn.Parameter(weights)
        biases = torch.stack([torch.empty(embed_dim) for _ in range(max_vars)])
        self.proj_biases = nn.Parameter(biases)
        self.reset_parameters()

    def reset_parameters(self):
        for idx in range(self.max_vars):
            nn.init.kaiming_uniform_(self.proj_weights[idx], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.proj_weights[idx])
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.proj_biases[idx], -bound, bound)

    def forward(self, x, var_ids=None):
        B, C, H, W = x.shape
        if var_ids is None:
            var_ids = list(range(self.max_vars))
        weights = self.proj_weights[var_ids].flatten(0, 1)
        biases = self.proj_biases[var_ids].flatten(0, 1)
        groups = len(var_ids)
        proj = F.conv2d(x, weights, biases, groups=groups, stride=self.patch_size)
        proj = proj.reshape(B, groups, -1, *proj.shape[-2:])
        proj = proj.flatten(3).transpose(2, 3)  # B, V, L, D
        return proj


# ============================================================================
# FIXED — ViT Backbone Components (from timm)
# ============================================================================

class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DropPath(nn.Module):
    """Stochastic depth."""

    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


# ============================================================================
# EDITABLE SECTION — Variable Aggregation Module (lines 310 to 351)
# ============================================================================
# This module takes per-variable patch embeddings and aggregates them into a
# single representation per spatial location. The input is x: [B, V, L, D]
# where B=batch, V=num_variables, L=num_patches, D=embed_dim. The output must
# be [B, L, D].
#
# You may define any helper classes/functions within this section. The
# VariableAggregator class MUST implement:
#   __init__(self, embed_dim, num_heads, num_vars)
#   forward(self, x)  where x: [B, V, L, D] -> returns [B, L, D]

class VariableAggregator(nn.Module):
    """Aggregates per-variable patch embeddings into a unified representation.

    Default: learnable query with single-layer cross-attention (ClimaX default).

    Args:
        embed_dim (int): Embedding dimension D.
        num_heads (int): Number of attention heads for cross-attention.
        num_vars (int): Number of input variables V.
    """

    def __init__(self, embed_dim, num_heads, num_vars):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_vars = num_vars
        # Learnable query token for cross-attention aggregation
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        """Aggregate variable embeddings.

        Args:
            x: [B, V, L, D] — per-variable patch embeddings.

        Returns:
            [B, L, D] — aggregated representation.
        """
        b, v, l, d = x.shape
        # Reshape to treat each spatial location independently
        x = x.permute(0, 2, 1, 3)   # B, L, V, D
        x = x.reshape(b * l, v, d)  # B*L, V, D

        # Cross-attention: query attends to all variable tokens
        query = self.var_query.expand(b * l, -1, -1)  # B*L, 1, D
        out, _ = self.var_agg(query, x, x)             # B*L, 1, D
        out = out.squeeze(1)                            # B*L, D

        out = out.reshape(b, l, d)  # B, L, D
        return out


# ============================================================================
# FIXED — ClimaX Model (uses VariableAggregator from editable section)
# ============================================================================

class ClimaXModel(nn.Module):
    """ClimaX weather forecasting model with pluggable variable aggregation."""

    def __init__(self, default_vars, img_size=(32, 64), patch_size=2,
                 embed_dim=1024, depth=8, decoder_depth=2, num_heads=16,
                 mlp_ratio=4.0, drop_path=0.1, drop_rate=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.default_vars = default_vars
        self.embed_dim = embed_dim
        num_vars = len(default_vars)

        # Per-variable patch embedding
        self.token_embeds = ParallelVarPatchEmbed(num_vars, img_size, patch_size, embed_dim)
        self.num_patches = self.token_embeds.num_patches

        # Variable embedding
        self.var_embed = nn.Parameter(torch.zeros(1, num_vars, embed_dim), requires_grad=True)
        self.var_map = {var: i for i, var in enumerate(default_vars)}

        # Variable aggregation (EDITABLE component)
        self.var_aggregator = VariableAggregator(embed_dim, num_heads, num_vars)

        # Positional embedding and lead time embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)
        self.lead_time_embed = nn.Linear(1, embed_dim)

        # ViT backbone
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                             drop=drop_rate, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Prediction head
        head_layers = []
        for _ in range(decoder_depth):
            head_layers.append(nn.Linear(embed_dim, embed_dim))
            head_layers.append(nn.GELU())
        head_layers.append(nn.Linear(embed_dim, num_vars * patch_size ** 2))
        self.head = nn.Sequential(*head_layers)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.img_size[0] // self.patch_size,
            self.img_size[1] // self.patch_size,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        var_embed = get_1d_sincos_pos_embed_from_grid(
            self.var_embed.shape[-1], np.arange(len(self.default_vars))
        )
        self.var_embed.data.copy_(torch.from_numpy(var_embed).float().unsqueeze(0))

        for i in range(len(self.token_embeds.proj_weights)):
            w = self.token_embeds.proj_weights[i].data
            nn.init.trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars_tuple, device):
        ids = np.array([self.var_map[var] for var in vars_tuple])
        return torch.from_numpy(ids).to(device)

    def unpatchify(self, x):
        p = self.patch_size
        c = len(self.default_vars)
        h = self.img_size[0] // p
        w = self.img_size[1] // p
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward_encoder(self, x, lead_times, variables):
        if isinstance(variables, list):
            variables = tuple(variables)
        var_ids = self.get_var_ids(variables, x.device)

        # Per-variable tokenization
        x = self.token_embeds(x, var_ids)  # B, V, L, D

        # Add variable embedding
        var_embed = self.var_embed[:, var_ids, :]
        x = x + var_embed.unsqueeze(2)  # B, V, L, D

        # Variable aggregation (EDITABLE)
        x = self.var_aggregator(x)  # B, L, D

        # Add positional embedding
        x = x + self.pos_embed

        # Add lead time embedding
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))  # B, D
        x = x + lead_time_emb.unsqueeze(1)

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x, lead_times, variables):
        out = self.forward_encoder(x, lead_times, variables)
        preds = self.head(out)
        preds = self.unpatchify(preds)
        return preds


# ============================================================================
# FIXED — Metrics (Latitude-weighted RMSE)
# ============================================================================

def lat_weighted_mse(pred, y, lat):
    """Latitude-weighted MSE for training loss."""
    error = (pred - y) ** 2
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(
        dtype=error.dtype, device=error.device
    )
    return (error * w_lat.unsqueeze(1)).mean(dim=1).mean()


def lat_weighted_rmse(pred, y, lat, out_variables):
    """Latitude-weighted RMSE per variable (for evaluation)."""
    error = (pred - y) ** 2
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(
        dtype=error.dtype, device=error.device
    )
    results = {}
    for i, var in enumerate(out_variables):
        rmse = torch.mean(torch.sqrt(torch.mean(error[:, i] * w_lat, dim=(-2, -1))))
        results[var] = rmse.item()
    return results


# ============================================================================
# FIXED — Training and Evaluation Script
# ============================================================================

def load_pretrained_weights(model, pretrained_path, img_size):
    """Load pretrained ClimaX checkpoint, handling key name mismatches."""
    if not pretrained_path or not os.path.exists(pretrained_path):
        print(f"No pretrained weights at {pretrained_path}, training from scratch.")
        return

    print(f"Loading pretrained weights from: {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    # Remove 'net.' prefix if present (Lightning checkpoint convention)
    new_state = {}
    for k, v in state.items():
        nk = k.replace("net.", "") if k.startswith("net.") else k
        # Map channel -> var naming
        nk = nk.replace("channel", "var")
        new_state[nk] = v

    # Handle pos_embed interpolation if size differs
    if "pos_embed" in new_state:
        pe = new_state["pos_embed"]
        if pe.shape != model.pos_embed.shape:
            print(f"Interpolating pos_embed from {pe.shape} to {model.pos_embed.shape}")
            orig_num = pe.shape[1]
            p = model.patch_size
            orig_h = int((orig_num // 2) ** 0.5)
            orig_w = 2 * orig_h
            new_h = img_size[0] // p
            new_w = img_size[1] // p
            pe = pe.reshape(1, orig_h, orig_w, -1).permute(0, 3, 1, 2)
            pe = F.interpolate(pe, size=(new_h, new_w), mode="bicubic", align_corners=False)
            new_state["pos_embed"] = pe.permute(0, 2, 3, 1).flatten(1, 2)

    # Remap non-parallel token_embeds to parallel format
    proj_weights = []
    proj_biases = []
    for i in range(len(model.default_vars)):
        w_key = f"token_embeds.{i}.proj.weight"
        b_key = f"token_embeds.{i}.proj.bias"
        if w_key in new_state:
            proj_weights.append(new_state.pop(w_key))
            proj_biases.append(new_state.pop(b_key))
    if proj_weights:
        new_state["token_embeds.proj_weights"] = torch.stack(proj_weights)
        new_state["token_embeds.proj_biases"] = torch.stack(proj_biases)

    # Filter out keys that don't match the model
    model_state = model.state_dict()

    # Map old aggregate_variables keys to new var_aggregator keys
    remap = {}
    for k in list(new_state.keys()):
        if k == "var_query":
            remap["var_aggregator.var_query"] = new_state.pop(k)
        elif k == "var_agg.in_proj_weight":
            remap["var_aggregator.var_agg.in_proj_weight"] = new_state.pop(k)
        elif k == "var_agg.in_proj_bias":
            remap["var_aggregator.var_agg.in_proj_bias"] = new_state.pop(k)
        elif k == "var_agg.out_proj.weight":
            remap["var_aggregator.var_agg.out_proj.weight"] = new_state.pop(k)
        elif k == "var_agg.out_proj.bias":
            remap["var_aggregator.var_agg.out_proj.bias"] = new_state.pop(k)
    new_state.update(remap)

    # Map old timm Block keys to our TransformerBlock keys
    block_remap = {}
    for k in list(new_state.keys()):
        nk = k
        if ".attn.qkv." in k:
            nk = k  # same name
        elif ".attn.proj." in k:
            nk = k  # same name
        if nk.startswith("blocks."):
            block_remap[nk] = new_state.pop(k)
    new_state.update(block_remap)

    filtered = {}
    skipped = []
    for k, v in new_state.items():
        if k in model_state and v.shape == model_state[k].shape:
            filtered[k] = v
        else:
            skipped.append(k)

    if skipped:
        print(f"Skipped {len(skipped)} keys: {skipped[:10]}...")

    msg = model.load_state_dict(filtered, strict=False)
    print(f"Loaded {len(filtered)} keys. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")


if __name__ == "__main__":
    # ── Configuration from environment ──
    output_dir = os.environ.get("OUTPUT_DIR", "out")
    seed = int(os.environ.get("SEED", 42))
    data_dir = os.environ.get("DATA_DIR", "/data/era5_5.625deg")
    weight_dir = os.environ.get("WEIGHT_DIR", "/data/climax_weights")
    pretrained_path = os.path.join(weight_dir, "5.625deg.ckpt")

    # Target variable and predict range from environment
    out_var = os.environ.get("OUT_VAR", "geopotential_500")
    predict_range = int(os.environ.get("PREDICT_RANGE", 72))  # hours
    predict_range_steps = predict_range // 6  # ERA5 data is 6-hourly

    # Training hyperparameters
    max_epochs = int(os.environ.get("MAX_EPOCHS", 50))
    batch_size = int(os.environ.get("BATCH_SIZE", 64))
    lr = float(os.environ.get("LR", 5e-4))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 1e-5))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 200))
    patience = int(os.environ.get("PATIENCE", 5))
    grad_clip = float(os.environ.get("GRAD_CLIP", 1.0))

    # ========================================================================
    # EDITABLE — CONFIG_OVERRIDES
    # ========================================================================
    # CONFIG_OVERRIDES: override training hyperparameters for your method.
    # Allowed keys: learning_rate, weight_decay, warmup_steps, patience, grad_clip.
    CONFIG_OVERRIDES = {}
    # ========================================================================
    # FIXED — Apply overrides and continue training setup
    # ========================================================================
    for _k, _v in CONFIG_OVERRIDES.items():
        if _k == 'learning_rate': lr = _v
        elif _k == 'weight_decay': weight_decay = _v
        elif _k == 'warmup_steps': warmup_steps = _v
        elif _k == 'patience': patience = _v
        elif _k == 'grad_clip': grad_clip = _v

    # Model config
    img_size = [32, 64]
    patch_size = 2
    embed_dim = 1024
    depth = 8
    decoder_depth = 2
    num_heads = 16
    mlp_ratio = 4.0
    drop_path = 0.1
    drop_rate = 0.1

    # ── Setup ──
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    variables = DEFAULT_VARS
    out_variables = [out_var]

    print(f"Task: predict {out_var} at {predict_range}h lead time")
    print(f"Variables: {len(variables)}, device: {device}")

    # ── Build data loaders ──
    norm_in = build_normalize(data_dir, variables)
    norm_out = build_normalize(data_dir, out_variables)

    def list_npz_files(split_dir):
        import glob
        return sorted(f for f in glob.glob(os.path.join(split_dir, "*.npz"))
                       if "climatology" not in os.path.basename(f))

    train_files = list_npz_files(os.path.join(data_dir, "train"))
    val_files = list_npz_files(os.path.join(data_dir, "val"))
    test_files = list_npz_files(os.path.join(data_dir, "test"))

    print(f"Data shards — train: {len(train_files)}, val: {len(val_files)}, test: {len(test_files)}")

    # Wrap datasets with normalization applied inline
    class NormalizedDataset(IterableDataset):
        def __init__(self, base_ds, norm_in, norm_out):
            self.base_ds = base_ds
            self.norm_in = norm_in
            self.norm_out = norm_out

        def __iter__(self):
            for inp, out, lead in self.base_ds:
                yield self.norm_in(inp), self.norm_out(out), lead

    train_ds = NormalizedDataset(
        ShuffleBuffer(
            NpyShardDataset(train_files, variables, out_variables,
                            predict_range_steps, hrs_each_step=6, shuffle=True),
            buffer_size=5000
        ),
        norm_in, norm_out
    )
    val_ds = NormalizedDataset(
        NpyShardDataset(val_files, variables, out_variables, predict_range_steps, hrs_each_step=6),
        norm_in, norm_out
    )
    test_ds = NormalizedDataset(
        NpyShardDataset(test_files, variables, out_variables, predict_range_steps, hrs_each_step=6),
        norm_in, norm_out
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_fn,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_fn,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn,
                             num_workers=0, pin_memory=True)

    # ── Load lat/lon ──
    lat = np.load(os.path.join(data_dir, "lat.npy"))
    lon = np.load(os.path.join(data_dir, "lon.npy"))

    # ── Build model ──
    model = ClimaXModel(
        default_vars=variables,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        decoder_depth=decoder_depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_path=drop_path,
        drop_rate=drop_rate,
    )

    # Load pretrained weights
    load_pretrained_weights(model, pretrained_path, img_size)

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {num_params:.2f}M")

    # ── Optimizer and scheduler ──
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if "var_embed" in name or "pos_embed" in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "lr": lr, "weight_decay": weight_decay},
        {"params": no_decay_params, "lr": lr, "weight_decay": 0.0},
    ], betas=(0.9, 0.99))

    # Denormalization for RMSE evaluation
    def denormalize_pred(pred):
        """Denormalize predictions to original scale."""
        m = norm_out.mean.to(pred.device).view(1, -1, 1, 1)
        s = norm_out.std.to(pred.device).view(1, -1, 1, 1)
        return pred * s + m

    # ── Training loop ──
    global_step = 0
    best_val_rmse = float("inf")
    epochs_no_improve = 0
    scaler = torch.amp.GradScaler(enabled=True)

    print("Starting training...")
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            inp, out, lead = [b.to(device) for b in batch]

            # Warm-up + cosine decay LR schedule
            if global_step < warmup_steps:
                cur_lr = lr * (global_step + 1) / warmup_steps
            else:
                progress = (global_step - warmup_steps) / max(1, warmup_steps * 20 - warmup_steps)
                cur_lr = lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
                cur_lr = max(cur_lr, 1e-8)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                preds = model(inp, lead, variables)
                # Select output variable channels
                var_ids = model.get_var_ids(tuple(out_variables), preds.device)
                preds = preds[:, var_ids]
                loss = lat_weighted_mse(preds, out, lat)

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % 500 == 0:
                avg = epoch_loss / n_batches
                print(f"TRAIN_METRICS step={global_step} epoch={epoch} "
                      f"train_loss={avg:.6f} lr={cur_lr:.2e}", flush=True)

        epoch_time = time.time() - t0
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch}: train_loss={avg_loss:.6f}, time={epoch_time:.1f}s, steps={global_step}")

        # ── Validation ──
        model.eval()
        val_rmse_accum = {}
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                inp, out, lead = [b.to(device) for b in batch]
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    preds = model(inp, lead, variables)
                    var_ids = model.get_var_ids(tuple(out_variables), preds.device)
                    preds = preds[:, var_ids]
                # Denormalize for RMSE
                preds_denorm = denormalize_pred(preds.float())
                out_denorm = denormalize_pred(out.float())
                rmse_dict = lat_weighted_rmse(preds_denorm, out_denorm, lat, out_variables)
                for k, v in rmse_dict.items():
                    val_rmse_accum[k] = val_rmse_accum.get(k, 0.0) + v
                val_count += 1

        val_rmse_avg = {k: v / val_count for k, v in val_rmse_accum.items()}
        val_rmse_mean = np.mean(list(val_rmse_avg.values()))
        rmse_str = ", ".join(f"{k}={v:.4f}" for k, v in val_rmse_avg.items())
        print(f"  Val RMSE: {rmse_str} (mean={val_rmse_mean:.4f})")
        print(f"TRAIN_METRICS step={global_step} epoch={epoch} "
              f"train_loss={avg_loss:.6f} val_rmse={val_rmse_mean:.4f}", flush=True)

        if val_rmse_mean < best_val_rmse:
            best_val_rmse = val_rmse_mean
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            print(f"  New best model saved (val_rmse={val_rmse_mean:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping after {patience} epochs without improvement.")
                break

    # ── Final evaluation on test set ──
    print("Loading best model for test evaluation...")
    best_ckpt = os.path.join(output_dir, "best_model.pt")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    model.eval()
    test_rmse_accum = {}
    test_count = 0
    with torch.no_grad():
        for batch in test_loader:
            inp, out, lead = [b.to(device) for b in batch]
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                preds = model(inp, lead, variables)
                var_ids = model.get_var_ids(tuple(out_variables), preds.device)
                preds = preds[:, var_ids]
            preds_denorm = denormalize_pred(preds.float())
            out_denorm = denormalize_pred(out.float())
            rmse_dict = lat_weighted_rmse(preds_denorm, out_denorm, lat, out_variables)
            for k, v in rmse_dict.items():
                test_rmse_accum[k] = test_rmse_accum.get(k, 0.0) + v
            test_count += 1

    test_rmse_avg = {k: v / test_count for k, v in test_rmse_accum.items()}
    test_rmse_mean = np.mean(list(test_rmse_avg.values()))
    rmse_str = ", ".join(f"{k}={v:.4f}" for k, v in test_rmse_avg.items())
    print(f"Test RMSE: {rmse_str} (mean={test_rmse_mean:.4f})")

    # Output metrics in TEST_METRICS format
    metrics_parts = [f"w_rmse_{k}={v:.4f}" for k, v in test_rmse_avg.items()]
    print(f"TEST_METRICS: {', '.join(metrics_parts)}", flush=True)

    print("Done.")
