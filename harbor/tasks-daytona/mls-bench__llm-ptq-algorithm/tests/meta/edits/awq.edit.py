"""AWQ baseline -- Activation-Aware Weight Quantization.

Identifies salient weight channels by analyzing activation magnitudes from
calibration data. Weights corresponding to high-activation channels are
"protected" by applying per-channel scaling factors before quantization:
scale up salient weights (making them easier to quantize accurately) and
scale down the rest. After quantization, the inverse scaling is applied
so the layer output is preserved.

The key insight is that not all weight channels are equally important:
channels that are consistently activated with large magnitudes during
inference contribute more to the output and should be quantized more
carefully. AWQ achieves this via a simple per-channel scaling trick that
requires no Hessian computation.

Reference: Lin et al., "AWQ: Activation-Aware Weight Quantization for
LLM Compression and Acceleration" (MLSys 2024)
"""

_FILE = "gptq/custom_ptq.py"

_AWQ_CODE = """\

# ── Helper: basic quantize/dequantize primitives ──────────────────────────────

def quantize_tensor(x, scale, zero_point, qmin, qmax):
    \"\"\"Quantize a float tensor to integers given scale and zero point.\"\"\"
    x_int = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
    return x_int


def dequantize_tensor(x_int, scale, zero_point):
    \"\"\"Dequantize integer tensor back to float.\"\"\"
    return (x_int - zero_point) * scale


def find_scale_zero(weight, num_bits=4, group_size=-1, symmetric=True):
    \"\"\"Compute per-channel (or per-group) quantization parameters.\"\"\"
    qmin = -(1 << (num_bits - 1))
    qmax = (1 << (num_bits - 1)) - 1

    if group_size > 0:
        out_features, in_features = weight.shape
        assert in_features % group_size == 0
        w_groups = weight.reshape(out_features, -1, group_size)
        if symmetric:
            w_max = w_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            scale = w_max / qmax
            zero_point = torch.zeros_like(scale)
        else:
            w_min = w_groups.amin(dim=-1, keepdim=True)
            w_max = w_groups.amax(dim=-1, keepdim=True)
            w_range = (w_max - w_min).clamp(min=1e-12)
            scale = w_range / (qmax - qmin)
            zero_point = torch.round(qmin - w_min / scale)
        scale = scale.reshape(out_features, -1).repeat_interleave(group_size, dim=1)
        zero_point = zero_point.reshape(out_features, -1).repeat_interleave(group_size, dim=1)
    else:
        if symmetric:
            w_max = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
            scale = w_max / qmax
            zero_point = torch.zeros_like(scale)
        else:
            w_min = weight.amin(dim=1, keepdim=True)
            w_max = weight.amax(dim=1, keepdim=True)
            w_range = (w_max - w_min).clamp(min=1e-12)
            scale = w_range / (qmax - qmin)
            zero_point = torch.round(qmin - w_min / scale)

    return scale, zero_point, qmin, qmax


class LayerQuantizer:
    \"\"\"AWQ quantizer -- faithful to mit-han-lab/llm-awq.

    Pipeline:
      1. add_batch: accumulate per-channel mean |X|; reservoir-sample raw input
         tokens (up to N_SAMPLE_TOKEN rows) so we can use real activations as
         the loss signal during search.
      2. Per-channel scale alpha-search (auto_scale): for ratio in [0, 1):
             s = x_max^ratio  (clamped, range-normalized: s /= sqrt(max*min))
         loss = mean((X @ (W - W_final).T)^2)  on sampled X.
      3. Per-group max clip-search (auto_clip), on the post-scale weights:
         clip per-group max by 1 - i/N for i in 0..MAX_SHRINK*N, loss is
         per-(out_channel, group) output-error using sampled X:
             org_out[r, t, g] = sum_c W_scaled[r,g,c] * X[t,g,c]
             cur_out[r, t, g] = sum_c Q(clamp(W,±M))[r,g,c] * X[t,g,c]
             err[r, g] = mean_t (cur_out - org_out)^2
      4. Quantize with the clipped per-group scales, undo channel scaling.

    Implemented to fit the LayerQuantizer interface (per-linear, no block ctx),
    so the loss is computed at linear-layer granularity (not full block).
    \"\"\"

    N_ALPHA = 20             # auto_scale grid size
    N_CLIP_GRID = 20         # auto_clip n_grid
    CLIP_MAX_SHRINK = 0.5    # auto_clip max_shrink (official default)
    N_SAMPLE_TOKEN = 256     # number of input tokens kept for loss computation
    OC_BATCH = 256           # output-channel batching for clip search (memory)

    def __init__(self, layer, num_bits=4, group_size=-1):
        self.layer = layer
        self.num_bits = num_bits
        self.group_size = group_size
        self.out_features, self.in_features = layer.weight.shape
        self.dev = layer.weight.device
        self.nsamples = 0

        # Per-channel sum of |activation| (averaged over tokens at quantize-time)
        self.act_sum = torch.zeros(
            self.in_features, device=self.dev, dtype=torch.float32
        )
        # Reservoir of input tokens (CPU to save GPU memory across layers)
        self._x_buf = []
        self._x_buf_rows = 0
        # Keep H for interface compatibility (unused by AWQ)
        self.H = torch.zeros(
            (self.in_features, self.in_features),
            device=self.dev, dtype=torch.float32
        )

    def add_batch(self, inp):
        \"\"\"Accumulate per-channel |X| stats and reservoir-sample raw inputs.\"\"\"
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp_f = inp.float()
        n = inp_f.shape[0]
        self.act_sum += inp_f.abs().sum(dim=0)
        self.nsamples += n
        # Keep ~4x N_SAMPLE_TOKEN candidate rows; we'll stride-sample at quantize.
        cap = self.N_SAMPLE_TOKEN * 4
        if self._x_buf_rows < cap:
            take = min(n, cap - self._x_buf_rows)
            # Take an evenly-spaced stride from this batch
            stride = max(1, n // max(take, 1))
            sampled = inp_f[::stride][:take].detach().to('cpu')
            self._x_buf.append(sampled)
            self._x_buf_rows += sampled.shape[0]

    def _get_x_samples(self):
        if not self._x_buf:
            return None
        X = torch.cat(self._x_buf, dim=0)
        if X.shape[0] > self.N_SAMPLE_TOKEN:
            stride = X.shape[0] // self.N_SAMPLE_TOKEN
            X = X[::stride][:self.N_SAMPLE_TOKEN]
        return X.to(self.dev)

    def quantize(self):
        \"\"\"AWQ: per-channel scale search + per-group clip search + quantize.\"\"\"
        W = self.layer.weight.data.clone().float()
        num_bits = self.num_bits
        group_size = self.group_size
        qmin = -(1 << (num_bits - 1))
        qmax = (1 << (num_bits - 1)) - 1

        if self.nsamples > 0:
            x_max = (self.act_sum / self.nsamples).clamp(min=1e-5)
        else:
            x_max = torch.ones(self.in_features, device=self.dev)

        X = self._get_x_samples()  # (T, in_features) on dev, may be None

        # ── (1) auto_scale: per-channel scale search ─────────────────────────
        best_err = float('inf')
        best_s = torch.ones(self.in_features, device=self.dev)

        for i in range(self.N_ALPHA):
            ratio = i / self.N_ALPHA
            s = x_max.pow(ratio).clamp(min=1e-4)
            s = s / (s.max() * s.min()).sqrt().clamp(min=1e-5)

            W_scaled = W * s.unsqueeze(0)
            scale_q, zp, _, _ = find_scale_zero(
                W_scaled, num_bits=num_bits, group_size=group_size, symmetric=True
            )
            W_q = quantize_tensor(W_scaled, scale_q, zp, qmin, qmax)
            W_dq = dequantize_tensor(W_q, scale_q, zp)
            W_final = W_dq / s.unsqueeze(0)

            if X is not None:
                # Output-error: ||X @ (W - W_final).T||^2 / (T * out)
                delta = (W - W_final).to(X.dtype)
                err = (X @ delta.T).pow(2).mean().item()
            else:
                err = (W - W_final).pow(2).mul(x_max.unsqueeze(0).pow(2)).sum().item()

            if err < best_err:
                best_err = err
                best_s = s.clone()

        # Apply best per-channel scaling
        W_scaled = W * best_s.unsqueeze(0)

        # ── (2) auto_clip: per-group max clip search ─────────────────────────
        if group_size > 0:
            n_groups = self.in_features // group_size
            gs = group_size
        else:
            n_groups = 1
            gs = self.in_features

        W_groups = W_scaled.reshape(self.out_features, n_groups, gs)  # (O, G, gs)
        base_max = W_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        best_max = base_max.clone()

        if X is not None:
            X_groups = X.reshape(X.shape[0], n_groups, gs)  # (T, G, gs)

            n_clip_iters = max(1, int(self.CLIP_MAX_SHRINK * self.N_CLIP_GRID))
            oc_batch = self.OC_BATCH
            if self.out_features % oc_batch != 0:
                # fall back to a divisor of out_features
                for cand in (128, 64, 32, 16, 8, 4, 2, 1):
                    if self.out_features % cand == 0:
                        oc_batch = cand
                        break

            for i_b in range(0, self.out_features, oc_batch):
                W_b = W_groups[i_b:i_b + oc_batch]                # (B, G, gs)
                base_max_b = base_max[i_b:i_b + oc_batch]          # (B, G, 1)
                # org_out[r, t, g] = sum_c W_b[r,g,c] * X_groups[t,g,c]
                org_out = torch.einsum('rgc,tgc->rtg', W_b, X_groups.float())
                min_errs = torch.full_like(base_max_b, float('inf'))
                best_max_b = base_max_b.clone()
                for i_s in range(n_clip_iters):
                    cur_max = base_max_b * (1 - i_s / self.N_CLIP_GRID)  # (B, G, 1)
                    cur_w = torch.clamp(W_b, -cur_max, cur_max)
                    scale_b = (cur_max / qmax).clamp(min=1e-12)
                    q_w = (
                        torch.clamp(torch.round(cur_w / scale_b), qmin, qmax) * scale_b
                    )
                    cur_out = torch.einsum('rgc,tgc->rtg', q_w, X_groups.float())
                    err_b = (cur_out - org_out).pow(2).mean(dim=1, keepdim=True)
                    err_b = err_b.permute(0, 2, 1).contiguous()  # (B, G, 1)
                    mask = err_b < min_errs
                    min_errs = torch.where(mask, err_b, min_errs)
                    best_max_b = torch.where(mask, cur_max, best_max_b)
                best_max[i_b:i_b + oc_batch] = best_max_b
                del org_out, cur_out, q_w, cur_w
            del X_groups
        # else: no calibration samples — fall back to base_max (no clipping)

        # ── (3) Final quantization with clipped scales ───────────────────────
        scale_g = (best_max / qmax).clamp(min=1e-12)
        scale_q = scale_g.expand_as(W_groups).reshape(self.out_features, self.in_features)
        zp = torch.zeros_like(scale_q)

        # Clamp scaled weights to the searched per-group range, then quantize
        W_clamped = torch.clamp(
            W_scaled,
            -best_max.expand_as(W_groups).reshape(self.out_features, self.in_features),
            best_max.expand_as(W_groups).reshape(self.out_features, self.in_features),
        )
        W_q = quantize_tensor(W_clamped, scale_q, zp, qmin, qmax)
        W_dq = dequantize_tensor(W_q, scale_q, zp)
        W_final = W_dq / best_s.unsqueeze(0)

        return W_final.to(self.layer.weight.dtype)

    def free(self):
        \"\"\"Release calibration buffers.\"\"\"
        del self.H
        del self.act_sum
        del self._x_buf
        self.H = None
        self.act_sum = None
        self._x_buf = None

"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 26,
        "end_line": 157,
        "content": _AWQ_CODE,
    },
]
