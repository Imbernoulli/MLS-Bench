"""QSGD (Quantized SGD) baseline, sized to the 100x step budget.

Stochastic quantization that maps each gradient element to a discrete set
of levels, using randomized rounding to preserve the expected value (unbiased),
sent in the paper's Elias-coded sparse encoding. Each tensor uses the largest
number of levels whose packet fits its share of the step budget.

Reference:
- Alistarh et al., "QSGD: Communication-Efficient SGD via Gradient
  Quantization and Encoding", NeurIPS 2017

GRACE reference: grace_dl/torch/compressor/qsgd.py
"""

_FILE = "pytorch-vision/custom_compressor.py"

_QSGD = """\
class Compressor:
    \"\"\"QSGD with Elias coding, at the step budget.

    Unbiased stochastic quantization of |g_i| / ||g|| to s levels
    (Alistarh et al., NeurIPS 2017), sent in the paper's sparse encoding:
    sign-and-level codes (a codebook of the values +-l * ||g|| / s and 0)
    at Elias-gamma coded positions. Each tensor uses the largest
    power-of-two s (up to 256) whose expected packet, measured on its
    previous gradient, fits its share of the step budget; the packet
    carries at most K entries, K the most that fit that share (should a
    gradient have more nonzero levels, the largest are kept). No error
    feedback (QSGD is unbiased). Per-tensor gradient clipping keeps
    quantization noise from diverging.
    \"\"\"

    def __init__(self, compress_ratio, param_numels, budget_bits):
        self.compress_ratio = compress_ratio
        self.clip_norm = 1.0
        self.levels = [2 ** j for j in range(9)]  # s = 1 .. 256
        total = sum(param_numels.values())
        floor = {n: self._cost(1, 1, n) for n in set(param_numels.values())}
        spare = budget_bits - sum(floor[n] for n in param_numels.values())
        self.share = {name: floor[n] + spare * n // total
                      for name, n in param_numels.items()}
        self.expected = {}  # name -> (host tensor of E[nnz] per s, event)

    @staticmethod
    def _cost(k, s, n):
        bits = math.ceil(math.log2(2 * s + 1))
        return k * bits + 32 * (2 * s + 1) + elias_gamma_bound(k, n)

    def _largest_k(self, s, n, share):
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._cost(mid, s, n) <= share:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def compress(self, tensor, name):
        flat = tensor.flatten()
        n = flat.numel()
        share = self.share[name]
        s = 1
        if name in self.expected:
            host, done = self.expected[name]
            if done is not None:
                done.synchronize()
            for s_try, e in zip(self.levels, host.tolist()):
                if self._cost(min(n, math.ceil(1.1 * e) + 8), s_try, n) <= share:
                    s = s_try
        k = min(n, self._largest_k(s, n, share))
        flat = flat * (self.clip_norm / (flat.norm() + 1e-6)).clamp(max=1.0)
        norm = flat.norm()
        unit = flat.abs() / norm.clamp(min=1e-30)
        # Expected nonzeros at each s, read at this tensor's next call.
        s_vec = torch.tensor(self.levels, dtype=unit.dtype).to(unit.device)
        e = (s_vec.unsqueeze(1) * unit.unsqueeze(0)).clamp(max=1.0).sum(1)
        done = None
        if e.is_cuda:
            e = torch.empty(e.shape, dtype=e.dtype,
                            pin_memory=True).copy_(e, non_blocking=True)
            done = torch.cuda.Event()
            done.record()
        self.expected[name] = (e, done)
        level = s * unit
        level = level.floor() + (torch.rand_like(level) < level - level.floor()).float()
        idx = torch.topk((level > 0).float() * (1.0 + unit), k,
                         sorted=False).indices
        values = flat[idx].sign() * level[idx] * (norm / s)
        return {"values": values, "bits": math.ceil(math.log2(2 * s + 1)),
                "indices": idx}
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 182,
        "end_line": 224,
        "content": _QSGD,
    },
]
