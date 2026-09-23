"""SignSGD baseline, sized to the 100x step budget.

Scaled sign compression with error feedback. One sign bit for every entry is
32x, not 100x, so each tensor sends the signs of its largest-magnitude
error-corrected entries (as many as its share of the step budget carries,
at Elias-gamma coded positions), scaled by their mean magnitude.

Reference:
- Bernstein et al., "signSGD: Compressed Optimisation for Non-Convex Problems",
  ICML 2018
- Karimireddy et al., "Error Feedback Fixes SignSGD and Other Gradient
  Compression Schemes", ICML 2019

GRACE reference: grace_dl/torch/compressor/signsgd.py
"""

_FILE = "pytorch-vision/custom_compressor.py"

_SIGNSGD = """\
class Compressor:
    \"\"\"Scaled signSGD with error feedback, at the step budget.

    Sends one sign bit per entry, scaled by the mean magnitude of the sent
    entries (a two-value codebook), with error feedback carrying what the
    sign loses into the next gradient. One bit for every entry would be
    32x, not the 100x budget, so each tensor sends the signs of its K
    largest-magnitude error-corrected entries, K being the largest count
    whose worst-case packet (sign bits, codebook, Elias-gamma coded
    positions) fits the tensor's share of the step budget.
    \"\"\"

    def __init__(self, compress_ratio, param_numels, budget_bits):
        self.compress_ratio = compress_ratio
        self.residuals = {}
        self.k = self._plan(param_numels, budget_bits,
                            lambda k, n: k + 64 + elias_gamma_bound(k, n))

    @staticmethod
    def _plan(param_numels, budget_bits, cost):
        total = sum(param_numels.values())
        spare = budget_bits - sum(cost(1, n) for n in param_numels.values())
        plan = {}
        for name, n in param_numels.items():
            share = cost(1, n) + spare * n // total
            lo, hi = 1, n
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if cost(mid, n) <= share:
                    lo = mid
                else:
                    hi = mid - 1
            plan[name] = lo
        return plan

    def compress(self, tensor, name):
        if name in self.residuals:
            tensor = tensor + self.residuals[name]
        flat = tensor.flatten()
        _, indices = torch.topk(flat.abs(), self.k[name], sorted=False)
        picked = flat[indices]
        scale = picked.abs().mean()
        values = torch.where(picked >= 0, scale, -scale)
        sent = torch.zeros_like(flat).scatter_(0, indices, values)
        self.residuals[name] = (flat - sent).view(tensor.shape)
        return {"values": values, "bits": 1, "indices": indices}
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 182,
        "end_line": 224,
        "content": _SIGNSGD,
    },
]
