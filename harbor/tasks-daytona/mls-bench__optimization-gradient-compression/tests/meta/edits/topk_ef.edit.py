"""TopK Sparsification with Error Feedback baseline.

Keeps only the top-K largest-magnitude gradient elements and zeros the rest,
with K sized to the 100x step budget (float32 values, Elias-gamma coded
positions).
Error feedback accumulates compression residuals and adds them to the next
iteration's gradient, which is critical for convergence with biased compressors.

Reference:
- Alistarh et al., "The Convergence of Sparsified Gradient Methods", NeurIPS 2018
- Stich et al., "Sparsified SGD with Memory", NeurIPS 2018
- Aji & Heafield, "Sparse Communication for Distributed Gradient Descent", EMNLP 2017

GRACE reference: grace_dl/torch/compressor/topk.py + grace_dl/dist/memory/residual.py
"""

_FILE = "pytorch-vision/custom_compressor.py"

_TOPK_EF = """\
class Compressor:
    \"\"\"TopK sparsification with error feedback (EF-TopK).

    Keeps the K largest-magnitude entries of each (error-corrected)
    gradient, sent as float32 values with Elias-gamma coded positions.
    K is the largest count whose worst-case packet fits the tensor's share
    of the step budget (one entry per tensor first, the rest of the budget
    split in proportion to tensor size). Error feedback accumulates what
    was not sent and adds it to the next gradient before compression.
    \"\"\"

    def __init__(self, compress_ratio, param_numels, budget_bits):
        self.compress_ratio = compress_ratio
        self.residuals = {}
        self.k = self._plan(param_numels, budget_bits,
                            lambda k, n: 32 * k + elias_gamma_bound(k, n))

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
        values = flat[indices]
        sent = torch.zeros_like(flat).scatter_(0, indices, values)
        self.residuals[name] = (flat - sent).view(tensor.shape)
        return {"values": values, "bits": 32, "indices": indices}
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 182,
        "end_line": 224,
        "content": _TOPK_EF,
    },
]
