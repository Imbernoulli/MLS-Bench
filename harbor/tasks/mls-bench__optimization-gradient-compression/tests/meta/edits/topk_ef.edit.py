"""TopK Sparsification with Error Feedback baseline.

Keeps only the top-K largest-magnitude gradient elements and zeros the rest.
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

    Keeps the K largest-magnitude gradient elements per tensor.
    Error feedback accumulates the compression error (original - decompressed)
    and adds it to the next gradient before compression, ensuring convergence.
    \"\"\"

    def __init__(self, compress_ratio=0.01):
        self.compress_ratio = compress_ratio
        self.residuals = {}

    def compress(self, tensor, name):
        # Error feedback: add accumulated residual
        if name in self.residuals:
            tensor = tensor + self.residuals[name]

        shape = tensor.shape
        tensor_flat = tensor.flatten()
        numel = tensor_flat.numel()
        k = max(1, int(numel * self.compress_ratio))

        # Select top-k by magnitude
        _, indices = torch.topk(tensor_flat.abs(), k, sorted=False)
        values = tensor_flat[indices]

        # Update residual: store what was NOT communicated
        decompressed_flat = torch.zeros_like(tensor_flat)
        decompressed_flat.scatter_(0, indices, values)
        self.residuals[name] = tensor_flat - decompressed_flat
        self.residuals[name] = self.residuals[name].view(shape)

        return [values, indices], (numel, shape)

    def decompress(self, compressed_tensors, ctx):
        values, indices = compressed_tensors
        numel, shape = ctx
        tensor_decompressed = torch.zeros(
            numel, dtype=values.dtype, device=values.device)
        tensor_decompressed.scatter_(0, indices, values)
        return tensor_decompressed.view(shape)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 182,
        "end_line": 232,
        "content": _TOPK_EF,
    },
]
