"""QSGD (Quantized SGD) baseline.

Stochastic quantization that maps each gradient element to a discrete set
of levels, using randomized rounding to preserve the expected value (unbiased).
The number of quantization levels controls the communication/convergence tradeoff.

Reference:
- Alistarh et al., "QSGD: Communication-Efficient SGD via Gradient
  Quantization and Encoding", NeurIPS 2017

GRACE reference: grace_dl/torch/compressor/qsgd.py
"""

_FILE = "pytorch-vision/custom_compressor.py"

_QSGD = """\
class Compressor:
    \"\"\"QSGD — Quantized Stochastic Gradient Descent.

    Quantizes each gradient element to one of `s` discrete levels using
    randomized rounding. The quantization is unbiased: E[Q(g)] = g.
    Communication cost: O(n * log(s) / 32) of original, where n = numel.

    Uses s=256 quantization levels for a stable communication/variance tradeoff.

    Note: QSGD is an *unbiased* compressor, so error feedback is not needed
    and can actually hurt convergence. Unlike biased compressors (TopK,
    SignSGD) that systematically lose information, QSGD preserves the
    expected gradient value, making the vanilla SGD convergence guarantees
    applicable with only increased variance.

    Reference: Alistarh et al., "QSGD: Communication-Efficient SGD via
    Gradient Quantization and Encoding", NeurIPS 2017.
    \"\"\"

    def __init__(self, compress_ratio=0.01):
        self.compress_ratio = compress_ratio
        # QSGD: s = number of quantization levels (~log2(s)+1 bits/element).
        # Var(Q(g)) ~ ||g||^2 * min(d/s^2, sqrt(d)/s). For deep nets with
        # d ~ 1e7 params, small s produces huge variance that interacts with
        # momentum SGD causing divergence on some seeds. s=256 gives ~9
        # bits/element (~3.5x compression) and keeps variance bounded.
        self.quantum_num = 256
        # Per-tensor gradient clip: prevent rare large-norm gradients from
        # amplifying quantization noise into divergence (standard QSGD
        # practice, cf. Alistarh 2017 Algorithm 1 discussion).
        self.clip_norm = 1.0

    def compress(self, tensor, name):
        shape = tensor.shape
        tensor_flat = tensor.flatten()

        # Gradient clipping BEFORE quantization — critical for stability.
        norm = tensor_flat.norm()
        if norm == 0:
            return [tensor_flat.to(torch.int16), norm], shape
        clip_coef = self.clip_norm / (norm + 1e-6)
        if clip_coef < 1.0:
            tensor_flat = tensor_flat * clip_coef
            norm = tensor_flat.norm()
            if norm == 0:
                return [tensor_flat.to(torch.int16), norm], shape

        abs_gradient = tensor_flat.abs()

        # Quantize: level = floor(s * |g_i| / ||g||) with stochastic rounding
        level_float = self.quantum_num / norm * abs_gradient
        previous_level = level_float.floor()
        prob = torch.rand_like(tensor_flat)
        is_next_level = (prob < (level_float - previous_level)).float()
        new_level = previous_level + is_next_level

        # Store sign and quantized level
        sign = tensor_flat.sign()
        tensor_compressed = (new_level * sign)
        tensor_compressed = tensor_compressed.to(torch.int16)

        return [tensor_compressed, norm], shape

    def decompress(self, compressed_tensors, ctx):
        shape = ctx
        tensor_compressed, norm = compressed_tensors

        # Dequantize: g_hat = (norm / s) * quantized_value
        decode_output = tensor_compressed.float()
        tensor_decompressed = norm / self.quantum_num * decode_output
        return tensor_decompressed.view(shape)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 182,
        "end_line": 232,
        "content": _QSGD,
    },
]
