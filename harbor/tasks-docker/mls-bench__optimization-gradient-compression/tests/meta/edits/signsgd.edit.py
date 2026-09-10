"""SignSGD baseline.

Transmits only the sign of each gradient element, achieving 32x compression
(1 bit per element vs 32 bits for float32). Uses majority vote aggregation
in the distributed setting. Combined with error feedback for better convergence.

Reference:
- Bernstein et al., "signSGD: Compressed Optimisation for Non-Convex Problems",
  ICML 2018
- Bernstein et al., "signSGD with Majority Vote is Communication Efficient
  and Fault Tolerant", ICLR 2019

GRACE reference: grace_dl/torch/compressor/signsgd.py
"""

_FILE = "pytorch-vision/custom_compressor.py"

_SIGNSGD = """\
class Compressor:
    \"\"\"SignSGD with error feedback.

    Compresses each gradient element to its sign (+1 or -1), achieving
    32x compression. Error feedback accumulates the magnitude information
    lost during sign extraction, improving convergence.

    The compress_ratio parameter is not used for sign compression (always
    1-bit), but the error feedback momentum can be tuned.
    \"\"\"

    def __init__(self, compress_ratio=0.01):
        self.compress_ratio = compress_ratio
        self.residuals = {}
        # Error feedback momentum
        self.ef_beta = 1.0

    def compress(self, tensor, name):
        # Error feedback: add accumulated residual
        if name in self.residuals:
            tensor = tensor + self.ef_beta * self.residuals[name]

        shape = tensor.shape
        tensor_flat = tensor.flatten()

        # Sign compression: 1 bit per element
        signs = (tensor_flat >= 0).to(torch.uint8)

        # Scale by mean magnitude for better reconstruction
        mean_magnitude = tensor_flat.abs().mean()

        # Update residual: original - reconstructed
        sign_float = signs.float() * 2 - 1  # map {0,1} -> {-1,+1}
        reconstructed = sign_float * mean_magnitude
        self.residuals[name] = (tensor_flat - reconstructed).view(shape)

        return [signs, mean_magnitude], shape

    def decompress(self, compressed_tensors, ctx):
        shape = ctx
        signs, mean_magnitude = compressed_tensors

        # Reconstruct: sign * mean_magnitude
        sign_float = signs.float() * 2 - 1
        tensor_decompressed = sign_float * mean_magnitude
        return tensor_decompressed.view(shape)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 182,
        "end_line": 232,
        "content": _SIGNSGD,
    },
]
