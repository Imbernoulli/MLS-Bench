"""finetune_then_ptq baseline -- FP fine-tune followed by RTN PTQ.

This is a *control* baseline that isolates the contribution of the QAT
fake-quant signal from the contribution of plain fine-tuning on the
WikiText-2 train split.  It uses the exact same training hyperparameters
as the QAT baselines (STE/LSQ/PACT/...), but the forward pass during
training is full precision -- no fake quantization, no QAT signal at
all.  After training, the real per-group symmetric round-to-nearest
quantize-dequantize (identical to the ``no_qat`` PTQ) is applied to
every wrapped linear weight before evaluation.

If this baseline matches a QAT method at every bit-width, that QAT
method's "improvement over no_qat" was really just the finetune
talking, not the QAT signal.  If it lags behind a QAT method
(especially at INT2/INT3, where RTN PTQ collapses), that gap is the
genuine QAT contribution.

Reference: same RTN PTQ scheme as Jacob et al., "Quantization and
Training of Neural Networks for Efficient Integer-Arithmetic-Only
Inference" (CVPR 2018), but applied to weights produced by a brief
full-precision finetune on the calibration text rather than to the
original pretrained weights.
"""

_FILE = "llm-qat-runtime/custom_qat.py"

_CODE = """\

# ── Finetune-then-PTQ control baseline ────────────────────────────────────────
# Forward pass during training is pure FP (no fake quant), but the same
# training schedule as STE/LSQ/PACT is run.  After training, real RTN
# QDQ is applied to materialize the integer model.

CONFIG_OVERRIDES = {
    "learning_rate": 2e-5,
    "num_steps": 500,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "max_grad_norm": 1.0,
    "warmup_steps": 50,
    "weight_decay": 0.0,
}


def _qrange(num_bits):
    qmax = (1 << (num_bits - 1)) - 1
    qmin = -(1 << (num_bits - 1))
    return qmin, qmax


def fake_quantize_weight(weight, num_bits, group_size):
    # Identity: no fake quant in forward -- pure FP finetune.
    return weight


def fake_quantize_activation(x, num_bits):
    return x


def quantize_dequantize_weight(weight, num_bits, group_size):
    qmin, qmax = _qrange(num_bits)
    out_features, in_features = weight.shape
    assert in_features % group_size == 0
    with torch.no_grad():
        w = weight.float().reshape(out_features, -1, group_size)
        w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = w_max / qmax
        w_q = torch.clamp(torch.round(w / scale), qmin, qmax) * scale
        return w_q.reshape(out_features, in_features).to(weight.dtype)


class QATWrapper(nn.Module):
    def __init__(self, linear, num_bits, group_size):
        super().__init__()
        self.linear = linear
        self.num_bits = num_bits
        self.group_size = group_size

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x):
        # Pure FP forward during training (no fake quant).  At eval time
        # the real QDQ has already been applied to ``linear.weight``, so
        # this still produces the genuine INT-N output.
        return F.linear(x, self.linear.weight, self.linear.bias)


def prepare_qat_model(model, num_bits, group_size):
    from transformers.pytorch_utils import Conv1D

    def _replace(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                setattr(parent, name, QATWrapper(child, num_bits=num_bits, group_size=group_size))
            elif isinstance(child, Conv1D):
                in_f, out_f = child.weight.shape
                lin = nn.Linear(in_f, out_f, bias=child.bias is not None,
                                device=child.weight.device, dtype=child.weight.dtype)
                with torch.no_grad():
                    lin.weight.copy_(child.weight.t().contiguous())
                    if child.bias is not None:
                        lin.bias.copy_(child.bias)
                setattr(parent, name, QATWrapper(lin, num_bits=num_bits, group_size=group_size))
            else:
                _replace(child)

    _replace(model)
    for head_attr in ("lm_head", "embed_out"):
        head = getattr(model, head_attr, None)
        if isinstance(head, QATWrapper):
            setattr(model, head_attr, head.linear)
    return model

"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 33,
        "end_line": 176,
        "content": _CODE,
    },
]
