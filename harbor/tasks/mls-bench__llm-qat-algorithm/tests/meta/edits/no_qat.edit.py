"""no_qat baseline -- PTQ-only reference (no fine-tuning).

Sets ``num_steps=0`` so the QAT training loop is a no-op, then relies
on the fixed real-quantize-dequantize roundtrip to materialize the
INT-N model.  This is a PTQ-only control for the QAT baselines.

Reference: standard round-to-nearest PTQ as in
Jacob et al., "Quantization and Training of Neural Networks for
Efficient Integer-Arithmetic-Only Inference" (CVPR 2018).
"""

_FILE = "llm-qat-runtime/custom_qat.py"

_CODE = """\

# ── PTQ-only baseline: no QAT fine-tune, real QDQ at eval time ────────────────

CONFIG_OVERRIDES = {
    "learning_rate": 0.0,
    "num_steps": 0,
    "batch_size": 2,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "warmup_steps": 0,
    "weight_decay": 0.0,
}


def _qrange(num_bits):
    qmax = (1 << (num_bits - 1)) - 1
    qmin = -(1 << (num_bits - 1))
    return qmin, qmax


def fake_quantize_weight(weight, num_bits, group_size):
    qmin, qmax = _qrange(num_bits)
    out_features, in_features = weight.shape
    assert in_features % group_size == 0
    w = weight.float().reshape(out_features, -1, group_size)
    w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = w_max / qmax
    w_q = torch.clamp(torch.round(w / scale), qmin, qmax) * scale
    w_dq = w + (w_q - w).detach()
    return w_dq.reshape(out_features, in_features).to(weight.dtype)


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
        # PTQ-only: in eval the real QDQ has already been applied to
        # linear.weight, so we just call the underlying linear.  During
        # the (zero-step) training phase this is a no-op anyway.
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
