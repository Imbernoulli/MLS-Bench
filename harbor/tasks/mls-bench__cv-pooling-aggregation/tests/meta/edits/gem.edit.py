"""Generalized Mean (GeM) Pooling baseline.

Learnable generalized mean pooling with parameter p (initialized to 3.0).
When p=1, equivalent to average pooling; as p->inf, approaches max pooling.

Reference: Radenovic et al., "Fine-tuning CNN Image Retrieval with No Human
Annotation" (TPAMI 2018)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_pool.py"

_CONTENT = """\
class CustomPool(nn.Module):
    \"\"\"Generalized Mean (GeM) Pooling.

    Learnable generalized mean with parameter p (init=3.0).
    Interpolates between average pooling (p=1) and max pooling (p->inf).

    \"\"\"

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * 3.0)
        self.eps = 1e-6

    def forward(self, x):
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps)
        return F.adaptive_avg_pool2d(x.pow(p), 1).pow(1.0 / p).view(x.size(0), -1)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 31,
        "end_line": 48,
        "content": _CONTENT,
    },
]
