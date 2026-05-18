"""Vanilla residual stream baseline — standard Pre-LN residual connections.

The template already uses x + sublayer(x), so this baseline only performs
an identity replacement on the block loop to satisfy the rigorous_codebase
requirement that every baseline has at least one OPS entry.
"""

_FILE = "nanoGPT/custom_pretrain.py"

_VANILLA_LOOP = """\
        # ── Residual stream: iterate through transformer blocks ──
        for block in self.transformer.h:
            x = block(x)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 162,
        "end_line": 164,
        "content": _VANILLA_LOOP,
    },
]
