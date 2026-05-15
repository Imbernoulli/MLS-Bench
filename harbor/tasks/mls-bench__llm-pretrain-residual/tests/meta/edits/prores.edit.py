"""ProRes (Progressive Residual Warmup) baseline.

Scales each transformer block's contribution by a layer-dependent warmup
factor: alpha(layer, step) = min(step / (T * layer_idx), 1.0), where
T=1000 and layer_idx is 1-indexed.  Deep layers see heavily damped
updates early in training, gradually ramping to full contribution.

This stabilizes deep Pre-LN training by preventing early noisy updates
from deep layers from corrupting the residual stream.

Reference: "Progressive Residual Warmup" (arXiv:2603.05369, 2026)
  - reported perplexity gains on Pre-LN language models.

Changes:
  - Block: Unchanged — vanilla Pre-LN residual.
  - GPT.__init__: Add warmup period T and a step counter buffer.
  - GPT.forward: Progressive scaling of per-block residual contributions.
  - GPT.configure_optimizers: Unchanged (no new learnable params).
"""

_FILE = "nanoGPT/custom_pretrain.py"

# ── 1. GPT.__init__: ProRes parameters (lines 128-130) ───────────────────
_INIT = """\
        # ── ProRes: progressive residual warmup ──
        # T controls the warmup period; deeper layers take T*layer_idx steps
        # to reach full contribution.  step counter is a non-parameter buffer.
        self.prores_T = 1000
        self.register_buffer('_prores_step', torch.zeros(1, dtype=torch.long))
"""

# ── 2. GPT.forward: ProRes loop (lines 162-164) ──────────────────────────
_FORWARD_LOOP = """\
        # ── ProRes: progressive residual warmup per block ──
        # Increment step counter once per forward (training only).
        if self.training:
            self._prores_step += 1
        step = self._prores_step.item()
        T = self.prores_T
        for i, block in enumerate(self.transformer.h):
            block_out = block(x)
            if self.training and step < T * (i + 1):
                # alpha ramps from 0 to 1 over T * layer_idx steps
                layer_idx = i + 1
                alpha = min(step / (T * layer_idx), 1.0)
                # block_out = x + delta (Pre-LN residual), so delta = block_out - x
                x = x + alpha * (block_out - x)
            else:
                x = block_out
"""

# Ordered bottom-to-top so line numbers remain stable across ops.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 162,
        "end_line": 164,
        "content": _FORWARD_LOOP,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 128,
        "end_line": 130,
        "content": _INIT,
    },
]
