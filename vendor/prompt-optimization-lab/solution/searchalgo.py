"""Instruction SEARCH ALGORITHM (iterative / beam refinement under a dev budget) surface (agent-editable) — automatic prompt optimization.

Frozen Qwen2.5-0.5B-Instruct, inference-only, calibrated zero-shot execution;
disjoint train proposal/selection and full official evaluation. You design ONLY the `search(ctx) -> str` budgeted generative search; everything else is
fixed. there is NO fixed pool — you generate candidates (induce/paraphrase) and search under a DEV-evaluation budget. A one-shot blind proposal generalizes only by luck; an iterative/beam search that uses dev feedback to keep and refine the best surfaces an instruction that generalizes. Keep it deterministic. Both complete official dataset settings run serially on one GPU.
Reference baselines: vendor/prompt-optimization-lab/baselines/
"""
from __future__ import annotations

import common  # noqa: F401


# === EDITABLE REGION — implement search below ===
def search(ctx):
    # Weak: propose exactly ONE candidate and return it blindly — no dev evaluation,
    # no refinement; generalizes only by luck.
    cands = ctx["induce"](1)
    return cands[0] if cands else ""
# === END EDITABLE REGION ===
