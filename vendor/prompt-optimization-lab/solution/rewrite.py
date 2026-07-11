"""Instruction PARAPHRASE vs from-scratch rewrite surface (agent-editable) — automatic prompt optimization.

Frozen Qwen2.5-0.5B-Instruct, inference-only, calibrated zero-shot execution;
disjoint train proposal/selection and full official evaluation. You design ONLY the `rewrite(seed, ctx) -> list[str]` paraphraser; everything else is
fixed. the scored pool is `[seed] + your rewrites`; to beat the fixed seed you must produce meaning-preserving paraphrases the small LM follows better, while echoing the seed (or emitting garbage) cannot improve over it. Keep it deterministic. Both complete official dataset settings run serially on one GPU.
Reference baselines: vendor/prompt-optimization-lab/baselines/
"""
from __future__ import annotations

import common  # noqa: F401


# === EDITABLE REGION — implement rewrite below ===
def rewrite(seed, ctx):
    # Weak: echo the seed unchanged — no paraphrase, so dev-selection just keeps the
    # seed and you cannot improve over the plain fixed instruction.
    return [seed]
# === END EDITABLE REGION ===
