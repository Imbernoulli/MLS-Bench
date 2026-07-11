"""Shared PROPOSAL-vs-EVALUATION BUDGET ALLOCATION surface (agent-editable) — automatic prompt optimization.

Frozen Qwen2.5-0.5B-Instruct, inference-only, calibrated zero-shot execution;
disjoint train proposal/selection and full official evaluation. You design ONLY the `allocate(ctx) -> str` proposal/eval budget allocator; everything else is
fixed. ONE shared budget covers BOTH proposal (induction) and dev evaluation. Spending it all proposing many candidates leaves ~0 dev eval each (blind pick); balancing a few proposals with enough dev eval each picks the candidate that generalizes. The harness aborts on budget overrun. Keep it deterministic. Both complete official dataset settings run serially on one GPU.
Reference baselines: vendor/prompt-optimization-lab/baselines/
"""
from __future__ import annotations

import common  # noqa: F401


# === EDITABLE REGION — implement allocate below ===
def allocate(ctx):
    # Weak: spend the ENTIRE shared budget proposing candidates; nothing remains for
    # dev evaluation, so the choice is blind (first proposal) and rarely generalizes.
    cands = ctx["propose"](ctx["budget"])
    return cands[0] if cands else ""
# === END EDITABLE REGION ===
