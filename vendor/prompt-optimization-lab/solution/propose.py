"""Candidate GENERATION (the APE proposer) surface (agent-editable) — automatic prompt optimization.

Frozen Qwen2.5-0.5B-Instruct, inference-only, calibrated zero-shot execution;
disjoint train proposal/selection and full official evaluation. You design ONLY the `propose(ctx) -> list[str]` pool generator; everything else is
fixed. the candidate POOL and fixed dev-selection pick whichever instruction your proposer surfaces; a thin/generic pool gives selection nothing good, while a diverse LM-induced pool from the labeled pool surfaces a generalizing instruction. Keep it deterministic. Both complete official dataset settings run serially on one GPU.
Reference baselines: vendor/prompt-optimization-lab/baselines/
"""
from __future__ import annotations

import common  # noqa: F401


# === EDITABLE REGION — implement propose below ===
def propose(ctx):
    # Weak: a single generic instruction — fixed dev-selection has nothing useful
    # to pick, so it returns a vague prompt that lands near the class prior.
    return ["Classify the text."]
# === END EDITABLE REGION ===
