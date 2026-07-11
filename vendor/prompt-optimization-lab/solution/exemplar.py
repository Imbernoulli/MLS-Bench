"""Few-shot EXEMPLAR selection for reverse-mode induction surface (agent-editable) — automatic prompt optimization.

Frozen Qwen2.5-0.5B-Instruct, inference-only, calibrated zero-shot execution;
disjoint train proposal/selection and full official evaluation. You design ONLY the `select_exemplars(pool, ctx) -> list[row]` exemplar selector; everything else is
fixed. the harness induces one candidate from each of several shuffles of YOUR exemplar set; a single/random exemplar yields a narrow, noisy induction, while a small label-balanced diverse set yields a robust instruction. Keep it deterministic. Both complete official dataset settings run serially on one GPU.
Reference baselines: vendor/prompt-optimization-lab/baselines/
"""
from __future__ import annotations

import common  # noqa: F401


# === EDITABLE REGION — implement select_exemplars below ===
def select_exemplars(pool, ctx):
    # Weak: a single arbitrary exemplar — narrow, noisy induction that rarely yields
    # a generalizing instruction.
    return [pool[0]]
# === END EDITABLE REGION ===
