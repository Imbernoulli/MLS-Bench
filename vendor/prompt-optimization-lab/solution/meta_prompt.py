"""META-PROMPT design for reverse-mode instruction induction surface (agent-editable) — automatic prompt optimization.

Frozen Qwen2.5-0.5B-Instruct, inference-only, calibrated zero-shot execution;
disjoint train proposal/selection and full official evaluation. You design ONLY the `meta_prompt(examples, ctx) -> str` induction-prompt designer; everything else is
fixed. the harness fills `{demo}` and `{labels}` and induces under YOUR meta-prompt over fixed balanced example sets; a vague prompt elicits off-task text, while a structured reverse-mode prompt elicits clean task instructions. Keep it deterministic. Both complete official dataset settings run serially on one GPU.
Reference baselines: vendor/prompt-optimization-lab/baselines/
"""
from __future__ import annotations

import common  # noqa: F401


# === EDITABLE REGION — implement meta_prompt below ===
def meta_prompt(examples, ctx):
    # Weak: a vague prompt — the LM rambles off-task instead of emitting a clean task
    # instruction, so induction yields noise and dev-selection has nothing good.
    return "Say something about these examples."
# === END EDITABLE REGION ===
