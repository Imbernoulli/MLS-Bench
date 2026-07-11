"""Leaky (weak) baseline edit for ape-calibration-input: replace the editable
function in prompt-optimization-lab/solution/calibration.py (lines 14-17).
Reference: vendor/prompt-optimization-lab/baselines/calibration-input_weak.py
2026-07-09 probe2 ARM D: ["This review is wonderful. It is positive."] → sst2 weak 0.7600.
"""

_FILE = "prompt-optimization-lab/solution/calibration.py"

_CONTENT = r'''def calibration_inputs(ctx):
    # Weak: a LABEL-BIASED gushing sentence as the "content-free" input — it injects a
    # positive label prior into the dev calibration, so the ranking picks a distractor.
    return ["This review is wonderful. It is positive."]'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 14, "end_line": 17, "content": _CONTENT},
]
