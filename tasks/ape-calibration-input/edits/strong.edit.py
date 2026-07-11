"""Contentfree (strong) baseline edit for ape-calibration-input: replace the editable
function in prompt-optimization-lab/solution/calibration.py (lines 14-17).
Reference: vendor/prompt-optimization-lab/baselines/calibration-input_strong.py
2026-07-09 probe2 ARM D: ["N/A","the"] → sst2 strong 0.9100 (old 4-input strong only
reached 0.8333 on sst2 — fixed).
"""

_FILE = "prompt-optimization-lab/solution/calibration.py"

_CONTENT = r'''def calibration_inputs(ctx):
    # Strong: genuinely CONTENT-FREE inputs whose mean label distribution is a pure
    # prior — subtracting it debiases the dev ranking (Zhao et al. 2021) and surfaces
    # the candidate that also wins on the held-out test (fixed "N/A" test calibration).
    return ["N/A", "the"]'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 14, "end_line": 17, "content": _CONTENT},
]
