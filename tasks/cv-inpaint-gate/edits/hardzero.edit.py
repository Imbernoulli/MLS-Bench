"""Unmeasured full-protocol anchor candidate.

Retained for worker-side calibration. No score, ordering, or full-protocol runtime
is asserted before a complete 256x256, 100,000-step run.
"""

_FILE = "image-inpainting/solution/gate.py"

_CONTENT = '''    hard = (gate > 0.7).float()          # high threshold -> drop most locations
    return feat * hard * mask * 0.5      # also zero the hole + halve -> destroys signal'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 17, "end_line": 17, "content": _CONTENT},
]
