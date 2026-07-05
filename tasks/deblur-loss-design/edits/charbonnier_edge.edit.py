"""DEPRECATED alias -- superseded by edits/sharp.edit.py (the strong baseline).

Kept only so any tooling that enumerates edit files finds a VALID op targeting the
current editable region. The live loss lever is the sharp-vs-over-smoothed TARGET
(see solution/loss.py); this alias applies the strong (true sharp target) baseline.
"""

_FILE = "image-deblur/solution/loss.py"

_CONTENT = '''def get_loss_config():
    # Sharp target (no over-smoothing) + Charbonnier + edge -> high deblur PSNR.
    return {"kind": "charbonnier", "edge_weight": 0.1, "target_smooth": 0.0}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 35, "end_line": 37, "content": _CONTENT},
]
