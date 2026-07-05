"""Mid / weak-real baseline (score floor) for vfi-synthesis: motion-compensated flow-warp.

Estimate the bidirectional flow to t=0.5 with a learnable flow net, backward-warp frame0 and
frame2, and average the two motion-compensated candidates (fixed 0.5 blend). Compensates the
bulk motion (far above the naive blend) but its blind 0.5-average still ghosts at
disocclusion boundaries -> below the learned refinement. Reference:
vendor/video-frame-interp/baselines/synthesis_flow_warp.py
"""

_FILE = "video-frame-interp/solution/synthesis.py"

_CONTENT = '''def get_synthesis_config():
    # Flow-warp both frames to t=0.5 + fixed-0.5 average (motion-compensated, no learned mask).
    return {"method": "flow_warp"}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 34, "end_line": 36, "content": _CONTENT},
]
