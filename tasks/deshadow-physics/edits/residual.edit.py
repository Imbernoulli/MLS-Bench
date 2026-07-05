"""WEAK physics baseline: predict a free 3-ch residual added to the shadowed input (unconstrained, does not respect the multiplicative illumination structure).
Reference: vendor/image-deshadow/solution/physics.py
"""

_FILE = "image-deshadow/solution/physics.py"

_CONTENT = 'def get_physics_config():\n    # Weak: free unconstrained 3-ch residual.\n    return {"mode": "residual"}'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 30, "end_line": 32, "content": _CONTENT},
]
