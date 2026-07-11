"""Literal-only state-initialization plan for ``mamba-state-init``.

Trusted verifier code applies the selected scheme; this file is never imported.
"""

def surface_config():
    return {"scheme": "constant_rate"}
