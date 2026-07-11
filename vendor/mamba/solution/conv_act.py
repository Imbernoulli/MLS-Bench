"""Literal-only activation plan for ``mamba-conv-act``.

Trusted verifier code builds the selected activation; this file is never imported.
"""

def surface_config():
    return {"activation": "identity"}
