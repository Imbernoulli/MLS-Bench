"""Literal-only read/write coupling plan for ``mamba-bc-coupling``.

Trusted verifier code builds the selected coupling; this file is never imported.
"""

def surface_config():
    return {"coupling": "tied"}
