"""Degenerate baseline: lowercase everything (mismatches cased refs)."""
def build_postproc() -> str:
    return "lowercase"
