"""Unmeasured full-protocol candidate: stripped source as the cluster key."""

_FILE = "code-generation-lab/solution/policy_consensus.py"
_CONTENT = '''def canonical(program):
    """Return a hashable key used to cluster one candidate program."""
    return (program or "").strip()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 5, "end_line": 7, "content": _CONTENT},
]
