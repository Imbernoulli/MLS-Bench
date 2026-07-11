"""Select the abstractive source policy for the measured reference run."""

OPS = [
    {
        "op": "replace",
        "file": "abstractive-summarization/solution/source.py",
        "start_line": 25,
        "end_line": 27,
        "content": "def build_source_policy() -> str:\n    return \"abstractive\"",
    }
]
