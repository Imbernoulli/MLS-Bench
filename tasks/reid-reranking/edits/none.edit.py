"""Weak re-ranking: identity / no-op (return the raw cosine distances).
Reference: vendor/torchreid-reid/baselines/rerank_none.py
"""
_FILE = "torchreid-reid/solution/rerank.py"
_CONTENT = '''def build_rerank():
    def rerank(distmat, qf, gf):
        return distmat

    rerank.name = "no_rerank"
    return rerank'''
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 12, "end_line": 17, "content": _CONTENT},
]
