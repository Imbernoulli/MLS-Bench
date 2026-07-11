"""Weak re-ranking: identity / no-op (return the raw cosine distances).
Reference: vendor/torchreid-reid/baselines/rerank_none.py

Standalone reference implementation of this baseline (also applied as an edit).
"""


def build_rerank():
    def rerank(distmat, qf, gf):
        return distmat

    rerank.name = "no_rerank"
    return rerank
