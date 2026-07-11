"""Medium re-ranking: Average Query Expansion (AQE).

A lightweight, widely-used retrieval trick (Chum et al., "Total Recall", 2007):
each query is augmented by the mean of its top-k gallery neighbours, then re-scored.
Cheaper than k-reciprocal and a solid intermediate. Reference:
vendor/torchreid-reid/baselines/rerank_aqe.py
"""
_FILE = "torchreid-reid/solution/rerank.py"
_CONTENT = '''def build_rerank():
    import numpy as np

    def rerank(distmat, qf, gf, topk=5):
        # for each query, average it with its top-k gallery neighbours, renormalise
        order = np.argsort(distmat, axis=1)
        new_q = np.zeros_like(qf)
        for i in range(qf.shape[0]):
            nn = order[i, :topk]
            v = qf[i] + gf[nn].sum(0)
            n = np.linalg.norm(v) + 1e-12
            new_q[i] = v / n
        return 1.0 - new_q @ gf.T

    rerank.name = "aqe"
    return rerank'''
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 12, "end_line": 17, "content": _CONTENT},
]
