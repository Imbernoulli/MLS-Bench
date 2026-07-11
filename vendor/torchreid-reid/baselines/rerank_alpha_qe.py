"""Memory-bounded alpha-weighted query expansion for full Market-1501.

Each query is expanded with its most similar gallery descriptors using the
standard alpha query-expansion weighting used in image retrieval. Processing
queries in bounded chunks avoids constructing a gallery-by-gallery matrix.
"""


def build_rerank():
    import numpy as np

    def rerank(distmat, qf, gf, topk=10, alpha=3.0, chunk_size=64):
        if distmat.ndim != 2 or qf.ndim != 2 or gf.ndim != 2:
            raise ValueError("re-ranking inputs must be rank-2 arrays")
        if distmat.shape != (qf.shape[0], gf.shape[0]):
            raise ValueError("distance and feature inventories disagree")
        if qf.shape[1] != gf.shape[1] or gf.shape[0] < topk:
            raise ValueError("feature dimensions or gallery size are invalid")

        expanded = np.empty_like(qf, dtype=np.float32)
        for start in range(0, qf.shape[0], chunk_size):
            stop = min(start + chunk_size, qf.shape[0])
            block = distmat[start:stop]
            indices = np.argpartition(block, kth=topk - 1, axis=1)[:, :topk]
            distances = np.take_along_axis(block, indices, axis=1)
            order = np.lexsort((indices, distances), axis=1)
            indices = np.take_along_axis(indices, order, axis=1)
            distances = np.take_along_axis(distances, order, axis=1)

            weights = np.maximum(1.0 - distances, 0.0) ** alpha
            weight_sum = weights.sum(axis=1, keepdims=True)
            normalized = np.full_like(weights, 1.0 / topk, dtype=np.float32)
            valid = weight_sum[:, 0] > 1e-12
            normalized[valid] = weights[valid] / weight_sum[valid]
            neighbours = np.sum(
                gf[indices] * normalized[:, :, np.newaxis], axis=1
            )
            vectors = qf[start:stop] + neighbours
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
                raise RuntimeError("query expansion produced an invalid norm")
            expanded[start:stop] = vectors / norms

        result = 1.0 - expanded @ gf.T
        return result.astype(np.float32, copy=False)

    rerank.name = "alpha_qe"
    return rerank

