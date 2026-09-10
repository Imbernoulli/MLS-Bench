"""Evaluation metrics for directed causal graph recovery."""
import numpy as np


def compute_metrics(B_est, B_true, threshold=0.01):
    """Compute SHD, F1, precision, and recall for directed edge recovery.

    Convention: B[i, j] != 0 means j -> i.

    SHD definition (each type counts as exactly 1 error):
        - Reversed edge : correct skeleton edge but wrong direction
        - Extra edge    : present in estimate but absent in truth (non-reversal)
        - Missing edge  : present in truth but absent in estimate (non-reversal)

    F1 / precision / recall are computed on the directed edge set
    (skeleton + direction both must be correct for a true positive).

    Parameters
    ----------
    B_est     : ndarray (n, n)  estimated adjacency matrix
    B_true    : ndarray (n, n)  ground-truth adjacency matrix
    threshold : float           |B[i,j]| > threshold is treated as a present edge

    Returns
    -------
    dict with keys: shd (int), f1 (float), precision (float), recall (float)
    """
    def to_edge_set(B):
        mask = np.abs(B) > threshold
        if not mask.any():
            return set()
        return set(zip(*np.where(mask)))

    est  = to_edge_set(B_est)
    true = to_edge_set(B_true)

    tp     = len(est & true)
    fp_set = est - true
    fn_set = true - est

    # Reversed edges: (i,j) in fp_set AND (j,i) in fn_set
    reversed_edges = {(i, j) for (i, j) in fp_set if (j, i) in fn_set}
    extra_edges    = fp_set - reversed_edges
    missing_edges  = fn_set - {(j, i) for (i, j) in reversed_edges}

    shd       = len(reversed_edges) + len(extra_edges) + len(missing_edges)
    precision = tp / (tp + len(fp_set)) if (tp + len(fp_set)) > 0 else 0.0
    recall    = tp / (tp + len(fn_set)) if (tp + len(fn_set)) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {"shd": shd, "f1": f1, "precision": precision, "recall": recall}
