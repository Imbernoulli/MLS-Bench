"""Alternative strong baseline: dev ANSWER LOG-LIKELIHOOD estimator.

Rank each candidate by the mean gold-label log-probability on the DEV set (a smoother
selection signal than 0/1 accuracy — Zhou et al. discuss log-prob as a candidate
estimator). Like execution-accuracy it surfaces a genuinely good instruction and
avoids the misleading distractors, generalizing to the held-out TEST set.
Reference: vendor/prompt-optimization-lab/baselines/scoring_loglik.py
"""

import numpy as np


def score_candidate(instruction, ctx) -> float:
    dev = ctx["dev"]
    mat = ctx["executor"].label_logprob_matrix(instruction, dev)  # [n, n_class]
    gold = np.array([r["label"] for r in dev])
    return float(np.mean(mat[np.arange(len(dev)), gold]))
