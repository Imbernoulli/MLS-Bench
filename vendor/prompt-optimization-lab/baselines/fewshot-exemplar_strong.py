"""Strong baseline for ape-fewshot-exemplar (Few-shot EXEMPLAR selection for reverse-mode induction).

Reference: pasted into solution/exemplar.py via the edit op.
"""

import common  # noqa: F401


def select_exemplars(pool, ctx):
    # Strong: a small LABEL-BALANCED, shuffled exemplar set (one per class, topped up)
    # — diverse conditioning that induces a robust, generalizing instruction.
    rng = ctx["rng"]
    by = {}
    for r in pool:
        by.setdefault(r["label"], []).append(r)
    labels = sorted(by)
    for v in by.values():
        rng.shuffle(v)
    out, i = [], 0
    while len(out) < 5 and any(by[l] for l in labels):
        lab = labels[i % len(labels)]
        if by[lab]:
            out.append(by[lab].pop())
        i += 1
    return out[:5]
