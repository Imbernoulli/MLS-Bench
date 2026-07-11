"""Strong baseline for ape-candidate-generation (Candidate GENERATION (the APE proposer)).

Reference: pasted into solution/propose.py via the edit op.
"""

import common  # noqa: F401


def propose(ctx):
    # Strong: LM-induce diverse candidates from the labeled pool (reverse-mode APE)
    # and add a few hand-written task descriptions; fixed dev-selection then surfaces
    # the one that generalizes to the held-out test set.
    ds = ctx["dataset"]
    cands = list(ctx["induce_instructions"](6))
    if ds.task == "sentiment":
        cands += ["Read the review and judge whether the opinion expressed is favorable or unfavorable.",
                  "Decide whether the reviewer liked or disliked the movie.",
                  "Determine the overall attitude of the review toward the film."]
    else:
        cands += ["Read the news article and identify which subject area it belongs to.",
                  "Classify this news story into its news section by its main subject.",
                  "Determine the general topic that best describes the news text below."]
    seen, out = set(), []
    for c in cands:
        k = (c or "").strip().lower()
        if k and k not in seen:
            seen.add(k); out.append(c.strip())
    return out
