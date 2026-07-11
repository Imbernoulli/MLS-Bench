"""Strong baseline for ape-paraphrase-rewrite (Instruction PARAPHRASE vs from-scratch rewrite).

Reference: pasted into solution/rewrite.py via the edit op.
"""

import common  # noqa: F401


def rewrite(seed, ctx):
    # Strong: LM-paraphrase the seed (meaning-preserving rewrites) plus a couple of
    # crisp task descriptions; dev-selection picks whichever the small LM follows best.
    out = [seed]
    out += common.paraphrase_instruction(ctx["executor"], seed, 4, seed=42)
    if ctx["dataset"].task == "sentiment":
        out += ["Read the review and judge whether the opinion expressed is favorable or unfavorable.",
                "Decide whether the reviewer liked or disliked the movie."]
    else:
        out += ["Read the news article and identify which subject area it belongs to.",
                "Determine the general topic that best describes the news text below."]
    seen, uniq = set(), []
    for c in out:
        k = c.strip().lower()
        if k and k not in seen:
            seen.add(k); uniq.append(c.strip())
    return uniq
