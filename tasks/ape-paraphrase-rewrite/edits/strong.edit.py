"""Paraphrase baseline edit for ape-paraphrase-rewrite: replace the editable
function in prompt-optimization-lab/solution/rewrite.py (lines 14-17).
Reference: vendor/prompt-optimization-lab/baselines/
"""

_FILE = "prompt-optimization-lab/solution/rewrite.py"

_CONTENT = r'''def rewrite(seed, ctx):
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
    return uniq'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 14, "end_line": 17, "content": _CONTENT},
]
