"""Strong baseline: APE (Zhou et al. 2022) — propose LM-induced candidates from the
labeled pool, add one hand-written task description, and SELECT the candidate with
the highest DEV execution-accuracy.

Reverse-mode instruction induction (Honovich et al. 2022) proposes candidates by
showing the frozen LM few-shot input/output examples and asking for the instruction;
selection by dev execution-accuracy (the canonical APE score) surfaces the candidate
that actually helps the small LM. The chosen instruction is scored on the disjoint
held-out TEST set, so a dev-overfit candidate does not win.
Reference: vendor/prompt-optimization-lab/baselines/search_ape.py
"""


def optimize(ctx) -> str:
    ex = ctx["executor"]
    ds = ctx["dataset"]
    dev = ctx["dev"]
    # Propose: LM-induced candidates + a few hand-written task-specific anchors.
    cands = list(ctx["induce_instructions"](5))
    if ds.task == "sentiment":
        cands += [
            "Read the review and judge whether the opinion expressed is favorable or unfavorable.",
            "Decide whether the reviewer liked or disliked the movie.",
            "Determine the overall attitude of the review toward the film.",
        ]
    else:
        cands += [
            "Read the news article and identify which subject area it belongs to.",
            "Classify this news story into its news section by its main subject.",
            "Determine the general topic that best describes the news text below.",
        ]
    # Deduplicate.
    seen, uniq = set(), []
    for c in cands:
        k = (c or "").strip().lower()
        if k and k not in seen:
            seen.add(k); uniq.append(c.strip())
    # Select: highest dev execution-accuracy (APE selection metric).
    best, best_acc = uniq[0], -1.0
    for c in uniq:
        a = ex.dev_accuracy(c, dev)
        if a > best_acc:
            best_acc, best = a, c
    return best
