"""Strong reference candidate: deterministic label-balanced exemplars."""

_FILE = "prompt-optimization-lab/solution/exemplar.py"
_CONTENT = '''def select_exemplars(pool, ctx):
    # Select a deterministic label-balanced set for reverse-mode induction.
    rng = ctx["rng"]
    by_label = {}
    for row in pool:
        by_label.setdefault(row["label"], []).append(row)
    for rows in by_label.values():
        rng.shuffle(rows)
    labels = sorted(by_label)
    selected = []
    while len(selected) < 5:
        label = labels[len(selected) % len(labels)]
        selected.append(by_label[label].pop())
    return selected'''
OPS = [{"op": "replace", "file": _FILE, "start_line": 14, "end_line": 17, "content": _CONTENT}]
