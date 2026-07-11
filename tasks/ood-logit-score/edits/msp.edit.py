"""FLOOR logit score for ood-logit-score (the pristine start): MSP.

Maximum Softmax Probability (Hendrycks & Gimpel, ICLR 2017). Softmax normalises away the
absolute logit magnitude, so overconfident OOD inputs score as high as ID -> weak AUROC.
Reference: vendor/ood-detection-lab/baselines/msp.py
"""

_FILE = "ood-detection-lab/solution/logit_score.py"

_CONTENT = '''class Scorer:
    def fit(self, ctx):
        return self

    def score(self, logits):
        # MSP -- maximum softmax probability (WEAK; overconfident on OOD).
        import torch
        probs = torch.softmax(logits, dim=-1)
        return probs.max(dim=-1).values.numpy()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 13, "end_line": 18, "content": _CONTENT},
]
