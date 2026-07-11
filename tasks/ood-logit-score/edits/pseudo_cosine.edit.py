"""Strong ID-only fitted logit anchor: pseudo-class cosine prototypes."""

_FILE = "ood-detection-lab/solution/logit_score.py"

_CONTENT = '''class Scorer:
    def fit(self, ctx):
        logits = ctx.tr_logits.detach().float()
        pseudo_labels = logits.argmax(dim=1)
        prototypes = []
        for label in range(ctx.num_classes):
            members = logits[pseudo_labels == label]
            if members.shape[0] == 0:
                raise RuntimeError(f"ID-fit logits have no pseudo-class {label}")
            prototypes.append(members.mean(dim=0))
        self.prototypes = torch.nn.functional.normalize(
            torch.stack(prototypes), dim=1,
        )
        return self

    def score(self, logits):
        normalized = torch.nn.functional.normalize(logits.float(), dim=1)
        return (normalized @ self.prototypes.T).max(dim=1).values.numpy()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 13, "end_line": 18, "content": _CONTENT},
]
