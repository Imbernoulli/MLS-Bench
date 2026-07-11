"""Pseudo-class cosine-prototype score fitted only from ID training logits."""
import torch


class Scorer:
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
        return (normalized @ self.prototypes.T).max(dim=1).values.numpy()
