"""Prototypical Networks baseline — rigorous codebase edit ops.

Compute class prototypes as the mean of support features, then classify
query images based on negative Euclidean distance to prototypes.

Reference: easyfsl/methods/prototypical_networks.py

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "easy-few-shot-learning/custom_fewshot.py"

_PROTONET = """\
class CustomFewShotMethod(FewShotClassifier):
    \"\"\"Prototypical Networks (Snell et al., 2017).

    Compute class prototypes as the mean feature vector of support examples,
    then classify queries by negative Euclidean distance to prototypes.
    \"\"\"

    def __init__(self):
        backbone = make_backbone(use_pooling=True)
        super().__init__(backbone=backbone, use_softmax=False)

    def process_support_set(self, support_images: Tensor, support_labels: Tensor):
        self.compute_prototypes_and_store_support_set(support_images, support_labels)

    def forward(self, query_images: Tensor) -> Tensor:
        query_features = self.compute_features(query_images)
        scores = self.l2_distance_to_prototypes(query_features)
        return self.softmax_if_specified(scores)

    @staticmethod
    def is_transductive() -> bool:
        return False

    def compute_loss(self, scores: Tensor, labels: Tensor) -> Tensor:
        return F.cross_entropy(scores, labels)

"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 225,
        "end_line": 286,
        "content": _PROTONET,
    },
]
