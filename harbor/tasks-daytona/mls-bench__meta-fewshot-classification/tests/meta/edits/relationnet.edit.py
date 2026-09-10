"""Relation Networks baseline — rigorous codebase edit ops.

Concatenates query and prototype feature maps and feeds them through a
relation module (CNN) to produce relation scores. Uses MSE loss.

Reference: easyfsl/methods/relation_networks.py

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "easy-few-shot-learning/custom_fewshot.py"

_RELATIONNET = """\
class _RelationModule(nn.Module):
    \"\"\"CNN relation module from Sung et al. (2018).\"\"\"

    def __init__(self, feature_dimension: int, inner_channels: int = 8):
        super().__init__()
        self.module = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(feature_dimension * 2, feature_dimension, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature_dimension, momentum=1, affine=True),
                nn.ReLU(),
                nn.AdaptiveMaxPool2d((5, 5)),
            ),
            nn.Sequential(
                nn.Conv2d(feature_dimension, feature_dimension, kernel_size=3, padding=0),
                nn.BatchNorm2d(feature_dimension, momentum=1, affine=True),
                nn.ReLU(),
                nn.AdaptiveMaxPool2d((1, 1)),
            ),
            nn.Flatten(),
            nn.Linear(feature_dimension, inner_channels),
            nn.ReLU(),
            nn.Linear(inner_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.module(x)


class CustomFewShotMethod(FewShotClassifier):
    \"\"\"Relation Networks (Sung et al., 2018).

    Extracts feature maps (not pooled vectors) from support and query images.
    Computes class prototypes as mean feature maps, concatenates each query-prototype
    pair, and feeds them through a learned relation module to get relation scores.
    Uses MSE loss since output represents relation scores in [0, 1].
    \"\"\"

    def __init__(self):
        backbone = make_backbone(use_pooling=False)  # Need feature maps, not vectors
        super().__init__(backbone=backbone, use_softmax=False)
        self.feature_dimension = FEATURE_DIMENSION
        self.relation_module = _RelationModule(self.feature_dimension)

    def process_support_set(self, support_images: Tensor, support_labels: Tensor):
        support_features = self.compute_features(support_images)
        n_way = len(torch.unique(support_labels))
        self.prototypes = torch.cat(
            [
                support_features[support_labels == label].mean(0, keepdim=True)
                for label in range(n_way)
            ]
        )

    def forward(self, query_images: Tensor) -> Tensor:
        query_features = self.compute_features(query_images)
        n_queries = query_features.shape[0]
        n_prototypes = self.prototypes.shape[0]

        # Build pairs: [n_queries * n_prototypes, 2 * C, H, W]
        query_prototype_pairs = torch.cat(
            (
                self.prototypes.unsqueeze(0).expand(n_queries, -1, -1, -1, -1),
                query_features.unsqueeze(1).expand(-1, n_prototypes, -1, -1, -1),
            ),
            dim=2,
        ).view(-1, 2 * self.feature_dimension, *query_features.shape[2:])

        relation_scores = self.relation_module(query_prototype_pairs).view(
            n_queries, n_prototypes
        )
        return self.softmax_if_specified(relation_scores)

    @staticmethod
    def is_transductive() -> bool:
        return False

    def compute_loss(self, scores: Tensor, labels: Tensor) -> Tensor:
        # RelationNet uses MSE with one-hot labels
        one_hot = F.one_hot(labels, num_classes=scores.shape[1]).float()
        return F.mse_loss(scores, one_hot)

"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 225,
        "end_line": 286,
        "content": _RELATIONNET,
    },
]
