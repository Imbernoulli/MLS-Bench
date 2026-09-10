# Meta-Learning: Few-Shot Image Classification

## Research Question
Design a novel few-shot image classifier that, given a small support set of N classes with K labeled examples each, generalizes to query examples of those classes. The contribution should be a reusable algorithmic component (a way of summarizing the support set, comparing query to support, or doing task-level adaptation), not a dataset-specific trick.

## Background
Few-shot classification recognizes new classes from a handful of labeled examples. Episodic evaluation samples N-way K-shot tasks: K support examples per class, then unlabeled query images to classify into one of the N classes. Common design axes:
- **Feature comparison**: Euclidean distance, cosine similarity, learned metric.
- **Support encoding**: per-class prototypes, attention, graph neural networks.
- **Query adaptation**: cross-attention, transductive inference, LSTM context.

Reference baselines (provided as read-only modules):
- **Prototypical Networks** — Snell, Swersky, Zemel, NeurIPS 2017 ([arXiv:1703.05175](https://arxiv.org/abs/1703.05175)). Class prototype = mean embedding of support; query classified by negative squared Euclidean distance.
- **Matching Networks** — Vinyals, Blundell, Lillicrap, Kavukcuoglu, Wierstra, NeurIPS 2016 ([arXiv:1606.04080](https://arxiv.org/abs/1606.04080)). Cosine attention over support embeddings; weighted-sum label prediction (no fine-tuning at test time).
- **Relation Networks** — Sung, Yang, Zhang, Xiang, Torr, Hospedales, CVPR 2018 ([arXiv:1711.06025](https://arxiv.org/abs/1711.06025)). A learned MLP scores the relation between query feature and class prototype, replacing fixed metrics.

## Model Interface
Implement `CustomFewShotMethod` in `custom_fewshot.py`:
```python
class CustomFewShotMethod(FewShotClassifier):
    def __init__(self):
        backbone = make_backbone(use_pooling=True)  # ResNet-12, 640-dim features
        super().__init__(backbone=backbone)

    def process_support_set(self, support_images, support_labels):
        # Extract and store support set information for forward()
        ...

    def forward(self, query_images) -> Tensor:
        # Return classification scores of shape (n_query, n_way)
        ...

    def compute_loss(self, scores, labels) -> Tensor:
        # Default: cross-entropy
        ...
```

## Available Utilities
- `self.compute_features(images)` — pass through `self.backbone`.
- `self.l2_distance_to_prototypes(features)` — negative Euclidean distance to `self.prototypes`.
- `self.cosine_distance_to_prototypes(features)` — cosine similarity to `self.prototypes`.
- `compute_prototypes(features, labels)` — mean feature per class.
- `self.compute_prototypes_and_store_support_set(images, labels)` — convenience method.
- `make_backbone(use_pooling=True/False)` — ResNet-12 with 640-dim feature vector or feature maps.

## Fixed Training & Evaluation Pipeline
- Backbone: ResNet-12 (640-dim).
- Episodic training: 500 tasks/epoch for 200 epochs, 5-way 5-shot tasks.
- Evaluation: mean classification accuracy over 600 test episodes per benchmark (higher is better).
- Benchmarks: **miniImageNet** (100 ImageNet classes), **CIFAR-FS** (100 classes from CIFAR-100), **CUB-200** (200 fine-grained bird species). All evaluated 5-way 5-shot.
