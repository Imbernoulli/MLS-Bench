# Meta-Learning: Inner-Loop Optimization Algorithm Design

## Research Question
Design a novel inner-loop adaptation algorithm for gradient-based meta-learning. The contribution is the *adaptation rule itself* (which parameters change, how gradients are scaled or transformed, what state is carried across inner steps), not changes to the data loader, backbone, or outer-loop schedule.

## Background
Gradient-based meta-learning (MAML-style) learns a model initialization that can be quickly adapted to new tasks via a few gradient steps. The **inner loop** is the adaptation step on the support set; the **outer loop** optimizes the initialization (and any optimizer state) across tasks.

Reference baselines (provided as read-only modules in `learn2learn/algorithms/`):
- **MAML** — Finn, Abbeel, Levine, ICML 2017 ([arXiv:1703.03400](https://arxiv.org/abs/1703.03400)). Inner loop = differentiable SGD with a fixed scalar learning rate; outer loop optimizes only the initialization.
- **Meta-SGD** — Li, Zhou, Chen, Li, 2017 ([arXiv:1707.09835](https://arxiv.org/abs/1707.09835)). Per-parameter learnable inner-loop learning rates (one rate vector per parameter tensor), meta-trained jointly with the initialization.
- **ANIL (Almost No Inner Loop)** — Raghu, Raghu, Bengio, Vinyals, ICLR 2020 ([arXiv:1909.09157](https://arxiv.org/abs/1909.09157)). Adapts only the classification head in the inner loop; the feature extractor is frozen during adaptation, exploiting feature reuse.

Design axes worth considering:
- **What to adapt**: all parameters, head only, learned subset/mask.
- **How to scale gradients**: fixed LR, per-parameter LR, preconditioning matrix, learned transform.
- **Memory across inner steps**: momentum-like state, recurrent updates, second-order info.
- **Regularization**: trust-region constraints, support-set overfitting penalties.

## Implementation Contract
Modify `InnerLoopOptimizer` in `learn2learn/custom_maml.py`:

```python
class InnerLoopOptimizer:
    def __init__(self, model: nn.Module, inner_lr: float):
        # model: base model (for parameter shape inspection)
        # inner_lr: default learning rate
        # Create any learnable parameters here.
        ...

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        # model is a CLONE — safe to modify in-place.
        # MUST use differentiable ops (torch.autograd.grad), NOT torch.optim.
        # Return the adapted model.
        ...

    def meta_parameters(self) -> List[Tensor]:
        # Learnable optimizer parameters for the outer loop.
        # Return [] if optimizer has no learnable state (vanilla MAML).
        ...
```

Available reference code: `learn2learn/algorithms/maml.py`, `meta_sgd.py`, `gbml.py`.

## Fixed Pipeline & Evaluation
- Backbone: CNN4.
- Meta-training: 60,000 iterations, 4 tasks per meta-batch.
- Inner loop: 5 steps during training, 10 steps during evaluation.
- Benchmarks: **miniImageNet 5-way 1-shot**, **miniImageNet 5-way 5-shot**, **CIFAR-FS 5-way 5-shot**.
- Metric: mean classification accuracy over 600 test episodes (higher is better).
