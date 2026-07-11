# Network Pruning: Curvature-Aware Importance

## Research Question

Design a non-negative per-weight importance using the supplied weight, gradient, and
Fisher-diagonal proxy.

## Implementation Contract

Modify `importance2(name, weight, grad, fisher)` in
`prune-lab/solution/second_order.py`. Return a finite, non-negative tensor with the
same shape and device as `weight`.

## Fixed Protocol

- Complete CIFAR-10 is required: 50,000 training images and 10,000 held-out test
  images. The model is torchvision ResNet-18 with a CIFAR stem.
- A pinned checkpoint carrying the `cifar10-resnet18-200ep-v1` provenance is
  loaded strictly and its reported dense accuracy is recomputed on all 10,000 test
  images before pruning.
- The harness enforces the declared global sparsity or structured MAC budget and a
  complete 160-epoch recovery budget. Missing required training, incomplete
  schedules, or a structured model that does not reduce measured MACs is invalid.
- Settings `cifar10` (seed 42) and `cifar10_seed1` (seed 1) both participate in
  scoring. Each setting processes the full data inventory.
- The scored metric is final held-out accuracy. Enforced sparsity, dense accuracy,
  parameter count, and measured MACs are diagnostic proof.
- Missing, crashing, wrong-shape, out-of-range, negative, or non-finite editable
  output invalidates verification. The harness never substitutes another
  implementation.
