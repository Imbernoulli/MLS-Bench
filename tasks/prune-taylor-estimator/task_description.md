# Network Pruning: Data-Aware Importance Estimation

## Research Question

Design how the complete supplied calibration pass is used to estimate a
non-negative importance tensor for every prunable parameter.

## Implementation Contract

Modify `estimate_importance(model, batches, params)` in
`prune-lab/solution/taylor_estimator.py`. Return a dictionary containing one
finite, non-negative, same-shape tensor for every `(name, parameter)` entry.

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
