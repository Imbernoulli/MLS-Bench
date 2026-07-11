# Network Pruning: Layer Budget Allocation

## Research Question

Allocate a fixed global weight-sparsity budget across ResNet-18 layers. The
instruction does not assert which allocation is best.

## Implementation Contract

Modify `layer_sparsity(layer_names)` in
`prune-lab/solution/layer_budget.py`. Return a dictionary whose known-layer values
are finite sparsity ratios in `[0,1)`. Missing layers use the global target. The
harness parameter-weights the relative allocation and enforces the exact global
pruned-weight count.

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
