# Sparse Adversarial Attack (L0 Constraint)

## Research Question
Can you design a stronger sparse adversarial attack that fools image classifiers by perturbing only a small number of spatial pixels?

## Background
Sparse adversarial attacks differ from dense `L_p` attacks in that the perturbation is restricted in `L0` rather than `L_inf` or `L2`: only a handful of input pixels may be modified, but each modified pixel can change by an arbitrary amount within `[0, 1]`. The sparsity constraint matches threat models such as physical patches, image-tag manipulation, and pixel-level corruption, and it is also informative because gradient-based attacks tend to spread perturbations across many pixels and are not well suited to it.

Representative sparse-attack algorithms include the Jacobian Saliency Map Attack JSMA (Papernot et al., 2016, arXiv:1511.07528), the differential-evolution One-Pixel attack (Su et al., 2019, arXiv:1710.08864), the geometry-inspired SparseFool (Modas et al., CVPR 2019, arXiv:1811.02248), the random-search Sparse-RS framework (Croce et al., AAAI 2022, arXiv:2006.12834), and Pixle, a fast pixel-rearrangement black-box attack (Pomponi et al., 2022, arXiv:2202.02236).

## Objective
Implement a stronger sparse attack in `bench/custom_attack.py`. The method should maximize attack success rate (ASR) under a strict `L0` perturbation budget:

- Threat model: full model access for custom attack implementation (gradients permitted).
- Norm constraint: number of modified spatial pixels is bounded.
- Budget: `L0(x_adv, x) <= pixels`, where `pixels = 24` (the canonical Sparse-RS CIFAR-10 L0 budget, Croce et al., AAAI 2022). A pixel is counted as modified if any of its channels changes.

## Editable Interface
You must implement:

`run_attack(model, images, labels, pixels, device, n_classes) -> adv_images`

Inputs:
- `images`: tensor of shape `(N, C, H, W)`, values in `[0, 1]`.
- `labels`: tensor of shape `(N,)`.
- `pixels`: maximum number of modified spatial pixels per sample.
- `n_classes`: 10 (CIFAR-10).

Output:
- `adv_images`: same shape as `images`, also in `[0, 1]`.

## Evaluation Protocol
Each evaluation script:
1. Loads one adversarially-robust pretrained model.
2. Collects up to 150 samples that are initially classified correctly.
3. Runs your `run_attack`.
4. Checks `L0` validity (`<= pixels` modified spatial pixels) and `[0, 1]` range.
5. Reports `clean_acc`, `robust_acc`, and `asr = 1 - robust_acc`.

Important:
- ASR denominator is the number of initially correct samples.
- Invalid adversarial outputs (shape mismatch, non-finite values, or violated budget) are treated as failure.

## Evaluation Scenarios
Three CIFAR-10 settings, each an adversarially-robust target model from the
RobustBench L2 model zoo (this is the canonical Sparse-RS L0 threat model,
Croce et al., AAAI 2022, Table 2 / App. A.5 — on standard undefended models
a strong L0 attack trivially saturates):

- `Rebuffi-R18-L2`: l2-AT PreActResNet-18 (Rebuffi et al., 2021) — the exact model the Sparse-RS paper evaluates L0 against.
- `Augustin-L2`: l2-robust model (Augustin et al., 2020).
- `Engstrom-L2`: l2-robust model (Engstrom et al., 2019).

## Baselines
The baselines below run inside the same harness via edit ops; reference implementations are in `torchattacks`:

- `onepixel`: One-Pixel attack (Su et al., 2019, arXiv:1710.08864). Differential-evolution sparse attack with default population and iteration settings from the paper.
- `sparsefool`: SparseFool (Modas et al., CVPR 2019, arXiv:1811.02248). Geometry-inspired sparse attack using DeepFool-like linearization.
- `jsma`: JSMA (Papernot et al., 2016, arXiv:1511.07528). Jacobian saliency map-based targeted sparse attack.
- `pixle`: Pixle (Pomponi et al., 2022, arXiv:2202.02236). Pixel-rearrangement-based black-box sparse attack.
- `sparse_rs`: Sparse-RS (Croce et al., AAAI 2022, arXiv:2006.12834). Random-search L0 attack from https://github.com/fra31/sparse-rs.

The goal is to improve ASR while respecting the L0 budget.
