# White-Box Evasion Attack under Linf Constraint

## Research Question
Can you design a stronger white-box `L_inf` evasion attack that increases attack success rate (ASR) under a small `eps` budget, where weak attacks already saturate near 100% on undefended models but strong baselines (PGD, AutoAttack) leave headroom on some architectures?

## Background
White-box evasion attacks assume the attacker has full access to the model, including its parameters and gradients. The classical first-order attack is FGSM (Goodfellow et al., 2015, arXiv:1412.6572), a one-step sign-of-gradient attack. Iterative variants such as PGD (Madry et al., 2018, arXiv:1706.06083) and momentum iterative FGSM, MI-FGSM (Dong et al., CVPR 2018, arXiv:1710.06081), refine the perturbation through multiple gradient steps. AutoAttack (Croce and Hein, ICML 2020, arXiv:2003.01690) is a parameter-free ensemble of two adaptive PGD variants together with FAB and Square Attack and is widely used as a strong reference attack.

## Objective
Implement a stronger white-box `L_inf` attack in `torchattacks/bench/custom_attack.py`. The method should maximize ASR under a strict perturbation budget:

- Threat model: white-box (full model access, including gradients).
- Norm constraint: `||x_adv - x||_inf <= eps`.
- Budget: `eps = 2/255`. RobustBench uses `8/255` for *defended* models, which saturates ASR to ~1.0 on undefended models and leaves no headroom for agents; the `2/255` regime is used here to differentiate attack quality on undefended classifiers.

## Editable Interface
You must implement:

`run_attack(model, images, labels, eps, device, n_classes) -> adv_images`

Inputs:
- `images`: tensor of shape `(N, C, H, W)`, values in `[0, 1]`.
- `labels`: tensor of shape `(N,)`.
- `n_classes`: 10 for CIFAR-10, 100 for CIFAR-100.

Output:
- `adv_images`: same shape as `images`, also in `[0, 1]`.

## Evaluation Protocol
Each evaluation script:
1. Loads one pretrained model.
2. Collects up to 1000 samples that are initially classified correctly.
3. Runs your `run_attack`.
4. Checks `L_inf` validity and `[0, 1]` range.
5. Reports `clean_acc`, `robust_acc`, and `asr = 1 - robust_acc`.

Important:
- ASR denominator is the number of initially correct samples.
- Invalid adversarial outputs (shape mismatch, non-finite values, or violated norm) are treated as failure.

## Evaluation Scenarios
Each scenario is a (model, dataset) pair drawn from {ResNet20, VGG11-BN, MobileNetV2} x {CIFAR-10, CIFAR-100}, using publicly available pretrained checkpoints.

## Baselines
The baselines below run inside the same harness via edit ops; reference implementations are in `torchattacks`:

- `fgsm`: Fast Gradient Sign Method (Goodfellow et al., 2015, arXiv:1412.6572). One-step sign-of-gradient attack.
- `pgd`: PGD (Madry et al., 2018, arXiv:1706.06083). Iterative projected gradient descent on the cross-entropy loss with random start, 40 inner steps and step size `eps/4`.
- `mifgsm`: MI-FGSM (Dong et al., CVPR 2018, arXiv:1710.06081). Iterative FGSM with momentum on the gradient direction.
- `autoattack`: AutoAttack (Croce and Hein, ICML 2020, arXiv:2003.01690). `torchattacks.AutoAttack(version="standard")`, the parameter-free ensemble of APGD-CE, APGD-DLR, FAB and Square Attack.

## Note on per-architecture natural robustness
At `eps=2/255`, ASR differs substantially across architectures on undefended models. This is an architectural property, not an evaluation bug: VGG-style wider-but-shallower activations at low-resolution feature maps absorb small `L_inf` perturbations more robustly than bottlenecked ResNet or depthwise-separable MobileNetV2, so PGD and AutoAttack saturate on the latter but leave meaningful headroom on the former.
