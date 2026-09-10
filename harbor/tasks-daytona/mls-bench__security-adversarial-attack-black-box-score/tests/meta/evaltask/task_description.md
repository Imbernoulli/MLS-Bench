# Score-Based Query Black-Box Attack under Linf Constraint

## Research Question
Can you design a stronger score-based query black-box attack that improves attack success rate (ASR) under a fixed query budget and `L_inf` perturbation constraint?

## Background
Score-based query black-box attacks assume the attacker can query the victim model and observe its logits (or softmax scores) but cannot access gradients or weights. The attacker must search the input space using only forward queries while staying inside an `L_inf` ball around the clean image. This regime models realistic threat scenarios such as MLOps APIs that expose only prediction confidences.

Representative algorithms include the random-search-based Square Attack (Andriushchenko et al., 2020, arXiv:1912.00049), gradient-free SPSA-based attacks (Uesato et al., 2018, arXiv:1802.05666), and pixel-coordinate random search baselines. Across these methods the central tradeoff is between per-step exploration (which helps escape local minima) and per-step exploitation (which keeps the query budget low).

## Objective
Implement a better black-box attack in `torchattacks/bench/custom_attack.py`:

- Threat model: query black-box (no gradient access, only model logits).
- Constraint: `||x_adv - x||_inf <= eps`.
- Budget: `n_queries` is a per-sample query budget.
- Primary metric: maximize `ASR` under the fixed budget.
- Tie-break: for similar ASR, lower `avg_queries` is better.

## Editable Interface
You must implement:

`run_attack(model, images, labels, eps, n_queries, device, n_classes) -> adv_images`

Inputs:
- `model`: black-box wrapper that returns logits only.
- `images`: tensor of shape `(N, C, H, W)`, in `[0, 1]`.
- `labels`: tensor of shape `(N,)`.
- `n_classes`: 10 for CIFAR-10, 100 for CIFAR-100.

Output:
- `adv_images`: tensor with same shape as `images`, values in `[0, 1]`.

## Trusted Evaluation Logic
The evaluation logic in `bench/run_eval.py` is fixed and not editable.

- It tracks all model queries through a wrapper.
- If a batch exceeds query budget (`batch_size * n_queries`), the entire batch is marked as attack failure.
- `L_inf` and `[0, 1]` validity are checked per sample; only invalid samples are marked as attack failure.

Wrapper behavior and evaluation logic are fixed. Improvements should be confined to the attack algorithm in `custom_attack.py`.

## Query Semantics
- One call to `model(x)` consumes `x.shape[0]` queries.
- Repeated calls on the same sample still consume additional queries.
- Different batch partitioning is treated as equivalent total budget usage.

## Evaluation Scenarios
Each scenario is a (model, dataset) pair drawn from {ResNet20, VGG11-BN, MobileNetV2} x {CIFAR-10, CIFAR-100}, using publicly available pretrained checkpoints.

## Metrics
Reported metrics line format:

`ATTACK_METRICS asr=... clean_acc=... robust_acc=... avg_queries=...`

- `asr`: attack success rate (higher is better) — primary metric.
- `clean_acc`: accuracy of the model on the unperturbed batch (sanity check).
- `robust_acc`: `1 - asr`.
- `avg_queries`: average number of model queries consumed per sample (lower is better, used as tie-break).
