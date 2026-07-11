# OOD Energy Temperature

Choose the temperature in a fixed free-energy logit score.

Edit `ood-detection-lab/solution/temperature_score.py` and implement
`select_temperature() -> float`. The value must be finite and in `[1e-3, 1e4]`.
The energy formula, classifier, ID fit data, and OOD splits are fixed.

Evaluation reports AUROC. Higher AUROC is
preferred, and every configured evaluation must complete.

The verifier uses one frozen OpenOOD-style CIFAR-10 `ResNet18_32x32` and an authenticated full inventory: 50,000 ID-fit images, 10,000 ID evaluation images, and three OOD domains containing 26,032, 10,000, and 10,000 images. It performs real inference over all 106,032 images in one serial command on one GPU. All three configured domains produce metrics and all three contribute to the task score. The dataset archive, checkpoint, preprocessing, seed 42, batch size 128, and model-forward inventory are fixed; incomplete, failed, duplicated, non-finite, or unauthenticated evaluations produce no metrics.
