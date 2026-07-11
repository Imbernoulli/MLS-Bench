# Gradient-Representation OOD Score

Design a post-hoc OOD score from a fixed per-sample final-layer gradient
representation.

Edit `ood-detection-lab/solution/gradient_score.py`. `Scorer.fit(ctx)` receives
all 50,000 ID-fit representations as `ctx.tr_gradients`, a finite `[50000, 10]`
tensor. `Scorer.score(gradients)` receives a finite `[N, 10]` tensor and must
return exactly one finite scalar per sample, with higher values indicating
in-distribution inputs. The verifier fixes the classifier, uniform-target
gradient construction, fit inventory, and evaluation splits. For logits `z`
and penultimate feature `h`, component `j` of the exposed representation is
`g_j = |softmax(z)_j - 0.1| * ||h||_1`; these ten values are the per-class L1
contributions to the final fully connected weight gradient against a uniform
target.

Evaluation reports ID-vs-OOD AUROC. Higher
AUROC is preferred, and every configured evaluation must complete.

The verifier uses one frozen OpenOOD-style CIFAR-10 `ResNet18_32x32` and an authenticated full inventory: 50,000 ID-fit images, 10,000 ID evaluation images, and three OOD domains containing 26,032, 10,000, and 10,000 images. It performs real inference over all 106,032 images in one serial command on one GPU. All three configured domains produce metrics and all three contribute to the task score. The dataset archive, checkpoint, preprocessing, seed 42, batch size 128, and model-forward inventory are fixed; incomplete, failed, duplicated, non-finite, or unauthenticated evaluations produce no metrics.
