# Feature-Only OOD Score

Design a post-hoc OOD score using only the frozen classifier's penultimate
features.

Edit `ood-detection-lab/solution/feature_score.py`. `Scorer.fit(ctx)` receives
ID-fit features and labels. `Scorer.score(feats)` must return one finite scalar
per sample, with higher values indicating in-distribution inputs. Logits, early
features, raw pixels, the model, and OOD data are not passed to the surface.

Evaluation reports ID-vs-OOD AUROC. Higher AUROC is preferred.

The verifier uses one frozen OpenOOD-style CIFAR-10 `ResNet18_32x32` and an authenticated full inventory: 50,000 ID-fit images, 10,000 ID evaluation images, and three OOD domains containing 26,032, 10,000, and 10,000 images. It performs real inference over all 106,032 images in one serial command on one GPU. All three configured domains produce metrics and all three contribute to the task score. The dataset archive, checkpoint, preprocessing, seed 42, batch size 128, and model-forward inventory are fixed; incomplete, failed, duplicated, non-finite, or unauthenticated evaluations produce no metrics.
