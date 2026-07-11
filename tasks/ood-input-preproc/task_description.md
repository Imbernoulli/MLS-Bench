# ODIN Input Perturbation Size

Choose the input-space perturbation magnitude used by a fixed ODIN preprocessing
pipeline.

Edit `ood-detection-lab/solution/input_preproc_score.py` and implement
`select_preprocess_epsilon() -> float`. The value must be finite and in
`[0, 0.1]`. Gradient direction, temperature, classifier, recomputed forward pass,
energy formula, and splits are fixed.

Evaluation reports AUROC. Higher AUROC is
preferred, and every configured evaluation must complete.

The verifier uses one frozen OpenOOD-style CIFAR-10 `ResNet18_32x32` and an authenticated full inventory: 50,000 ID-fit images, 10,000 ID evaluation images, and three OOD domains containing 26,032, 10,000, and 10,000 images. It first performs real inference over all 106,032 images, then performs the fixed two-forward ODIN path over the ID evaluation split and every OOD split, for 218,096 model-forward images and 1,714 batches in total. All three configured domains run serially in one command on one GPU and all three contribute to the task score. The dataset archive, checkpoint, preprocessing, seed 42, batch size 128, and forward inventory are fixed; incomplete, failed, duplicated, non-finite, or unauthenticated evaluations produce no metrics.
