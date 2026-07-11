# Image Inpainting: Training Loss

## Objective

Investigate the training objective used by the repository's fixed training and evaluation pipeline. Modify only the declared editable file. The task does not prescribe a candidate ordering or an expected implementation.

## Editable Surface

- File: `image-inpainting/solution/loss.py`
- Required symbol: `compute_loss(out, gt, mask)`

The callable must return one finite, non-negative scalar tensor that retains a gradient path to the network output. A load error, exception, malformed result, missing output, or non-finite value invalidates verification. The verifier never substitutes another implementation after the selected callable fails.

If the editing agent exits before changing the file, evaluation of the repository's untouched native implementation is valid.

## Fixed Protocol

The protocol is `places365-val256-fullres-v1`: the complete canonical Places365-Standard `val_256` archive, pinned by SHA-256 `24b4e639ef12a0012af525bc4cb443e4ab4aaea8369a1fb009b70e4a4aad5d48`, is used at its native 256x256 resolution. A seeded, disjoint split uses 32,000 images for optimization and 4,500 for evaluation. Every run uses seed 42, batch size 8, Adam with learning rate `1e-4`, and exactly 100,000 optimizer steps.

A single trained checkpoint is evaluated sequentially on all three mask settings, and all three participate in scoring: small square holes, large square holes, and irregular strokes. The one verification command uses one GPU; training is not repeated per mask setting. Verification only reads the prebuilt image artifact and does not install packages, download data, extract archives, or prepare data.

This is a research-scale real-image proxy over a canonical Places365 archive, not a claimed reproduction of the larger Places2 corpus used by DeepFillv2. Full-protocol runtime and baseline quality still require worker-side measurement.

## Scoring State

Masked-region L1 is the objective; lower is better. The proof parser requires the pinned protocol record, all ten training-progress records, three ordered blocks of exactly 4,500 per-image records, recomputable aggregate metrics, and a terminal successful shell record. Missing, duplicated, malformed, non-finite, inconsistent, or failed verification produces no metrics and therefore exactly zero score.

The leaderboard is intentionally header-only while fresh full-protocol anchors are pending. Until complete finite anchors are recorded, even an otherwise valid run maps to exactly zero rather than using obsolete reduced-scale measurements.

Do not modify the verifier, scorer, data, scripts, or unrelated solution files.
