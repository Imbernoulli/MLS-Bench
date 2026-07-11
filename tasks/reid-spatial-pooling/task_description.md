# Person Re-Identification: Spatial Pooling

## Objective

Investigate the spatial pooling inside the repository's fixed training and evaluation pipeline. Modify only the declared editable file. No candidate ordering or expected implementation is supplied; select a design from the public contract and the feedback produced by valid runs.

## Editable Surface

- File: `torchreid-reid/solution/pooling.py`
- Public symbol: `build_pooling`

Preserve the callable signature in that file and satisfy the input/output contract enforced by the fixed harness. Returned tensors and arrays must have the declared type, shape, dtype, and device, contain only finite values, and remain deterministic under the fixed seed. Configuration values and indices must be complete and within their enforced ranges.

The selected surface is active. A load error, exception, malformed return, missing output, or NaN/Inf invalidates the run; the harness does not replace a failed active implementation with another implementation.

## Evaluation

The fixed harness fine-tunes an ImageNet-pretrained ResNet-50 on the complete
official benchmark training split for 60 epochs with a fixed P x K sampler,
optimizer, schedule, loss, image geometry, and seed. Every official query is
assigned to one of three retrieval-difficulty groups and evaluated against the
complete official gallery. All three groups participate in scoring. The harness
reports mAP and CMC retrieval metrics; higher values are better.

A non-zero score requires authenticated full data and model inventories, exactly
60 epochs, 11,003 updates and 704,192 sampled training images, every query and
gallery item, all required finite metrics, and the final
training and evaluation completion records. An interrupted or malformed run is
invalid.

Scoring uses the official bounded mAP scale directly for every required group;
it does not depend on baseline ordering or unproven calibration rows. The
separate provenance record documents the retained terminal runtime evidence.

Do not modify the harness, scorer, data, scripts, or unrelated solution files.
