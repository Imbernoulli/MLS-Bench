# Image Captioning: Prefix Mapping Optimization

## Objective

Investigate the optimizer and learning-rate trajectory used to fit the visual-prefix mapping. Modify only the declared editable file. No candidate ordering,
expected implementation, or evaluation category is supplied.

## Static Configuration Contract

The editable file contains one literal `CONFIG = {...}` assignment. Choose AdamW or SGD and provide all required finite hyperparameters. Schedules may be constant, cosine, or cosine after warmup; warmup is zero for the first two choices.
Only literal strings, booleans, integers, and finite floating-point values are
accepted. Imports, calls, comprehensions, extra statements, missing keys, unknown
keys, malformed values, and incomplete configurations invalidate the run.

The verifier does not import or execute this file. It trains the same frozen-encoder,
frozen-decoder prefix captioner for ten complete epochs over every official training
image-caption pair, then evaluates every image in the fixed evaluation partition.
It reports CIDEr and BLEU-4 only after data hashes, training completion, prediction
count, and metric finiteness are proven.

Do not modify the harness, data, scripts, scorer, or unrelated files.
