# Image Captioning: Prefix Mapping Initialization

## Objective

Investigate how the trainable visual-prefix mapping should be initialized before caption training. Modify only the declared editable file. No candidate ordering,
expected implementation, or evaluation category is supplied.

## Static Configuration Contract

The editable file contains one literal `CONFIG = {...}` assignment. Set `scheme` to `pytorch_default`, `xavier_uniform`, `kaiming_uniform`, or `caption_mean`. The last option initializes the output prefix from frozen caption-token embeddings.
Only literal strings, booleans, integers, and finite floating-point values are
accepted. Imports, calls, comprehensions, extra statements, missing keys, unknown
keys, malformed values, and incomplete configurations invalidate the run.

The verifier does not import or execute this file. It trains the same frozen-encoder,
frozen-decoder prefix captioner for ten complete epochs over every official training
image-caption pair, then evaluates every image in the fixed evaluation partition.
It reports CIDEr and BLEU-4 only after data hashes, training completion, prediction
count, and metric finiteness are proven.

Do not modify the harness, data, scripts, scorer, or unrelated files.
