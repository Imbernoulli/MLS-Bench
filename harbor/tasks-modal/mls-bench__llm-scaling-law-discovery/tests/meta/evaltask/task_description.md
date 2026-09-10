# SLDBench Scaling Law Discovery

## Research Question
Design a better scaling-law model that extrapolates on `SLDBench` scaling tasks while keeping a single shared functional form per task and fitting group-specific coefficients from the observed trials. The intended contribution is a compact symbolic law per benchmark family — not generic tabular regression.

## Background
This task is built on the SLDBench benchmark from Liu et al., "Can Language Models Discover Scaling Laws?", 2025, arXiv:2507.21184 (project page: https://linhaowei1.github.io/scaling_law_discovery/). SLDBench collects ~5,000 LLM training experiments from existing scaling-law literature and turns them into symbolic-regression tasks: given numeric experiment descriptors and a categorical group, predict held-out training losses on extrapolation regions.

We use three representative and harder subsets (less saturated than the original `parallel` / `moe` / `sft` trio):

- **`sld-vocab`** — vocabulary scaling law: unigram-normalised loss as a function of non-vocabulary parameters `N`, vocabulary size `V`, and training characters `D`. Reference: Tao et al., "Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies", 2024, arXiv:2407.13623.
- **`sld-lrbsz`** — learning-rate & batch-size scaling law: LM loss as a joint function of learning rate, batch size, training tokens, and non-embedding parameters.
- **`sld-dataconstrained`** — data-constrained scaling law: loss as a function of unique tokens `U`, parameters `N`, and total tokens `D`, where `D` can exceed `U` (data repetition). Reference: Muennighoff et al., "Scaling Data-Constrained Language Models", NeurIPS 2023, arXiv:2305.16264.

## What you can modify
The `ScalingLawModel` class in `custom_scaling_law.py`. Your model receives:

- `X_num` — raw numeric inputs (per-benchmark list below).
- `X_cat` — categorical metadata (primarily the `group`).
- `y` — observed target losses on the training split.

The runtime loads the official `SLDBench` train/test splits from `/data/scaling_law/*.jsonl`. The observed training trials are also mirrored into the editable workspace as read-only files for direct inspection:

- `scaling-law-lab/observed_trials/sld_vocab_train.jsonl`
- `scaling-law-lab/observed_trials/sld_lrbsz_train.jsonl`
- `scaling-law-lab/observed_trials/sld_dataconstrained_train.jsonl`

Inspect these raw trials directly and discover benchmark-specific symbolic laws. Large pretrained LMs are not allowed.

### Benchmarks
- `sld-vocab` — numeric: `non_vocab_parameters`, `vocab_size`, `num_characters`; categorical: `group`; target: `unigram_normalized_loss` (can be negative — do not clip).
- `sld-lrbsz` — numeric: `lr`, `bsz`, `data_size`, `non_embedding_param_size`; categorical: `group`; target: `lm_loss`.
- `sld-dataconstrained` — numeric: `unique_tokens`, `params`, `tokens`; categorical: `group`; target: `loss`.

### Interface
```python
class ScalingLawModel:
    def __init__(self, benchmark_name, numeric_names, categorical_names):
        ...
    def fit(self, X_num, X_cat, y):
        return self
    def predict(self, X_num, X_cat):
        return y_pred
```
`benchmark_name` lets you use different law families for `vocab`, `lrbsz`, and `dataconstrained` while still keeping one shared symbolic expression per benchmark and fitting group-specific coefficients.

## Evaluation
- **Primary**: held-out test `R^2` per benchmark (higher is better).
- **Secondary**: `MAE`, `RMSE`, `NMAE` (lower is better).

Strong solutions usually:
- fit coefficients per `group` rather than collapsing all groups together;
- preserve sensible asymptotics on larger or denser test points (good extrapolation, not memorization).
