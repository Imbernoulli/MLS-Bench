# MLS-Bench: llm-scaling-law-discovery

# Scaling Law Discovery

## Research Question
Design a scaling-law model that extrapolates well on held-out scaling tasks
while keeping a single shared functional form per task and fitting
group-specific coefficients from observed trials. The intended contribution is
a compact symbolic law per task family — not generic tabular regression.

## Background
Given numeric experiment descriptors and a categorical `group`, predict
training-loss-style targets on extrapolation regions. The model receives raw
numeric inputs and a group label, and should produce a symbolic law that
generalizes across group-specific coefficients.

## What you can modify
The `ScalingLawModel` class in `custom_scaling_law.py`. Your model receives:

- `X_num` — raw numeric inputs.
- `X_cat` — categorical metadata (primarily the `group`).
- `y` — observed target values on the training split.

The runtime loads train/test splits from `/data/scaling_law/*.jsonl`. Inspect
observed trials directly and discover task-specific symbolic laws. Large
pretrained LMs are not allowed.

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
`benchmark_name` lets you use different law families for different task
variants while still keeping one shared symbolic expression per variant and
fitting group-specific coefficients.

Strong solutions usually:
- fit coefficients per `group` rather than collapsing all groups together;
- preserve sensible asymptotics on larger or denser inputs (good
  extrapolation, not memorization).

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/scaling-law-lab/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be rejected.

- `scaling-law-lab/custom_scaling_law.py`
- editable lines **183–211**

## Readable Context

### `scaling-law-lab/custom_scaling_law.py`  [EDITABLE — lines 183–211 only]

```python
   182:
   183: # ============================================================
   184: # Scaling Law Model (EDITABLE)
   185: # ============================================================
   186:
   187: class ScalingLawModel:
   188:     """Editable task-specific symbolic law scaffold.
   189:
   190:     You may implement different symbolic forms for each task variant
   191:     passed via `benchmark_name`.
   192:     """
   193:
   194:     def __init__(self, benchmark_name: str, numeric_names=None, categorical_names=None):
   195:         self.benchmark_name = benchmark_name
   196:         self.numeric_names = list(numeric_names or [])
   197:         self.categorical_names = list(categorical_names or [])
   198:
   199:     def fit(self, X_num, X_cat, y):
   200:         self.mean_ = float(np.mean(y))
   201:         return self
   202:
   203:     def predict(self, X_num, X_cat):
   204:         return np.full(len(X_num), self.mean_)
```

## Tips

- Keep the function/class signatures of the editable regions identical;
  they are imported by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- Aim for an *algorithmic* contribution — many hyperparameters are locked
  outside the editable surface anyway.

Good luck.
