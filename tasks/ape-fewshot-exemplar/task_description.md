# Automatic Prompt Optimization: Induction Exemplars

## Research Question
Choose the labeled training examples used for reverse-mode instruction induction. The frozen LM induces candidates from those examples, the fixed selector ranks candidates only on dev, and exactly one selected instruction is evaluated on each complete official evaluation split.

The editable function is in `prompt-optimization-lab/solution/exemplar.py`. The model, executor, examples,
candidate-selection procedure, and evaluation data are fixed. The function must be
deterministic and may use only the provided context.

## Evaluation Protocol
- **agnews** - 128 proposal and 200 selection rows from train; all 7,600 official test rows for evaluation.
- **sst2** - 128 proposal and 200 selection rows from train; all 872 labeled official validation rows for evaluation.

The three text inventories are pairwise disjoint. The evaluation labels are not used
to generate or rank candidates. Only the single instruction selected on train/dev is
run on the evaluation split. Both settings participate in scoring and execute
serially on one GPU. The metric is full-split zero-shot classification accuracy.
