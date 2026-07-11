# Automatic Prompt Optimization: Meta-Prompt Design

## Research Question
Design the reverse-mode meta-prompt used by the frozen LM to induce candidate instructions from fixed labeled training examples. Candidate ranking uses only dev; exactly one selected instruction is evaluated on each complete official evaluation split.

The editable function is in `prompt-optimization-lab/solution/meta_prompt.py`. The model, executor, examples,
candidate-selection procedure, and evaluation data are fixed. The function must be
deterministic and may use only the provided context.

## Evaluation Protocol
- **agnews** - 128 proposal and 200 selection rows from train; all 7,600 official test rows for evaluation.
- **sst2** - 128 proposal and 200 selection rows from train; all 872 labeled official validation rows for evaluation.

The three text inventories are pairwise disjoint. The evaluation labels are not used
to generate or rank candidates. Only the single instruction selected on train/dev is
run on the evaluation split. Both settings participate in scoring and execute
serially on one GPU. The metric is full-split zero-shot classification accuracy.
