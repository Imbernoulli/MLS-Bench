# MathRuler

*A light-weight tool for evaluating LLMs in rule-based ways.*

## Installation

We use [vLLM](https://github.com/vllm-project/vllm) to accelerate the generation.

```bash
git clone https://github.com/hiyouga/MathRuler.git
cd MathRuler
pip install .
```

## Datasets

- [MATH](https://github.com/hendrycks/math): 500 problems.
- [GSM8K](https://github.com/openai/grade-school-math): 1319 problems.
- [AIME24](https://huggingface.co/datasets/HuggingFaceH4/aime_2024): 30 problems.
- [AIME25](https://huggingface.co/datasets/math-ai/aime25): 30 problems.

## Generate

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 mathruler gen Qwen/Qwen2.5-Math-7B-Instruct
```

Example output:

> Processed prompts: 100%|██████| 500/500 [00:36<00:00, 13.75it/s, est. speed input: 15765.84 toks/s, output: 5299.80 toks/s]

### Optional Arguments

- **json_path** (str): path to the eval file, defaults to `data/math_splits/test.jsonl`
- **save_path** (str): path to the predicted file, defaults to `predicts/test.jsonl`
- **n_shot** (int): number of few-shot examples, defaults to `0`
- **demo_split** (str): split to build few-shot examples, defaults to `math`
- **system** (str): system message for generation, defaults to `Please reason step by step, and put your final answer within \boxed{}.`
- **temperature** (float): decode temperature value, defaults to `0.0`
- **top_p** (float): decode top p value, defaults to `1.0`
- **max_tokens** (int): maximum number of generated tokens, defaults to `4096`
- **sample_num** (int): best-of-n evaluation, defaults to `1`

## Evaluate

```bash
mathruler eval predicts/test.jsonl
```

Example output:

> Processing sample: 100%|██████| 500/500 [00:00<00:00, 926.32it/s]
>
> Accuracy: 413/500 = 82.60%.

## Experimental Results

### MATH Dataset

|                  Command                       | Measured Acc | Reported Acc |
| ---------------------------------------------- | ------------ | ------------ |
| mathruler gen meta-llama/Meta-Llama-3-8B       | 29.2%        | 29.1%*       |
| mathruler gen meta-llama/Llama-3.1-8B-Instruct | 50.8%        | 51.9%*       |
| mathruler gen meta-llama/Llama-3.2-3B-Instruct | 48.4%        | 48.0%**      |
| mathruler gen Qwen/Qwen2.5-Math-7B-Instruct    | 82.6%        | 83.6%***     |

|                  Command                                                     | Measured Acc |
| ---------------------------------------------------------------------------- | ------------ |
| mathruler gen Qwen/Qwen2.5-Math-7B-Instruct --temperature 0.5 --sample_num 4 | 90.4%        |

### GSM8K Dataset

> Use `--json_path data/gsm8k_splits/test.jsonl` to evaluate models on the GSM8K dataset.

|                  Command                       | Measured Acc | Reported Acc |
| ---------------------------------------------- | ------------ | ------------ |
| mathruler gen meta-llama/Meta-Llama-3-8B       | 65.3%        | 80.6%*       |
| mathruler gen meta-llama/Llama-3.1-8B-Instruct | 81.7%        | 84.5%*       |
| mathruler gen meta-llama/Llama-3.2-3B-Instruct | 74.6%        | 77.7%**      |
| mathruler gen Qwen/Qwen2.5-Math-7B-Instruct    | 95.6%        | 95.2%***     |

> [!NOTE]
> For the GSM8K dataset, we evaluate all the models in zero-shot CoT setting, while the reported values of the Llama models are extracted from 8-shot CoT setting (*).

- *: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- **: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- ***: https://qwenlm.github.io/blog/qwen2.5-math/

## Example Use

```python
from mathruler.grader import extract_boxed_content, grade_answer

grade_answer(given_answer: str, ground_truth: str)
grade_answer(extract_boxed_content(generated_result: str), answer: str)
```

## Acknowledgement

- [openai/prm800k](https://github.com/openai/prm800k)
- [openai/grade-school-math](https://github.com/openai/grade-school-math)
- [QwenLM/Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math)
- [vllm-project/vllm](https://github.com/vllm-project/vllm)

## Citation

```bibtex
@Misc{mathruler,
  title = {MathRuler},
  author = {hiyouga},
  howpublished = {\url{https://github.com/hiyouga/MathRuler}},
  year = {2025}
}
```
