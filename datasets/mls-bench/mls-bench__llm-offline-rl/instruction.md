# MLS-Bench: llm-offline-rl

# LLM Offline RL: Preference Optimization for Math Reasoning

## Objective
Design a custom preference loss for offline preference optimization of a math
LLM. Implement your loss in the `compute_preference_loss` method of
`trainer.py` (selected via `pref_loss=custom`, registered in
`finetuning_args.py`).

## Background
DPO and its variants (Hinge, IPO, KTO-pair, ORPO, SimPO) directly optimize a
preference loss on chosen/rejected response pairs without a separate reward
model. Each variant trades off stability, length bias, calibration, and
reference dependence differently. This task asks you to design a single
preference loss that improves math reasoning over the standard variants.

## Setup (Step-DPO recipe — Lai et al. 2024, arXiv:2406.18629)
- **Base model**: `Qwen2.5-Math-1.5B-Instruct` (math-specialized SFT, 1.5B
  params). Trained with the `qwen` chat template.
- **Preference data**: `xinlai/Math-Step-DPO-10K` (~10K math problems with
  step-level chosen/rejected solutions). We use the full responses as
  response-level chosen/rejected pairs, so all DPO variants apply directly.
- **Training**: full-parameter, 4× GPU, ZeRO-2, β=0.1 (or variant-specific),
  4 epochs, lr=5e-7, cosine schedule.

## Evaluation (judge-free)
Three math reasoning benchmarks — graded by MathRuler's sympy + mathd
checker, no LLM judge involved:

1. **GSM8K** — grade-school math (1.32K problems). Metric: `gsm8k_accuracy`.
2. **MATH-500** — 500-problem subset of MATH competition. Metric: `math500_accuracy`.
3. **AIME 2024** — 30 American Invitational Math Exam problems. Metric: `aime2024_accuracy`.

A single vLLM engine is loaded once per evaluation pass and runs greedy
decoding (temperature=0) for all three benchmarks.

## Baselines
| Name  | `pref_loss` | Reference            |
|-------|-------------|----------------------|
| dpo   | sigmoid     | Rafailov et al. 2023 |
| simpo | simpo       | Meng et al. 2024     |
| ipo   | ipo         | Azar et al. 2023     |
| orpo  | orpo        | Hong et al. 2024     |

Your `pref_loss=custom` implementation should beat at least one of these on
the average of the three math benchmarks.


## Your Workspace

You are working inside `/app`. The package source tree
`/app/LLaMA-Factory/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `LLaMA-Factory/src/llamafactory/train/dpo/trainer.py`
- editable lines **187–216**
- `LLaMA-Factory/src/llamafactory/hparams/finetuning_args.py`
- editable lines **556–556**


Other files you may **read** for context (do not modify):
- `scripts/train.sh`


## Readable Context


### `LLaMA-Factory/src/llamafactory/train/dpo/trainer.py`  [EDITABLE — lines 187–216 only]

```python
     1: # Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
     2: #
     3: # This code is inspired by the HuggingFace's TRL library.
     4: # https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/dpo_trainer.py
     5: #
     6: # Licensed under the Apache License, Version 2.0 (the "License");
     7: # you may not use this file except in compliance with the License.
     8: # You may obtain a copy of the License at
     9: #
    10: #     http://www.apache.org/licenses/LICENSE-2.0
    11: #
    12: # Unless required by applicable law or agreed to in writing, software
    13: # distributed under the License is distributed on an "AS IS" BASIS,
    14: # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    15: # See the License for the specific language governing permissions and
    16: # limitations under the License.
    17: 
    18: import warnings
    19: from collections import defaultdict
    20: from contextlib import nullcontext
    21: from types import MethodType
    22: from typing import TYPE_CHECKING, Literal, Optional, Union
    23: 
    24: import torch
    25: import torch.nn.functional as F
    26: from transformers import Trainer
    27: from trl import DPOTrainer
    28: from trl.models.utils import prepare_deepspeed, prepare_fsdp
    29: from trl.trainer import disable_dropout_in_model
    30: from typing_extensions import override
    31: 
    32: from ...extras.constants import IGNORE_INDEX
    33: from ...extras.packages import is_transformers_version_greater_than
    34: from ..callbacks import SaveProcessorCallback
    35: from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps, nested_detach
    36: 
    37: 
    38: if TYPE_CHECKING:
    39:     from transformers import PreTrainedModel, ProcessorMixin
    40: 
    41:     from ...hparams import FinetuningArguments
    42: 
    43: 
    44: class CustomDPOTrainer(DPOTrainer):
    45:     def __init__(
    46:         self,
    47:         model: Union["PreTrainedModel", torch.nn.Module],
    48:         ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
    49:         finetuning_args: "FinetuningArguments",
    50:         processor: Optional["ProcessorMixin"],
    51:         disable_dropout: bool = True,
    52:         **kwargs,
    53:     ):
    54:         if is_transformers_version_greater_than("4.46"):
    55:             kwargs["processing_class"] = kwargs.pop("tokenizer")
    56: 
    57:         if disable_dropout:
    58:             disable_dropout_in_model(model)
    59:             if ref_model is not None:
    60:                 disable_dropout_in_model(ref_model)
    61: 
    62:         self.finetuning_args = finetuning_args
    63:         self.f_divergence_type = "reverse_kl"
    64:         self.reference_free = False
    65:         self.use_dpo_data_collator = True  # hack to avoid warning
    66:         self.generate_during_eval = False  # disable at evaluation
    67:         self.label_pad_token_id = IGNORE_INDEX
    68:         self.padding_value = 0
    69:         self.is_encoder_decoder = model.config.is_encoder_decoder
    70:         self.precompute_ref_log_probs = False
    71:         self._precomputed_train_ref_log_probs = False
    72:         self._precomputed_eval_ref_log_probs = False
    73:         self._peft_has_been_casted_to_bf16 = False
    74: 
    75:         self.ref_model = ref_model
    76:         self._stored_metrics = defaultdict(lambda: defaultdict(list))
    77: 
    78:         # dpo hyperparams
    79:         self.beta = finetuning_args.pref_beta
    80:         self.loss_type = finetuning_args.pref_loss
    81:         self.ftx_gamma = finetuning_args.pref_ftx
    82:         self.bco_gemma = finetuning_args.pref_bco_weight
    83:         self.label_smoothing = finetuning_args.dpo_label_smoothing
    84:         self.simpo_gamma = finetuning_args.simpo_gamma
    85:         self.ld_alpha = finetuning_args.ld_alpha
    86: 
    87:         Trainer.__init__(self, model=model, **kwargs)
    88:         self.model_accepts_loss_kwargs = False  # overwrite trainer's default behavior
    89:         if not hasattr(self, "accelerator"):
    90:             raise AttributeError("Please update `transformers`.")
    91: 
    92:         warnings.simplefilter("ignore")  # remove gc warnings on ref model
    93: 
    94:         if ref_model is not None:
    95:             if self.is_deepspeed_enabled:
    96:                 if not (
    97:                     getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
    98:                 ):  # quantized models are already set on the correct device
    99:                     self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
   100:             elif self.is_fsdp_enabled:
   101:                 if self.accelerator.is_fsdp2:
   102:                     from accelerate.utils.fsdp_utils import fsdp2_prepare_model
   103: 
   104:                     self.ref_model = fsdp2_prepare_model(self.accelerator, self.ref_model)
   105:                 else:
   106:                     self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
   107:             else:
   108:                 self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
   109:                 self.ref_model.eval()
   110: 
   111:         if processor is not None:
   112:             self.add_callback(SaveProcessorCallback(processor))
   113: 
   114:         if finetuning_args.use_badam:
   115:             from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore
   116: 
   117:             self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
   118:             self.add_callback(BAdamCallback)
   119: 
   120:         if self.bco_gemma >= 1e-6:
   121:             from trl.trainer import RunningMoments
   122: 
   123:             self.running = RunningMoments(self.accelerator)
   124: 
   125:     @override
   126:     def create_optimizer(self) -> "torch.optim.Optimizer":
   127:         if self.optimizer is None:
   128:             self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
   129:         return super().create_optimizer()
   130: 
   131:     @override
   132:     def create_scheduler(
   133:         self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
   134:     ) -> "torch.optim.lr_scheduler.LRScheduler":
   135:         create_custom_scheduler(self.args, num_training_steps, optimizer)
   136:         return super().create_scheduler(num_training_steps, optimizer)
   137: 
   138:     @override
   139:     def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
   140:         if self.finetuning_args.disable_shuffling:
   141:             return torch.utils.data.SequentialSampler(self.train_dataset)
   142: 
   143:         return super()._get_train_sampler(*args, **kwargs)
   144: 
   145:     @override
   146:     def get_batch_samples(self, *args, **kwargs):
   147:         r"""Replace the method of DPO Trainer with the one of the standard Trainer."""
   148:         return Trainer.get_batch_samples(self, *args, **kwargs)
   149: 
   150:     def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
   151:         r"""Compute ORPO's odds ratio (OR) loss for batched log probabilities of the policy model."""
   152:         log_odds = (chosen_logps - rejected_logps) - (
   153:             torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
   154:         )
   155:         sft_loss = -chosen_logps
   156:         odds_ratio_loss = -F.logsigmoid(log_odds)
   157:         orpo_loss = sft_loss + self.beta * odds_ratio_loss
   158:         return orpo_loss
   159: 
   160:     def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
   161:         r"""Compute SimPO loss for batched log probabilities of the policy model."""
   162:         pi_logratios = chosen_logps - rejected_logps
   163:         gamma_logratios = self.simpo_gamma / self.beta
   164:         logits = pi_logratios - gamma_logratios
   165:         simpo_loss = -F.logsigmoid(self.beta * logits)
   166:         return simpo_loss
   167: 
   168:     def bco_loss(
   169:         self,
   170:         chosen_logps: "torch.Tensor",
   171:         rejected_logps: "torch.Tensor",
   172:         reference_chosen_logps: "torch.Tensor",
   173:         reference_rejected_logps: "torch.Tensor",
   174:     ) -> "torch.Tensor":
   175:         chosen_logratios = chosen_logps - reference_chosen_logps
   176:         rejected_logratios = rejected_logps - reference_rejected_logps
   177:         chosen_rewards = self.beta * chosen_logratios
   178:         rejected_rewards = self.beta * rejected_logratios
   179:         rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
   180:         self.running.update(rewards)  # update baseline
   181:         delta = self.running.mean
   182:         bco_loss = -F.logsigmoid((self.beta * chosen_logratios) - delta) - F.logsigmoid(
   183:             -(self.beta * rejected_logratios - delta)
   184:         )
   185:         return bco_loss
   186: 
   187:     def compute_preference_loss(
   188:         self,
   189:         policy_chosen_logps: "torch.Tensor",
   190:         policy_rejected_logps: "torch.Tensor",
   191:         reference_chosen_logps: Optional["torch.Tensor"],
   192:         reference_rejected_logps: Optional["torch.Tensor"],
   193:     ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
   194:         r"""Compute loss for preference learning."""
   195:         if not self.finetuning_args.use_ref_model:
   196:             if self.loss_type == "orpo":
   197:                 losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
   198:             elif self.loss_type == "simpo":
   199:                 losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
   200:             else:
   201:                 raise NotImplementedError(f"Unknown loss type: {self.loss_type}.")
   202: 
   203:             chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
   204:             rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
   205:         else:
   206:             losses, chosen_rewards, rejected_rewards = self.dpo_loss(
   207:                 policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
   208:             )
   209: 
   210:             if self.bco_gemma > 1e-6:
   211:                 bco_losses = self.bco_loss(
   212:                     policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
   213:                 )
   214:                 losses = (losses + bco_losses * self.bco_gemma) / (1.0 + self.bco_gemma)  # re-weight W_p and W_q
   215: 
   216:         return losses, chosen_rewards, rejected_rewards
   217: 
   218:     @override
   219:     def concatenated_forward(
   220:         self, model: "PreTrainedModel", batch: dict[str, "torch.Tensor"], is_ref_model: bool = False
   221:     ) -> dict[str, "torch.Tensor"]:
   222:         r"""Compute the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.
   223: 
   224:         Otherwise the average log probabilities.
   225:         """
   226:         if self.finetuning_args.use_ref_model:
   227:             batch = nested_detach(batch, clone=True)  # avoid error
   228: 
   229:         labels = batch.pop("labels")  # dpo do not need compute loss in forward
   230:         all_logits: torch.Tensor = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
   231:         all_logps, valid_length = get_batch_logps(
   232:             logits=all_logits, labels=labels, ld_alpha=(self.ld_alpha if not is_ref_model else None)
   233:         )
   234:         if self.loss_type in ["ipo", "orpo", "simpo"]:
   235:             all_logps = all_logps / valid_length
   236: 
   237:         batch_size = batch["input_ids"].size(0) // 2
   238:         chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
   239:         chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
   240:         chosen_length, _ = valid_length.split(batch_size, dim=0)
   241:         if self.loss_type in ["ipo", "orpo", "simpo"]:
   242:             chosen_logps_avg = chosen_logps
   243:         else:
   244:             chosen_logps_avg = chosen_logps / chosen_length
   245: 
   246:         return {
   247:             "chosen_logps": chosen_logps,
   248:             "rejected_logps": rejected_logps,
   249:             "chosen_logits": chosen_logits,
   250:             "rejected_logits": rejected_logits,
   251:             "chosen_logps_avg": chosen_logps_avg,
   252:         }
   253: 
   254:     @override
   255:     def compute_reference_log_probs(
   256:         self, model: "PreTrainedModel", batch: dict[str, "torch.Tensor"]
   257:     ) -> tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
   258:         r"""Compute log probabilities of the reference model."""
   259:         if not self.finetuning_args.use_ref_model:
   260:             return None, None
   261: 
   262:         if self.ref_model is None:
   263:             ref_model = model
   264:             ref_context = self.accelerator.unwrap_model(model).disable_adapter()
   265:         else:
   266:             ref_model = self.ref_model
   267:             ref_context = nullcontext()
   268: 
   269:         with torch.no_grad(), ref_context:
   270:             ref_output = self.concatenated_forward(ref_model, batch, is_ref_model=True)
   271:             reference_chosen_logps = ref_output["chosen_logps"]
   272:             reference_rejected_logps = ref_output["rejected_logps"]
   273: 
   274:         return reference_chosen_logps, reference_rejected_logps
   275: 
   276:     @override
   277:     def get_batch_loss_metrics(
   278:         self,
   279:         model: "PreTrainedModel",
   280:         batch: dict[str, "torch.Tensor"],
   281:         train_eval: Literal["train", "eval"] = "train",
   282:     ) -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
   283:         r"""Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
   284:         metrics = {}
   285: 
   286:         model_output = self.concatenated_forward(model, batch)
   287:         policy_chosen_logps = model_output["chosen_logps"]
   288:         policy_rejected_logps = model_output["rejected_logps"]
   289:         policy_chosen_logits = model_output["chosen_logits"]
   290:         policy_rejected_logits = model_output["rejected_logits"]
   291:         policy_chosen_logps_avg = model_output["chosen_logps_avg"]
   292: 
   293:         reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)
   294:         losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
   295:             policy_chosen_logps,
   296:             policy_rejected_logps,
   297:             reference_chosen_logps,
   298:             reference_rejected_logps,
   299:         )
   300:         sft_loss = -policy_chosen_logps_avg
   301:         if self.ftx_gamma > 1e-6:
   302:             losses += self.ftx_gamma * sft_loss
   303: 
   304:         prefix = "eval_" if train_eval == "eval" else ""
   305:         metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
   306:         metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
   307:         metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item()
   308:         metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
   309:         metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item()
   310:         metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item()
   311:         metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().item()
   312:         metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().item()
   313:         if self.loss_type == "orpo":
   314:             metrics[f"{prefix}sft_loss"] = sft_loss.mean().item()
   315:             metrics[f"{prefix}odds_ratio_loss"] = ((losses - sft_loss) / self.beta).mean().item()
   316: 
   317:         return losses.mean(), metrics
   318: 
   319:     @override
   320:     def compute_loss(
   321:         self, model: "PreTrainedModel", inputs: dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
   322:     ) -> Union["torch.Tensor", tuple["torch.Tensor", list["torch.Tensor"]]]:
   323:         r"""Subclass and override to accept extra kwargs."""
   324:         return super().compute_loss(model, inputs, return_outputs)
   325: 
   326:     @override
   327:     def log(self, logs: dict[str, float], *args, **kwargs) -> None:
   328:         r"""Log `logs` on the various objects watching training, including stored metrics."""
   329:         # logs either has "loss" or "eval_loss"
   330:         train_eval = "train" if "loss" in logs else "eval"
   331:         # Add averaged stored metrics to logs
   332:         key_list, metric_list = [], []
   333:         for key, metrics in self._stored_metrics[train_eval].items():
   334:             key_list.append(key)
   335:             metric_list.append(torch.tensor(metrics, dtype=torch.float).to(self.accelerator.device).mean().item())
   336: 
   337:         del self._stored_metrics[train_eval]
   338:         if len(metric_list) < 10:  # pad to for all reduce
   339:             for i in range(10 - len(metric_list)):
   340:                 key_list.append(f"dummy_{i}")
   341:                 metric_list.append(0.0)
   342: 
   343:         metric_list = torch.tensor(metric_list, dtype=torch.float).to(self.accelerator.device)
   344:         metric_list = self.accelerator.reduce(metric_list, "mean").tolist()
   345:         for key, metric in zip(key_list, metric_list):  # add remaining items
   346:             if not key.startswith("dummy_"):
   347:                 logs[key] = metric
   348: 
   349:         return Trainer.log(self, logs, *args, **kwargs)
```

### `LLaMA-Factory/src/llamafactory/hparams/finetuning_args.py`  [EDITABLE — lines 556–556 only]

```python
Lines 443-602:
   443: @dataclass
   444: class FinetuningArguments(
   445:     SwanLabArguments,
   446:     BAdamArgument,
   447:     ApolloArguments,
   448:     GaloreArguments,
   449:     RLHFArguments,
   450:     LoraArguments,
   451:     OFTArguments,
   452:     FreezeArguments,
   453: ):
   454:     r"""Arguments pertaining to which techniques we are going to fine-tuning with."""
   455: 
   456:     pure_bf16: bool = field(
   457:         default=False,
   458:         metadata={"help": "Whether or not to train model in purely bf16 precision (without AMP)."},
   459:     )
   460:     stage: Literal["pt", "sft", "rm", "ppo", "dpo", "kto"] = field(
   461:         default="sft",
   462:         metadata={"help": "Which stage will be performed in training."},
   463:     )
   464:     finetuning_type: Literal["lora", "oft", "freeze", "full"] = field(
   465:         default="lora",
   466:         metadata={"help": "Which fine-tuning method to use."},
   467:     )
   468:     use_llama_pro: bool = field(
   469:         default=False,
   470:         metadata={"help": "Whether or not to make only the parameters in the expanded blocks trainable."},
   471:     )
   472:     use_adam_mini: bool = field(
   473:         default=False,
   474:         metadata={"help": "Whether or not to use the Adam-mini optimizer."},
   475:     )
   476:     use_mca: bool = field(
   477:         default=False,
   478:         metadata={
   479:             "help": (
   480:                 "Whether or not to use MCA (Megatron Core Adapter) training. "
   481:                 "Controlled by USE_MCA environment variable."
   482:             )
   483:         },
   484:     )
   485:     use_muon: bool = field(
   486:         default=False,
   487:         metadata={"help": "Whether or not to use the Muon optimizer."},
   488:     )
   489:     use_dft_loss: bool = field(
   490:         default=False,
   491:         metadata={"help": "Whether to use the DFT loss."},
   492:     )
   493:     use_asft_loss: bool = field(
   494:         default=False,
   495:         metadata={"help": "Whether to use the ASFT loss."},
   496:     )
   497:     asft_alpha: float = field(
   498:         default=0.1,
   499:         metadata={"help": "The alpha parameter for ASFT loss to control the power of adaptive weight."},
   500:     )
   501:     use_eaft_loss: bool = field(
   502:         default=False,
   503:         metadata={"help": "Whether to use the EAFT loss."},
   504:     )
   505:     eaft_alpha: float = field(
   506:         default=1.0,
   507:         metadata={"help": "The alpha parameter for EAFT loss to control the power of adaptive weight."},
   508:     )
   509:     freeze_vision_tower: bool = field(
   510:         default=True,
   511:         metadata={"help": "Whether ot not to freeze the vision tower in MLLM training."},
   512:     )
   513:     freeze_multi_modal_projector: bool = field(
   514:         default=True,
   515:         metadata={"help": "Whether or not to freeze the multi modal projector in MLLM training."},
   516:     )
   517:     freeze_language_model: bool = field(
   518:         default=False,
   519:         metadata={"help": "Whether or not to freeze the language model in MLLM training."},
   520:     )
   521:     compute_accuracy: bool = field(
   522:         default=False,
   523:         metadata={"help": "Whether or not to compute the token-level accuracy at evaluation."},
   524:     )
   525:     disable_shuffling: bool = field(
   526:         default=False,
   527:         metadata={"help": "Whether or not to disable the shuffling of the training set."},
   528:     )
   529:     early_stopping_steps: int | None = field(
   530:         default=None,
   531:         metadata={"help": "Number of steps to stop training if the `metric_for_best_model` does not improve."},
   532:     )
   533:     plot_loss: bool = field(
   534:         default=False,
   535:         metadata={"help": "Whether or not to save the training loss curves."},
   536:     )
   537:     include_effective_tokens_per_second: bool = field(
   538:         default=False,
   539:         metadata={"help": "Whether or not to compute effective tokens per second."},
   540:     )
   541: 
   542:     def __post_init__(self):
   543:         def split_arg(arg):
   544:             if isinstance(arg, str):
   545:                 return [item.strip() for item in arg.split(",")]
   546:             return arg
   547: 
   548:         self.freeze_trainable_modules: list[str] = split_arg(self.freeze_trainable_modules)
   549:         self.freeze_extra_modules: list[str] | None = split_arg(self.freeze_extra_modules)
   550:         self.lora_alpha: int = self.lora_alpha or self.lora_rank * 2
   551:         self.lora_target: list[str] = split_arg(self.lora_target)
   552:         self.oft_target: list[str] = split_arg(self.oft_target)
   553:         self.additional_target: list[str] | None = split_arg(self.additional_target)
   554:         self.galore_target: list[str] = split_arg(self.galore_target)
   555:         self.apollo_target: list[str] = split_arg(self.apollo_target)
   556:         self.use_ref_model = self.stage == "dpo" and self.pref_loss not in ["orpo", "simpo"]
   557: 
   558:         assert self.finetuning_type in ["lora", "oft", "freeze", "full"], "Invalid fine-tuning method."
   559:         assert self.ref_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."
   560:         assert self.reward_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."
   561: 
   562:         if self.stage == "ppo" and self.reward_model is None:
   563:             raise ValueError("`reward_model` is necessary for PPO training.")
   564: 
   565:         if self.stage == "ppo" and self.reward_model_type == "lora" and self.finetuning_type != "lora":
   566:             raise ValueError("`reward_model_type` cannot be lora for Freeze/Full PPO training.")
   567: 
   568:         if self.stage == "ppo" and self.reward_model_type == "oft" and self.finetuning_type != "oft":
   569:             raise ValueError("`reward_model_type` cannot be oft for Freeze/Full PPO training.")
   570: 
   571:         if self.stage == "dpo" and self.pref_loss != "sigmoid" and self.dpo_label_smoothing > 1e-6:
   572:             raise ValueError("`dpo_label_smoothing` is only valid for sigmoid loss function.")
   573: 
   574:         if self.use_llama_pro and self.finetuning_type == "full":
   575:             raise ValueError("`use_llama_pro` is only valid for Freeze or LoRA training.")
   576: 
   577:         if self.finetuning_type == "lora" and (self.use_galore or self.use_apollo or self.use_badam):
   578:             raise ValueError("Cannot use LoRA with GaLore, APOLLO or BAdam together.")
   579: 
   580:         if int(self.use_galore) + int(self.use_apollo) + (self.use_badam) > 1:
   581:             raise ValueError("Cannot use GaLore, APOLLO or BAdam together.")
   582: 
   583:         if self.pissa_init and (self.stage in ["ppo", "kto"] or self.use_ref_model):
   584:             raise ValueError("Cannot use PiSSA for current training stage.")
   585: 
   586:         if self.finetuning_type != "lora":
   587:             if self.loraplus_lr_ratio is not None:
   588:                 raise ValueError("`loraplus_lr_ratio` is only valid for LoRA training.")
   589: 
   590:             if self.use_rslora:
   591:                 raise ValueError("`use_rslora` is only valid for LoRA training.")
   592: 
   593:             if self.use_dora:
   594:                 raise ValueError("`use_dora` is only valid for LoRA training.")
   595: 
   596:             if self.pissa_init:
   597:                 raise ValueError("`pissa_init` is only valid for LoRA training.")
   598: 
   599:     def to_dict(self) -> dict[str, Any]:
   600:         args = asdict(self)
   601:         args = {k: f"<{k.upper()}>" if k.endswith("api_key") else v for k, v in args.items()}
   602:         return args
```


## Adapter Warnings

Some reference context could not be rendered completely:

- `scripts/train.sh` read context source file was not found


## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **train** — wall-clock budget `4:00:00`, compute share `2`
- **math_eval** — wall-clock budget `1:30:00`, compute share `0.5`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.





## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
