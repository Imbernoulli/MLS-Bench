#!/usr/bin/env python3
"""Evaluate a nanoGPT checkpoint on standard benchmarks using lm-evaluation-harness."""

import argparse
import importlib.util
import os
import sys

# Ensure pip-installed lm_eval is used, not the local repo source
sys.path = [p for p in sys.path if p and 'lm-evaluation-harness' not in os.path.abspath(p)]

import torch
import tiktoken
import lm_eval
from lm_eval.api.model import LM


class NanoGPTLM(LM):
    """lm-evaluation-harness wrapper for nanoGPT models."""

    def __init__(self, checkpoint_path, source_path, device="cuda", batch_size=1):
        super().__init__()

        # Load model architecture from saved source file
        spec = importlib.util.spec_from_file_location("nanogpt_src", source_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Reconstruct and load model
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_args = ckpt["model_args"]
        config = mod.GPTConfig(**model_args)
        model = mod.GPT(config)
        # Strip _orig_mod. prefix from torch.compile'd checkpoints
        state_dict = ckpt["model_state_dict"]
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        self._model = model
        self._device_name = device
        self._batch_size = batch_size
        self._max_length = config.block_size

        # GPT-2 tokenizer via tiktoken
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self._eot_token_id = self.tokenizer.eot_token

    @property
    def eot_token_id(self):
        return self._eot_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device_name

    def tok_encode(self, string, **kwargs):
        return self.tokenizer.encode(string, allowed_special={"<|endoftext|>"})

    def tok_decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens)

    def _model_forward(self, input_ids):
        """Run forward pass, return logits for ALL positions.

        nanoGPT returns only last-token logits when targets=None (generation
        optimisation).  Pass a dummy target tensor so the model returns the
        full [B, T, V] logits needed by lm-eval's loglikelihood scoring.
        """
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Dummy targets to force full-sequence logits
                dummy_targets = input_ids
                logits, _ = self._model(input_ids, dummy_targets)
        return logits

    def loglikelihood(self, requests):
        results = []
        for req in requests:
            ctx, cont = req.args
            ctx_enc = self.tok_encode(ctx) if ctx else []
            cont_enc = self.tok_encode(cont)

            all_enc = (ctx_enc + cont_enc)[-self._max_length:]
            ctx_len = max(len(all_enc) - len(cont_enc), 0)

            if len(all_enc) < 2:
                results.append((-1e10, False))
                continue

            inp = torch.tensor([all_enc], device=self._device_name)
            logits = self._model_forward(inp)
            log_probs = torch.nn.functional.log_softmax(logits[0].float(), dim=-1)

            cont_log_prob = 0.0
            greedy_correct = True
            start = max(ctx_len - 1, 0)
            for i in range(start, len(all_enc) - 1):
                tok_id = all_enc[i + 1]
                cont_log_prob += log_probs[i, tok_id].item()
                if log_probs[i].argmax().item() != tok_id:
                    greedy_correct = False

            results.append((cont_log_prob, greedy_correct))
        return results

    def loglikelihood_rolling(self, requests):
        results = []
        for req in requests:
            string = req.args[0]
            tokens = self.tok_encode(string)

            total_log_prob = 0.0
            for start in range(0, max(1, len(tokens) - 1), self._max_length):
                chunk = tokens[start:start + self._max_length + 1]
                if len(chunk) < 2:
                    continue
                inp = torch.tensor([chunk[:-1]], device=self._device_name)
                logits = self._model_forward(inp)
                log_probs = torch.nn.functional.log_softmax(logits[0].float(), dim=-1)
                for i in range(len(chunk) - 1):
                    total_log_prob += log_probs[i, chunk[i + 1]].item()

            results.append((total_log_prob,))
        return results

    def generate_until(self, requests):
        results = []
        for req in requests:
            ctx, gen_kwargs = req.args
            max_gen = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            stop = gen_kwargs.get("until", [])

            tokens = self.tok_encode(ctx)[-self._max_length:]
            inp = torch.tensor([tokens], device=self._device_name)
            generated = []

            for _ in range(max_gen):
                logits = self._model_forward(inp[:, -self._max_length:])
                next_tok = logits[0, -1].argmax().item()
                generated.append(next_tok)
                inp = torch.cat(
                    [inp, torch.tensor([[next_tok]], device=self._device_name)], dim=1
                )
                decoded = self.tok_decode(generated)
                if any(s in decoded for s in stop):
                    break

            results.append(self.tok_decode(generated))
        return results


def main():
    parser = argparse.ArgumentParser(description="nanoGPT lm-eval benchmark")
    parser.add_argument("--checkpoint", required=True, help="Path to ckpt_{label}.pt")
    parser.add_argument("--source", required=True, help="Path to model_source_{label}.py")
    parser.add_argument("--tasks", default="hellaswag,arc_easy,piqa,winogrande")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}")
    print(f"Model source: {args.source}")

    model = NanoGPTLM(
        checkpoint_path=args.checkpoint,
        source_path=args.source,
        device=args.device,
        batch_size=args.batch_size,
    )

    task_list = [t.strip() for t in args.tasks.split(",")]
    print(f"Running tasks: {task_list} (num_fewshot={args.num_fewshot})")

    results = lm_eval.simple_evaluate(
        model=model,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
    )

    # Print per-task results
    metrics = {}
    for task_name, task_results in results["results"].items():
        # Prefer acc_norm if available (e.g. hellaswag), else acc
        acc = task_results.get("acc_norm,none", task_results.get("acc,none"))
        if acc is not None:
            metrics[task_name] = round(acc * 100, 2)
            print(f"{task_name}: {acc:.4f} ({acc*100:.2f}%)")

    # Print TEST_METRICS line for parser
    metrics_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
    print(f"TEST_METRICS: {metrics_str}", flush=True)


if __name__ == "__main__":
    main()
