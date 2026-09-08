"""Custom GPT-2 pretraining script for KV-structural reduction tasks.

Based on Andrej Karpathy's nanoGPT, with a narrow editable region for KV
structure changes such as grouped KV heads and latent KV compression.
"""

import ast
import inspect
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


# ── Editable region: KV structure design ──────────────────────────────────
# BEGIN KV EDITABLE REGION
def build_kv_heads(config):
    """Return the number of KV heads and per-head dimension."""

    n_kv_head = config.n_head
    head_dim = config.n_embd // config.n_head
    return n_kv_head, head_dim


def cross_layer_share(layer_idx, config):
    """Optionally reuse KV structure across layers.

    Default: no cross-layer KV sharing.
    """

    return False


def latent_kv_project(k, v, config):
    """Optional latent KV bottleneck.

    Default: identity projection.
    """

    return k, v, 1.0


def expand_kv_to_q_heads(tensor, target_heads):
    """Expand KV heads to query heads while remaining safe for any head count."""

    current_heads = tensor.size(1)
    if current_heads == target_heads:
        return tensor
    full_repeats = target_heads // current_heads
    remainder = target_heads % current_heads
    parts = []
    if full_repeats > 0:
        parts.append(tensor.repeat_interleave(full_repeats, dim=1))
    if remainder > 0:
        parts.append(tensor[:, :remainder, :, :])
    return torch.cat(parts, dim=1)


class CausalSelfAttention(nn.Module):
    _shared_kv_cache = {}

    def __init__(self, config, layer_idx=0):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.layer_idx = layer_idx
        self.n_kv_head, self.head_dim = build_kv_heads(config)
        self.share_across_layers = cross_layer_share(layer_idx, config)

        q_dim = config.n_embd
        kv_dim = 2 * self.n_kv_head * self.head_dim
        self.c_attn = nn.Linear(config.n_embd, q_dim + kv_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )
        self.use_pos_emb = True
        self.head_sharing_ratio = self.n_head / max(self.n_kv_head, 1)

    def forward(self, x):
        bsz, seq_len, channels = x.size()
        qkv = self.c_attn(x)
        q, kv = qkv.split(
            [self.n_embd, 2 * self.n_kv_head * self.head_dim],
            dim=2,
        )
        k, v = kv.chunk(2, dim=2)

        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)

        reused_previous = False
        if self.share_across_layers and (self.layer_idx - 1) in self._shared_kv_cache:
            k, v = self._shared_kv_cache[self.layer_idx - 1]
            reused_previous = True
        else:
            self._shared_kv_cache[self.layer_idx] = (k.detach(), v.detach())

        if self.n_kv_head != self.n_head:
            k = expand_kv_to_q_heads(k, self.n_head)
            v = expand_kv_to_q_heads(v, self.n_head)

        k, v, latent_ratio = latent_kv_project(k, v, self)
        self._last_latent_rank_ratio = float(latent_ratio)
        self._last_kv_storage_ratio = 0.0 if reused_previous else float(latent_ratio)
        self._uses_latent_compression = bool(latent_ratio < 0.999)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
        y = self.resid_dropout(self.c_proj(y))
        return y
# END KV EDITABLE REGION


def _validate_kv_editable_region():
    allowed_names = {
        "build_kv_heads",
        "cross_layer_share",
        "latent_kv_project",
        "expand_kv_to_q_heads",
        "MLARMSNorm",
        "rotate_half",
        "build_rotary_cache",
        "apply_rotary_pos_emb_interleave",
        "CausalSelfAttention",
    }
    required_names = {
        "build_kv_heads",
        "cross_layer_share",
        "latent_kv_project",
        "CausalSelfAttention",
    }
    with open(__file__, 'r') as _f:
        source = _f.read()
    start_marker = "# BEGIN KV EDITABLE REGION"
    end_marker = "# END KV EDITABLE REGION"
    start = source.index(start_marker) + len(start_marker)
    end = source.index(end_marker)
    snippet = source[start:end]
    parsed = ast.parse(snippet)
    seen_names = set()
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef):
            seen_names.add(node.name)
            if node.name not in allowed_names:
                raise RuntimeError(f"Forbidden top-level function in KV region: {node.name}")
        elif isinstance(node, ast.ClassDef):
            seen_names.add(node.name)
            if node.name not in allowed_names:
                raise RuntimeError(f"Forbidden top-level class in KV region: {node.name}")
        else:
            raise RuntimeError(
                "KV editable region may only contain top-level function/class definitions"
            )
    missing = required_names - seen_names
    if missing:
        raise RuntimeError(
            f"KV editable region is missing required definitions: {sorted(missing)}"
        )


_validate_kv_editable_region()


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, layer_idx=layer_idx)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config, layer_idx=i) for i in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def structural_metrics(self):
        head_sharing = []
        latent_rank = []
        kv_bytes = []
        for block in self.transformer.h:
            attn = block.attn
            n_head = int(getattr(attn, "n_head", self.config.n_head))
            n_kv_head = int(getattr(attn, "n_kv_head", n_head))
            head_dim = int(getattr(attn, "head_dim", self.config.n_embd // self.config.n_head))
            head_sharing.append(n_head / max(n_kv_head, 1))

            if getattr(attn, "share_across_layers", False):
                # This layer borrows KV from the previous layer; no new KV storage needed.
                latent_rank.append(0.0)
                kv_bytes.append(0.0)
            elif hasattr(attn, "kv_a_proj_with_mqa") and hasattr(attn, "kv_b_proj"):
                if hasattr(attn, "kv_a_layernorm"):
                    kv_lora_rank = int(attn.kv_a_layernorm.weight.numel())
                else:
                    kv_lora_rank = int(getattr(attn, "kv_lora_rank", head_dim))
                qk_rope_head_dim = int(attn.kv_a_proj_with_mqa.out_features - kv_lora_rank)
                qk_head_dim = int(getattr(attn, "qk_head_dim", head_dim + qk_rope_head_dim))
                latent_rank.append(kv_lora_rank / max(qk_head_dim, 1))
                kv_bytes.append(float(2 * (kv_lora_rank + qk_rope_head_dim)))
            else:
                latent_rank.append(1.0)
                kv_bytes.append(float(2 * n_kv_head * head_dim * 2))
        return {
            "head_sharing_ratio": sum(head_sharing) / len(head_sharing),
            "latent_rank_ratio": sum(latent_rank) / len(latent_rank),
            "kv_bytes_per_token": sum(kv_bytes) / len(kv_bytes),
        }

    def forward(self, idx, targets=None):
        device = idx.device
        _, t = idx.size()
        assert t <= self.config.block_size
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)
        use_pos = getattr(self.transformer.h[0].attn, "use_pos_emb", True)
        if use_pos:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            x = x + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer


def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def get_batch(split, data_dir, batch_size, block_size, device):
    data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
    if "cuda" in str(device):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


def get_named_eval_batch(dataset_path, batch_size, block_size, device):
    data = np.memmap(dataset_path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
    if "cuda" in str(device):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


if __name__ == "__main__":
    output_dir = os.environ.get("OUTPUT_DIR", "out")
    seed = int(os.environ.get("SEED", 1337))
    _DATA_ROOT = os.environ.get("DATA_ROOT", "/data")
    data_dir = os.environ.get("DATA_DIR", os.path.join(_DATA_ROOT, "climbmix"))
    eval_dir = os.environ.get("EVAL_DIR", os.path.join(_DATA_ROOT, "eval"))
    n_layer = int(os.environ.get("N_LAYER", 12))
    n_head = int(os.environ.get("N_HEAD", 12))
    n_embd = int(os.environ.get("N_EMBD", 768))
    max_iters = int(os.environ.get("MAX_ITERS", 5000))
    eval_interval = int(os.environ.get("EVAL_INTERVAL", 500))
    eval_iters = 200
    log_interval = 10
    batch_size = int(os.environ.get("BATCH_SIZE", 12))
    block_size = int(os.environ.get("BLOCK_SIZE", 1024))
    gradient_accumulation_steps = int(os.environ.get("GRAD_ACCUM", 5))
    run_aux_eval = os.environ.get("RUN_AUX_EVAL", "0") == "1"
    aux_eval_datasets = [
        item.strip() for item in os.environ.get("AUX_EVAL_DATASETS", "wikitext2").split(",") if item.strip()
    ]
    aux_eval_iters = int(os.environ.get("AUX_EVAL_ITERS", 64))
    aux_eval_batch_size = int(os.environ.get("AUX_EVAL_BATCH_SIZE", 4))
    learning_rate = float(os.environ.get("LEARNING_RATE", 6e-4))
    min_lr = learning_rate / 10
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0
    warmup_iters = int(max_iters * 0.04)
    lr_decay_iters = max_iters
    # torch.compile on MLA-style dynamic split/reshape (qk_nope+qk_rope,
    # kv_a_proj split, broadcast-expand across heads) generated more graph
    # breaks than speedup (6s/iter compiled vs 3.4s eager on H200). Keep
    # compile off here; sibling llm-pretrain-attention etc. are free to flip.
    compile_model = False
    dtype = "bfloat16"

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP

        dist.init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"

    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size

    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    if master_process:
        print(f"tokens per iteration will be: {tokens_per_iter:,}")

    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=False,
        vocab_size=50304,
        dropout=0.0,
    )
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
    raw_model = model.module if ddp else model

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                x, y = get_batch(split, data_dir, batch_size, block_size, device)
                with ctx:
                    _, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    @torch.no_grad()
    def evaluate_aux_metrics():
        # Heldout cross-entropy on each named eval dataset (WikiText-2/103 +
        # LAMBADA). Generation throughput is intentionally not measured: in
        # pure-PyTorch eager mode it tracks per-layer op count more than the
        # KV-structure design merit (see kv_bytes_per_token for the actual
        # efficiency axis).
        model.eval()
        heldout_values = []
        dataset_metrics = {}
        for dataset_name in aux_eval_datasets:
            dataset_path = os.path.join(eval_dir, f"{dataset_name}.bin")
            heldout_losses = torch.zeros(aux_eval_iters)
            for k in range(aux_eval_iters):
                x, y = get_named_eval_batch(dataset_path, aux_eval_batch_size, block_size, device)
                with ctx:
                    _, loss = model(x, y)
                heldout_losses[k] = loss.item()
            heldout_loss_value = float(heldout_losses.mean().item())
            heldout_values.append(heldout_loss_value)
            dataset_metrics[f"heldout_loss_{dataset_name}"] = heldout_loss_value
        model.train()
        return {
            "heldout_loss": float(sum(heldout_values) / len(heldout_values)),
            **dataset_metrics,
        }

    t0 = time.time()
    iter_num = 0
    while True:
        lr = get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(
                f"TRAIN_METRICS: step={iter_num} train_loss={losses['train']:.4f} val_loss={losses['val']:.4f}",
                flush=True,
            )

        for micro_step in range(gradient_accumulation_steps):
            x, y = get_batch("train", data_dir, batch_size, block_size, device)
            with ctx:
                _, loss = model(x, y)
                loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()

        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if iter_num % log_interval == 0 and master_process:
            dt = time.time() - t0
            print(f"iter {iter_num}: loss {loss.item() * gradient_accumulation_steps:.4f}, time {dt*1000:.2f}ms")
            t0 = time.time()

        iter_num += 1
        if iter_num > max_iters:
            break

    if master_process:
        losses = estimate_loss()
        stats = raw_model.structural_metrics()
        aux_metrics = evaluate_aux_metrics() if run_aux_eval else {
            "heldout_loss": float("nan"),
        }
        final_metrics = {
            "val_loss": float(losses["val"]),
            "kv_bytes_per_token": float(stats["kv_bytes_per_token"]),
            "head_sharing_ratio": float(stats["head_sharing_ratio"]),
            "latent_rank_ratio": float(stats["latent_rank_ratio"]),
            "heldout_loss": float(aux_metrics["heldout_loss"]),
        }
        if run_aux_eval:
            for dataset_name in aux_eval_datasets:
                final_metrics[f"heldout_loss_{dataset_name}"] = float(aux_metrics[f"heldout_loss_{dataset_name}"])
        metric_parts = []
        for key, value in final_metrics.items():
            if isinstance(value, float):
                if key in {"kv_bytes_per_token", "head_sharing_ratio", "latent_rank_ratio"}:
                    metric_parts.append(f"{key}={value:.2f}")
                else:
                    metric_parts.append(f"{key}={value:.6f}")
            else:
                metric_parts.append(f"{key}={value}")
        print("TEST_METRICS: " + " ".join(metric_parts), flush=True)

        env_label = os.environ.get("ENV", "model")
        save_model = raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model
        ckpt_data = {"model_state_dict": save_model.state_dict(), "model_args": model_args}
        ckpt_path = os.path.join(output_dir, f"ckpt_{env_label}.pt")
        torch.save(ckpt_data, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}", flush=True)

        src_path = os.path.join(output_dir, f"model_source_{env_label}.py")
        with open(os.path.abspath(__file__), "r", encoding="utf-8") as src, open(
            src_path, "w", encoding="utf-8"
        ) as dst:
            dst.write(src.read())
        print(f"Model source saved to {src_path}", flush=True)

        metrics_path = os.path.join(output_dir, f"metrics_{env_label}.json")
        metrics_snapshot = dict(final_metrics)
        metrics_snapshot.update({
            "_mlsbench_schema": "llm-kv-structural-reduction-replay-v1",
            "_mlsbench_env_label": env_label,
            "_mlsbench_seed": seed,
        })
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_snapshot, f, indent=2, sort_keys=True)
        print(f"Metrics snapshot saved to {metrics_path}", flush=True)

    if ddp:
        dist.destroy_process_group()
