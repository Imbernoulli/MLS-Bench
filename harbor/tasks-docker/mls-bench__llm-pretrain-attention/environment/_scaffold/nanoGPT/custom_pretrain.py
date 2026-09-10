"""Custom GPT-2 Pretraining Script
Based on Andrej Karpathy's nanoGPT, evaluated on FineWeb dataset.
"""

import math
import inspect
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# ============================================================================
# Model Components
# ============================================================================

# ── Normalization ──────────────────────────────────────────────────────────
class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

# ── Self-Attention ─────────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        # Set to False if using custom position encoding (e.g. RoPE)
        self.use_pos_emb = True

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

# ── Feed-Forward Network ──────────────────────────────────────────────────
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

# ── Transformer Block ─────────────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# ============================================================================
# GPT Model
# ============================================================================

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
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
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

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)
        # Conditionally add learned position embeddings
        use_pos = getattr(self.transformer.h[0].attn, 'use_pos_emb', True)
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

    # ── Optimizer Configuration ────────────────────────────────────────────
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

# ── Learning Rate Schedule ─────────────────────────────────────────────────
def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    """Cosine learning rate schedule with linear warmup."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# ============================================================================
# Data Loading
# ============================================================================

def get_batch(data, batch_size, block_size, device):
    """Get a random batch from a pre-opened memmap (nanoGPT style)."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y

# ============================================================================
# Training Script
# ============================================================================

if __name__ == '__main__':
    # ── Configuration from environment ──
    output_dir = os.environ.get('OUTPUT_DIR', 'out')
    seed = int(os.environ.get('SEED', 1337))
    data_dir = os.environ.get('DATA_DIR', '/data/climbmix')

    # Model config from environment
    n_layer = int(os.environ.get('N_LAYER', 12))
    n_head = int(os.environ.get('N_HEAD', 12))
    n_embd = int(os.environ.get('N_EMBD', 768))

    # Training hyperparameters (overridable via env for different model sizes)
    max_iters = int(os.environ.get('MAX_ITERS', 5000))
    eval_interval = int(os.environ.get('EVAL_INTERVAL', 500))
    eval_iters = 200
    log_interval = 10
    batch_size = int(os.environ.get('BATCH_SIZE', 12))
    block_size = 1024
    gradient_accumulation_steps = int(os.environ.get('GRAD_ACCUM', 5))
    learning_rate = float(os.environ.get('LEARNING_RATE', 6e-4))
    min_lr = learning_rate / 10
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0
    warmup_iters = int(max_iters * 0.04)
    lr_decay_iters = max_iters
    # CONFIG_OVERRIDES: override training hyperparameters for your method.
    # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
    CONFIG_OVERRIDES = {}

    # Apply per-method hyperparameter overrides
    for _k, _v in CONFIG_OVERRIDES.items():
        if _k == 'learning_rate': learning_rate = _v; min_lr = learning_rate / 10
        elif _k == 'weight_decay': weight_decay = _v
        elif _k == 'warmup_iters': warmup_iters = _v
        elif _k == 'min_lr': min_lr = _v
        elif _k == 'grad_clip': grad_clip = _v

    compile_model = True
    dtype = 'bfloat16'

    # ── DDP Setup ──
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        dist.init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        master_process = True
        device = 'cuda'
        seed_offset = 0

    # ── Setup ──
    device_type = 'cuda'
    torch.manual_seed(seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    if master_process:
        os.makedirs(output_dir, exist_ok=True)

    tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
    if ddp:
        tokens_per_iter *= int(os.environ.get('WORLD_SIZE', 1))
    if master_process:
        print(f"tokens per iteration will be: {tokens_per_iter:,}")

    # ── Load Data ──
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    if master_process:
        print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")

    # ── Model Init ──
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                      block_size=block_size, bias=False, vocab_size=50304, dropout=0.0)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.to(device)


    scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

    if compile_model:
        if master_process:
            print("compiling the model...")
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

    # ── Evaluation ──
    @torch.no_grad()
    def estimate_loss():
        out = {}
        raw = model.module if ddp else model
        raw.eval()
        for split, data in [('train', train_data), ('val', val_data)]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(data, batch_size, block_size, device)
                with ctx:
                    logits, loss = raw(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        raw.train()
        return out

    # ── Training Loop ──
    t0 = time.time()
    best_val_loss = 1e9

    for iter_num in range(max_iters + 1):
        lr = get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss()
            train_loss = losses['train'].item()
            val_loss = losses['val'].item()
            print(f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
            print(f"TRAIN_METRICS: step={iter_num}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}", flush=True)
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                X, Y = get_batch(train_data, batch_size, block_size, device)
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()

        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and iter_num > 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.6f}")

    # ── Free training state to reclaim GPU memory ──
    del optimizer, scaler
    import gc; gc.collect()
    torch.cuda.empty_cache()

    # ── Final Evaluation ──
    if master_process:
        losses = estimate_loss()
        val_loss = losses['val'].item()
        train_loss = losses['train'].item()
        print(f"Final: train loss {train_loss:.4f}, val loss {val_loss:.4f}, best val loss {best_val_loss:.4f}")

        # ── PPL on benchmark datasets ──
        eval_dir = os.environ.get('EVAL_DIR', '/data/eval')
        raw = model.module if ddp else model
        raw.eval()
        eval_datasets = ['wikitext2', 'lambada']
        ppl_results = {}
        for ds_name in eval_datasets:
            ds_path = os.path.join(eval_dir, f'{ds_name}.bin')
            if not os.path.exists(ds_path):
                print(f"Eval dataset not found: {ds_path}")
                continue
            data = np.memmap(ds_path, dtype=np.uint16, mode='r')
            n_tokens = len(data)
            # Process in non-overlapping chunks of block_size
            total_loss = 0.0
            n_chunks = 0
            with torch.no_grad():
                for start in range(0, n_tokens - block_size, block_size):
                    x = torch.from_numpy(data[start:start+block_size].astype(np.int64)).unsqueeze(0).to(device)
                    y = torch.from_numpy(data[start+1:start+1+block_size].astype(np.int64)).unsqueeze(0).to(device)
                    with ctx:
                        _, loss = raw(x, y)
                    total_loss += loss.item()
                    n_chunks += 1
            avg_loss = total_loss / n_chunks
            ppl = math.exp(avg_loss)
            ppl_results[ds_name] = ppl
            print(f"PPL {ds_name}: {ppl:.2f} (avg_loss={avg_loss:.4f}, {n_chunks} chunks)")

        ppl_str = ', '.join(f'{k}_ppl={v:.2f}' for k, v in ppl_results.items())
        print(f"TEST_METRICS: val_loss={val_loss:.4f}, {ppl_str}", flush=True)

        # ── Save checkpoint for downstream evaluation (lm-eval-harness) ──
        import shutil
        env_label = os.environ.get('ENV', 'model')
        # Unwrap torch.compile to get clean state_dict keys
        save_model = raw._orig_mod if hasattr(raw, '_orig_mod') else raw
        ckpt_data = {'model_state_dict': save_model.state_dict(), 'model_args': model_args}
        ckpt_path = os.path.join(output_dir, f'ckpt_{env_label}.pt')
        torch.save(ckpt_data, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")
        src_path = os.path.join(output_dir, f'model_source_{env_label}.py')
        shutil.copy2(os.path.abspath(__file__), src_path)
        print(f"Model source saved to {src_path}")

    if ddp:
        dist.destroy_process_group()
