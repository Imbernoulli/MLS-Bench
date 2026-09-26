"""Gradient Compression for Communication-Efficient Distributed Training.

Self-contained benchmark: trains standard vision models on CIFAR datasets
using data-parallel SGD with a pluggable gradient compressor.

The script simulates distributed training on a single node by:
1. Computing gradients normally
2. Encoding each gradient with the editable Compressor into a packet whose
   size fixed code charges in bits against a per-step budget
3. Decoding the packet in fixed code and using it for the optimizer step

This measures the effect of gradient compression on convergence quality at
a fixed communication budget, without requiring multi-node infrastructure.
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ============================================================================
# Model Definitions (FIXED)
# ============================================================================


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.linear(out)


class VGG(nn.Module):
    """VGG-11 with batch normalization."""

    def __init__(self, num_classes=100):
        super().__init__()
        cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend([
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                ])
                in_channels = v
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def build_model(model_name, num_classes, device):
    if model_name == 'resnet20':
        model = ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)
    elif model_name == 'resnet56':
        model = ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)
    elif model_name == 'vgg11':
        model = VGG(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


# ============================================================================
# Data Loading (FIXED)
# ============================================================================

def get_dataloaders(dataset_name, batch_size, num_workers=2):
    if dataset_name == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        num_classes = 10
        Dataset = datasets.CIFAR10
    elif dataset_name == 'cifar100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        num_classes = 100
        Dataset = datasets.CIFAR100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    _data_root = os.environ.get("DATA_ROOT", "/data")
    train_set = Dataset(_data_root + '/cifar', train=True, download=False,
                        transform=train_transform)
    test_set = Dataset(_data_root + '/cifar', train=False, download=False,
                       transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, num_classes


# ============================================================================
# EDITABLE SECTION — Gradient Compressor (lines 182-224)
# ============================================================================

class Compressor:
    """Gradient compressor. Default: plain Top-K sparsification (no error
    feedback), each tensor's k sized to its share of the step budget.

    Interface contract (enforced by the fixed code below this class):
    - __init__(compress_ratio, param_numels, budget_bits): `param_numels`
      maps every parameter name to its number of entries; `budget_bits` is
      the most one training step may transmit for all parameters together
      (compress_ratio x 32 bits x total entries, i.e. 100x at 0.01).
    - compress(tensor, name) -> packet, or a list of packets whose decodings
      are summed. Fixed code decodes the packet; only its contents reach
      the optimizer. A packet is one of
        {"values": V, "bits": b}                dense: V has one entry per
                                                gradient entry
        {"values": V, "bits": b, "indices": I}  sparse: I = distinct flat
                                                positions of the entries V
        {"factors": (P, Q)}                     low rank: P @ Q.T viewed as
                                                (shape[0], numel/shape[0])
      Values are sent as float32. Cost in bits (fixed `packet_cost`):
      len(V) * b, plus 32 per distinct value when b < 32 (the codebook;
      at most 2**b distinct values), plus Elias-gamma coded gaps of I,
      plus 32 per factor entry. A step over budget_bits stops the run.
    - The compressor may keep state across calls (e.g. error feedback).
      `name` identifies the parameter.
    """

    def __init__(self, compress_ratio, param_numels, budget_bits):
        self.compress_ratio = compress_ratio
        total = sum(param_numels.values())
        self.k = {}
        for name, numel in param_numels.items():
            share = budget_bits * numel // total
            k = 0
            while (k < numel and
                   32 * (k + 1) + elias_gamma_bound(k + 1, numel) <= share):
                k += 1
            self.k[name] = k

    def compress(self, tensor, name):
        flat = tensor.flatten()
        _, indices = torch.topk(flat.abs(), self.k[name], sorted=False)
        return {"values": flat[indices], "bits": 32, "indices": indices}


# ============================================================================
# FIXED SECTION — Transmission Format, Bit Accounting and Decoding
# ============================================================================

class CompressionContractError(RuntimeError):
    """A packet is malformed, or a step transmits more than its budget."""


def step_budget_bits(compress_ratio, param_numels):
    """Bits one step may transmit: compress_ratio x the dense float32 payload."""
    return int(compress_ratio * 32 * sum(param_numels.values()))


def elias_gamma_bound(k, numel):
    """Most Elias-gamma bits any k distinct positions in [0, numel) can cost
    (equal gaps are the worst case)."""
    if k <= 0:
        return 0
    return k + int(math.floor(2 * k * math.log2(numel / k) + 1e-9))


def _to_device(values, dtype, device):
    """A host list as a device tensor without waiting on the device queue."""
    t = torch.tensor(values, dtype=dtype)
    if device.type == "cuda":
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


def _parts(packet, shape, device):
    """Check a packet's structure and copy its tensors off the compressor.
    Returns a list of ("dense", v, b) / ("sparse", v, b, idx) /
    ("lowrank", P, Q) with float32 values and int64 indices.

    Containers must be plain dicts / lists / tuples, `bits` a plain int and
    every tensor exactly a torch.Tensor (no subclass): the compressor's
    tensors are first touched only through the unbound torch.Tensor.detach,
    so neither a subclass nor an instance attribute (say `t.numel = ...`)
    can make what is charged differ from what is decoded."""
    numel = shape.numel()
    if type(packet) is dict:
        packet = [packet]
    if not (type(packet) in (list, tuple) and packet
            and all(type(p) is dict for p in packet)):
        raise CompressionContractError(
            "compress must return a packet dict or a non-empty list of them")
    out = []
    for p in packet:
        if "factors" in p:
            if (set(p) != {"factors"} or type(p["factors"]) not in (tuple, list)
                    or len(p["factors"]) != 2):
                raise CompressionContractError(
                    "a low-rank packet is {'factors': (P, Q)}")
            P, Q = p["factors"]
            if type(P) is not torch.Tensor or type(Q) is not torch.Tensor:
                raise CompressionContractError(
                    "factors must be torch.Tensor (not a subclass)")
            P, Q = torch.Tensor.detach(P), torch.Tensor.detach(Q)
            rows = shape[0] if len(shape) >= 1 else 1
            cols = numel // rows
            if not (P.dim() == 2 and Q.dim() == 2
                    and P.shape[0] == rows and Q.shape[0] == cols
                    and P.shape[1] == Q.shape[1] and P.shape[1] >= 1):
                raise CompressionContractError(
                    f"factors must be P ({rows} x r) and Q ({cols} x r)")
            out.append(("lowrank",
                        P.to(device=device, dtype=torch.float32) + 0.0,
                        Q.to(device=device, dtype=torch.float32) + 0.0))
            continue
        if not (set(p) <= {"values", "bits", "indices"}
                and "values" in p and "bits" in p):
            raise CompressionContractError(
                "a packet is {'values', 'bits'[, 'indices']} or {'factors'}")
        b = p["bits"]
        if type(b) is not int or not 1 <= b <= 32:
            raise CompressionContractError("'bits' must be an int in [1, 32]")
        v = p["values"]
        if type(v) is not torch.Tensor:
            raise CompressionContractError(
                "'values' must be a torch.Tensor (not a subclass)")
        v = torch.Tensor.detach(v)
        if v.is_complex() or v.dtype == torch.bool:
            raise CompressionContractError("'values' must be a real tensor")
        v = v.reshape(-1).to(device=device, dtype=torch.float32) + 0.0
        idx = p.get("indices")
        if idx is None:
            if v.numel() != numel:
                raise CompressionContractError(
                    f"a dense packet needs {numel} values, got {v.numel()}")
            out.append(("dense", v, b))
            continue
        if type(idx) is not torch.Tensor:
            raise CompressionContractError(
                "'indices' must be a torch.Tensor (not a subclass)")
        idx = torch.Tensor.detach(idx)
        if idx.is_floating_point() or idx.is_complex() or idx.dtype == torch.bool:
            raise CompressionContractError("'indices' must be an integer tensor")
        idx = idx.reshape(-1).to(device=device, dtype=torch.int64).clone()
        if idx.numel() != v.numel():
            raise CompressionContractError(
                "'indices' and 'values' differ in length")
        out.append(("sparse", v, b, idx))
    return out


def _account(tensors, device, *, _to_device=_to_device):
    """Charge and decode the packets of one step, all on the device.

    `tensors` is a list of (numel, parts), one per gradient tensor. Cost:
    len(values) * bits, plus 32 bits per distinct value when bits < 32
    (the codebook, at most 2**bits entries), plus the Elias-gamma code of
    each sparse packet's sorted position gaps, plus 32 bits per low-rank
    factor entry. Returns (bits, ok, flat): ok = [finite, indices valid,
    codebook valid], flat = the decoded gradients, concatenated."""
    t_true = torch.ones((), dtype=torch.bool, device=device)
    total = sum(numel for numel, _ in tensors)
    flat = torch.zeros(total, dtype=torch.float32, device=device)
    static = 0
    finite, cb_vals, cb_caps = [], [], []
    sp_idx, sp_val, sp_meta = [], [], []  # meta: len, numel, off, part off
    off = part_off = 0
    for numel, parts in tensors:
        for part in parts:
            if part[0] == "lowrank":
                P, Q = part[1], part[2]
                static += 32 * (torch.Tensor.numel(P) + torch.Tensor.numel(Q))
                finite += [P.reshape(-1), Q.reshape(-1)]
                flat[off:off + numel].add_((P @ Q.t()).reshape(-1))
                continue
            v, b = part[1], part[2]
            static += torch.Tensor.numel(v) * b
            finite.append(v)
            if b < 32:
                cb_vals.append(v)
                cb_caps.append(2 ** b)
            if part[0] == "dense":
                flat[off:off + numel].add_(v)
            else:
                sp_idx.append(part[3])
                sp_val.append(v)
                sp_meta.append((torch.Tensor.numel(v), numel, off, part_off))
                part_off += numel
        off += numel
    bits = torch.full((), float(static), dtype=torch.float64, device=device)
    fin_ok = torch.isfinite(torch.cat(finite)).all() if finite else t_true
    cb_ok, idx_ok = t_true, t_true
    if cb_vals:
        lens = [torch.Tensor.numel(v) for v in cb_vals]
        v = torch.cat(cb_vals)
        meta = _to_device([lens, cb_caps], torch.int64, device)
        seg = torch.repeat_interleave(
            torch.arange(len(lens), device=device), meta[0],
            output_size=sum(lens))
        order = torch.sort(v, stable=True).indices
        order = order[torch.sort(seg[order], stable=True).indices]
        vs, ss = v[order], seg[order]
        start = torch.ones_like(vs, dtype=torch.bool)
        start[1:] = (vs[1:] != vs[:-1]) | (ss[1:] != ss[:-1])
        levels = torch.zeros(len(lens), dtype=torch.int64, device=device)
        levels.index_add_(0, ss, start.long())
        cb_ok = (levels <= meta[1]).all()
        bits = bits + 32 * levels.sum()
    if sp_idx:
        lens = [m[0] for m in sp_meta]
        meta = _to_device([list(col) for col in zip(*sp_meta)],
                          torch.int64, device)
        seg = torch.repeat_interleave(
            torch.arange(len(lens), device=device), meta[0],
            output_size=sum(lens))
        idx = torch.cat(sp_idx)
        numels = meta[1][seg]
        in_range = ((idx >= 0) & (idx < numels)).all()
        # Invalid indices fail the check; clamp so that, until the run
        # stops, the scatter stays inside the packet's own tensor.
        safe = torch.minimum(idx.clamp(min=0), numels - 1)
        flat.index_add_(0, safe + meta[2][seg], torch.cat(sp_val))
        order = torch.sort(safe + meta[3][seg]).indices
        local, ss = idx[order], seg[order]
        first = torch.ones_like(local, dtype=torch.bool)
        first[1:] = ss[1:] != ss[:-1]
        gap = torch.where(first, local + 1, local - torch.roll(local, 1))
        idx_ok = in_range & (gap > 0).all()
        bits = bits + (2 * torch.floor(torch.log2(
            gap.clamp(min=1).double())) + 1).sum()
    return bits, torch.stack([fin_ok, idx_ok, cb_ok]), flat


def _check(ok):
    if not ok[0]:
        raise CompressionContractError("a packet carries non-finite values")
    if not ok[1]:
        raise CompressionContractError(
            "packet indices must be distinct and inside the tensor")
    if not ok[2]:
        raise CompressionContractError(
            "a packet has more distinct values than 2**bits")


def packet_cost(packet, shape, *, _parts=_parts, _account=_account,
                _check=_check):
    """Bits `packet` (a packet or a list of them) costs for a tensor of
    `shape`, exactly as the per-step budget check charges it."""
    shape = torch.Size(shape)
    first = packet if isinstance(packet, dict) else (packet or [{}])[0]
    t = first.get("values") if isinstance(first, dict) else None
    if t is None and isinstance(first, dict) and first.get("factors"):
        t = first["factors"][0]
    device = t.device if torch.is_tensor(t) else torch.device("cpu")
    bits, ok, _ = _account([(shape.numel(), _parts(packet, shape, device))],
                           device)
    _check(ok.tolist())
    return int(bits.item())


def apply_gradient_compression(model, compressor, check_bindings, *,
                               _parts=_parts, _account=_account):
    """Encode every gradient with the compressor and replace it with the
    decoding of its packet: decoding is fixed, so the optimizer sees only
    what was transmitted. Returns the step's receipt (bits, then the three
    validity checks), filled asynchronously; `settle_step` reads it after
    the training loop's next host sync and stops the run on a violation.
    `check_bindings` runs after every compress call (see `bindings_check`)."""
    staged = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        packet = compressor.compress(param.grad.detach(), name)
        check_bindings()
        staged.append((param, _parts(packet, param.shape, param.device)))
    if not staged:
        return torch.tensor([0.0, 1.0, 1.0, 1.0], dtype=torch.float64), None
    device = staged[0][0].device
    bits, ok, flat = _account([(p.numel(), parts) for p, parts in staged],
                              device)
    receipt, done = torch.cat([bits.reshape(1), ok.double()]), None
    if device.type == "cuda":
        host = torch.empty(4, dtype=torch.float64, pin_memory=True)
        receipt = host.copy_(receipt, non_blocking=True)
        done = torch.cuda.Event()
        done.record()
    off = 0
    for param, _ in staged:
        n = param.numel()
        param.grad = flat[off:off + n].view(param.shape).to(param.dtype)
        off += n
    return receipt, done


def settle_step(receipt, budget_bits, *, _check=_check):
    """Check a step's receipt against the contract and the budget; returns
    the bits the step transmitted. Cheap after a host sync (loss.item())."""
    receipt, done = receipt
    if done is not None:
        done.synchronize()
    result = receipt.tolist()
    _check([bool(x) for x in result[1:]])
    if result[0] > budget_bits:
        raise CompressionContractError(
            f"a step transmitted {result[0]:.0f} bits, over the budget of "
            f"{budget_bits} bits (compress_ratio x 32 bits x parameter entries)")
    return result[0]


# ============================================================================
# FIXED SECTION — Training Loop
# ============================================================================

def cosine_lr(optimizer, epoch, total_epochs, warmup_epochs, base_lr, min_lr=0.0):
    """Cosine learning rate schedule with linear warmup."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / (warmup_epochs + 1)
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels, reduction='sum')
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc = 100.0 * correct / total
    avg_loss = total_loss / total
    return acc, avg_loss


import collections  # noqa: E402  (fixed code only)
import torch.nn.modules.module as _torch_module_py  # noqa: E402
import torch.optim.optimizer as _torch_optimizer_py  # noqa: E402

# The fixed functions reach one another through default arguments bound
# when they are defined, so compressor code that rebinds a module-level name
# (`global settle_step`, say) does not change what they run; the check below
# also stops the run if it tries.
_FIXED_NAMES = (
    "__builtins__", "argparse", "math", "os", "time", "torch", "nn", "F",
    "optim", "DataLoader", "datasets", "transforms", "conv3x3", "BasicBlock",
    "ResNet", "VGG", "build_model", "get_dataloaders",
    "CompressionContractError", "step_budget_bits", "elias_gamma_bound",
    "_to_device", "_parts", "_account", "_check", "packet_cost",
    "apply_gradient_compression", "settle_step", "cosine_lr", "evaluate")
# The builtins the fixed code calls; no module-level name may shadow them.
_FIXED_BUILTINS = (
    "all", "bool", "dict", "enumerate", "float", "int", "isinstance", "len",
    "list", "max", "print", "range", "set", "str", "sum", "tuple", "type",
    "zip")


def bindings_check(names=_FIXED_NAMES, builtin_names=_FIXED_BUILTINS, *,
                   _g=globals(), _error=CompressionContractError,
                   _hook_modules=(_torch_optimizer_py, _torch_module_py)):
    """Snapshot the module-level `names` and torch's global optimizer and
    module hook registries; the returned check stops the run if any of the
    names has since been rebound, if a module-level name shadows one of
    `builtin_names`, or if a global hook is registered (one could hand the
    optimizer something other than the decoded gradient)."""
    bound = tuple((name, _g[name]) for name in names)
    hooks = tuple((vars(mod), name, reg) for mod in _hook_modules
                  for name, reg in vars(mod).items()
                  if name.startswith("_global_") and "_hooks" in name)
    for _, name, reg in hooks:
        if type(reg) not in (dict, collections.OrderedDict):
            raise _error(f"torch's hook registry {name!r} was replaced")

    def check():
        for name, obj in bound:
            if _g.get(name) is not obj:
                raise _error(f"compressor code rebound the fixed name {name!r}")
        for name in builtin_names:
            if name in _g:
                raise _error(f"a module-level {name!r} shadows the builtin "
                             f"the fixed code calls")
        for mod_vars, name, reg in hooks:
            if reg or mod_vars.get(name) is not reg:
                raise _error(f"a global hook is registered in torch's {name!r}")
    return check


def train(args, *, apply_gradient_compression=apply_gradient_compression,
          settle_step=settle_step, evaluate=evaluate, cosine_lr=cosine_lr,
          check_bindings=bindings_check()):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    train_loader, test_loader, num_classes = get_dataloaders(
        args.dataset, args.batch_size)
    model = build_model(args.model, num_classes, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}, Dataset: {args.dataset}, "
          f"Parameters: {n_params:,}, Compress ratio: {args.compress_ratio}")

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=args.weight_decay)

    param_numels = {name: p.numel() for name, p in model.named_parameters()}
    budget_bits = step_budget_bits(args.compress_ratio, param_numels)
    dense_bits = 32 * n_params
    print(f"Communication budget: {budget_bits} bits per step "
          f"({dense_bits / budget_bits:.1f}x below the dense float32 "
          f"gradient of {dense_bits} bits)", flush=True)
    compressor = Compressor(args.compress_ratio, dict(param_numels),
                            budget_bits)
    total_bits = 0.0
    total_steps = 0

    best_acc = 0.0
    for epoch in range(args.epochs):
        lr = cosine_lr(optimizer, epoch, args.epochs, args.warmup_epochs,
                       args.lr, min_lr=args.lr * 0.01)
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_bits = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()

            # Apply gradient compression before optimizer step
            receipt = apply_gradient_compression(model, compressor,
                                                 check_bindings)

            optimizer.step()

            running_loss += loss.item()
            step_bits = settle_step(receipt, budget_bits)
            total_bits += step_bits
            total_steps += 1
            epoch_bits += step_bits
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total
        train_loss = running_loss / len(train_loader)
        comm_x = dense_bits * len(train_loader) / max(epoch_bits, 1.0)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs - 1:
            test_acc, test_loss = evaluate(model, test_loader, device)
            if test_acc > best_acc:
                best_acc = test_acc
            print(f"TRAIN_METRICS epoch={epoch+1} lr={lr:.6f} "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.2f} "
                  f"comm_x={comm_x:.1f} "
                  f"test_acc={test_acc:.2f} test_loss={test_loss:.4f}",
                  flush=True)
        else:
            print(f"TRAIN_METRICS epoch={epoch+1} lr={lr:.6f} "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.2f} "
                  f"comm_x={comm_x:.1f}", flush=True)

    # Final evaluation
    test_acc, test_loss = evaluate(model, test_loader, device)
    if test_acc > best_acc:
        best_acc = test_acc
    compression = dense_bits * total_steps / max(total_bits, 1.0)
    print(f"TEST_METRICS test_acc={test_acc:.2f} best_acc={best_acc:.2f} "
          f"test_loss={test_loss:.4f} compression={compression:.2f}",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description='Gradient Compression Benchmark')
    parser.add_argument('--model', type=str, default='resnet20',
                        choices=['resnet20', 'resnet56', 'vgg11'])
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--compress-ratio', type=float, default=0.01,
                        help='Compression ratio (fraction of gradient to keep)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
