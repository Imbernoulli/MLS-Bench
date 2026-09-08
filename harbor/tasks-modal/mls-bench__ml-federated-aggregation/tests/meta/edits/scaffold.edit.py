"""SCAFFOLD baseline — full Algorithm 1.

SCAFFOLD (Karimireddy et al. ICML 2020) corrects client drift by adding
a per-step control-variate correction during local SGD. Because this
template now exposes the full client loop through the Strategy API, we
can implement Alg 1 exactly: plain SGD (no momentum) with corrected
gradient ``g + c - c_i`` at each step, Option-II control-variate update.

Reference: Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging
for Federated Learning", ICML 2020. arXiv:1910.06378.
"""

_FILE = "flower/custom_fl_aggregation.py"

_SCAFFOLD_STRATEGY = """\
class Strategy:
    \"\"\"SCAFFOLD — Alg 1 with Option-II control-variate update.\"\"\"

    def __init__(self, global_model, args):
        self.args = args
        self.num_clients = args.num_clients
        self.global_control = OrderedDict(
            (k, torch.zeros_like(v, device=\"cpu\"))
            for k, v in global_model.state_dict().items()
        )
        self.client_controls = {}   # client_idx -> OrderedDict (CPU)

    def _zero_like_state(self, state_dict):
        return OrderedDict(
            (k, torch.zeros_like(v, device=\"cpu\")) for k, v in state_dict.items()
        )

    def _get_client_control(self, client_idx, reference_state):
        c_i = self.client_controls.get(client_idx)
        if c_i is None:
            c_i = self._zero_like_state(reference_state)
            self.client_controls[client_idx] = c_i
        return c_i

    def _ensure_global_control_on(self, device, reference_state):
        # Lazily move global_control to the model's device once and keep it there.
        if (not hasattr(self, \"_gc_dev\")) or self._gc_dev_id != id(device):
            self.global_control = OrderedDict(
                (k, v.to(device) if v.is_floating_point() else v)
                for k, v in self.global_control.items()
            )
            self._gc_dev = device
            self._gc_dev_id = id(device)

    def _get_client_control_on(self, client_idx, reference_state, device):
        c_i = self.client_controls.get(client_idx)
        if c_i is None:
            c_i = OrderedDict(
                (k, torch.zeros_like(v, device=device) if v.is_floating_point()
                 else torch.zeros_like(v, device=\"cpu\"))
                for k, v in reference_state.items()
            )
            self.client_controls[client_idx] = c_i
        elif any(v.device != device for v in c_i.values() if v.is_floating_point()):
            c_i = OrderedDict(
                (k, v.to(device) if v.is_floating_point() else v)
                for k, v in c_i.items()
            )
            self.client_controls[client_idx] = c_i
        return c_i

    def client_local_train(self, global_state_dict, client_dataset, model_fn,
                           loss_fn, local_epochs, local_lr, local_batch_size,
                           device, client_idx):
        model = model_fn()
        model.load_state_dict(global_state_dict)
        model.to(device)
        model.train()

        # Move global_control + c_i to device once; keep them resident.
        self._ensure_global_control_on(device, model.state_dict())
        c_i = self._get_client_control_on(client_idx, model.state_dict(), device)

        # Snapshot global params x ON DEVICE for Option-II later.
        x_dev = OrderedDict(
            (n, p.detach().clone())
            for n, p in model.named_parameters()
            if n in self.global_control
        )

        # Pre-compute (c - c_i) on device once per client.
        correction_dev = {}
        for name, p in model.named_parameters():
            if name in self.global_control:
                correction_dev[id(p)] = self.global_control[name] - c_i[name]

        optimizer = optim.SGD(model.parameters(), lr=local_lr)  # plain SGD
        loader = DataLoader(client_dataset, batch_size=local_batch_size,
                            shuffle=True, drop_last=False, num_workers=0)

        total_loss, total_samples, local_steps = 0.0, 0, 0
        for _ in range(local_epochs):
            for batch_data in loader:
                if len(batch_data) != 2:
                    continue
                inputs, targets = batch_data
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                if outputs.dim() == 3:
                    outputs = outputs.view(-1, outputs.size(-1))
                    targets = targets.view(-1)
                loss = loss_fn(outputs, targets)
                loss.backward()
                # Corrected gradient: g + (c - c_i). Pure in-place add on device.
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    corr = correction_dev.get(id(p))
                    if corr is not None:
                        p.grad.add_(corr)
                optimizer.step()
                local_steps += 1
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        # Option-II update — stay on device.
        if local_steps > 0 and local_lr > 0.0:
            denom = local_steps * local_lr
            new_ci = OrderedDict()
            delta_c = OrderedDict()
            for name, p in model.named_parameters():
                if name not in self.global_control:
                    continue
                # c_i+ = c_i - c + (x - y) / (K * eta)
                update = c_i[name] - self.global_control[name] + (x_dev[name] - p.detach()) / denom
                delta_c[name] = (update - c_i[name]).clone()
                new_ci[name] = update
            # Carry over non-FP buffer keys unchanged from existing c_i.
            for k, v in c_i.items():
                if k not in new_ci:
                    new_ci[k] = v
                    delta_c[k] = torch.zeros_like(v)
            self._pending_delta_c = getattr(self, \"_pending_delta_c\", {})
            self._pending_delta_c[client_idx] = delta_c
            self.client_controls[client_idx] = new_ci

        # Single GPU→CPU transfer for the returned state_dict (server aggregates on CPU).
        final_state = OrderedDict(
            (k, v.detach().cpu()) for k, v in model.state_dict().items()
        )
        avg_loss = total_loss / max(total_samples, 1)
        return final_state, len(client_dataset), avg_loss

    def aggregate(self, global_state_dict, client_updates, round_num):
        # FedAvg-style weighted model average.
        total_samples = sum(max(upd[1], 1) for upd in client_updates)
        new_state = OrderedDict()
        for key, ref in global_state_dict.items():
            if not ref.is_floating_point():
                new_state[key] = client_updates[0][0][key].detach().clone()
                continue
            acc = torch.zeros_like(ref, device=\"cpu\", dtype=torch.float32)
            for st, n, _ in client_updates:
                acc += st[key].detach().cpu().float() * (max(n, 1) / total_samples)
            new_state[key] = acc.to(ref.dtype)

        # Server-side global c update: c <- c + (|S|/N) * mean_i Δc_i.
        # δc_i are on the same device as global_control — stay there.
        deltas = getattr(self, \"_pending_delta_c\", {})
        if deltas:
            weight = len(client_updates) / max(self.num_clients, 1)
            n_updates = len(deltas)
            for key in self.global_control:
                if not self.global_control[key].is_floating_point():
                    continue
                acc = None
                for dc in deltas.values():
                    if key in dc and dc[key].is_floating_point():
                        contrib = dc[key].to(self.global_control[key].device)
                        acc = contrib.clone() if acc is None else acc + contrib
                if acc is not None:
                    self.global_control[key] = (
                        self.global_control[key] + (weight / n_updates) * acc
                    )
            self._pending_delta_c = {}
        return new_state

    def select_clients(self, num_available, num_to_select, round_num):
        return random.sample(range(num_available), min(num_to_select, num_available))
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 340,
        "end_line": 420,
        "content": _SCAFFOLD_STRATEGY,
    },
]
