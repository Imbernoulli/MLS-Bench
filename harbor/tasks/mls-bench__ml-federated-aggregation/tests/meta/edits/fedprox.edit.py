"""FedProx baseline.

FedProx adds a proximal term ``(mu/2) * ||w - w_global||^2`` to the local
objective to penalise drift from the global model under heterogeneous data.
Server-side aggregation is identical to FedAvg (sample-weighted mean).

Reference: Li et al., "Federated Optimization in Heterogeneous Networks",
MLSys 2020. arXiv:1812.06127.
"""

_FILE = "flower/custom_fl_aggregation.py"

_FEDPROX_STRATEGY = """\
class Strategy:
    \"\"\"FedProx — plain SGD + proximal term in the local objective.\"\"\"

    def __init__(self, global_model, args):
        self.args = args
        self.mu = 0.01  # Li et al. 2020 suggested range 0.001-0.1 on CIFAR/FEMNIST.

    def client_local_train(self, global_state_dict, client_dataset, model_fn,
                           loss_fn, local_epochs, local_lr, local_batch_size,
                           device, client_idx):
        model = model_fn()
        model.load_state_dict(global_state_dict)
        model.to(device)
        model.train()
        # Freeze copies of the global parameters for the prox term.
        global_params = [
            p.detach().clone() for p in model.parameters() if p.requires_grad
        ]
        mu_half = 0.5 * self.mu

        def prox_loss(m):
            prox = 0.0
            for w, w0 in zip(
                [p for p in m.parameters() if p.requires_grad],
                global_params,
            ):
                prox = prox + (w - w0).pow(2).sum()
            return mu_half * prox

        loader = DataLoader(client_dataset, batch_size=local_batch_size,
                            shuffle=True, drop_last=False, num_workers=0)
        avg_loss, _ = _default_client_sgd(
            model, loader, loss_fn, local_epochs, local_lr, device,
            loss_aug=prox_loss,
        )
        return model.cpu().state_dict(), len(client_dataset), avg_loss

    def aggregate(self, global_state_dict, client_updates, round_num):
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
        "content": _FEDPROX_STRATEGY,
    },
]
