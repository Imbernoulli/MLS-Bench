"""FedAvg (Federated Averaging) baseline.

Plain SGD client training + sample-count-weighted average on the server.

Reference: McMahan et al., "Communication-Efficient Learning of Deep Networks
from Decentralized Data", AISTATS 2017. arXiv:1602.05629.
"""

_FILE = "flower/custom_fl_aggregation.py"

_FEDAVG_STRATEGY = """\
class Strategy:
    \"\"\"FedAvg — plain SGD + weighted average of client state dicts.\"\"\"

    def __init__(self, global_model, args):
        self.args = args

    def client_local_train(self, global_state_dict, client_dataset, model_fn,
                           loss_fn, local_epochs, local_lr, local_batch_size,
                           device, client_idx):
        model = model_fn()
        model.load_state_dict(global_state_dict)
        model.to(device)
        model.train()
        loader = DataLoader(client_dataset, batch_size=local_batch_size,
                            shuffle=True, drop_last=False, num_workers=0)
        avg_loss, _ = _default_client_sgd(
            model, loader, loss_fn, local_epochs, local_lr, device)
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
        "content": _FEDAVG_STRATEGY,
    },
]
