"""TRA (Temporal Routing Adaptor) baseline — rigorous codebase edit ops.

Faithful reproduction of qlib's official TRAModel from:
  qlib/contrib/model/pytorch_tra.py
with benchmark hyperparameters from:
  examples/benchmarks/TRA/workflow_config_tra_Alpha158.yaml

Two OPS:
  1. Replace model code in custom_model.py with TRA implementation
  2. Replace only the editable dataset adapter / processor blocks to use
     MTSDatasetH with Alpha158+FilterCol (20 features)
"""

_MODEL_FILE = "qlib/custom_model.py"
_WORKFLOW_FILE = "qlib/workflow_config.yaml"

_TRA_MODEL = """\
# =====================================================================
# EDITABLE: CustomModel — implement your stock prediction model here
# =====================================================================
import io
import os
import copy
import math
import json
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from tqdm import tqdm

from qlib.constant import EPS
from qlib.log import get_module_logger

device = "cuda" if torch.cuda.is_available() else "cpu"


class RNN(nn.Module):
    \"\"\"RNN Model — verbatim from qlib/contrib/model/pytorch_tra.py.

    Args:
        input_size (int): input size (# features)
        hidden_size (int): hidden size
        num_layers (int): number of hidden layers
        rnn_arch (str): rnn architecture
        use_attn (bool): whether use attention layer.
            we use concat attention as https://github.com/fulifeng/Adv-ALSTM/
        dropout (float): dropout rate
    \"\"\"

    def __init__(
        self,
        input_size=16,
        hidden_size=64,
        num_layers=2,
        rnn_arch="GRU",
        use_attn=True,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_arch = rnn_arch
        self.use_attn = use_attn

        if hidden_size < input_size:
            # compression
            self.input_proj = nn.Linear(input_size, hidden_size)
        else:
            self.input_proj = None

        self.rnn = getattr(nn, rnn_arch)(
            input_size=min(input_size, hidden_size),
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        if self.use_attn:
            self.W = nn.Linear(hidden_size, hidden_size)
            self.u = nn.Linear(hidden_size, 1, bias=False)
            self.output_size = hidden_size * 2
        else:
            self.output_size = hidden_size

    def forward(self, x):
        if self.input_proj is not None:
            x = self.input_proj(x)

        rnn_out, last_out = self.rnn(x)
        if self.rnn_arch == "LSTM":
            last_out = last_out[0]
        last_out = last_out.mean(dim=0)

        if self.use_attn:
            laten = self.W(rnn_out).tanh()
            scores = self.u(laten).softmax(dim=1)
            att_out = (rnn_out * scores).sum(dim=1)
            last_out = torch.cat([last_out, att_out], dim=1)

        return last_out


class PositionalEncoding(nn.Module):
    # reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    \"\"\"Transformer Model — verbatim from qlib/contrib/model/pytorch_tra.py.

    Args:
        input_size (int): input size (# features)
        hidden_size (int): hidden size
        num_layers (int): number of transformer layers
        num_heads (int): number of heads in transformer
        dropout (float): dropout rate
    \"\"\"

    def __init__(
        self,
        input_size=16,
        hidden_size=64,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.input_proj = nn.Linear(input_size, hidden_size)

        self.pe = PositionalEncoding(input_size, dropout)
        layer = nn.TransformerEncoderLayer(
            nhead=num_heads, dropout=dropout, d_model=hidden_size, dim_feedforward=hidden_size * 4
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.output_size = hidden_size

    def forward(self, x):
        x = x.permute(1, 0, 2).contiguous()  # the first dim need to be time
        x = self.pe(x)

        x = self.input_proj(x)
        out = self.encoder(x)

        return out[-1]


class TRA(nn.Module):
    \"\"\"Temporal Routing Adaptor (TRA) — verbatim from qlib/contrib/model/pytorch_tra.py.

    TRA takes historical prediction errors & latent representation as inputs,
    then routes the input sample to a specific predictor for training & inference.

    Args:
        input_size (int): input size (RNN/Transformer's hidden size)
        num_states (int): number of latent states (i.e., trading patterns)
            If `num_states=1`, then TRA falls back to traditional methods
        hidden_size (int): hidden size of the router
        tau (float): gumbel softmax temperature
        src_info (str): information for the router
    \"\"\"

    def __init__(
        self,
        input_size,
        num_states=1,
        hidden_size=8,
        rnn_arch="GRU",
        num_layers=1,
        dropout=0.0,
        tau=1.0,
        src_info="LR_TPE",
    ):
        super().__init__()

        assert src_info in ["LR", "TPE", "LR_TPE"], "invalid `src_info`"

        self.num_states = num_states
        self.tau = tau
        self.rnn_arch = rnn_arch
        self.src_info = src_info

        self.predictors = nn.Linear(input_size, num_states)

        if self.num_states > 1:
            if "TPE" in src_info:
                self.router = getattr(nn, rnn_arch)(
                    input_size=num_states,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout,
                )
                self.fc = nn.Linear(hidden_size + input_size if "LR" in src_info else hidden_size, num_states)
            else:
                self.fc = nn.Linear(input_size, num_states)

    def reset_parameters(self):
        for child in self.children():
            child.reset_parameters()

    def forward(self, hidden, hist_loss):
        preds = self.predictors(hidden)

        if self.num_states == 1:  # no need for router when having only one prediction
            return preds, None, None

        if "TPE" in self.src_info:
            out = self.router(hist_loss)[1]  # TPE
            if self.rnn_arch == "LSTM":
                out = out[0]
            out = out.mean(dim=0)
            if "LR" in self.src_info:
                out = torch.cat([hidden, out], dim=-1)  # LR_TPE
        else:
            out = hidden  # LR

        out = self.fc(out)

        choice = F.gumbel_softmax(out, dim=-1, tau=self.tau, hard=True)
        prob = torch.softmax(out / self.tau, dim=-1)

        return preds, choice, prob


def evaluate(pred):
    pred = pred.rank(pct=True)  # transform into percentiles
    score = pred.score
    label = pred.label
    diff = score - label
    MSE = (diff**2).mean()
    MAE = (diff.abs()).mean()
    IC = score.corr(label, method="spearman")
    return {"MSE": MSE, "MAE": MAE, "IC": IC}


def shoot_infs(inp_tensor):
    \"\"\"Replaces inf by maximum of tensor\"\"\"
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf, as_tuple=False)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor


def sinkhorn(Q, n_iters=3, epsilon=0.1):
    # epsilon should be adjusted according to logits value's scale
    with torch.no_grad():
        Q = torch.exp(Q / epsilon)
        Q = shoot_infs(Q)
        for i in range(n_iters):
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= Q.sum(dim=1, keepdim=True)
    return Q


def loss_fn(pred, label):
    mask = ~torch.isnan(label)
    if len(pred.shape) == 2:
        label = label[:, None]
    return (pred[mask] - label[mask]).pow(2).mean(dim=0)


def minmax_norm(x):
    xmin = x.min(dim=-1, keepdim=True).values
    xmax = x.max(dim=-1, keepdim=True).values
    mask = (xmin == xmax).squeeze()
    x = (x - xmin) / (xmax - xmin + EPS)
    x[mask] = 1
    return x


def transport_sample(all_preds, label, choice, prob, hist_loss, count, transport_method, alpha, training=False):
    \"\"\"
    sample-wise transport — verbatim from qlib/contrib/model/pytorch_tra.py.
    \"\"\"
    assert all_preds.shape == choice.shape
    assert len(all_preds) == len(label)
    assert transport_method in ["oracle", "router"]

    all_loss = torch.zeros_like(all_preds)
    mask = ~torch.isnan(label)
    all_loss[mask] = (all_preds[mask] - label[mask, None]).pow(2)  # [sample x states]

    L = minmax_norm(all_loss.detach())
    Lh = L * alpha + minmax_norm(hist_loss) * (1 - alpha)  # add hist loss for transport
    Lh = minmax_norm(Lh)
    P = sinkhorn(-Lh)
    del Lh

    if transport_method == "router":
        if training:
            pred = (all_preds * choice).sum(dim=1)  # gumbel softmax
        else:
            pred = all_preds[range(len(all_preds)), prob.argmax(dim=-1)]  # argmax
    else:
        pred = (all_preds * P).sum(dim=1)

    if transport_method == "router":
        loss = loss_fn(pred, label)
    else:
        loss = (all_loss * P).sum(dim=1).mean()

    return loss, pred, L, P


def transport_daily(all_preds, label, choice, prob, hist_loss, count, transport_method, alpha, training=False):
    \"\"\"
    daily transport — verbatim from qlib/contrib/model/pytorch_tra.py.
    \"\"\"
    assert len(prob) == len(count)
    assert len(all_preds) == sum(count)
    assert transport_method in ["oracle", "router"]

    all_loss = []  # loss of all predictions
    start = 0
    for i, cnt in enumerate(count):
        slc = slice(start, start + cnt)  # samples from the i-th day
        start += cnt
        tloss = loss_fn(all_preds[slc], label[slc])  # loss of the i-th day
        all_loss.append(tloss)
    all_loss = torch.stack(all_loss, dim=0)  # [days x states]

    L = minmax_norm(all_loss.detach())
    Lh = L * alpha + minmax_norm(hist_loss) * (1 - alpha)  # add hist loss for transport
    Lh = minmax_norm(Lh)
    P = sinkhorn(-Lh)
    del Lh

    pred = []
    start = 0
    for i, cnt in enumerate(count):
        slc = slice(start, start + cnt)  # samples from the i-th day
        start += cnt
        if transport_method == "router":
            if training:
                tpred = all_preds[slc] @ choice[i]  # gumbel softmax
            else:
                tpred = all_preds[slc][:, prob[i].argmax(dim=-1)]  # argmax
        else:
            tpred = all_preds[slc] @ P[i]
        pred.append(tpred)
    pred = torch.cat(pred, dim=0)  # [samples]

    if transport_method == "router":
        loss = loss_fn(pred, label)
    else:
        loss = (all_loss * P).sum(dim=1).mean()

    return loss, pred, L, P


def load_state_dict_unsafe(model, state_dict):
    \"\"\"
    Load state dict to provided model while ignore exceptions.
    \"\"\"

    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model)
    load = None  # break load->load reference cycle

    return {"unexpected_keys": unexpected_keys, "missing_keys": missing_keys, "error_msgs": error_msgs}


def plot(P):
    assert isinstance(P, pd.DataFrame)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    P.plot.area(ax=axes[0], xlabel="")
    P.idxmax(axis=1).value_counts().sort_index().plot.bar(ax=axes[1], xlabel="")
    plt.tight_layout()

    with io.BytesIO() as buf:
        plt.savefig(buf, format="png")
        buf.seek(0)
        img = plt.imread(buf)
        plt.close()

    return np.uint8(img * 255)


class CustomModel(Model):
    \"\"\"TRA Model — faithful to qlib's official TRAModel (pytorch_tra.py).

    Hyperparameters from official benchmark:
    examples/benchmarks/TRA/workflow_config_tra_Alpha158.yaml

    The workflow provides MTSDatasetH with Alpha158+FilterCol (20 features).
    fit()/predict() use dataset.prepare() to get MTSDataSampler loaders.
    \"\"\"

    def __init__(self):
        super().__init__()
        self.logger = get_module_logger("TRA")

        # Official benchmark hyperparameters
        self.model_config = {
            "input_size": 20,
            "hidden_size": 64,
            "num_layers": 2,
            "rnn_arch": "LSTM",
            "use_attn": True,
            "dropout": 0.0,
        }
        self.tra_config = {
            "num_states": 3,
            "rnn_arch": "LSTM",
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "tau": 1.0,
            "src_info": "LR_TPE",
        }
        self.model_type = "RNN"
        self.lr = 1e-3
        self.n_epochs = 100
        self.early_stop = 20
        self.update_freq = 1
        self.max_steps_per_epoch = None
        self.lamb = 1.0
        self.rho = 0.99
        self.alpha = 0.5
        self.seed = int(os.environ.get("SEED", "42"))
        self.logdir = None
        self.eval_train = False
        self.eval_test = True
        self.pretrain = True
        self.init_state = None
        self.reset_router = False
        self.freeze_model = False
        self.freeze_predictors = False
        self.transport_method = "router"
        self.use_daily_transport = False  # memory_mode=sample
        self.transport_fn = transport_sample

        self._writer = None

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self._init_model()

    def _init_model(self):
        self.logger.info("init TRAModel...")

        self.model = eval(self.model_type)(**self.model_config).to(device)
        print(self.model)

        self.tra = TRA(self.model.output_size, **self.tra_config).to(device)
        print(self.tra)

        if self.init_state:
            self.logger.warning(f"load state dict from `init_state`")
            state_dict = torch.load(self.init_state, map_location="cpu")
            self.model.load_state_dict(state_dict["model"])
            res = load_state_dict_unsafe(self.tra, state_dict["tra"])
            self.logger.warning(str(res))

        if self.reset_router:
            self.logger.warning(f"reset TRA.router parameters")
            self.tra.fc.reset_parameters()
            self.tra.router.reset_parameters()

        if self.freeze_model:
            self.logger.warning(f"freeze model parameters")
            for param in self.model.parameters():
                param.requires_grad_(False)

        if self.freeze_predictors:
            self.logger.warning(f"freeze TRA.predictors parameters")
            for param in self.tra.predictors.parameters():
                param.requires_grad_(False)

        self.logger.info("# model params: %d" % sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        self.logger.info("# tra params: %d" % sum(p.numel() for p in self.tra.parameters() if p.requires_grad))

        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.tra.parameters()), lr=self.lr)

        self.fitted = False
        self.global_step = -1

    def train_epoch(self, epoch, data_set, is_pretrain=False):
        self.model.train()
        self.tra.train()
        data_set.train()
        self.optimizer.zero_grad()

        P_all = []
        prob_all = []
        choice_all = []
        max_steps = len(data_set)
        if self.max_steps_per_epoch is not None:
            if epoch == 0 and self.max_steps_per_epoch < max_steps:
                self.logger.info(f"max steps updated from {max_steps} to {self.max_steps_per_epoch}")
            max_steps = min(self.max_steps_per_epoch, max_steps)

        cur_step = 0
        total_loss = 0
        total_count = 0
        for batch in tqdm(data_set, total=max_steps):
            cur_step += 1
            if cur_step > max_steps:
                break

            if not is_pretrain:
                self.global_step += 1

            data, state, label, count = batch["data"], batch["state"], batch["label"], batch["daily_count"]
            index = batch["daily_index"] if self.use_daily_transport else batch["index"]

            with torch.set_grad_enabled(not self.freeze_model):
                hidden = self.model(data)

            all_preds, choice, prob = self.tra(hidden, state)

            if is_pretrain or self.transport_method != "none":
                # NOTE: use oracle transport for pre-training
                loss, pred, L, P = self.transport_fn(
                    all_preds,
                    label,
                    choice,
                    prob,
                    state.mean(dim=1),
                    count,
                    self.transport_method if not is_pretrain else "oracle",
                    self.alpha,
                    training=True,
                )
                data_set.assign_data(index, L)  # save loss to memory
                if self.use_daily_transport:  # only save for daily transport
                    P_all.append(pd.DataFrame(P.detach().cpu().numpy(), index=index))
                    prob_all.append(pd.DataFrame(prob.detach().cpu().numpy(), index=index))
                    choice_all.append(pd.DataFrame(choice.detach().cpu().numpy(), index=index))
                decay = self.rho ** (self.global_step // 100)  # decay every 100 steps
                lamb = 0 if is_pretrain else self.lamb * decay
                reg = prob.log().mul(P).sum(dim=1).mean()  # train router to predict TO assignment
                if self._writer is not None and not is_pretrain:
                    self._writer.add_scalar("training/router_loss", -reg.item(), self.global_step)
                    self._writer.add_scalar("training/reg_loss", loss.item(), self.global_step)
                    self._writer.add_scalar("training/lamb", lamb, self.global_step)
                    if not self.use_daily_transport:
                        P_mean = P.mean(axis=0).detach()
                        self._writer.add_scalar("training/P", P_mean.max() / P_mean.min(), self.global_step)
                loss = loss - lamb * reg
            else:
                pred = all_preds.mean(dim=1)
                loss = loss_fn(pred, label)

            (loss / self.update_freq).backward()
            if cur_step % self.update_freq == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self._writer is not None and not is_pretrain:
                self._writer.add_scalar("training/total_loss", loss.item(), self.global_step)

            total_loss += loss.item()
            total_count += 1

        if self.use_daily_transport and len(P_all) > 0:
            P_all = pd.concat(P_all, axis=0)
            prob_all = pd.concat(prob_all, axis=0)
            choice_all = pd.concat(choice_all, axis=0)
            P_all.index = data_set.restore_daily_index(P_all.index)
            prob_all.index = P_all.index
            choice_all.index = P_all.index
            if not is_pretrain:
                self._writer.add_image("P", plot(P_all), epoch, dataformats="HWC")
                self._writer.add_image("prob", plot(prob_all), epoch, dataformats="HWC")
                self._writer.add_image("choice", plot(choice_all), epoch, dataformats="HWC")

        total_loss /= total_count

        if self._writer is not None and not is_pretrain:
            self._writer.add_scalar("training/loss", total_loss, epoch)

        return total_loss

    def test_epoch(self, epoch, data_set, return_pred=False, prefix="test", is_pretrain=False):
        self.model.eval()
        self.tra.eval()
        data_set.eval()

        preds = []
        probs = []
        P_all = []
        metrics = []
        for batch in tqdm(data_set):
            data, state, label, count = batch["data"], batch["state"], batch["label"], batch["daily_count"]
            index = batch["daily_index"] if self.use_daily_transport else batch["index"]

            with torch.no_grad():
                hidden = self.model(data)
                all_preds, choice, prob = self.tra(hidden, state)

            if is_pretrain or self.transport_method != "none":
                loss, pred, L, P = self.transport_fn(
                    all_preds,
                    label,
                    choice,
                    prob,
                    state.mean(dim=1),
                    count,
                    self.transport_method if not is_pretrain else "oracle",
                    self.alpha,
                    training=False,
                )
                data_set.assign_data(index, L)  # save loss to memory
                if P is not None and return_pred:
                    P_all.append(pd.DataFrame(P.cpu().numpy(), index=index))
            else:
                pred = all_preds.mean(dim=1)

            X = np.c_[pred.cpu().numpy(), label.cpu().numpy(), all_preds.cpu().numpy()]
            columns = ["score", "label"] + ["score_%d" % d for d in range(all_preds.shape[1])]
            pred = pd.DataFrame(X, index=batch["index"], columns=columns)

            metrics.append(evaluate(pred))

            if return_pred:
                preds.append(pred)
                if prob is not None:
                    columns = ["prob_%d" % d for d in range(all_preds.shape[1])]
                    probs.append(pd.DataFrame(prob.cpu().numpy(), index=index, columns=columns))

        metrics = pd.DataFrame(metrics)
        metrics = {
            "MSE": metrics.MSE.mean(),
            "MAE": metrics.MAE.mean(),
            "IC": metrics.IC.mean(),
            "ICIR": metrics.IC.mean() / metrics.IC.std(),
        }

        if self._writer is not None and epoch >= 0 and not is_pretrain:
            for key, value in metrics.items():
                self._writer.add_scalar(prefix + "/" + key, value, epoch)

        if return_pred:
            preds = pd.concat(preds, axis=0)
            preds.index = data_set.restore_index(preds.index)
            preds.index = preds.index.swaplevel()
            preds.sort_index(inplace=True)

            if probs:
                probs = pd.concat(probs, axis=0)
                if self.use_daily_transport:
                    probs.index = data_set.restore_daily_index(probs.index)
                else:
                    probs.index = data_set.restore_index(probs.index)
                    probs.index = probs.index.swaplevel()
                    probs.sort_index(inplace=True)

            if len(P_all):
                P_all = pd.concat(P_all, axis=0)
                if self.use_daily_transport:
                    P_all.index = data_set.restore_daily_index(P_all.index)
                else:
                    P_all.index = data_set.restore_index(P_all.index)
                    P_all.index = P_all.index.swaplevel()
                    P_all.sort_index(inplace=True)

        return metrics, preds, probs, P_all

    def _fit(self, train_set, valid_set, test_set, evals_result, is_pretrain=True):
        best_score = -1
        best_epoch = 0
        stop_rounds = 0
        best_params = {
            "model": copy.deepcopy(self.model.state_dict()),
            "tra": copy.deepcopy(self.tra.state_dict()),
        }
        # train
        if not is_pretrain and self.transport_method != "none":
            self.logger.info("init memory...")
            self.test_epoch(-1, train_set)

        for epoch in range(self.n_epochs):
            self.logger.info("Epoch %d:", epoch)

            self.logger.info("training...")
            self.train_epoch(epoch, train_set, is_pretrain=is_pretrain)

            self.logger.info("evaluating...")
            # NOTE: during evaluating, the whole memory will be refreshed
            if not is_pretrain and (self.transport_method == "router" or self.eval_train):
                train_set.clear_memory()  # NOTE: clear the shared memory
                train_metrics = self.test_epoch(epoch, train_set, is_pretrain=is_pretrain, prefix="train")[0]
                evals_result["train"].append(train_metrics)
                self.logger.info("train metrics: %s" % train_metrics)

            valid_metrics = self.test_epoch(epoch, valid_set, is_pretrain=is_pretrain, prefix="valid")[0]
            evals_result["valid"].append(valid_metrics)
            self.logger.info("valid metrics: %s" % valid_metrics)

            if self.eval_test:
                test_metrics = self.test_epoch(epoch, test_set, is_pretrain=is_pretrain, prefix="test")[0]
                evals_result["test"].append(test_metrics)
                self.logger.info("test metrics: %s" % test_metrics)

            if valid_metrics["IC"] > best_score:
                best_score = valid_metrics["IC"]
                stop_rounds = 0
                best_epoch = epoch
                best_params = {
                    "model": copy.deepcopy(self.model.state_dict()),
                    "tra": copy.deepcopy(self.tra.state_dict()),
                }
                if self.logdir is not None:
                    torch.save(best_params, self.logdir + "/model.bin")
            else:
                stop_rounds += 1
                if stop_rounds >= self.early_stop:
                    self.logger.info("early stop @ %s" % epoch)
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_params["model"])
        self.tra.load_state_dict(best_params["tra"])

        return best_score

    def fit(self, dataset, evals_result=dict()):
        # MTSDatasetH is provided by the workflow (Alpha158+FilterCol, 20 features).
        train_set, valid_set, test_set = dataset.prepare(["train", "valid", "test"])

        self.fitted = True
        self.global_step = -1

        evals_result["train"] = []
        evals_result["valid"] = []
        evals_result["test"] = []

        if self.pretrain:
            self.logger.info("pretraining...")
            self.optimizer = optim.Adam(
                list(self.model.parameters()) + list(self.tra.predictors.parameters()), lr=self.lr
            )
            self._fit(train_set, valid_set, test_set, evals_result, is_pretrain=True)

            # reset optimizer
            self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.tra.parameters()), lr=self.lr)

        self.logger.info("training...")
        best_score = self._fit(train_set, valid_set, test_set, evals_result, is_pretrain=False)

        self.logger.info("inference")
        train_metrics, train_preds, train_probs, train_P = self.test_epoch(-1, train_set, return_pred=True)
        self.logger.info("train metrics: %s" % train_metrics)

        valid_metrics, valid_preds, valid_probs, valid_P = self.test_epoch(-1, valid_set, return_pred=True)
        self.logger.info("valid metrics: %s" % valid_metrics)

        test_metrics, test_preds, test_probs, test_P = self.test_epoch(-1, test_set, return_pred=True)
        self.logger.info("test metrics: %s" % test_metrics)

    def predict(self, dataset, segment="test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        test_set = dataset.prepare(segment)

        metrics, preds, _, _ = self.test_epoch(-1, test_set, return_pred=True)
        self.logger.info("test metrics: %s" % metrics)

        return preds
"""

_TRA_DATASET_HEADER = """\
  dataset:
    class: MTSDatasetH
    module_path: qlib.contrib.data.dataset
    kwargs:
      handler:
        class: Alpha158
        module_path: qlib.contrib.data.handler
        kwargs:
"""

_TRA_DATASET_CONFIG = """\
          infer_processors:
            - class: FilterCol
              kwargs:
                fields_group: feature
                col_list:
                  - RESI5
                  - WVMA5
                  - RSQR5
                  - KLEN
                  - RSQR10
                  - CORR5
                  - CORD5
                  - CORR10
                  - ROC60
                  - RESI10
                  - VSTD5
                  - RSQR60
                  - CORR60
                  - WVMA60
                  - STD5
                  - RSQR20
                  - CORD60
                  - CORD10
                  - CORR20
                  - KLOW
            - class: RobustZScoreNorm
              kwargs:
                fields_group: feature
                clip_outlier: true
            - class: Fillna
              kwargs:
                fields_group: feature
          learn_processors:
            - class: CSRankNorm
              kwargs:
                fields_group: label
          label: ["Ref($close, -2) / Ref($close, -1) - 1"]
      seq_len: 60
      num_states: 3
      batch_size: 1024
      memory_mode: "sample"
      drop_last: true
"""

OPS = [
    {
        "op": "replace",
        "file": _MODEL_FILE,
        "start_line": 16,
        "end_line": 103,
        "content": _TRA_MODEL,
    },
    {
        "op": "replace",
        "file": _WORKFLOW_FILE,
        "start_line": 19,
        "end_line": 26,
        "content": _TRA_DATASET_HEADER,
    },
    {
        "op": "replace",
        "file": _WORKFLOW_FILE,
        "start_line": 32,
        "end_line": 45,
        "content": _TRA_DATASET_CONFIG,
    },
]
