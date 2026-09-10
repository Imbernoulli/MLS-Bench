"""Transformer baseline — rigorous codebase edit ops.

Faithful reproduction of qlib's official TransformerModel from:
  qlib/contrib/model/pytorch_transformer.py  (NON-TS version, uses DatasetH)
with benchmark hyperparameters from:
  examples/benchmarks/Transformer/workflow_config_transformer_Alpha360.yaml

Only replaces the CustomModel class (model code).
Workflow config is unchanged — uses DatasetH with Alpha360 (same as official).
"""

_FILE = "qlib/custom_model.py"

_TRANSFORMER_MODEL = """\
# =====================================================================
# EDITABLE: CustomModel — implement your stock prediction model here
# =====================================================================
import copy
import math
import os
import torch.optim as optim


class PositionalEncoding(nn.Module):
    \"\"\"Positional encoding — verbatim from qlib/contrib/model/pytorch_transformer.py.\"\"\"

    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]


class Transformer(nn.Module):
    \"\"\"Transformer network — verbatim from qlib/contrib/model/pytorch_transformer.py.

    Reshapes flat Alpha360 features internally:
    [N, F*T] -> [N, d_feat, T] -> [N, T, d_feat]
    \"\"\"

    def __init__(
        self, d_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.5, device=None
    ):
        super(Transformer, self).__init__()
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.decoder_layer = nn.Linear(d_model, 1)
        self.device = device
        self.d_feat = d_feat

    def forward(self, src):
        # src [N, F*T] --> [N, T, F]
        src = src.reshape(len(src), self.d_feat, -1).permute(0, 2, 1)
        src = self.feature_layer(src)

        # src [N, T, F] --> [T, N, F]
        src = src.transpose(1, 0)  # not batch first

        mask = None

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)

        # [T, N, F] --> [N, T*F]
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])

        return output.squeeze()


class CustomModel(Model):
    \"\"\"Transformer model — faithful to qlib's official TransformerModel
    (pytorch_transformer.py).

    Uses DatasetH with Alpha360 features. The Transformer reshapes the flat
    360-dim feature vector internally: [N, 360] -> [N, 6, 60] -> [N, 60, 6].

    Hyperparameters from official benchmark:
    examples/benchmarks/Transformer/workflow_config_transformer_Alpha360.yaml
    \"\"\"

    def __init__(self):
        super().__init__()
        # Official Alpha360 benchmark hyperparameters
        self.d_feat = 6
        self.d_model = 64
        self.nhead = 2
        self.num_layers = 2
        self.dropout = 0
        self.n_epochs = 100
        self.lr = 0.0001
        self.metric = ""
        self.batch_size = 2048
        self.early_stop = 5
        self.loss = "mse"
        self.reg = 1e-3
        self.seed = int(os.environ.get("SEED", "42"))
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.model = Transformer(
            self.d_feat,
            self.d_model,
            self.nhead,
            self.num_layers,
            self.dropout,
            self.device,
        )
        self.train_optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.reg
        )
        self.fitted = False
        self.model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred.float() - label.float()) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])
        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)
        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])
        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, x_train, y_train):
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)

        self.model.train()

        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = (
                torch.from_numpy(x_train_values[indices[i : i + self.batch_size]])
                .float()
                .to(self.device)
            )
            label = (
                torch.from_numpy(y_train_values[indices[i : i + self.batch_size]])
                .float()
                .to(self.device)
            )

            pred = self.model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        self.model.eval()

        scores = []
        losses = []

        indices = np.arange(len(x_values))

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = (
                torch.from_numpy(x_values[indices[i : i + self.batch_size]])
                .float()
                .to(self.device)
            )
            label = (
                torch.from_numpy(y_values[indices[i : i + self.batch_size]])
                .float()
                .to(self.device)
            )

            with torch.no_grad():
                pred = self.model(feature)
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(self, dataset: DatasetH):
        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError(
                "Empty data from dataset, please check your dataset config."
            )

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        stop_steps = 0
        best_score = -np.inf
        best_epoch = 0
        best_param = None

        self.fitted = True

        for step in range(self.n_epochs):
            self.train_epoch(x_train, y_train)
            train_loss, train_score = self.test_epoch(x_train, y_train)
            val_loss, val_score = self.test_epoch(x_valid, y_valid)
            print(
                "Epoch%d: train %.6f, valid %.6f"
                % (step, train_score, val_score)
            )

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    print("early stop")
                    break

        print("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_param)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH, segment="test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(
            segment, col_set="feature", data_key=DataHandlerLP.DK_I
        )
        index = x_test.index
        self.model.eval()
        x_values = x_test.values
        sample_num = x_values.shape[0]
        preds = []

        for begin in range(sample_num)[:: self.batch_size]:
            if sample_num - begin < self.batch_size:
                end = sample_num
            else:
                end = begin + self.batch_size

            x_batch = (
                torch.from_numpy(x_values[begin:end]).float().to(self.device)
            )

            with torch.no_grad():
                pred = self.model(x_batch).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 16,
        "end_line": 103,
        "content": _TRANSFORMER_MODEL,
    },
]
