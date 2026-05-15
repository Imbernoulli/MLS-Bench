"""GATs baseline -- rigorous codebase edit ops.

Faithful reproduction of qlib's official GATs model from:
  qlib/contrib/model/pytorch_gats.py  (DatasetH version)
with benchmark hyperparameters from:
  examples/benchmarks/GATs/workflow_config_gats_Alpha360.yaml

Replaces ONLY the model code. The default workflow config (Alpha360/DatasetH)
is compatible with GATs, so only the model code needs replacing.

Key: model_path controls optional pre-trained LSTM loading; empty trains end-to-end.
"""

_MODEL_FILE = "qlib/custom_model.py"

_GATS_MODEL = """\
# =====================================================================
# EDITABLE: CustomModel -- implement your stock prediction model here
# =====================================================================
import copy
import torch.optim as optim


class LSTMModel(nn.Module):
    \"\"\"LSTM backbone -- verbatim from qlib/contrib/model/pytorch_lstm.py.\"\"\"

    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)
        self.d_feat = d_feat

    def forward(self, x):
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()


class GRUModel(nn.Module):
    \"\"\"GRU backbone -- verbatim from qlib/contrib/model/pytorch_gru.py.\"\"\"

    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)
        self.d_feat = d_feat

    def forward(self, x):
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()


class GATModel(nn.Module):
    \"\"\"GAT network -- verbatim from qlib/contrib/model/pytorch_gats.py.\"\"\"

    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"):
        super().__init__()

        if base_model == "GRU":
            self.rnn = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError("unknown base model name `%s`" % base_model)

        self.hidden_size = hidden_size
        self.d_feat = d_feat
        self.transformation = nn.Linear(self.hidden_size, self.hidden_size)
        self.a = nn.Parameter(torch.randn(self.hidden_size * 2, 1))
        self.a.requires_grad = True
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def cal_attention(self, x, y):
        x = self.transformation(x)
        y = self.transformation(y)

        sample_num = x.shape[0]
        dim = x.shape[1]
        e_x = x.expand(sample_num, sample_num, dim)
        e_y = torch.transpose(e_x, 0, 1)
        attention_in = torch.cat((e_x, e_y), 2).view(-1, dim * 2)
        self.a_t = torch.t(self.a)
        attention_out = self.a_t.mm(torch.t(attention_in)).view(sample_num, sample_num)
        attention_out = self.leaky_relu(attention_out)
        att_weight = self.softmax(attention_out)
        return att_weight

    def forward(self, x):
        # x: [N, F*T] -- DatasetH provides flattened features
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        hidden = out[:, -1, :]
        att_weight = self.cal_attention(hidden, hidden)
        hidden = att_weight.mm(hidden) + hidden
        hidden = self.fc(hidden)
        hidden = self.leaky_relu(hidden)
        return self.fc_out(hidden).squeeze()


class CustomModel(Model):
    \"\"\"GATs model -- faithful to qlib's official GATs (pytorch_gats.py).

    Hyperparameters from official benchmark:
    examples/benchmarks/GATs/workflow_config_gats_Alpha360.yaml

    Uses DatasetH with Alpha360 (d_feat=6, 360 features reshaped to 60x6).
    Daily batching via get_daily_inter() for graph attention across stocks.
    \"\"\"

    def __init__(self):
        super().__init__()
        # Official benchmark hyperparameters (Alpha360)
        self.d_feat = 6
        self.hidden_size = 64
        self.num_layers = 2
        self.dropout = 0.7
        self.n_epochs = 200
        self.lr = 1e-4
        self.metric = "loss"
        self.early_stop = 20
        self.loss = "mse"
        self.base_model = "LSTM"
        self.model_path = "examples/benchmarks/LSTM/model_lstm_csi300.pkl"
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.GAT_model = GATModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            base_model=self.base_model,
        )
        self.train_optimizer = optim.Adam(
            self.GAT_model.parameters(), lr=self.lr
        )
        self.fitted = False
        self.GAT_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred - label) ** 2
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

    def get_daily_inter(self, df, shuffle=False):
        # organize the train data into daily batches
        daily_count = df.groupby(level=0, group_keys=False).size().values
        daily_index = np.roll(np.cumsum(daily_count), 1)
        daily_index[0] = 0
        if shuffle:
            # shuffle data
            daily_shuffle = list(zip(daily_index, daily_count))
            np.random.shuffle(daily_shuffle)
            daily_index, daily_count = zip(*daily_shuffle)
        return daily_index, daily_count

    def train_epoch(self, x_train, y_train):
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)
        self.GAT_model.train()

        # organize the train data into daily batches
        daily_index, daily_count = self.get_daily_inter(x_train, shuffle=True)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_train_values[batch]).float().to(self.device)
            label = torch.from_numpy(y_train_values[batch]).float().to(self.device)

            pred = self.GAT_model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.GAT_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        # prepare training data
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)
        self.GAT_model.eval()

        scores = []
        losses = []

        # organize the test data into daily batches
        daily_index, daily_count = self.get_daily_inter(data_x, shuffle=False)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_values[batch]).float().to(self.device)
            label = torch.from_numpy(y_values[batch]).float().to(self.device)

            with torch.no_grad():
                pred = self.GAT_model(feature)
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(self, dataset: DatasetH):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        stop_steps = 0
        best_score = -np.inf
        best_epoch = 0
        best_param = None

        # optionally load pretrained base_model
        if self.base_model == "LSTM":
            pretrained_model = LSTMModel(d_feat=self.d_feat, hidden_size=self.hidden_size, num_layers=self.num_layers)
        elif self.base_model == "GRU":
            pretrained_model = GRUModel(d_feat=self.d_feat, hidden_size=self.hidden_size, num_layers=self.num_layers)
        else:
            raise ValueError("unknown base model name `%s`" % self.base_model)

        if self.model_path:
            pretrained_model.load_state_dict(torch.load(self.model_path, map_location=self.device))

        model_dict = self.GAT_model.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_model.state_dict().items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        self.GAT_model.load_state_dict(model_dict)

        # train
        self.fitted = True

        for step in range(self.n_epochs):
            self.train_epoch(x_train, y_train)
            train_loss, train_score = self.test_epoch(x_train, y_train)
            val_loss, val_score = self.test_epoch(x_valid, y_valid)
            print("Epoch%d: train %.6f, valid %.6f" % (step, train_score, val_score))

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.GAT_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    print("early stop")
                    break

        print("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.GAT_model.load_state_dict(best_param)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH, segment="test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        index = x_test.index
        self.GAT_model.eval()
        x_values = x_test.values
        preds = []

        # organize the data into daily batches
        daily_index, daily_count = self.get_daily_inter(x_test, shuffle=False)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            x_batch = torch.from_numpy(x_values[batch]).float().to(self.device)

            with torch.no_grad():
                pred = self.GAT_model(x_batch).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)
"""

OPS = [
    {
        "op": "replace",
        "file": _MODEL_FILE,
        "start_line": 58,
        "end_line": 156,
        "content": _GATS_MODEL,
    },
]
