"""LightGBM baseline -- rigorous codebase edit ops.

Faithful reproduction of qlib's official LGBModel (qlib/contrib/model/gbdt.py)
with benchmark hyperparameters from:
  examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha360.yaml

Replaces the CustomModel class and resets only the editable workflow
preprocessing block to the official LightGBM Alpha360 pipeline. The default
graph-model workflow keeps neural-model infer_processors that do not belong
in the LightGBM baseline.
"""

_FILE = "qlib/custom_model.py"
_WORKFLOW_FILE = "qlib/workflow_config.yaml"

_LGBM_MODEL = """\
# =====================================================================
# EDITABLE: CustomModel -- implement your stock prediction model here
# =====================================================================
class CustomModel(Model):
    \"\"\"LightGBM model -- faithful to qlib's official LGBModel (gbdt.py).

    Hyperparameters from official benchmark:
    examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha360.yaml
    \"\"\"

    def __init__(self):
        super().__init__()
        # Official benchmark kwargs (passed to lgb.train via self.params)
        self.params = {
            "objective": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
            "verbosity": -1,
        }
        self.early_stopping_rounds = 50
        self.num_boost_round = 1000
        self.model = None

    def _prepare_data(self, dataset):
        \"\"\"Prepare LightGBM datasets -- matches LGBModel._prepare_data().\"\"\"
        import lightgbm as lgb

        ds_l = []
        for key in ["train", "valid"]:
            if key in dataset.segments:
                df = dataset.prepare(
                    key, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
                )
                if df.empty:
                    raise ValueError(
                        "Empty data from dataset, please check your dataset config."
                    )
                x, y = df["feature"], df["label"]
                # Lightgbm need 1D array as its label
                if y.values.ndim == 2 and y.values.shape[1] == 1:
                    y = np.squeeze(y.values)
                else:
                    raise ValueError(
                        "LightGBM doesn't support multi-label training"
                    )
                ds_l.append(
                    (lgb.Dataset(x.values, label=y, free_raw_data=False), key)
                )
        return ds_l

    def fit(self, dataset: DatasetH):
        import lightgbm as lgb

        ds_l = self._prepare_data(dataset)
        ds, names = list(zip(*ds_l))
        early_stopping_callback = lgb.early_stopping(
            self.early_stopping_rounds
        )
        verbose_eval_callback = lgb.log_evaluation(period=20)
        evals_result = {}
        evals_result_callback = lgb.record_evaluation(evals_result)
        self.model = lgb.train(
            self.params,
            ds[0],  # training dataset
            num_boost_round=self.num_boost_round,
            valid_sets=ds,
            valid_names=names,
            callbacks=[
                early_stopping_callback,
                verbose_eval_callback,
                evals_result_callback,
            ],
        )

    def predict(self, dataset: DatasetH, segment="test"):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(
            segment, col_set="feature", data_key=DataHandlerLP.DK_I
        )
        return pd.Series(self.model.predict(x_test.values), index=x_test.index)
"""

_LGBM_HANDLER = """\
          infer_processors: []
          learn_processors:
            - class: DropnaLabel
            - class: CSRankNorm
              kwargs:
                fields_group: label
          label: ["Ref($close, -2) / Ref($close, -1) - 1"]
"""

OPS = [
    {
        "op": "replace",
        "file": _WORKFLOW_FILE,
        "start_line": 32,
        "end_line": 45,
        "content": _LGBM_HANDLER,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 58,
        "end_line": 156,
        "content": _LGBM_MODEL,
    },
]
