# Custom stock prediction model for MLS-Bench
#
# EDITABLE section: CustomModel class with fit() and predict() methods.
# FIXED sections: imports below.
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================================
# EDITABLE: CustomModel — implement your stock prediction model here
# =====================================================================
class CustomModel(Model):
    """Custom stock prediction model.

    You must implement:
        fit(dataset)    — train the model on the training data
        predict(dataset, segment="test") — return predictions as pd.Series

    The dataset is a qlib DatasetH with Alpha360 features (360 features per
    stock per day). The 360 features come from 6 base features
    (open/close/high/low/volume/vwap ratios) x 60 days of history.

    For temporal models, features can be reshaped:
        x.reshape(N, 6, 60).permute(0, 2, 1) -> [N, 60, 6]
    giving 60 time steps of 6 features each.

    Segments: "train", "valid", "test".

    Getting data from the dataset:
        df_train = dataset.prepare("train", col_set=["feature", "label"],
                                    data_key=DataHandlerLP.DK_L)
        features = df_train["feature"]   # DataFrame: (n_samples, 360)
        labels = df_train["label"]       # DataFrame: (n_samples, 1)

    The label is: Ref($close, -2) / Ref($close, -1) - 1
    (i.e., the return from T+1 to T+2, predicted at time T)

    predict() must return a pd.Series indexed by (datetime, instrument)
    matching the target segment's index.

    Available imports: torch, torch.nn, numpy, pandas, lightgbm, sklearn, scipy
    All network definitions and training logic go in this class.
    """

    def __init__(self):
        super().__init__()
        self.fitted = False
        # --- Default: Ridge regression baseline ---
        from sklearn.linear_model import Ridge

        self.model = Ridge(alpha=1.0)

    def fit(self, dataset: DatasetH):
        """Train the model.

        Args:
            dataset: DatasetH with "train" and "valid" segments.
        """
        df_train = dataset.prepare(
            "train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        features = df_train["feature"].values
        labels = df_train["label"].values.ravel()

        # Remove NaN rows
        mask = ~(np.isnan(features).any(axis=1) | np.isnan(labels))
        features = features[mask]
        labels = labels[mask]

        self.model.fit(features, labels)
        self.fitted = True

    def predict(self, dataset: DatasetH, segment="test"):
        """Generate predictions.

        Args:
            dataset: DatasetH with the target segment.
            segment: Which segment to predict on (default: "test").

        Returns:
            pd.Series of predictions, indexed by (datetime, instrument).
        """
        if not self.fitted:
            raise ValueError("Model is not fitted yet!")

        df_test = dataset.prepare(
            segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_I
        )
        features = df_test["feature"]
        index = features.index

        features_np = features.values
        features_np = np.nan_to_num(features_np, nan=0.0)

        preds = self.model.predict(features_np)
        return pd.Series(preds, index=index, name="score")
