# Custom graph-based stock prediction model for MLS-Bench
#
# EDITABLE section: CustomModel class with fit() and predict() methods.
# FIXED sections: imports and stock-concept graph loading below.
import os
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
# FIXED: Stock-concept graph data loading utilities
# =====================================================================
# Paths to pre-downloaded graph data
STOCK2CONCEPT_PATH = os.path.expanduser("~/.qlib/qlib_data/qlib_csi300_stock2concept.npy")
STOCK_INDEX_PATH = os.path.expanduser("~/.qlib/qlib_data/qlib_csi300_stock_index.npy")

# Load the stock-concept mapping matrix and stock index
# stock2concept_matrix: shape (num_stocks, num_concepts), binary membership
# stock_index_dict: dict mapping instrument name -> integer index
_stock2concept_matrix = np.load(STOCK2CONCEPT_PATH)
_stock_index_dict = np.load(STOCK_INDEX_PATH, allow_pickle=True).item()


def get_stock_index(instruments, default_index=733):
    """Map instrument names to integer indices for stock2concept lookup.

    Args:
        instruments: array-like of instrument name strings
        default_index: fallback index for unknown instruments (733 = padding)

    Returns:
        np.ndarray of integer indices
    """
    indices = np.array([_stock_index_dict.get(inst, default_index)
                        for inst in instruments])
    return indices.astype(int)


def get_concept_matrix(stock_indices):
    """Get the concept membership matrix for given stock indices.

    Args:
        stock_indices: np.ndarray of integer stock indices

    Returns:
        np.ndarray of shape (len(stock_indices), num_concepts), float32
    """
    return _stock2concept_matrix[stock_indices].astype(np.float32)


# =====================================================================
# EDITABLE: CustomModel — implement your stock prediction model here
# =====================================================================
class CustomModel(Model):
    """Custom graph-based stock prediction model.

    You must implement:
        fit(dataset)    — train the model on the training data
        predict(dataset, segment="test") — return predictions as pd.Series

    The dataset is a qlib DatasetH with Alpha360 features (6 base features x 60
    days = 360 features per stock per day). Segments: "train", "valid", "test".

    Getting data from the dataset:
        df_train = dataset.prepare("train", col_set=["feature", "label"],
                                    data_key=DataHandlerLP.DK_L)
        features = df_train["feature"]   # DataFrame: (n_samples, 360)
        labels = df_train["label"]       # DataFrame: (n_samples, 1)

    Stock-concept graph data (loaded above):
        - _stock2concept_matrix: (num_stocks, num_concepts) binary matrix
        - _stock_index_dict: maps instrument name -> stock index
        - get_stock_index(instruments): maps instrument names to indices
        - get_concept_matrix(stock_indices): returns concept membership matrix

    Usage in training (daily batches for graph-based models):
        daily_count = df.groupby(level=0).size().values
        daily_index = np.roll(np.cumsum(daily_count), 1)
        daily_index[0] = 0
        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = features.values[batch]
            instruments = features.index.get_level_values("instrument")[batch]
            stock_idx = get_stock_index(instruments)
            concept_mat = get_concept_matrix(stock_idx)
            # concept_mat shape: (batch_stocks, num_concepts)

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
        # --- Default: Ridge regression baseline (ignores graph) ---
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
