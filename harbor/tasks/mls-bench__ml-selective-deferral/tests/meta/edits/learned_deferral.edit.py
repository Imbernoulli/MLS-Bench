"""Learned deferral baseline."""

_FILE = "scikit-learn/custom_selective.py"

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 253,
        "end_line": 287,
        "content": """class SelectivePolicy:
    \"\"\"Compact learned gate that predicts correctness from confidence features.\"\"\"

    def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT, random_state: int = 0):
        self.target_coverage = float(target_coverage)
        self.random_state = int(random_state)
        self.threshold_: float = 0.5
        self.group_thresholds_: dict[int, float] = {}
        self.meta_model_ = None
        self.strategy_name = \"learned_deferral\"

    def fit(self, probs: np.ndarray, y_true: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> \"SelectivePolicy\":
        features = _confidence_features(probs, groups, X)
        correct = (np.argmax(probs, axis=1) == y_true).astype(int)
        self.meta_model_ = Pipeline(
            steps=[
                (\"scale\", StandardScaler()),
                (
                    \"clf\",
                    LogisticRegression(
                        max_iter=1000,
                        solver=\"lbfgs\",
                        class_weight=\"balanced\",
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        self.meta_model_.fit(features, correct)
        scores = self.meta_model_.predict_proba(features)[:, 1]
        quantile = float(np.clip(1.0 - self.target_coverage, 0.0, 1.0))
        self.threshold_ = float(np.quantile(scores, quantile))
        self.group_thresholds_ = {}
        return self

    def acceptance_score(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        if self.meta_model_ is None:
            return np.max(probs, axis=1)
        features = _confidence_features(probs, groups, X)
        return self.meta_model_.predict_proba(features)[:, 1]

    def predict_accept(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        return self.acceptance_score(probs, groups, X) >= self.threshold_

    def calibration_summary(self) -> dict[str, float]:
        return {\"threshold\": float(self.threshold_)}
""",
    }
]
