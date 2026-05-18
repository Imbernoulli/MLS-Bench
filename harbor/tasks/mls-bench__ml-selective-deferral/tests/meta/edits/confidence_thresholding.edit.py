"""Confidence-thresholding baseline."""

_FILE = "scikit-learn/custom_selective.py"

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 253,
        "end_line": 287,
        "content": """class SelectivePolicy:
    \"\"\"Global confidence threshold tuned on the calibration set.\"\"\"

    def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT, random_state: int = 0):
        self.target_coverage = float(target_coverage)
        self.random_state = int(random_state)
        self.threshold_: float = 0.5
        self.group_thresholds_: dict[int, float] = {}
        self.meta_model_ = None
        self.strategy_name = \"confidence_thresholding\"

    def fit(self, probs: np.ndarray, y_true: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> \"SelectivePolicy\":
        scores = self.acceptance_score(probs, groups, X)
        quantile = float(np.clip(1.0 - self.target_coverage, 0.0, 1.0))
        self.threshold_ = float(np.quantile(scores, quantile))
        self.group_thresholds_ = {}
        self.meta_model_ = None
        return self

    def acceptance_score(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        return np.max(probs, axis=1)

    def predict_accept(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        return self.acceptance_score(probs, groups, X) >= self.threshold_

    def calibration_summary(self) -> dict[str, float]:
        return {\"threshold\": float(self.threshold_)}
""",
    }
]
