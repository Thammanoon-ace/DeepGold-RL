"""Validation package: walk-forward, time-series splitters, multi-dataset eval."""
from .splitters import Fold, TimeSeriesSplitter
from .walk_forward import WalkForwardValidator, WalkForwardResult
from .multi_dataset import MultiDatasetEvaluator
from .robustness import (
    RobustnessReport,
    compute_robustness,
    robustness_score,
    bootstrap_median_ci,
)
from .grid import GridEvaluator, GridResult

__all__ = [
    "Fold",
    "TimeSeriesSplitter",
    "WalkForwardValidator",
    "WalkForwardResult",
    "MultiDatasetEvaluator",
    "RobustnessReport",
    "compute_robustness",
    "robustness_score",
    "bootstrap_median_ci",
    "GridEvaluator",
    "GridResult",
]
