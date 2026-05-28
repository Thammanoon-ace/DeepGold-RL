"""
Robustness scoring & distributional evaluation (V3.5 experimental protocol).

The multi-seed walk-forward result (compounded −62%…+204% from the seed alone)
proved that point metrics are meaningless here. This module replaces "best run"
thinking with **distributions over the seed×fold grid** and a single, rank-able
**Robustness Score** that rewards consistency and penalises instability and
downside — never peak return.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence

import numpy as np


@dataclass
class RobustnessReport:
    """Distribution summary over a grid of (seed, fold) results."""

    n_cells: int
    median: float
    mean: float
    std: float
    iqr: float
    p10: float
    p90: float
    worst: float
    best: float
    pct_positive: float
    robustness_score: float
    vs_baseline_winrate: Optional[float] = None  # % of cells beating a baseline

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def pretty(self) -> str:
        base = "" if self.vs_baseline_winrate is None else (
            f"  Beats baseline   : {self.vs_baseline_winrate:.0f}% of cells\n")
        return (
            "Robustness Report (distribution over seed x fold)\n"
            "-------------------------------------------------\n"
            f"  Cells            : {self.n_cells}\n"
            f"  Median           : {self.median:+.2f}%\n"
            f"  Mean +/- std     : {self.mean:+.2f}% +/- {self.std:.2f}\n"
            f"  IQR              : {self.iqr:.2f}  (p10 {self.p10:+.2f} .. p90 {self.p90:+.2f})\n"
            f"  Worst / best     : {self.worst:+.2f}% / {self.best:+.2f}%\n"
            f"  Profitable cells : {self.pct_positive:.0f}%\n"
            + base +
            f"  ROBUSTNESS SCORE : {self.robustness_score:+.2f}\n"
        )


def robustness_score(returns: np.ndarray) -> float:
    """A single scalar ranking configs by *robustness*, not peak.

    ``score = median − 0.5·IQR − 1.0·max(0, −worst) + 25·(pct_positive − 0.5)``

    Rewards a high, consistent central tendency and a high share of profitable
    cells; penalises a wide interquartile spread and any large worst-case loss.
    Deliberately makes a single lucky +204% cell *not* dominate.
    """
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    median = np.median(r)
    iqr = np.percentile(r, 75) - np.percentile(r, 25)
    worst = r.min()
    pct_pos = float((r > 0).mean())
    downside = max(0.0, -worst)
    return float(median - 0.5 * iqr - 1.0 * downside + 25.0 * (pct_pos - 0.5))


def compute_robustness(
    returns: Sequence[float], baseline: Optional[float] = None
) -> RobustnessReport:
    """Summarize a grid of compounded returns (%) into a RobustnessReport."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        raise ValueError("No results to summarize.")
    return RobustnessReport(
        n_cells=int(r.size),
        median=float(np.median(r)),
        mean=float(r.mean()),
        std=float(r.std()),
        iqr=float(np.percentile(r, 75) - np.percentile(r, 25)),
        p10=float(np.percentile(r, 10)),
        p90=float(np.percentile(r, 90)),
        worst=float(r.min()),
        best=float(r.max()),
        pct_positive=float((r > 0).mean() * 100.0),
        robustness_score=robustness_score(r),
        vs_baseline_winrate=(float((r > baseline).mean() * 100.0)
                             if baseline is not None else None),
    )


def bootstrap_median_ci(
    returns: Sequence[float], n_boot: int = 5000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Bootstrap (1-alpha) confidence interval for the median return.

    With high variance and few cells this interval is usually wide and straddles
    zero — which is the honest message: the edge is not statistically established.
    """
    r = np.asarray(returns, dtype=float)
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(r, size=r.size, replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(meds, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
