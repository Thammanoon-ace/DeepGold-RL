"""
Time-series cross-validation splitters for walk-forward analysis (V2 / Phase 3).

These generate **non-overlapping, strictly ordered** train/test folds where
every test window lies *after* its training window — the only valid arrangement
for financial back-testing (a model is never trained on data from the future of
its evaluation period).

An optional ``gap`` (embargo) inserts a buffer between the end of training and
the start of testing so that rolling-window indicators computed near the
boundary cannot share inputs across the split.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Fold:
    """One walk-forward fold, expressed as integer row positions (half-open)."""

    index: int
    train_start: int
    train_end: int   # exclusive
    test_start: int
    test_end: int    # exclusive

    @property
    def train_len(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_len(self) -> int:
        return self.test_end - self.test_start


class TimeSeriesSplitter:
    """Generate expanding- or rolling-window walk-forward folds.

    Parameters
    ----------
    n_splits:
        Number of test folds.
    test_size:
        Bars in each test fold. If ``None``, set to ``n_samples // (n_splits+1)``.
    train_size:
        Bars in each training window for ``mode='rolling'``. Ignored for
        ``'expanding'`` (which always trains on everything up to the gap).
    mode:
        ``'expanding'`` (growing train set) or ``'rolling'`` (fixed-size, sliding).
    gap:
        Embargo bars between ``train_end`` and ``test_start`` (default 0).
    min_train_size:
        Minimum training bars required for the first fold; folds that would have
        less training data are dropped with a clear error if none remain.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        train_size: Optional[int] = None,
        mode: str = "expanding",
        gap: int = 0,
        min_train_size: Optional[int] = None,
    ) -> None:
        if mode not in ("expanding", "rolling"):
            raise ValueError("mode must be 'expanding' or 'rolling'.")
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1.")
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.mode = mode
        self.gap = gap
        self.min_train_size = min_train_size

    def split(self, n_samples: int) -> List[Fold]:
        """Return the list of folds for a series of ``n_samples`` rows."""
        test_size = self.test_size or (n_samples // (self.n_splits + 1))
        if test_size <= 0:
            raise ValueError(
                f"Derived test_size <= 0 for n_samples={n_samples}, "
                f"n_splits={self.n_splits}. Provide more data or fewer splits."
            )

        # The block of test folds occupies the tail of the series; the first
        # test fold starts here so the last one ends exactly at n_samples.
        anchor = n_samples - self.n_splits * test_size
        min_train = self.min_train_size if self.min_train_size is not None else test_size

        folds: List[Fold] = []
        for k in range(self.n_splits):
            test_start = anchor + k * test_size
            test_end = test_start + test_size
            train_end = test_start - self.gap

            if self.mode == "expanding":
                train_start = 0
            else:  # rolling
                if self.train_size is None:
                    # Default rolling window: same length as the test block prefix.
                    train_start = max(0, train_end - max(test_size * 3, min_train))
                else:
                    train_start = max(0, train_end - self.train_size)

            train_len = train_end - train_start
            if train_len < min_train or train_end <= train_start or test_end > n_samples:
                continue  # not enough history yet for this fold
            folds.append(
                Fold(len(folds), train_start, train_end, test_start, test_end)
            )

        if not folds:
            raise ValueError(
                "No valid folds produced. Reduce n_splits/min_train_size or "
                "supply a longer dataset."
            )
        return folds
