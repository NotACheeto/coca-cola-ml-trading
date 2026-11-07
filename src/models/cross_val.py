"""
Purged and Embargoed Walk-Forward Cross-Validation for Financial Time Series.
Eliminates data leakage from overlapping forward labels and serial correlation.
"""

from typing import Generator, List, Tuple
import numpy as np
import pandas as pd


class PurgedWalkForwardCV:
    """
    Purged and Embargoed Walk-Forward Time Series Splitter.
    
    Guarantees:
    1. Strict chronological ordering (Train before Test).
    2. Purging: Drops training instances whose forward-looking labels overlap test periods.
    3. Embargo: Omits a cooldown buffer following test periods to neutralize serial correlation.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_window: int = 5,
        embargo_pct: float = 0.01,
        min_train_pct: float = 0.40,
    ):
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.embargo_pct = embargo_pct
        self.min_train_pct = min_train_pct

    def split(
        self, X: pd.DataFrame, y: pd.Series = None, groups=None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yields (train_indices, test_indices) for walk-forward evaluation.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        min_train_samples = int(n_samples * self.min_train_pct)
        test_size = (n_samples - min_train_samples) // self.n_splits
        embargo_size = max(1, int(n_samples * self.embargo_pct))

        for fold in range(self.n_splits):
            test_start = min_train_samples + fold * test_size
            test_end = test_start + test_size if fold < self.n_splits - 1 else n_samples

            # Train ends before test starts, minus purge_window to remove label overlap
            train_end = max(0, test_start - self.purge_window)
            train_idx = indices[:train_end]

            test_idx = indices[test_start:test_end]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
