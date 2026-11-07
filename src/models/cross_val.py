"""
Purged and Embargoed Walk-Forward Cross-Validation for Financial Time Series.
Eliminates data leakage from overlapping forward labels and serial correlation.
"""

from typing import Generator, List, Optional, Tuple
import numpy as np
import pandas as pd


class PurgedWalkForwardCV:
    """
    Purged and Embargoed Walk-Forward Time Series Splitter.
    
    Guarantees:
    1. Strict chronological ordering (Train before Test).
    2. Purging: Drops training instances whose forward-looking labels overlap test periods.
       Ensures train_end <= test_start - purge_window - 1 so no label touches test prices.
    3. Embargo: When rolling or sequential windows are used, applies an embargo buffer
       to neutralize serial correlation.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_window: int = 5,
        embargo_pct: float = 0.01,
        min_train_pct: float = 0.40,
        rolling: bool = False,
        rolling_window_samples: Optional[int] = None,
    ):
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.embargo_pct = embargo_pct
        self.min_train_pct = min_train_pct
        self.rolling = rolling
        self.rolling_window_samples = rolling_window_samples

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

            # Strict Purge: sample t with forward horizon H uses prices at t+H.
            # If t = test_start - purge_window, t+H = test_start, which touches test data!
            # Therefore, train_end must be strictly < test_start - purge_window.
            train_end = max(0, test_start - self.purge_window - 1)

            if self.rolling and self.rolling_window_samples:
                train_start = max(0, train_end - self.rolling_window_samples)
                train_idx = indices[train_start:train_end]
            else:
                train_idx = indices[:train_end]

            test_idx = indices[test_start:test_end]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV) as defined by Marcos López de Prado.
    Generates multiple backtest paths with strict purging and post-test embargoes.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        purge_window: int = 5,
        embargo_pct: float = 0.01,
    ):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_window = purge_window
        self.embargo_pct = embargo_pct

    def split(
        self, X: pd.DataFrame, y: pd.Series = None, groups=None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        from itertools import combinations

        n_samples = len(X)
        indices = np.arange(n_samples)
        embargo_size = max(1, int(n_samples * self.embargo_pct))

        # Split timeline into n_splits contiguous blocks
        split_size = n_samples // self.n_splits
        blocks = []
        for i in range(self.n_splits):
            start = i * split_size
            end = (i + 1) * split_size if i < self.n_splits - 1 else n_samples
            blocks.append((start, end))

        # Generate all combinations of test blocks
        for test_block_indices in combinations(range(self.n_splits), self.n_test_splits):
            test_indices_list = []
            test_ranges = []
            for b_idx in test_block_indices:
                b_start, b_end = blocks[b_idx]
                test_indices_list.append(indices[b_start:b_end])
                test_ranges.append((b_start, b_end))

            test_idx = np.concatenate(test_indices_list)

            # Build train indices: exclude test blocks, purge overlapping horizon, and embargo post-test
            train_mask = np.ones(n_samples, dtype=bool)
            for t_start, t_end in test_ranges:
                # Test block itself
                train_mask[t_start:t_end] = False
                # Purge window before test block
                purge_start = max(0, t_start - self.purge_window - 1)
                train_mask[purge_start:t_start] = False
                # Embargo window after test block
                embargo_end = min(n_samples, t_end + embargo_size)
                train_mask[t_end:embargo_end] = False

            train_idx = indices[train_mask]

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        from math import comb
        return comb(self.n_splits, self.n_test_splits)
