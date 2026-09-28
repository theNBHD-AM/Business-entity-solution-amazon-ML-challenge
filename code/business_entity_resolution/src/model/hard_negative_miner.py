"""
Hard Negative Mining.

Near-miss negatives (high string similarity, but NOT a true match) hurt precision
most — these are the cases where the model must learn to say NO confidently.

Strategy:
  1. From all blocked candidate pairs, label positives vs. negatives using labels.
  2. Identify hard negatives: negatives with name_jaro_winkler > threshold.
  3. Oversample hard negatives during training by including them at higher weight.
"""
from __future__ import annotations

import numpy as np
from loguru import logger


class HardNegativeMiner:
    """
    Selects a balanced, hard-negative-rich training set from a feature matrix.
    """

    def __init__(
        self,
        hard_negative_ratio: int = 5,
        hard_threshold: float = 0.85,
        jw_feature_idx: int = 2,  # index of name_jaro_winkler in feature vector
        random_state: int = 42,
    ) -> None:
        self.hard_negative_ratio = hard_negative_ratio
        self.hard_threshold = hard_threshold
        self.jw_idx = jw_feature_idx
        self.rng = np.random.default_rng(random_state)

    def mine(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (X_train, y_train, sample_weights) with hard negatives prioritized.

        Strategy:
          - Keep ALL positives.
          - Identify hard negatives (JW > threshold).
          - Sample random negatives to fill out ratio.
          - Assign higher sample weight to hard negatives.

        Returns:
            X_train, y_train, sample_weights
        """
        pos_mask = y == 1
        neg_mask = y == 0
        hard_neg_mask = neg_mask & (X[:, self.jw_idx] >= self.hard_threshold)
        easy_neg_mask = neg_mask & ~hard_neg_mask

        n_pos = pos_mask.sum()
        n_hard = hard_neg_mask.sum()
        n_easy_target = max(0, self.hard_negative_ratio * n_pos - n_hard)

        logger.info(
            f"HardNegativeMiner: {n_pos} positives, {n_hard} hard negatives, "
            f"sampling {n_easy_target} easy negatives."
        )

        # Sample easy negatives
        easy_neg_indices = np.where(easy_neg_mask)[0]
        n_easy_sample = min(n_easy_target, len(easy_neg_indices))
        sampled_easy = self.rng.choice(
            easy_neg_indices, size=int(n_easy_sample), replace=False
        )

        selected = np.concatenate([
            np.where(pos_mask)[0],
            np.where(hard_neg_mask)[0],
            sampled_easy,
        ])

        X_train = X[selected]
        y_train = y[selected]

        # Sample weights: positives=1, hard_negatives=2, easy_negatives=0.5
        weights = np.ones(len(selected), dtype=np.float32)
        n_pos_sel = n_pos
        n_hard_sel = n_hard
        weights[n_pos_sel: n_pos_sel + n_hard_sel] = 2.0
        weights[n_pos_sel + n_hard_sel:] = 0.5

        logger.info(
            f"Training set: {len(selected):,} pairs "
            f"({n_pos} pos / {n_hard_sel} hard-neg / {n_easy_sample} easy-neg)"
        )
        return X_train, y_train, weights
