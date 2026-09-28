"""
Threshold Tuning for F_0.5 Maximization.

Searches for the decision threshold that maximizes macro-averaged F_0.5
on the held-out validation split.

Key insight: F_0.5 weights precision ~2x over recall, so the optimal threshold
is typically HIGHER than 0.5. This tuner finds the exact optimal value.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger
from tqdm import tqdm

from .metrics import EntityResolutionMetrics


class ThresholdTuner:
    """
    Grid-searches decision thresholds to maximize macro F_0.5.
    Handles per-entity decision making (not just global threshold).
    """

    def __init__(
        self,
        beta: float = 0.5,
        search_low: float = 0.3,
        search_high: float = 0.95,
        n_steps: int = 65,
    ) -> None:
        self.beta = beta
        self.thresholds = np.linspace(search_low, search_high, n_steps)
        self.metrics = EntityResolutionMetrics(beta=beta)
        self.best_threshold: float = 0.5
        self.best_score: float = 0.0

    def tune(
        self,
        val_candidates_df,          # polars DataFrame
        val_probs: np.ndarray,      # calibrated probs for each candidate row
        ground_truth: dict[str, list[str]],
        country_map: Optional[dict[str, str]] = None,
    ) -> float:
        """
        Grid-search thresholds; return the best threshold.

        Args:
            val_candidates_df: validation candidate pairs
            val_probs: calibrated probability for each pair (same row order)
            ground_truth: {source1_id: [true_match_ids]}
            country_map: optional for per-country breakdown

        Returns:
            best threshold (float)
        """
        # Build list of (source1_id, candidate_id, prob)
        pairs = list(val_candidates_df.iter_rows(named=True))
        assert len(pairs) == len(val_probs), "Mismatch in pairs vs. probs length."

        logger.info(
            f"Tuning threshold over {len(self.thresholds)} values "
            f"[{self.thresholds[0]:.2f}, {self.thresholds[-1]:.2f}]..."
        )

        scores = []
        for threshold in tqdm(self.thresholds, desc="Threshold search"):
            predictions = self._apply_threshold(pairs, val_probs, threshold)
            result = self.metrics.evaluate(
                predictions, ground_truth, country_map=country_map, verbose=False
            )
            scores.append((threshold, result["macro_f05"]))

        # Find best
        best_thresh, best_score = max(scores, key=lambda x: x[1])
        self.best_threshold = best_thresh
        self.best_score = best_score

        logger.info(
            f"Best threshold: {best_thresh:.4f} → F_0.5 = {best_score:.4f}"
        )

        # Log top-5 thresholds
        top5 = sorted(scores, key=lambda x: -x[1])[:5]
        logger.info("Top-5 thresholds:")
        for t, s in top5:
            logger.info(f"  threshold={t:.4f}  F_0.5={s:.4f}")

        return best_thresh

    def apply(
        self,
        candidates_df,
        probs: np.ndarray,
        threshold: Optional[float] = None,
    ) -> dict[str, set[str]]:
        """Apply threshold to produce final predictions."""
        t = threshold if threshold is not None else self.best_threshold
        pairs = list(candidates_df.iter_rows(named=True))
        return self._apply_threshold(pairs, probs, t)

    @staticmethod
    def _apply_threshold(
        pairs: list[dict],
        probs: np.ndarray,
        threshold: float,
    ) -> dict[str, set[str]]:
        predictions: dict[str, set[str]] = {}
        for row, prob in zip(pairs, probs):
            s1_id = row["source1_id"]
            if s1_id not in predictions:
                predictions[s1_id] = set()
            if prob >= threshold:
                predictions[s1_id].add(row["candidate_id"])
        return predictions
