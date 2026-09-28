"""
Probability Calibration.

Calibrates raw GBM output probabilities to be better-behaved for thresholding.
Uses isotonic regression (non-parametric, better for tree classifiers)
or Platt scaling (sigmoid) as fallback.

Calibrated probabilities allow stable threshold tuning for F_0.5 maximization.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    """
    Post-hoc calibration using OOF predictions from GBM training.
    """

    def __init__(self, method: str = "isotonic") -> None:
        """
        Args:
            method: 'isotonic' or 'sigmoid' (Platt scaling).
        """
        self.method = method
        self._calibrator = None

    def fit(
        self,
        raw_probs: np.ndarray,
        y_true: np.ndarray,
    ) -> "ProbabilityCalibrator":
        """
        Fit calibrator on OOF predictions.

        Args:
            raw_probs: out-of-fold probabilities from GBM (N,)
            y_true: binary ground-truth labels (N,)
        """
        logger.info(f"Fitting {self.method} calibrator on {len(y_true):,} OOF samples.")
        if self.method == "isotonic":
            self._calibrator = IsotonicRegression(out_of_bounds="clip")
            self._calibrator.fit(raw_probs, y_true)
        elif self.method == "sigmoid":
            # Platt scaling via logistic regression on log-odds
            raw_probs_clipped = np.clip(raw_probs, 1e-6, 1 - 1e-6)
            log_odds = np.log(raw_probs_clipped / (1 - raw_probs_clipped)).reshape(-1, 1)
            self._calibrator = LogisticRegression()
            self._calibrator.fit(log_odds, y_true)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        logger.info("Calibration fit complete.")
        return self

    def predict(
        self,
        raw_probs: np.ndarray,
    ) -> np.ndarray:
        """Transform raw probabilities to calibrated probabilities."""
        if self._calibrator is None:
            return raw_probs
        if self.method == "isotonic":
            return self._calibrator.predict(raw_probs).astype(np.float32)
        elif self.method == "sigmoid":
            raw_probs_clipped = np.clip(raw_probs, 1e-6, 1 - 1e-6)
            log_odds = np.log(raw_probs_clipped / (1 - raw_probs_clipped)).reshape(-1, 1)
            return self._calibrator.predict_proba(log_odds)[:, 1].astype(np.float32)
        return raw_probs

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Calibrator saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ProbabilityCalibrator":
        with open(path, "rb") as f:
            return pickle.load(f)
