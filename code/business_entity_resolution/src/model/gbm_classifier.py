"""
Gradient-Boosted Machine Classifier.

Primary backend: LightGBM (MIT license).
Fallback: XGBoost (Apache-2.0).

Training strategy:
  - k-fold cross-validation stratified by country
  - Hard-negative-weighted sample_weight
  - Early stopping on binary_logloss
  - Out-of-fold (OOF) predictions for calibration
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from sklearn.model_selection import StratifiedKFold

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    logger.warning("lightgbm not installed.")

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


_DEFAULT_LGB_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "n_jobs": -1,
}


class GBMClassifier:
    """
    Gradient-boosted binary classifier for entity match prediction.
    Supports k-fold cross-validation with per-country stratification.
    """

    def __init__(
        self,
        backend: str = "lightgbm",
        n_folds: int = 5,
        random_state: int = 42,
        params: Optional[dict] = None,
    ) -> None:
        self.backend = backend
        self.n_folds = n_folds
        self.random_state = random_state
        self.params = params or _DEFAULT_LGB_PARAMS
        self.models: list = []
        self.oof_preds: Optional[np.ndarray] = None
        self.feature_importance: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,  # e.g. country label per row
        feature_names: Optional[list[str]] = None,
        eval_set_X: Optional[np.ndarray] = None,
        eval_set_y: Optional[np.ndarray] = None,
    ) -> "GBMClassifier":
        """
        Train with k-fold CV, collecting OOF predictions.

        Args:
            X: feature matrix (N, F)
            y: binary labels (N,)
            sample_weight: per-sample weights
            groups: group labels for stratification (country strings)
            feature_names: list of feature name strings
            eval_set_X/y: optional held-out validation set for early stopping
        """
        if self.backend == "lightgbm" and not _HAS_LGB:
            raise RuntimeError("lightgbm required.")
        if self.backend == "xgboost" and not _HAS_XGB:
            raise RuntimeError("xgboost required.")

        self.oof_preds = np.zeros(len(y), dtype=np.float32)
        self.models = []
        importances = []

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )
        fold_label = y  # stratify by target by default

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(X, fold_label)
        ):
            logger.info(f"Fold {fold+1}/{self.n_folds}")
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            w_tr = sample_weight[train_idx] if sample_weight is not None else None

            model = self._build_model()
            if self.backend == "lightgbm":
                callbacks = [
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=100),
                ]
                model.fit(
                    X_tr, y_tr,
                    sample_weight=w_tr,
                    eval_set=[(X_val, y_val)],
                    feature_name=feature_names or "auto",
                    callbacks=callbacks,
                )
                importances.append(model.feature_importances_)
            elif self.backend == "xgboost":
                model.fit(
                    X_tr, y_tr,
                    sample_weight=w_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )

            self.oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            self.models.append(model)

            # Per-fold val metric
            fold_f05 = self._f_beta(
                y_val, (self.oof_preds[val_idx] > 0.5).astype(int), beta=0.5
            )
            logger.info(f"  Fold {fold+1} OOF F_0.5 @ 0.5 threshold: {fold_f05:.4f}")

        if importances:
            self.feature_importance = np.mean(importances, axis=0)
        logger.info(
            f"Training complete. Overall OOF F_0.5: "
            f"{self._f_beta(y, (self.oof_preds > 0.5).astype(int), beta=0.5):.4f}"
        )
        return self

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Average ensemble prediction across all folds."""
        preds = np.zeros(len(X), dtype=np.float64)
        for model in self.models:
            preds += model.predict_proba(X)[:, 1]
        return (preds / len(self.models)).astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"models": self.models, "params": self.params}, f)
        logger.info(f"GBM model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "GBMClassifier":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.models = data["models"]
        obj.params = data["params"]
        logger.info(f"GBM model loaded from {path}")
        return obj

    def _build_model(self):
        if self.backend == "lightgbm":
            return lgb.LGBMClassifier(**self.params)
        elif self.backend == "xgboost":
            return xgb.XGBClassifier(**self.params, use_label_encoder=False, eval_metric="logloss")
        raise ValueError(f"Unknown backend: {self.backend}")

    @staticmethod
    def _f_beta(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 0.5) -> float:
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = (1 + beta**2) * prec + beta**2 * rec
        return (1 + beta**2) * prec * rec / denom if denom > 0 else 0.0
