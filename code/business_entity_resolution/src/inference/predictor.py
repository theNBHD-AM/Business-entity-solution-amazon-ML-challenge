"""
Full Inference Pipeline — end-to-end prediction for test data.

Takes:
  - Test Source 1, 2, 3 DataFrames (normalized)
  - Pre-built FAISS index + blocking artifacts
  - Trained GBM + calibrator
  - Best threshold from validation tuning

Produces:
  - matching_results.tsv (every S1 entity, with matched IDs or empty)
  - candidate_pairs.tsv (all blocking output)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from loguru import logger

from ..features.feature_builder import FeatureBuilder
from ..model.gbm_classifier import GBMClassifier
from ..model.calibrator import ProbabilityCalibrator
from ..blocking.candidate_generator import CandidateGenerator


class EntityResolutionPredictor:
    """
    Orchestrates end-to-end inference.
    """

    def __init__(
        self,
        candidate_generator: CandidateGenerator,
        feature_builder: FeatureBuilder,
        classifier: GBMClassifier,
        calibrator: ProbabilityCalibrator,
        threshold: float = 0.5,
        dl_refiner=None,
        dl_refiner_low: float = 0.4,
        dl_refiner_high: float = 0.7,
    ) -> None:
        self.gen = candidate_generator
        self.feat = feature_builder
        self.clf = classifier
        self.cal = calibrator
        self.threshold = threshold
        self.dl_refiner = dl_refiner
        self.dl_low = dl_refiner_low
        self.dl_high = dl_refiner_high

    def predict(
        self,
        s1_df: pl.DataFrame,
        s23_df: pl.DataFrame,
        s1_embeddings: Optional[np.ndarray],
        s23_embeddings: Optional[np.ndarray],
        phonetic_index: Optional[dict] = None,
        geo_index: Optional[dict] = None,
        output_dir: str = "outputs",
    ) -> dict[str, set[str]]:
        """
        Full inference pass. Returns {source1_id: {matched_ids}}.
        Also writes candidate_pairs.tsv and matching_results.tsv to output_dir.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. Generate candidates
        logger.info("Step 1: Generating candidates...")
        candidates_df = self.gen.generate(
            s1_df=s1_df,
            s23_df=s23_df,
            s1_embeddings=s1_embeddings,
            s23_embeddings=s23_embeddings,
            phonetic_index=phonetic_index,
            geo_index=geo_index,
        )

        # Save candidate_pairs.tsv
        cand_path = output_path / "candidate_pairs.tsv"
        candidates_df.write_csv(str(cand_path), separator="\t")
        logger.info(f"candidate_pairs.tsv: {len(candidates_df):,} pairs written to {cand_path}")

        # 2. Build features
        logger.info("Step 2: Building features...")
        s1_lookup = self.feat.build_lookup(s1_df)
        s23_lookup = self.feat.build_lookup(s23_df)
        X = self.feat.build_features(candidates_df, s1_lookup, s23_lookup)

        # 3. Predict probabilities
        logger.info("Step 3: Predicting match probabilities...")
        raw_probs = self.clf.predict_proba(X)
        cal_probs = self.cal.predict(raw_probs)

        # 4. Optional DL refinement on ambiguous band
        if self.dl_refiner is not None:
            ambiguous_mask = (
                (cal_probs >= self.dl_low) & (cal_probs <= self.dl_high)
            )
            logger.info(
                f"DL refinement: {ambiguous_mask.sum():,} pairs in ambiguous band "
                f"[{self.dl_low}, {self.dl_high}]"
            )
            refined = self.dl_refiner.refine_batch(
                candidates_df=candidates_df,
                s1_lookup=s1_lookup,
                s23_lookup=s23_lookup,
                mask=ambiguous_mask,
            )
            cal_probs[ambiguous_mask] = refined

        # 5. Apply threshold
        logger.info(f"Step 4: Applying threshold {self.threshold:.4f}...")
        predictions: dict[str, set[str]] = {}
        all_s1_ids = s1_df["entity_id"].to_list()

        # Initialize all S1 entities (ensures singletons appear in output)
        for s1_id in all_s1_ids:
            predictions[s1_id] = set()

        pairs = list(candidates_df.iter_rows(named=True))
        for row, prob in zip(pairs, cal_probs):
            if prob >= self.threshold:
                predictions[row["source1_id"]].add(row["candidate_id"])

        # 6. Write matching_results.tsv
        self._write_matching_results(predictions, output_path / "matching_results.tsv")

        return predictions

    @staticmethod
    def _write_matching_results(
        predictions: dict[str, set[str]],
        path: Path,
    ) -> None:
        """Write matching_results.tsv. Every S1 entity appears exactly once."""
        rows = []
        for s1_id, matched in sorted(predictions.items()):
            matched_str = "|".join(sorted(matched)) if matched else ""
            rows.append({"source1_id": s1_id, "matched_ids": matched_str})
        df = pl.DataFrame(rows)
        df.write_csv(str(path), separator="\t")
        n_matched = sum(1 for v in predictions.values() if v)
        n_singletons = sum(1 for v in predictions.values() if not v)
        logger.info(
            f"matching_results.tsv written: {len(predictions):,} entities "
            f"({n_matched} with matches, {n_singletons} singletons) → {path}"
        )
