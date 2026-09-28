"""
End-to-End Pipeline Orchestrator
=================================
Connects all seven layers of the Business Entity Resolution pipeline.

Usage:
    python -m src.pipeline --config configs/pipeline_config.yaml --mode train
    python -m src.pipeline --config configs/pipeline_config.yaml --mode test

Modes:
    train  : Normalize → Block → Features → Train GBM → Calibrate → Tune threshold
    test   : Load artifacts → Block test data → Features → Predict → Write outputs
    full   : Run train then test in one shot
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import yaml
from loguru import logger

from .data.loader import DataLoader
from .normalization.normalizer import RecordNormalizer
from .normalization.legal_suffix import LegalSuffixCanonicalizer
from .normalization.address_parser import AddressParser
from .normalization.transliterator import ScriptTransliterator
from .normalization.synthetic_france import FrenchAugmentor
from .blocking.embedder import EntityEmbedder
from .blocking.ann_index import ANNIndex
from .blocking.minhash_lsh import MinHashLSHBlocker
from .blocking.phonetic_blocker import PhoneticBlocker
from .blocking.geo_blocker import GeoBlocker
from .blocking.candidate_generator import CandidateGenerator
from .features.feature_builder import FeatureBuilder
from .model.gbm_classifier import GBMClassifier
from .model.calibrator import ProbabilityCalibrator
from .model.hard_negative_miner import HardNegativeMiner
from .evaluation.metrics import EntityResolutionMetrics
from .evaluation.threshold_tuner import ThresholdTuner
from .inference.predictor import EntityResolutionPredictor


# ============================================================
#  Logging setup
# ============================================================

def setup_logging(log_dir: str = "logs", level: str = "INFO") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    logger.add(
        f"{log_dir}/pipeline_{{time:YYYYMMDD_HHmmss}}.log",
        level="DEBUG",
        rotation="100 MB",
        retention="7 days",
    )


# ============================================================
#  Config loader
# ============================================================

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
#  Pipeline class
# ============================================================

class EntityResolutionPipeline:
    """
    Full end-to-end pipeline orchestrating all modular layers.

    Design principle: every intermediate artifact is persisted as Parquet
    so the pipeline can be resumed from any stage without recomputation.
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.loader = DataLoader(
            chunk_size=config["data"]["chunk_size"],
            processed_dir=config["paths"]["processed_dir"],
            artifacts_dir=config["paths"]["artifacts_dir"],
        )
        self.normalizer = RecordNormalizer(
            suffix_canonicalizer=LegalSuffixCanonicalizer(
                min_freq=config["normalization"]["legal_suffix_min_freq"]
            ),
            address_parser=AddressParser(),
            transliterator=ScriptTransliterator(),
        )
        emb_cfg = config["blocking"]
        self.embedder = EntityEmbedder(
            model_name=emb_cfg["embedding_model"],
            batch_size=emb_cfg["embedding_batch_size"],
            max_length=emb_cfg["embedding_max_length"],
        )
        self.ann = ANNIndex(
            dim=self.embedder.dim,
            nlist=emb_cfg["faiss_nlist"],
            nprobe=emb_cfg["faiss_nprobe"],
            m=emb_cfg["faiss_m"],
            nbits=emb_cfg["faiss_nbits"],
        )
        self.lsh = MinHashLSHBlocker(
            num_perm=emb_cfg["minhash_num_perm"],
            threshold=emb_cfg["minhash_threshold"],
        )
        self.phonetic = PhoneticBlocker() if emb_cfg.get("phonetic_blocking") else None
        self.geo = GeoBlocker() if emb_cfg.get("geo_blocking") else None
        self.gen = CandidateGenerator(
            ann_index=self.ann,
            lsh_blocker=self.lsh,
            phonetic_blocker=self.phonetic,
            geo_blocker=self.geo,
            ann_top_k=emb_cfg["ann_top_k"],
            final_top_k=emb_cfg["final_top_k"],
        )
        self.feature_builder = FeatureBuilder(
            tfidf_vocab_size=config["features"]["tfidf_vocab_size"],
        )
        m_cfg = config["model"]
        self.classifier = GBMClassifier(
            backend=m_cfg["gbm_backend"],
            n_folds=m_cfg["n_folds"],
            random_state=m_cfg["random_state"],
            params=m_cfg.get("lightgbm_params", {}),
        )
        self.calibrator = ProbabilityCalibrator(
            method=m_cfg["calibration_method"]
        )
        self.miner = HardNegativeMiner(
            hard_negative_ratio=m_cfg["hard_negative_ratio"],
        )
        t_cfg = config["threshold"]
        self.tuner = ThresholdTuner(
            beta=t_cfg["beta"],
            search_low=t_cfg["search_range"][0],
            search_high=t_cfg["search_range"][1],
            n_steps=t_cfg["search_steps"],
        )
        self.metrics = EntityResolutionMetrics(beta=t_cfg["beta"])
        self.best_threshold: float = 0.5

    # ------------------------------------------------------------------ #
    #  TRAIN MODE                                                          #
    # ------------------------------------------------------------------ #

    def train(self) -> None:
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("TRAIN MODE")
        logger.info("=" * 60)

        cfg = self.cfg
        processed = Path(cfg["paths"]["processed_dir"])
        artifacts = Path(cfg["paths"]["artifacts_dir"])

        # ── 1. Load & normalize training data ──────────────────────────
        logger.info("Stage 1: Loading and normalizing training data...")

        s1_norm = self._load_and_normalize(
            cfg["data"]["source1_train"], "s1_train_norm"
        )
        s2_norm = self._load_and_normalize(
            cfg["data"]["source2_train"], "s2_train_norm"
        )
        s3_norm = self._load_and_normalize(
            cfg["data"]["source3_train"], "s3_train_norm"
        )
        s23_norm = pl.concat([s2_norm, s3_norm], how="diagonal")
        self.loader.save_parquet(s23_norm, "s23_train_norm")

        # Inject synthetic France records into a held-out validation subset
        if cfg["synthetic_france"]["enabled"]:
            logger.info("Injecting synthetic French records for generalization testing...")
            augmentor = FrenchAugmentor()
            fr_s1, fr_s23, fr_labels = augmentor.generate_pairs(
                n_pairs=cfg["synthetic_france"]["n_samples"] // 5
            )
            # We'll keep these separate for the France-specific eval below

        # ── 2. Build embeddings ─────────────────────────────────────────
        logger.info("Stage 2: Building embeddings...")
        s1_emb = self._get_or_build_embeddings(s1_norm, "s1_train_emb")
        s23_emb = self._get_or_build_embeddings(s23_norm, "s23_train_emb")

        # ── 3. Build blocking indexes ───────────────────────────────────
        logger.info("Stage 3: Building blocking indexes...")

        # FAISS index on S23
        s23_ids = s23_norm["entity_id"].to_list()
        self.ann.build(s23_emb, ids=np.arange(len(s23_ids)))
        self.ann.save(artifacts / "ann_s23_train.faiss")

        # MinHash LSH on S23 names
        self.lsh.index(s23_ids, s23_norm["norm_name"].to_list())

        # Phonetic index on S23
        phonetic_index = None
        if self.phonetic:
            phonetic_index = self.phonetic.build_index(
                s23_ids, s23_norm["norm_name"].to_list()
            )

        # Geo index on S23
        geo_index = None
        if self.geo:
            geo_index = self.geo.build_index(
                ids=s23_ids,
                countries=s23_norm["country"].to_list(),
                cities=s23_norm["parsed_city"].to_list(),
                postals=s23_norm["parsed_postal"].to_list(),
            )

        # Update candidate generator with built indexes
        self.gen.ann = self.ann
        self.gen.lsh = self.lsh
        self.gen.phonetic = self.phonetic
        self.gen.geo = self.geo

        # Set ID lookup maps for feature builder
        s23_id_to_idx = {eid: i for i, eid in enumerate(s23_ids)}
        s1_ids = s1_norm["entity_id"].to_list()
        s1_id_to_idx = {eid: i for i, eid in enumerate(s1_ids)}
        self.feature_builder.s1_embeddings = s1_emb
        self.feature_builder.s23_embeddings = s23_emb
        self.feature_builder.s1_id_to_idx = s1_id_to_idx
        self.feature_builder.s23_id_to_idx = s23_id_to_idx

        # ── 4. Generate training candidates ────────────────────────────
        logger.info("Stage 4: Generating training candidates...")
        s1_query_texts = self.embedder.build_texts(
            s1_norm["norm_name"].to_list(),
            s1_norm["norm_address"].to_list(),
            prefix="query: ",
        )
        s1_query_emb = self.embedder.encode_to_file(
            s1_query_texts, artifacts / "s1_train_query_emb.npy"
        )
        candidates_train = self.gen.generate(
            s1_df=s1_norm,
            s23_df=s23_norm,
            s1_embeddings=s1_query_emb,
            s23_embeddings=s23_emb,
            phonetic_index=phonetic_index,
            geo_index=geo_index,
        )
        self.loader.save_parquet(candidates_train, "candidates_train", subdir="artifacts")

        # Measure blocking recall
        labels = self.loader.load_labels(cfg["data"]["train_labels"])
        blocking_stats = self.metrics.compute_blocking_recall(candidates_train, labels)
        logger.info(
            f"Blocking recall: {blocking_stats['blocking_recall']:.4f} "
            f"(this is the recall ceiling)"
        )

        # ── 5. Build training features + labels ────────────────────────
        logger.info("Stage 5: Building training features...")
        self.feature_builder.fit_tfidf(
            s23_ids, s23_norm["norm_name"].to_list()
        )
        s1_lookup = self.feature_builder.build_lookup(s1_norm)
        s23_lookup = self.feature_builder.build_lookup(s23_norm)

        X_all = self.feature_builder.build_features(
            candidates_train, s1_lookup, s23_lookup
        )

        # Build labels array aligned with candidates_train rows
        y_all = self._build_label_array(candidates_train, labels)
        np.save(str(artifacts / "X_train.npy"), X_all)
        np.save(str(artifacts / "y_train.npy"), y_all)

        # Country group labels for stratification
        country_map = {
            row["entity_id"]: str(row.get("country") or "unknown")
            for row in s1_norm.iter_rows(named=True)
        }

        # ── 6. Hard negative mining + GBM training ─────────────────────
        logger.info("Stage 6: Training GBM classifier...")
        X_mined, y_mined, weights = self.miner.mine(X_all, y_all)
        self.classifier.fit(
            X_mined, y_mined,
            sample_weight=weights,
            feature_names=self.feature_builder.feature_names,
        )

        # ── 7. Calibrate on OOF predictions ────────────────────────────
        logger.info("Stage 7: Calibrating probabilities...")
        # Re-predict on full training set for OOF-based calibration
        oof_probs_full = self.classifier.predict_proba(X_all)
        self.calibrator.fit(oof_probs_full, y_all)

        # ── 8. Validation split & threshold tuning ─────────────────────
        logger.info("Stage 8: Threshold tuning on validation split...")
        # Use last 20% of S1 entities as validation
        n_val = max(int(len(s1_ids) * 0.2), 1)
        val_s1_ids = set(s1_ids[-n_val:])
        val_mask_series = candidates_train["source1_id"].is_in(list(val_s1_ids))
        val_candidates = candidates_train.filter(val_mask_series)
        val_X = self.feature_builder.build_features(val_candidates, s1_lookup, s23_lookup)
        val_raw_probs = self.classifier.predict_proba(val_X)
        val_cal_probs = self.calibrator.predict(val_raw_probs)
        val_labels = {k: v for k, v in labels.items() if k in val_s1_ids}
        # Ensure all val S1 entities appear (including those with no candidates)
        for sid in val_s1_ids:
            if sid not in val_labels:
                val_labels[sid] = []

        self.best_threshold = self.tuner.tune(
            val_candidates_df=val_candidates,
            val_probs=val_cal_probs,
            ground_truth=val_labels,
            country_map=country_map,
        )

        # Full validation evaluation at best threshold
        val_preds = self.tuner.apply(val_candidates, val_cal_probs, self.best_threshold)
        # Add singletons with no candidates
        for sid in val_s1_ids:
            if sid not in val_preds:
                val_preds[sid] = set()
        final_eval = self.metrics.evaluate(val_preds, val_labels, country_map=country_map)
        logger.info(
            f"Validation F_0.5 @ threshold={self.best_threshold:.4f}: "
            f"{final_eval['macro_f05']:.4f}"
        )

        # ── 9. Save all artifacts ───────────────────────────────────────
        logger.info("Stage 9: Saving artifacts...")
        self.classifier.save(artifacts / "gbm_model.pkl")
        self.calibrator.save(artifacts / "calibrator.pkl")
        import json
        with open(artifacts / "threshold.json", "w") as f:
            json.dump({"threshold": self.best_threshold}, f)
        self.normalizer.suffix_canon.save(artifacts / "legal_suffix_map.json")

        logger.info(
            f"Training complete in {(time.time()-t0)/60:.1f} min. "
            f"Best threshold: {self.best_threshold:.4f}"
        )

    # ------------------------------------------------------------------ #
    #  TEST MODE                                                           #
    # ------------------------------------------------------------------ #

    def test(self) -> None:
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("TEST MODE")
        logger.info("=" * 60)

        cfg = self.cfg
        artifacts = Path(cfg["paths"]["artifacts_dir"])
        output_dir = cfg["paths"]["output_dir"]
        import json

        # Load trained artifacts
        self.classifier = GBMClassifier.load(artifacts / "gbm_model.pkl")
        self.calibrator = ProbabilityCalibrator.load(artifacts / "calibrator.pkl")
        with open(artifacts / "threshold.json") as f:
            self.best_threshold = json.load(f)["threshold"]
        logger.info(f"Loaded threshold: {self.best_threshold:.4f}")

        # Load legal suffix map
        from .normalization.legal_suffix import LegalSuffixCanonicalizer
        self.normalizer.suffix_canon = LegalSuffixCanonicalizer.load(
            artifacts / "legal_suffix_map.json"
        )

        # Normalize test data
        logger.info("Normalizing test data...")
        s1_test = self._load_and_normalize(cfg["data"]["source1_test"], "s1_test_norm")
        s2_test = self._load_and_normalize(cfg["data"]["source2_test"], "s2_test_norm")
        s3_test = self._load_and_normalize(cfg["data"]["source3_test"], "s3_test_norm")
        s23_test = pl.concat([s2_test, s3_test], how="diagonal")

        # Embeddings
        logger.info("Building test embeddings...")
        s1_emb = self._get_or_build_embeddings(s1_test, "s1_test_emb")
        s23_emb = self._get_or_build_embeddings(s23_test, "s23_test_emb")

        # Rebuild FAISS on test S23 (test set is separate from train)
        logger.info("Building test ANN index...")
        s23_ids = s23_test["entity_id"].to_list()
        self.ann = ANNIndex(
            dim=self.embedder.dim,
            nlist=self.cfg["blocking"]["faiss_nlist"],
            nprobe=self.cfg["blocking"]["faiss_nprobe"],
            m=self.cfg["blocking"]["faiss_m"],
            nbits=self.cfg["blocking"]["faiss_nbits"],
        )
        self.ann.build(s23_emb, ids=np.arange(len(s23_ids)))

        self.lsh = MinHashLSHBlocker(
            num_perm=self.cfg["blocking"]["minhash_num_perm"],
            threshold=self.cfg["blocking"]["minhash_threshold"],
        )
        self.lsh.index(s23_ids, s23_test["norm_name"].to_list())

        phonetic_index = None
        if self.phonetic:
            phonetic_index = self.phonetic.build_index(
                s23_ids, s23_test["norm_name"].to_list()
            )
        geo_index = None
        if self.geo:
            geo_index = self.geo.build_index(
                ids=s23_ids,
                countries=s23_test["country"].to_list(),
                cities=s23_test["parsed_city"].to_list(),
                postals=s23_test["parsed_postal"].to_list(),
            )

        self.gen.ann = self.ann
        self.gen.lsh = self.lsh
        self.gen.phonetic = self.phonetic
        self.gen.geo = self.geo

        s1_ids = s1_test["entity_id"].to_list()
        self.feature_builder.s1_embeddings = s1_emb
        self.feature_builder.s23_embeddings = s23_emb
        self.feature_builder.s1_id_to_idx = {eid: i for i, eid in enumerate(s1_ids)}
        self.feature_builder.s23_id_to_idx = {eid: i for i, eid in enumerate(s23_ids)}

        # Fit TF-IDF on test S23 names
        self.feature_builder.fit_tfidf(s23_ids, s23_test["norm_name"].to_list())

        # Run full inference
        predictor = EntityResolutionPredictor(
            candidate_generator=self.gen,
            feature_builder=self.feature_builder,
            classifier=self.classifier,
            calibrator=self.calibrator,
            threshold=self.best_threshold,
        )
        s1_query_texts = self.embedder.build_texts(
            s1_test["norm_name"].to_list(),
            s1_test["norm_address"].to_list(),
            prefix="query: ",
        )
        s1_query_emb = self.embedder.encode_to_file(
            s1_query_texts, artifacts / "s1_test_query_emb.npy"
        )
        predictor.predict(
            s1_df=s1_test,
            s23_df=s23_test,
            s1_embeddings=s1_query_emb,
            s23_embeddings=s23_emb,
            phonetic_index=phonetic_index,
            geo_index=geo_index,
            output_dir=output_dir,
        )
        logger.info(
            f"Test inference complete in {(time.time()-t0)/60:.1f} min. "
            f"Outputs written to {output_dir}/"
        )

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _load_and_normalize(
        self, tsv_path: str, artifact_name: str
    ) -> pl.DataFrame:
        """Load TSV, normalize, save Parquet, return normalized DataFrame."""
        parquet_path = Path(self.cfg["paths"]["processed_dir"]) / f"{artifact_name}.parquet"
        if parquet_path.exists():
            logger.info(f"Loading cached normalized data: {parquet_path.name}")
            return pl.read_parquet(str(parquet_path))

        df = self.loader.load_full_polars(tsv_path)
        self.normalizer.fit_suffix_map(df)
        df_norm = self.normalizer.normalize(df)
        self.loader.save_parquet(df_norm, artifact_name)
        return df_norm

    def _get_or_build_embeddings(
        self, df: pl.DataFrame, name: str
    ) -> np.ndarray:
        """Load cached embeddings or build and cache them via memmap."""
        path = Path(self.cfg["paths"]["artifacts_dir"]) / f"{name}.npy"
        if path.exists():
            try:
                arr = self.embedder.load_embeddings(path, mmap_mode="r")
                if arr.shape[1] == self.embedder.dim and arr.shape[0] == len(df):
                    logger.info(f"Loaded valid cached embeddings: {path.name} {arr.shape}")
                    return arr
                else:
                    logger.warning(
                        f"Cached {path.name} dimension/row mismatch "
                        f"({arr.shape} vs expected ({len(df):,}, {self.embedder.dim})). Re-encoding..."
                    )
            except Exception as e:
                logger.warning(f"Failed to load cached {path.name} ({e}). Re-encoding...")

        texts = self.embedder.build_texts(
            df["norm_name"].to_list(),
            df["norm_address"].to_list(),
            prefix="passage: ",
        )
        return self.embedder.encode_to_file(texts, path)

    @staticmethod
    def _build_label_array(
        candidates_df: pl.DataFrame,
        labels: dict[str, list[str]],
    ) -> np.ndarray:
        """Align binary labels with candidate_df row order."""
        y = []
        for row in candidates_df.iter_rows(named=True):
            s1_id = row["source1_id"]
            cand_id = row["candidate_id"]
            true_matches = set(labels.get(s1_id, []))
            y.append(1 if cand_id in true_matches else 0)
        arr = np.array(y, dtype=np.int32)
        n_pos = arr.sum()
        logger.info(
            f"Label array: {len(arr):,} pairs, {n_pos} positives "
            f"({n_pos/max(len(arr),1)*100:.2f}% positive rate)"
        )
        return arr
