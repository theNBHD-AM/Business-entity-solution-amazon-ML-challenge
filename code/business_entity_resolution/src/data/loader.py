"""
Data Engineering Layer — chunked/streaming loader.

Supports multi-GB TSV files via polars lazy scan or pandas chunked reads.
Every intermediate result is persisted as Parquet for downstream efficiency.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
import polars as pl
from loguru import logger

from .corruption_detector import CorruptionDetector


COLUMNS = ["entity_id", "business_name", "business_address", "country"]


class DataLoader:
    """
    Streaming loader for business entity TSV files.

    Design choices:
    - Polars lazy scan is used when the file is on-disk (fast, zero-copy).
    - Falls back to chunked pandas reads for compatibility.
    - Corrupted rows are quarantined, not dropped from Source 1's perspective.
    """

    def __init__(
        self,
        chunk_size: int = 50_000,
        processed_dir: str = "data/processed",
        artifacts_dir: str = "data/artifacts",
    ) -> None:
        self.chunk_size = chunk_size
        self.processed_dir = Path(processed_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.corruption_detector = CorruptionDetector()
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def load_polars_lazy(self, tsv_path: str | Path) -> pl.LazyFrame:
        """Return a Polars LazyFrame — schema inferred, separator=tab."""
        return pl.scan_csv(
            str(tsv_path),
            separator="\t",
            null_values=["NULL", "N/A", "NA", "-", "NONE", "null", "none"],
            infer_schema_length=5000,
            ignore_errors=False,
        )

    def load_full_polars(self, tsv_path: str | Path) -> pl.DataFrame:
        """Eagerly collect a TSV into a Polars DataFrame with corruption tagging."""
        tsv_path = Path(tsv_path)
        logger.info(f"Loading (Polars eager): {tsv_path.name}")
        try:
            df = pl.read_csv(
                str(tsv_path),
                separator="\t",
                null_values=["NULL", "N/A", "NA", "-", "NONE", "null", "none"],
                infer_schema_length=5000,
                ignore_errors=True,
            )
        except Exception as exc:
            logger.warning(f"Polars read failed ({exc}), falling back to pandas.")
            df = self._pandas_fallback_to_polars(tsv_path)

        df = self._ensure_columns(df)
        df, corrupt_ids = self.corruption_detector.tag(df)
        if corrupt_ids:
            self._save_corrupted(corrupt_ids, tsv_path.stem)
        logger.info(f"  Loaded {len(df):,} rows ({len(corrupt_ids)} flagged corrupt).")
        return df

    def iter_chunks_pandas(
        self, tsv_path: str | Path
    ) -> Iterator[pd.DataFrame]:
        """Yield pandas DataFrame chunks for memory-constrained environments."""
        tsv_path = Path(tsv_path)
        logger.info(f"Streaming (pandas chunks={self.chunk_size:,}): {tsv_path.name}")
        reader = pd.read_csv(
            tsv_path,
            sep="\t",
            chunksize=self.chunk_size,
            dtype=str,
            keep_default_na=False,
            na_values=["NULL", "N/A", "NA", "-", "NONE", "null", "none"],
            on_bad_lines="warn",
            encoding="utf-8",
            encoding_errors="replace",
        )
        for i, chunk in enumerate(reader):
            chunk = self._ensure_columns_pd(chunk)
            yield chunk

    def save_parquet(
        self, df: pl.DataFrame | pd.DataFrame, name: str, subdir: str = "processed"
    ) -> Path:
        """Persist a DataFrame as Parquet; return path."""
        base = self.processed_dir if subdir == "processed" else self.artifacts_dir
        out = base / f"{name}.parquet"
        if isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df)
        df.write_parquet(str(out), compression="zstd")
        logger.debug(f"Saved {len(df):,} rows → {out}")
        return out

    def load_parquet(self, name: str, subdir: str = "processed") -> pl.DataFrame:
        base = self.processed_dir if subdir == "processed" else self.artifacts_dir
        path = base / f"{name}.parquet"
        return pl.read_parquet(str(path))

    def load_labels(self, labels_path: str | Path) -> dict[str, list[str]]:
        """
        Load ground-truth labels.
        Returns: {source1_id: [source2_or_3_ids, ...]}
        """
        df = pl.read_csv(
            str(labels_path),
            separator="\t",
            has_header=True,
        )
        # Expected columns: source1_entity_id / source1_id / entity_id
        # matched_entity_ids / matched_ids / match_ids (comma or pipe separated)
        labels: dict[str, list[str]] = {}
        for row in df.iter_rows(named=True):
            s1 = str(
                row.get("source1_entity_id")
                or row.get("source1_id")
                or row.get("entity_id")
                or ""
            )
            matched_raw = str(
                row.get("matched_entity_ids")
                or row.get("matched_ids")
                or row.get("match_ids")
                or ""
            )
            if matched_raw and matched_raw not in ("", "nan", "None"):
                # Handle both comma and pipe delimiters
                delims = "," if "," in matched_raw else "|"
                matched = [m.strip() for m in matched_raw.split(delims) if m.strip()]
            else:
                matched = []
            labels[s1] = matched
        logger.info(f"Loaded labels for {len(labels):,} Source 1 entities.")
        return labels

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _pandas_fallback_to_polars(self, path: Path) -> pl.DataFrame:
        chunks = []
        for chunk in self.iter_chunks_pandas(path):
            chunks.append(pl.from_pandas(chunk))
        return pl.concat(chunks)

    @staticmethod
    def _ensure_columns(df: pl.DataFrame) -> pl.DataFrame:
        for col in COLUMNS:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).alias(col).cast(pl.Utf8))
        return df.select(COLUMNS + [c for c in df.columns if c not in COLUMNS])

    @staticmethod
    def _ensure_columns_pd(df: pd.DataFrame) -> pd.DataFrame:
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df

    def _save_corrupted(
        self, corrupt_ids: list[str], source_name: str
    ) -> None:
        out = self.artifacts_dir / f"{source_name}_corrupted_ids.txt"
        with open(out, "w", encoding="utf-8") as f:
            for eid in corrupt_ids:
                f.write(eid + "\n")
        logger.warning(
            f"  {len(corrupt_ids)} corrupted entity IDs written to {out}"
        )
