"""
Master Record Normalizer — orchestrates all normalization sub-steps.

Applied to every record from Source 1, 2, and 3 before any downstream step.

Outputs a Polars DataFrame with both original columns and normalized columns:
  norm_name, norm_address, norm_name_no_suffix, canonical_suffix,
  parsed_city, parsed_state, parsed_postal, parsed_street_num,
  transliterated_name, transliterated_address
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

import polars as pl
from loguru import logger
from tqdm import tqdm

from .legal_suffix import LegalSuffixCanonicalizer
from .address_parser import AddressParser
from .transliterator import ScriptTransliterator

_NULL_TOKENS = frozenset([
    "null", "n/a", "na", "-", "none", "nan", ".", "\\", "unknown",
    "not available", "not applicable",
])


class RecordNormalizer:
    """
    Full normalization pipeline for a business record DataFrame.
    """

    def __init__(
        self,
        suffix_canonicalizer: Optional[LegalSuffixCanonicalizer] = None,
        address_parser: Optional[AddressParser] = None,
        transliterator: Optional[ScriptTransliterator] = None,
        batch_size: int = 10_000,
    ) -> None:
        self.suffix_canon = suffix_canonicalizer or LegalSuffixCanonicalizer()
        self.addr_parser = address_parser or AddressParser()
        self.translit = transliterator or ScriptTransliterator()
        self.batch_size = batch_size

    def fit_suffix_map(self, df: pl.DataFrame) -> "RecordNormalizer":
        """Learn additional suffix variants from training data."""
        names = df["business_name"].drop_nulls().to_list()
        self.suffix_canon.fit(names)
        return self

    def normalize(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply full normalization pipeline to every row.
        Returns df with additional normalized columns.
        """
        logger.info(f"Normalizing {len(df):,} records...")
        results = []
        rows = df.iter_rows(named=True)

        for i, row in enumerate(tqdm(rows, total=len(df), desc="Normalizing")):
            results.append(self._normalize_row(row))
            if (i + 1) % 50_000 == 0:
                logger.debug(f"  Normalized {i+1:,} rows...")

        # Build normalized columns as series
        new_cols = {
            k: [r[k] for r in results]
            for k in results[0].keys()
        }
        for col_name, values in new_cols.items():
            df = df.with_columns(pl.Series(col_name, values).alias(col_name))

        logger.info("Normalization complete.")
        return df

    def normalize_polars_lazy(
        self, df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Vectorized normalization of text columns using Polars expressions.
        Faster for simple transforms; row-level operations use normalize().
        """
        df = df.with_columns([
            # Replace NULL tokens with null
            pl.when(
                pl.col("business_name").str.to_lowercase().is_in(_NULL_TOKENS)
            ).then(None).otherwise(pl.col("business_name")).alias("business_name"),
            pl.when(
                pl.col("business_address").str.to_lowercase().is_in(_NULL_TOKENS)
            ).then(None).otherwise(pl.col("business_address")).alias("business_address"),
        ])
        return df

    # ------------------------------------------------------------------ #
    #  Private                                                             #
    # ------------------------------------------------------------------ #

    def _normalize_row(self, row: dict) -> dict:
        name_raw = str(row.get("business_name") or "")
        addr_raw = str(row.get("business_address") or "")
        country = str(row.get("country") or "")

        # --- Replace null tokens ---
        name_raw = "" if name_raw.lower().strip() in _NULL_TOKENS else name_raw
        addr_raw = "" if addr_raw.lower().strip() in _NULL_TOKENS else addr_raw

        # --- NFKC normalization ---
        name_nfkc = unicodedata.normalize("NFKC", name_raw)
        addr_nfkc = unicodedata.normalize("NFKC", addr_raw)

        # --- Case fold ---
        name_lower = name_nfkc.lower().strip()
        addr_lower = addr_nfkc.lower().strip()

        # --- Remove excess punctuation but preserve structure ---
        name_clean = re.sub(r"[^\w\s,&.'-]", " ", name_lower)
        name_clean = re.sub(r"\s+", " ", name_clean).strip()
        addr_clean = re.sub(r"[^\w\s,./'-]", " ", addr_lower)
        addr_clean = re.sub(r"\s+", " ", addr_clean).strip()

        # --- Legal suffix canonicalization ---
        name_no_suffix, canonical_suffix = self.suffix_canon.canonicalize(name_clean)

        # --- Address abbreviation expansion ---
        addr_expanded = self.addr_parser.expand_abbreviations(addr_clean)

        # --- Address structured parsing ---
        parsed = self.addr_parser.parse(addr_expanded, country=country)

        # --- Transliteration ---
        norm_name = self.translit.normalize(name_clean)
        norm_addr = self.translit.normalize(addr_expanded)
        name_no_suffix_t = self.translit.normalize(name_no_suffix)

        return {
            "norm_name": norm_name,
            "norm_address": norm_addr,
            "norm_name_no_suffix": name_no_suffix_t,
            "canonical_suffix": canonical_suffix,
            "parsed_city": parsed["city"],
            "parsed_state": parsed["state"],
            "parsed_postal": parsed["postal_code"],
            "parsed_street_num": parsed["street_number"],
            "parsed_street_name": parsed["street_name"],
            "full_norm_address": parsed["full_normalized"],
        }
