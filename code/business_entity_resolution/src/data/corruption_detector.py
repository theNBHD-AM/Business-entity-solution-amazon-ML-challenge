"""
Corruption Detection — identifies mis-encoded or structurally broken rows.

Detection signals:
  1. Replacement-character density (U+FFFD) > threshold → encoding corruption.
  2. Anomalously short or empty entity_id.
  3. Both name and address are null/empty.
  4. entity_id prefix not matching expected pattern (S1-/S2-/S3-).
  5. Non-UTF-8 control characters in text fields.

Corrective actions:
  - Flag rows with a `_corrupt` boolean column.
  - Attempt basic encoding repair (latin-1 re-encode) for mild cases.
  - Log all anomalies for manual inspection.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

import polars as pl
from loguru import logger

_REPLACEMENT = "\ufffd"
_CONTROL_PAT = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_EXPECTED_ID_PAT = re.compile(r"^S[123]-")


class CorruptionDetector:
    """
    Tags corrupted rows in a Polars DataFrame and returns the cleaned frame
    alongside a list of corrupted entity_ids.
    """

    def __init__(
        self,
        replacement_threshold: float = 0.05,
        min_name_len: int = 1,
    ) -> None:
        self.replacement_threshold = replacement_threshold
        self.min_name_len = min_name_len

    def tag(
        self, df: pl.DataFrame
    ) -> tuple[pl.DataFrame, list[str]]:
        """
        Add a `_corrupt` boolean column and return (df_with_flag, corrupt_ids).
        Does NOT drop corrupted rows — Source 1 entities must always appear in output.
        """
        flags = []
        for row in df.iter_rows(named=True):
            flags.append(self._is_corrupt(row))

        df = df.with_columns(pl.Series("_corrupt", flags, dtype=pl.Boolean))
        corrupt_ids = (
            df.filter(pl.col("_corrupt"))["entity_id"]
            .drop_nulls()
            .to_list()
        )
        if corrupt_ids:
            logger.warning(
                f"CorruptionDetector: {len(corrupt_ids)} rows flagged. "
                f"Sample IDs: {corrupt_ids[:5]}"
            )
        return df, corrupt_ids

    # ------------------------------------------------------------------ #
    #  Private                                                             #
    # ------------------------------------------------------------------ #

    def _is_corrupt(self, row: dict) -> bool:
        eid = str(row.get("entity_id") or "")
        name = str(row.get("business_name") or "")
        addr = str(row.get("business_address") or "")

        # Signal 1: no valid entity_id
        if len(eid) < 3 or not _EXPECTED_ID_PAT.match(eid):
            return True

        # Signal 2: both name and address empty
        if not name.strip() and not addr.strip():
            return True

        # Signal 3: replacement-character density in name+address
        combined = name + addr
        if combined:
            density = combined.count(_REPLACEMENT) / len(combined)
            if density > self.replacement_threshold:
                return True

        # Signal 4: control characters
        if _CONTROL_PAT.search(name) or _CONTROL_PAT.search(addr):
            return True

        return False

    @staticmethod
    def attempt_encoding_repair(text: str) -> str:
        """Try to recover latin-1 mis-decoded as UTF-8."""
        try:
            return text.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text
