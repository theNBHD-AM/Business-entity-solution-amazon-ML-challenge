"""
Script Transliteration — Devanagari → Latin and generic multi-script normalization.

Uses the `indic-transliteration` library (Apache-2.0) for Devanagari→IAST/HK,
then applies Unicode NFKD stripping to produce an ASCII-comparable form.

NO external database lookups — purely rule-based / library-based.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

from loguru import logger

_HAS_INDIC = False
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as indic_transliterate
    _HAS_INDIC = True
except ImportError:
    logger.warning(
        "indic-transliteration not installed. Devanagari normalization disabled."
        " Install with: pip install indic-transliteration"
    )

# Unicode block ranges
_DEVANAGARI_PAT = re.compile(r"[\u0900-\u097F]+")
_LATIN_PAT = re.compile(r"[a-zA-Z0-9\s]+")


class ScriptTransliterator:
    """
    Normalizes text containing Devanagari or other non-Latin scripts.

    Strategy:
      1. Detect Devanagari spans.
      2. Transliterate each span to HK (Harvard-Kyoto) scheme → ASCII.
      3. Apply NFKD decomposition + ASCII strip on the full result.
      4. This produces a consistent Latin key suitable for similarity hashing.
    """

    def __init__(self, target_scheme: str = "HK") -> None:
        """
        Args:
            target_scheme: transliteration target. HK (Harvard-Kyoto) is all-ASCII.
                           IAST produces diacritics (fine for cosine similarity,
                           but needs further stripping for phonetic keys).
        """
        self.target_scheme = target_scheme
        self._enabled = _HAS_INDIC

    def normalize(self, text: str) -> str:
        """
        Full normalization pipeline:
          NFKC → lowercase → transliterate Devanagari → NFKD strip diacritics.
        """
        if not text or not isinstance(text, str):
            return ""

        # Step 1: Unicode NFKC compose
        text = unicodedata.normalize("NFKC", text)

        # Step 2: Transliterate Devanagari spans
        if self._enabled and _DEVANAGARI_PAT.search(text):
            text = self._transliterate_devanagari(text)

        # Step 3: NFKD + strip non-ASCII diacritics
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", errors="ignore").decode("ascii")

        # Step 4: lowercase, collapse whitespace
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _transliterate_devanagari(self, text: str) -> str:
        """Replace each Devanagari span with its HK transliteration."""
        def _repl(m: re.Match) -> str:
            try:
                return indic_transliterate(
                    m.group(0),
                    sanscript.DEVANAGARI,
                    getattr(sanscript, self.target_scheme, sanscript.HK),
                )
            except Exception:
                return m.group(0)  # pass through on failure

        return _DEVANAGARI_PAT.sub(_repl, text)

    def detect_script(self, text: str) -> str:
        """Return 'devanagari', 'latin', or 'mixed'."""
        if not text:
            return "empty"
        has_dev = bool(_DEVANAGARI_PAT.search(text))
        has_lat = bool(re.search(r"[a-zA-Z]", text))
        if has_dev and has_lat:
            return "mixed"
        if has_dev:
            return "devanagari"
        return "latin"
