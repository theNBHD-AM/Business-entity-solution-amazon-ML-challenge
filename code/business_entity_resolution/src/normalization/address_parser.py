"""
Address parsing and abbreviation expansion.

Extracts structured sub-fields:
  - street_number, street_name, city, state, postal_code

Handles:
  - US, India, and France address conventions (country-agnostic heuristics)
  - Abbreviation expansion: Rd→Road, St→Street, Blvd→Boulevard, etc.
  - Devanagari/mixed-script addresses forwarded to transliterator downstream.
"""
from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Abbreviation expansion dictionary (generic — not tied to any data source)
# ---------------------------------------------------------------------------
ABBREV_EXPAND: dict[str, str] = {
    # Road types
    r"\bRd\b": "Road",
    r"\bRd\.\b": "Road",
    r"\bSt\b(?!\s+[A-Z][a-z])": "Street",  # avoid expanding "St" as saint prefix
    r"\bSt\.\b": "Street",
    r"\bAve\b": "Avenue",
    r"\bAve\.\b": "Avenue",
    r"\bBlvd\b": "Boulevard",
    r"\bBlvd\.\b": "Boulevard",
    r"\bDr\b": "Drive",
    r"\bDr\.\b": "Drive",
    r"\bLn\b": "Lane",
    r"\bLn\.\b": "Lane",
    r"\bCt\b": "Court",
    r"\bCt\.\b": "Court",
    r"\bPl\b": "Place",
    r"\bPl\.\b": "Place",
    r"\bHwy\b": "Highway",
    r"\bHwy\.\b": "Highway",
    r"\bFwy\b": "Freeway",
    r"\bPkwy\b": "Parkway",
    r"\bPkwy\.\b": "Parkway",
    r"\bTrl\b": "Trail",
    r"\bTer\b": "Terrace",
    r"\bTer\.\b": "Terrace",
    r"\bCir\b": "Circle",
    r"\bCir\.\b": "Circle",
    r"\bExpy\b": "Expressway",
    # Directional
    r"\bN\b": "North",
    r"\bS\b": "South",
    r"\bE\b": "East",
    r"\bW\b": "West",
    r"\bNE\b": "Northeast",
    r"\bNW\b": "Northwest",
    r"\bSE\b": "Southeast",
    r"\bSW\b": "Southwest",
    # Suite/Unit
    r"\bSte\b": "Suite",
    r"\bSte\.\b": "Suite",
    r"\bApt\b": "Apartment",
    r"\bApt\.\b": "Apartment",
    r"\bDept\b": "Department",
    # India specific
    r"\bNagar\b": "Nagar",
    r"\bMarg\b": "Marg",
    r"\bVihar\b": "Vihar",
    # France (generic, no lookup)
    r"\bRue\b": "Rue",
    r"\bAv\.\b": "Avenue",
    r"\bBd\b": "Boulevard",
    r"\bCedex\b": "Cedex",
}

_COMPILED_ABBREVS = [
    (re.compile(pat, re.IGNORECASE), expansion)
    for pat, expansion in ABBREV_EXPAND.items()
]

# Postal code patterns
_US_ZIP = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
_IN_PIN = re.compile(r"\b([1-9][0-9]{5})\b")
_FR_ZIP = re.compile(r"\b([0-9]{5})\b")
_GENERIC_ZIP = re.compile(r"\b([A-Z0-9]{3,10})\b")

# Street number
_STREET_NUM = re.compile(r"^(\d+[A-Za-z]?)\b")


class AddressParser:
    """Parses and normalizes business address strings."""

    def expand_abbreviations(self, text: str) -> str:
        """Expand road/directional abbreviations in place."""
        for pat, replacement in _COMPILED_ABBREVS:
            text = pat.sub(replacement, text)
        return text

    def parse(self, address: str, country: Optional[str] = None) -> dict:
        """
        Returns a dict with extracted sub-fields.
        All fields default to empty string if not found.
        """
        if not address or not isinstance(address, str):
            return self._empty()

        # Expand abbreviations
        expanded = self.expand_abbreviations(address)

        # Tokenize by commas and whitespace
        parts = [p.strip() for p in re.split(r"[,\n]", expanded) if p.strip()]

        result = self._empty()
        result["full_normalized"] = " ".join(parts)

        # Street number from first part
        if parts:
            m = _STREET_NUM.match(parts[0])
            if m:
                result["street_number"] = m.group(1)
                result["street_name"] = parts[0][m.end():].strip()
            else:
                result["street_name"] = parts[0]

        # Postal code detection (order: US, India, France, generic)
        country_upper = (country or "").upper()
        combined = " ".join(parts)
        if "US" in country_upper or not country_upper:
            m = _US_ZIP.search(combined)
            if m:
                result["postal_code"] = m.group(1)

        if not result["postal_code"] and "IN" in country_upper or "INDIA" in country_upper:
            m = _IN_PIN.search(combined)
            if m:
                result["postal_code"] = m.group(1)

        if not result["postal_code"] and "FR" in country_upper or "FRANCE" in country_upper:
            m = _FR_ZIP.search(combined)
            if m:
                result["postal_code"] = m.group(1)

        if not result["postal_code"]:
            m = _GENERIC_ZIP.search(combined)
            if m:
                result["postal_code"] = m.group(1)

        # City and state heuristics (last significant parts)
        if len(parts) >= 2:
            result["city"] = parts[-2] if len(parts) >= 2 else ""
            result["state"] = parts[-1] if len(parts) >= 3 else ""
        elif len(parts) == 1:
            # Single-part address: try to identify city
            result["city"] = parts[0]

        return result

    @staticmethod
    def _empty() -> dict:
        return {
            "full_normalized": "",
            "street_number": "",
            "street_name": "",
            "city": "",
            "state": "",
            "postal_code": "",
        }
