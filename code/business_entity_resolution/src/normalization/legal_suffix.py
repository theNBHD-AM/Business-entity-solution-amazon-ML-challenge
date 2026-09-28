"""
Legal Suffix Canonicalization.

Built entirely from frequency analysis of training data — no external registry.
Maps observed variant → canonical token (or empty string to strip).
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Optional

import polars as pl
from loguru import logger


# Seed map — augmented from frequency analysis at runtime
_SEED_MAP: dict[str, str] = {
    # US
    "llc": "llc",
    "l.l.c.": "llc",
    "l.l.c": "llc",
    "inc": "inc",
    "inc.": "inc",
    "incorporated": "inc",
    "corp": "corp",
    "corp.": "corp",
    "corporation": "corp",
    "co": "co",
    "co.": "co",
    "company": "co",
    "ltd": "ltd",
    "ltd.": "ltd",
    "limited": "ltd",
    "lp": "lp",
    "l.p.": "lp",
    "lllp": "lllp",
    "llp": "llp",
    "l.l.p.": "llp",
    "pllc": "pllc",
    "p.l.l.c.": "pllc",
    "pc": "pc",
    "p.c.": "pc",
    "pa": "pa",
    "p.a.": "pa",
    "associates": "associates",
    "assoc": "associates",
    "assoc.": "associates",
    "group": "group",
    "grp": "group",
    "holdings": "holdings",
    "enterprises": "enterprises",
    "trust": "trust",
    "dba": "",   # strip DBA markers from name field
    "d.b.a.": "",
    "d/b/a": "",
    "t/a": "",
    "ta": "",
    # India
    "pvt": "pvt",
    "pvt.": "pvt",
    "private": "pvt",
    "pvt ltd": "pvtltd",
    "pvt. ltd.": "pvtltd",
    "private limited": "pvtltd",
    "(pvt) ltd": "pvtltd",
    "(p) ltd": "pvtltd",
    "p. ltd.": "pvtltd",
    "lmtd": "ltd",
    # France (zero-shot generalization)
    "sarl": "sarl",
    "s.a.r.l.": "sarl",
    "sas": "sas",
    "s.a.s.": "sas",
    "sa": "sa",
    "s.a.": "sa",
    "eurl": "eurl",
    "e.u.r.l.": "eurl",
    "sasu": "sasu",
    "snc": "snc",
    "sci": "sci",
    "ste": "ste",  # société abbreviated
    "societe": "ste",
    "société": "ste",
    "ei": "ei",
    "micro-entreprise": "ei",
    "auto-entrepreneur": "ei",
}


class LegalSuffixCanonicalizer:
    """
    Two-pass canonicalization:
      Pass 1: apply seed map.
      Pass 2 (optional): learn additional variants from training data frequency.
    """

    def __init__(self, min_freq: int = 5) -> None:
        self.min_freq = min_freq
        self._map: dict[str, str] = dict(_SEED_MAP)
        # Compiled regex for full-token suffix stripping
        self._build_regex()

    def fit(self, names: list[str]) -> "LegalSuffixCanonicalizer":
        """
        Learn additional suffix variants from training data.
        Looks at the last 1-3 tokens of each name after lowercasing.
        """
        counter: Counter = Counter()
        for name in names:
            tokens = re.sub(r"[^\w\s]", " ", name.lower()).split()
            for n in (1, 2, 3):
                if len(tokens) >= n + 1:  # at least one non-suffix token
                    suffix = " ".join(tokens[-n:])
                    counter[suffix] += 1

        new_variants = 0
        for suffix, freq in counter.most_common():
            if freq >= self.min_freq and suffix not in self._map:
                # Heuristic: if it looks like a legal suffix (short, no digits)
                if len(suffix) <= 20 and not any(c.isdigit() for c in suffix):
                    # Check if any word in the suffix is already a known canonical
                    parts = suffix.split()
                    for known_canon in set(self._map.values()):
                        if known_canon and any(p == known_canon for p in parts):
                            self._map[suffix] = known_canon
                            new_variants += 1
                            break
        logger.info(
            f"LegalSuffixCanonicalizer: learned {new_variants} additional variants "
            f"from {len(names):,} names."
        )
        self._build_regex()
        return self

    def canonicalize(self, name: str) -> tuple[str, str]:
        """
        Return (name_without_suffix, canonical_suffix).
        Tries multi-word suffixes first, then single-word.
        """
        clean = re.sub(r"[^\w\s]", " ", name.lower()).strip()
        tokens = clean.split()
        for n in (3, 2, 1):
            if len(tokens) > n:
                candidate = " ".join(tokens[-n:])
                if candidate in self._map:
                    core = " ".join(tokens[:-n]).strip()
                    return core, self._map[candidate]
        return clean, ""

    def strip_suffix(self, name: str) -> str:
        """Return the name with legal suffix removed."""
        core, _ = self.canonicalize(name)
        return core

    def _build_regex(self) -> None:
        # Sort by length desc so longest variants match first
        sorted_keys = sorted(self._map.keys(), key=len, reverse=True)
        escaped = [re.escape(k) for k in sorted_keys]
        pat = r"(?i)\b(" + "|".join(escaped) + r")\s*$"
        self._regex = re.compile(pat)

    def save(self, path: str | Path) -> None:
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._map, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "LegalSuffixCanonicalizer":
        import json
        obj = cls()
        with open(path, "r", encoding="utf-8") as f:
            obj._map = json.load(f)
        obj._build_regex()
        return obj
