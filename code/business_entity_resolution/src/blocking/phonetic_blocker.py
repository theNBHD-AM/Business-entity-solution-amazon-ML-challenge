"""
Phonetic Blocker — Soundex / Metaphone key generation.

Provides a cheap, language-agnostic blocking key based on the first
phonetically significant token of the business name.

Uses jellyfish (MIT license).
"""
from __future__ import annotations

import re
from collections import defaultdict

from loguru import logger

try:
    import jellyfish
    _HAS_JELLYFISH = True
except ImportError:
    _HAS_JELLYFISH = False
    logger.warning("jellyfish not installed. Phonetic blocking disabled.")

_STOP_TOKENS = frozenset([
    "the", "a", "an", "of", "and", "&", "for", "in", "at",
    "by", "to", "de", "du", "la", "le", "les", "et",
])


class PhoneticBlocker:
    """
    Generates phonetic keys (Soundex + Metaphone) for blocking.
    Groups entities with the same phonetic key as candidates.
    """

    def __init__(self, method: str = "both") -> None:
        """
        Args:
            method: 'soundex', 'metaphone', or 'both' (union of both keys).
        """
        if not _HAS_JELLYFISH:
            raise RuntimeError("jellyfish required for phonetic blocking.")
        self.method = method

    def key(self, name: str) -> list[str]:
        """Compute phonetic key(s) for a normalized name."""
        # Extract significant first token
        tokens = re.sub(r"[^\w\s]", " ", name.lower()).split()
        sig_tokens = [t for t in tokens if t not in _STOP_TOKENS and len(t) > 1]
        if not sig_tokens:
            return []
        first = sig_tokens[0]
        keys = []
        if self.method in ("soundex", "both"):
            try:
                keys.append("SX:" + jellyfish.soundex(first))
            except Exception:
                pass
        if self.method in ("metaphone", "both"):
            try:
                keys.append("MP:" + jellyfish.metaphone(first))
            except Exception:
                pass
        return keys

    def build_index(
        self,
        ids: list[str],
        names: list[str],
    ) -> dict[str, list[str]]:
        """Build {phonetic_key: [entity_ids]} index."""
        index: dict[str, list[str]] = defaultdict(list)
        for eid, name in zip(ids, names):
            for k in self.key(name):
                index[k].append(eid)
        return dict(index)

    def get_candidates(
        self,
        query_name: str,
        phonetic_index: dict[str, list[str]],
        exclude_id: Optional[str] = None,
    ) -> list[str]:
        """Return IDs sharing a phonetic key with query_name."""
        cands = set()
        for k in self.key(query_name):
            cands.update(phonetic_index.get(k, []))
        if exclude_id:
            cands.discard(exclude_id)
        return list(cands)
