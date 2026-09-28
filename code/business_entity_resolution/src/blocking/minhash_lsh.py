"""
MinHash LSH Blocker — character n-gram based.

Captures typo-level matches that may not be in the ANN top-K.
Uses datasketch (MIT license) with configurable shingle size.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from loguru import logger

try:
    from datasketch import MinHash, MinHashLSH
    _HAS_DATASKETCH = True
except ImportError:
    _HAS_DATASKETCH = False
    logger.warning("datasketch not installed. MinHash LSH blocking disabled.")


class MinHashLSHBlocker:
    """
    Builds an LSH index on character-level shingles of normalized names.
    Returns candidate pairs with Jaccard similarity above threshold.
    """

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.25,
        ngram: int = 3,
    ) -> None:
        if not _HAS_DATASKETCH:
            raise RuntimeError("datasketch required for MinHash LSH blocking.")
        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram = ngram
        self.lsh: Optional[MinHashLSH] = None
        self._id_to_minhash: dict[str, MinHash] = {}

    def _text_to_minhash(self, text: str) -> MinHash:
        m = MinHash(num_perm=self.num_perm)
        shingles = {
            text[i: i + self.ngram]
            for i in range(max(len(text) - self.ngram + 1, 1))
        }
        for s in shingles:
            m.update(s.encode("utf-8"))
        return m

    def index(
        self,
        ids: list[str],
        texts: list[str],
    ) -> "MinHashLSHBlocker":
        """Build LSH index from a list of (id, text) pairs."""
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        logger.info(f"Building MinHash LSH for {len(ids):,} items...")
        for eid, text in zip(ids, texts):
            if not text:
                continue
            mh = self._text_to_minhash(text)
            self._id_to_minhash[eid] = mh
            try:
                self.lsh.insert(eid, mh)
            except ValueError:
                pass  # duplicate key
        logger.info("MinHash LSH index built.")
        return self

    def query(
        self,
        query_id: str,
        query_text: str,
    ) -> list[str]:
        """Return candidate IDs similar to query_text."""
        if self.lsh is None:
            return []
        mh = self._text_to_minhash(query_text)
        candidates = self.lsh.query(mh)
        return [c for c in candidates if c != query_id]

    def query_batch(
        self,
        query_ids: list[str],
        query_texts: list[str],
    ) -> dict[str, list[str]]:
        """Batch query; returns dict of query_id → candidate_ids."""
        results: dict[str, list[str]] = {}
        for qid, qt in zip(query_ids, query_texts):
            results[qid] = self.query(qid, qt)
        return results
