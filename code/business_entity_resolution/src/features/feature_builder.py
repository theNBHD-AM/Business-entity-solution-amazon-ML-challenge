"""
Per-pair Feature Engineering.

For each (Source1, Candidate) pair, computes:
  Name similarity:   exact, JW, Levenshtein, token-sort/set ratio, TF-IDF cosine,
                     embedding cosine, Jaccard, suffix-stripped exact match.
  Address similarity: token overlap, embedding cosine, field exact-match flags,
                      street-number match, length delta.
  Meta features:     country match, source pair (S2/S3), field-length deltas,
                     missing-field flags, multi-signal blocking source count.

All features are float32 for LightGBM efficiency.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl
from loguru import logger
from tqdm import tqdm

try:
    from rapidfuzz import fuzz as rfuzz
    from rapidfuzz import distance as rfdist
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False
    logger.warning("rapidfuzz not installed. String similarity features limited.")

try:
    import jellyfish
    _HAS_JELLY = True
except ImportError:
    _HAS_JELLY = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


_FEATURE_NAMES: list[str] = [
    # Name features
    "name_exact",
    "name_nosuffix_exact",
    "name_jaro_winkler",
    "name_levenshtein_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_jaccard_token",
    "name_tfidf_cosine",
    "name_emb_cosine",
    "name_len_delta",
    "name_s1_missing",
    "name_cand_missing",
    # Address features
    "addr_token_overlap",
    "addr_emb_cosine",
    "addr_city_exact",
    "addr_state_exact",
    "addr_postal_exact",
    "addr_street_num_exact",
    "addr_len_delta",
    "addr_s1_missing",
    "addr_cand_missing",
    # Meta features
    "country_match",
    "source_is_s2",
    "source_is_s3",
    "suffix_match",
    "blocking_source_count",
    "blocking_has_ann",
    "blocking_has_lsh",
    "blocking_has_phonetic",
    "blocking_has_geo",
    "ann_score",
    "rank",
]


class FeatureBuilder:
    """
    Computes a fixed-length feature vector for each candidate pair.
    """

    def __init__(
        self,
        tfidf_vocab_size: int = 50_000,
        s1_embeddings: Optional[np.ndarray] = None,
        s23_embeddings: Optional[np.ndarray] = None,
        s1_id_to_idx: Optional[dict[str, int]] = None,
        s23_id_to_idx: Optional[dict[str, int]] = None,
    ) -> None:
        self.tfidf_vocab_size = tfidf_vocab_size
        self.s1_embeddings = s1_embeddings
        self.s23_embeddings = s23_embeddings
        self.s1_id_to_idx = s1_id_to_idx or {}
        self.s23_id_to_idx = s23_id_to_idx or {}
        self._tfidf: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._tfidf_ids: list[str] = []

    @property
    def feature_names(self) -> list[str]:
        return list(_FEATURE_NAMES)

    def fit_tfidf(
        self,
        ids: list[str],
        texts: list[str],
    ) -> "FeatureBuilder":
        """Fit TF-IDF on S23 normalized names for cosine similarity."""
        if not _HAS_SKLEARN:
            return self
        logger.info(f"Fitting TF-IDF on {len(texts):,} documents...")
        self._tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=self.tfidf_vocab_size,
            min_df=2,
        )
        self._tfidf_matrix = self._tfidf.fit_transform(texts)
        self._tfidf_ids = ids
        self._tfidf_id_to_row = {eid: i for i, eid in enumerate(ids)}
        logger.info("TF-IDF fitted.")
        return self

    def build_features(
        self,
        candidates_df: pl.DataFrame,
        s1_lookup: dict[str, dict],
        s23_lookup: dict[str, dict],
    ) -> np.ndarray:
        """
        Build feature matrix.

        Args:
            candidates_df: output of CandidateGenerator (source1_id, candidate_id, ...)
            s1_lookup: {entity_id: normalized_row_dict}
            s23_lookup: {entity_id: normalized_row_dict}

        Returns:
            float32 array of shape (N_pairs, N_features)
        """
        n = len(candidates_df)
        X = np.zeros((n, len(_FEATURE_NAMES)), dtype=np.float32)
        feat_idx = {name: i for i, name in enumerate(_FEATURE_NAMES)}

        rows = candidates_df.iter_rows(named=True)
        for i, row in enumerate(tqdm(rows, total=n, desc="Building features")):
            s1_id = row["source1_id"]
            cand_id = row["candidate_id"]

            s1 = s1_lookup.get(s1_id, {})
            cand = s23_lookup.get(cand_id, {})

            s1_name = str(s1.get("norm_name", "") or "")
            cand_name = str(cand.get("norm_name", "") or "")
            s1_addr = str(s1.get("norm_address", "") or "")
            cand_addr = str(cand.get("norm_address", "") or "")
            s1_nosuffix = str(s1.get("norm_name_no_suffix", "") or "")
            cand_nosuffix = str(cand.get("norm_name_no_suffix", "") or "")
            s1_suffix = str(s1.get("canonical_suffix", "") or "")
            cand_suffix = str(cand.get("canonical_suffix", "") or "")
            s1_country = str(s1.get("country", "") or "").lower()
            cand_country = str(cand.get("country", "") or "").lower()
            sources_str = str(row.get("sources", "") or "")

            # ---- Name features ----
            X[i, feat_idx["name_exact"]] = float(s1_name == cand_name and s1_name != "")
            X[i, feat_idx["name_nosuffix_exact"]] = float(
                s1_nosuffix == cand_nosuffix and s1_nosuffix != ""
            )

            if _HAS_RAPIDFUZZ and s1_name and cand_name:
                X[i, feat_idx["name_jaro_winkler"]] = rfuzz.jaro_winkler(s1_name, cand_name) / 100.0
                X[i, feat_idx["name_levenshtein_ratio"]] = rfuzz.ratio(s1_name, cand_name) / 100.0
                X[i, feat_idx["name_token_sort_ratio"]] = rfuzz.token_sort_ratio(s1_name, cand_name) / 100.0
                X[i, feat_idx["name_token_set_ratio"]] = rfuzz.token_set_ratio(s1_name, cand_name) / 100.0
                X[i, feat_idx["name_partial_ratio"]] = rfuzz.partial_ratio(s1_name, cand_name) / 100.0
            elif _HAS_JELLY and s1_name and cand_name:
                X[i, feat_idx["name_jaro_winkler"]] = jellyfish.jaro_winkler_similarity(s1_name, cand_name)

            # Jaccard on token sets
            s1_tokens = set(s1_name.split())
            cand_tokens = set(cand_name.split())
            if s1_tokens or cand_tokens:
                union = len(s1_tokens | cand_tokens)
                if union > 0:
                    X[i, feat_idx["name_jaccard_token"]] = len(s1_tokens & cand_tokens) / union

            # TF-IDF cosine
            if self._tfidf is not None and s1_name:
                try:
                    s1_vec = self._tfidf.transform([s1_name])
                    cand_row_idx = self._tfidf_id_to_row.get(cand_id)
                    if cand_row_idx is not None:
                        cand_vec = self._tfidf_matrix[cand_row_idx]
                        sim = sk_cosine(s1_vec, cand_vec)[0, 0]
                        X[i, feat_idx["name_tfidf_cosine"]] = float(sim)
                except Exception:
                    pass

            # Embedding cosine
            s1_idx = self.s1_id_to_idx.get(s1_id)
            cand_idx = self.s23_id_to_idx.get(cand_id)
            if (s1_idx is not None and cand_idx is not None
                    and self.s1_embeddings is not None
                    and self.s23_embeddings is not None):
                e1 = self.s1_embeddings[s1_idx]
                e2 = self.s23_embeddings[cand_idx]
                X[i, feat_idx["name_emb_cosine"]] = float(np.dot(e1, e2))

            X[i, feat_idx["name_len_delta"]] = abs(len(s1_name) - len(cand_name)) / max(len(s1_name), len(cand_name), 1)
            X[i, feat_idx["name_s1_missing"]] = float(not s1_name)
            X[i, feat_idx["name_cand_missing"]] = float(not cand_name)

            # ---- Address features ----
            s1_addr_tokens = set(s1_addr.split())
            cand_addr_tokens = set(cand_addr.split())
            addr_union = len(s1_addr_tokens | cand_addr_tokens)
            if addr_union > 0:
                X[i, feat_idx["addr_token_overlap"]] = (
                    len(s1_addr_tokens & cand_addr_tokens) / addr_union
                )

            # Address embedding cosine (reuse same embeddings: name+addr are combined)
            if (s1_idx is not None and cand_idx is not None
                    and self.s1_embeddings is not None
                    and self.s23_embeddings is not None):
                X[i, feat_idx["addr_emb_cosine"]] = float(
                    np.dot(self.s1_embeddings[s1_idx], self.s23_embeddings[cand_idx])
                )

            # City/state/postal exact match
            X[i, feat_idx["addr_city_exact"]] = float(
                bool(s1.get("parsed_city")) and
                s1.get("parsed_city", "") == cand.get("parsed_city", "")
            )
            X[i, feat_idx["addr_state_exact"]] = float(
                bool(s1.get("parsed_state")) and
                s1.get("parsed_state", "") == cand.get("parsed_state", "")
            )
            X[i, feat_idx["addr_postal_exact"]] = float(
                bool(s1.get("parsed_postal")) and
                s1.get("parsed_postal", "") == cand.get("parsed_postal", "")
            )
            X[i, feat_idx["addr_street_num_exact"]] = float(
                bool(s1.get("parsed_street_num")) and
                s1.get("parsed_street_num", "") == cand.get("parsed_street_num", "")
            )
            X[i, feat_idx["addr_len_delta"]] = abs(len(s1_addr) - len(cand_addr)) / max(len(s1_addr), len(cand_addr), 1)
            X[i, feat_idx["addr_s1_missing"]] = float(not s1_addr)
            X[i, feat_idx["addr_cand_missing"]] = float(not cand_addr)

            # ---- Meta features ----
            X[i, feat_idx["country_match"]] = float(s1_country == cand_country and s1_country != "")
            X[i, feat_idx["source_is_s2"]] = float(cand_id.startswith("S2-"))
            X[i, feat_idx["source_is_s3"]] = float(cand_id.startswith("S3-"))
            X[i, feat_idx["suffix_match"]] = float(
                s1_suffix == cand_suffix and s1_suffix != ""
            )

            # Blocking signal meta
            srcs = set(sources_str.split("|")) if sources_str else set()
            X[i, feat_idx["blocking_source_count"]] = len(srcs)
            X[i, feat_idx["blocking_has_ann"]] = float("ann" in srcs)
            X[i, feat_idx["blocking_has_lsh"]] = float("lsh" in srcs)
            X[i, feat_idx["blocking_has_phonetic"]] = float("phonetic" in srcs)
            X[i, feat_idx["blocking_has_geo"]] = float("geo" in srcs)
            X[i, feat_idx["ann_score"]] = float(row.get("ann_score", 0.0) or 0.0)
            X[i, feat_idx["rank"]] = float(row.get("rank", 0) or 0)

        return X

    def build_lookup(
        self, df: pl.DataFrame
    ) -> dict[str, dict]:
        """Build {entity_id: row_dict} for fast per-pair lookup."""
        lookup = {}
        for row in df.iter_rows(named=True):
            lookup[row["entity_id"]] = row
        return lookup
