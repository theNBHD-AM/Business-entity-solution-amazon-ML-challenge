"""
Candidate Generator — merges signals from ANN, MinHash LSH, Phonetic, Geo
blockers and produces the final candidate pair list.

Output schema:
  source1_id, candidate_id, source, ann_score, final_rank

This module is the critical recall ceiling. All downstream precision
optimizations are bounded by what is present in this candidate set.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from loguru import logger
from tqdm import tqdm

from .ann_index import ANNIndex
from .minhash_lsh import MinHashLSHBlocker
from .phonetic_blocker import PhoneticBlocker
from .geo_blocker import GeoBlocker


class CandidateGenerator:
    """
    Merges all blocking signals and produces a candidate pair Polars DataFrame.

    Priority (recall vs. speed):
      1. ANN (dense semantic similarity) — broadest coverage
      2. MinHash LSH (character-level typo similarity)
      3. Phonetic (first-name phonetic key)
      4. Geo (country+city bucket)

    Final candidate set = UNION of all signals, capped at `final_top_k`
    per Source 1 entity after re-ranking by ANN score.
    """

    def __init__(
        self,
        ann_index: Optional[ANNIndex] = None,
        lsh_blocker: Optional[MinHashLSHBlocker] = None,
        phonetic_blocker: Optional[PhoneticBlocker] = None,
        geo_blocker: Optional[GeoBlocker] = None,
        ann_top_k: int = 50,
        final_top_k: int = 30,
    ) -> None:
        self.ann = ann_index
        self.lsh = lsh_blocker
        self.phonetic = phonetic_blocker
        self.geo = geo_blocker
        self.ann_top_k = ann_top_k
        self.final_top_k = final_top_k

    def generate(
        self,
        s1_df: pl.DataFrame,
        s23_df: pl.DataFrame,
        s1_embeddings: Optional[np.ndarray],
        s23_embeddings: Optional[np.ndarray],
        phonetic_index: Optional[dict[str, list[str]]] = None,
        geo_index: Optional[dict[str, list[str]]] = None,
    ) -> pl.DataFrame:
        """
        Main candidate generation entry point.

        Args:
            s1_df: Source 1 normalized DataFrame (includes parsed_city etc.)
            s23_df: Source 2+3 combined normalized DataFrame
            s1_embeddings: embeddings for s1_df rows (may be None)
            s23_embeddings: embeddings for s23_df rows (may be None)
            phonetic_index: pre-built {phonetic_key: [s23_ids]}
            geo_index: pre-built {geo_bucket: [s23_ids]}

        Returns:
            Polars DataFrame with columns:
            [source1_id, candidate_id, ann_score, sources, rank]
        """
        s23_ids = s23_df["entity_id"].to_list()
        s23_id_to_idx = {eid: i for i, eid in enumerate(s23_ids)}

        # ANN search: S1 queries against S23 index
        ann_results: dict[str, dict[str, float]] = defaultdict(dict)
        if self.ann is not None and s1_embeddings is not None:
            logger.info("Running ANN search...")
            distances, indices = self.ann.search(s1_embeddings, top_k=self.ann_top_k)
            for i, s1_row in enumerate(s1_df.iter_rows(named=True)):
                s1_id = s1_row["entity_id"]
                for dist, idx in zip(distances[i], indices[i]):
                    if idx < 0 or idx >= len(s23_ids):
                        continue
                    cand_id = s23_ids[idx]
                    ann_results[s1_id][cand_id] = float(dist)

        # LSH candidates
        lsh_results: dict[str, set[str]] = defaultdict(set)
        if self.lsh is not None:
            logger.info("Running MinHash LSH blocking...")
            s1_names = s1_df["norm_name"].to_list()
            s1_ids_list = s1_df["entity_id"].to_list()
            batch_results = self.lsh.query_batch(s1_ids_list, s1_names)
            for s1_id, cands in batch_results.items():
                for c in cands:
                    if c in s23_id_to_idx:
                        lsh_results[s1_id].add(c)

        # Phonetic candidates
        phonetic_results: dict[str, set[str]] = defaultdict(set)
        if self.phonetic is not None and phonetic_index is not None:
            logger.info("Running phonetic blocking...")
            for row in s1_df.iter_rows(named=True):
                s1_id = row["entity_id"]
                cands = self.phonetic.get_candidates(
                    row.get("norm_name", ""),
                    phonetic_index,
                    exclude_id=s1_id,
                )
                for c in cands:
                    if c in s23_id_to_idx:
                        phonetic_results[s1_id].add(c)

        # Geo candidates
        geo_results: dict[str, set[str]] = defaultdict(set)
        if self.geo is not None and geo_index is not None:
            logger.info("Running geo blocking...")
            for row in s1_df.iter_rows(named=True):
                s1_id = row["entity_id"]
                cands = self.geo.get_candidates(
                    row.get("country", ""),
                    row.get("parsed_city", ""),
                    row.get("parsed_postal", ""),
                    geo_index,
                    exclude_id=s1_id,
                )
                for c in cands:
                    if c in s23_id_to_idx:
                        geo_results[s1_id].add(c)

        # Merge all signals
        logger.info("Merging blocking signals...")
        all_s1_ids = s1_df["entity_id"].to_list()
        records = []
        for s1_id in tqdm(all_s1_ids, desc="Merging candidates"):
            merged: dict[str, dict] = {}

            # Add ANN with scores
            for cand_id, score in ann_results.get(s1_id, {}).items():
                merged[cand_id] = {"ann_score": score, "sources": set(["ann"])}

            # Add LSH
            for cand_id in lsh_results.get(s1_id, set()):
                if cand_id in merged:
                    merged[cand_id]["sources"].add("lsh")
                else:
                    merged[cand_id] = {"ann_score": 0.0, "sources": set(["lsh"])}

            # Add phonetic
            for cand_id in phonetic_results.get(s1_id, set()):
                if cand_id in merged:
                    merged[cand_id]["sources"].add("phonetic")
                else:
                    merged[cand_id] = {"ann_score": 0.0, "sources": set(["phonetic"])}

            # Add geo
            for cand_id in geo_results.get(s1_id, set()):
                if cand_id in merged:
                    merged[cand_id]["sources"].add("geo")
                else:
                    merged[cand_id] = {"ann_score": 0.0, "sources": set(["geo"])}

            # Sort by source count (multi-signal = stronger signal) then ANN score
            ranked = sorted(
                merged.items(),
                key=lambda x: (len(x[1]["sources"]), x[1]["ann_score"]),
                reverse=True,
            )[: self.final_top_k]

            for rank, (cand_id, info) in enumerate(ranked):
                records.append({
                    "source1_id": s1_id,
                    "candidate_id": cand_id,
                    "ann_score": info["ann_score"],
                    "sources": "|".join(sorted(info["sources"])),
                    "rank": rank,
                })

        if not records:
            return pl.DataFrame(schema={
                "source1_id": pl.Utf8, "candidate_id": pl.Utf8,
                "ann_score": pl.Float32, "sources": pl.Utf8, "rank": pl.Int32,
            })

        df_candidates = pl.DataFrame(records)
        logger.info(
            f"Candidate generation complete: {len(df_candidates):,} pairs "
            f"for {df_candidates['source1_id'].n_unique():,} S1 entities."
        )
        return df_candidates

    def save_candidates(
        self,
        candidates: pl.DataFrame,
        output_path: str | Path,
        as_tsv: bool = True,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if as_tsv:
            candidates.write_csv(str(path), separator="\t")
        else:
            candidates.write_parquet(str(path))
        logger.info(f"Candidates saved to {path} ({len(candidates):,} rows)")
        return path
