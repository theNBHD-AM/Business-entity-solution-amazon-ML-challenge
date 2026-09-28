"""
Submission Validator
=====================
Replicates the challenge's output validation checks.

Validates:
  1. matching_results.tsv schema and completeness
  2. candidate_pairs.tsv schema
  3. No Source 1 entity is missing from output
  4. All matched IDs belong to Source 2 or 3 in the test set
  5. No duplicate Source 1 entries
  6. No S1 IDs leaked into matched_ids column
  7. Pipe-separated matched_ids are individually valid

Usage:
    python utils/validate_submission.py \
        --results outputs/matching_results.tsv \
        --candidates outputs/candidate_pairs.tsv \
        --source1 data/raw/source1_test.tsv \
        --source2 data/raw/source2_test.tsv \
        --source3 data/raw/source3_test.tsv

Exit code 0 = valid, 1 = invalid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import polars as pl
from loguru import logger


def _load_ids(tsv_path: str) -> set[str]:
    df = pl.read_csv(tsv_path, separator="\t", has_header=True, dtype={"entity_id": pl.Utf8})
    return set(df["entity_id"].drop_nulls().to_list())


def validate_submission(
    results_path: str,
    candidates_path: str,
    source1_path: str,
    source2_path: str,
    source3_path: str,
) -> bool:
    errors: list[str] = []
    warnings: list[str] = []

    # ── Load expected IDs ──────────────────────────────────────────────
    logger.info("Loading source ID sets...")
    s1_ids = _load_ids(source1_path)
    s2_ids = _load_ids(source2_path)
    s3_ids = _load_ids(source3_path)
    valid_match_ids = s2_ids | s3_ids
    logger.info(
        f"Source counts: S1={len(s1_ids):,}, S2={len(s2_ids):,}, S3={len(s3_ids):,}"
    )

    # ── Validate matching_results.tsv ──────────────────────────────────
    logger.info(f"Validating: {results_path}")
    try:
        results_df = pl.read_csv(
            results_path, separator="\t", has_header=True,
            dtype={"source1_id": pl.Utf8, "matched_ids": pl.Utf8},
        )
    except Exception as e:
        errors.append(f"Cannot load matching_results.tsv: {e}")
        _report(errors, warnings)
        return False

    # Check required columns
    required_cols = {"source1_id", "matched_ids"}
    missing_cols = required_cols - set(results_df.columns)
    if missing_cols:
        errors.append(f"matching_results.tsv missing columns: {missing_cols}")

    if errors:
        _report(errors, warnings)
        return False

    result_s1_ids = set(results_df["source1_id"].drop_nulls().to_list())

    # Every S1 entity must appear exactly once
    missing_from_output = s1_ids - result_s1_ids
    extra_in_output = result_s1_ids - s1_ids
    if missing_from_output:
        errors.append(
            f"MISSING {len(missing_from_output):,} S1 entities from output. "
            f"Sample: {list(missing_from_output)[:5]}"
        )
    if extra_in_output:
        errors.append(
            f"EXTRA {len(extra_in_output):,} IDs in output not in source1_test. "
            f"Sample: {list(extra_in_output)[:5]}"
        )

    # No duplicate source1_id rows
    n_rows = len(results_df)
    n_unique = results_df["source1_id"].n_unique()
    if n_rows != n_unique:
        errors.append(
            f"DUPLICATE source1_id rows: {n_rows} rows but only {n_unique} unique IDs."
        )

    # Validate matched_ids
    invalid_match_ids: list[str] = []
    s1_in_matches: list[str] = []
    for row in results_df.iter_rows(named=True):
        raw = str(row.get("matched_ids") or "").strip()
        if not raw or raw in ("", "nan", "None"):
            continue  # singleton — OK
        matched = [m.strip() for m in raw.split("|") if m.strip()]
        for mid in matched:
            if mid not in valid_match_ids:
                invalid_match_ids.append(mid)
            if mid in s1_ids:
                s1_in_matches.append(mid)

    if invalid_match_ids:
        errors.append(
            f"INVALID matched_ids (not in S2/S3): {len(invalid_match_ids):,}. "
            f"Sample: {invalid_match_ids[:5]}"
        )
    if s1_in_matches:
        errors.append(
            f"S1 IDs found in matched_ids (forbidden): {s1_in_matches[:5]}"
        )

    # ── Validate candidate_pairs.tsv ───────────────────────────────────
    if candidates_path and Path(candidates_path).exists():
        logger.info(f"Validating: {candidates_path}")
        try:
            cand_df = pl.read_csv(
                candidates_path, separator="\t", has_header=True,
            )
            cand_required = {"source1_id", "candidate_id"}
            cand_missing = cand_required - set(cand_df.columns)
            if cand_missing:
                warnings.append(
                    f"candidate_pairs.tsv missing columns: {cand_missing}"
                )
            else:
                # Check that all S1 IDs in candidates are valid
                cand_s1 = set(cand_df["source1_id"].drop_nulls().to_list())
                invalid_cand_s1 = cand_s1 - s1_ids
                if invalid_cand_s1:
                    warnings.append(
                        f"candidate_pairs.tsv has {len(invalid_cand_s1)} unknown source1_ids."
                    )
                # Check candidate IDs are from S2/S3
                cand_ids_set = set(cand_df["candidate_id"].drop_nulls().to_list())
                invalid_cands = cand_ids_set - valid_match_ids
                if invalid_cands:
                    warnings.append(
                        f"candidate_pairs.tsv has {len(invalid_cands)} unknown candidate_ids."
                    )
                logger.info(
                    f"candidate_pairs.tsv: {len(cand_df):,} pairs, "
                    f"{cand_df['source1_id'].n_unique():,} unique S1 entities."
                )
        except Exception as e:
            warnings.append(f"candidate_pairs.tsv load error: {e}")
    else:
        warnings.append("candidate_pairs.tsv not found — skipping.")

    # ── Summary ────────────────────────────────────────────────────────
    _report(errors, warnings)
    if errors:
        return False

    # Stats summary
    n_matched_entities = 0
    n_total_matches = 0
    for row in results_df.iter_rows(named=True):
        raw = str(row.get("matched_ids") or "").strip()
        if raw and raw not in ("", "nan", "None"):
            ids = [m.strip() for m in raw.split("|") if m.strip()]
            if ids:
                n_matched_entities += 1
                n_total_matches += len(ids)

    n_singletons = n_unique - n_matched_entities
    logger.info(
        f"\n{'='*55}\n"
        f"  ✅ SUBMISSION VALID\n"
        f"  Total S1 entities : {n_unique:,}\n"
        f"  Entities w/ match : {n_matched_entities:,}\n"
        f"  Singleton entities: {n_singletons:,}\n"
        f"  Total match pairs : {n_total_matches:,}\n"
        f"{'='*55}"
    )
    return True


def _report(errors: list[str], warnings: list[str]) -> None:
    for w in warnings:
        logger.warning(f"⚠️  {w}")
    for e in errors:
        logger.error(f"❌  {e}")
    if not errors:
        logger.success("No validation errors found.")


@click.command()
@click.option("--results", "-r", required=True, help="Path to matching_results.tsv")
@click.option("--candidates", "-p", default="", help="Path to candidate_pairs.tsv")
@click.option("--source1", "-s1", required=True, help="Path to source1_test.tsv")
@click.option("--source2", "-s2", required=True, help="Path to source2_test.tsv")
@click.option("--source3", "-s3", required=True, help="Path to source3_test.tsv")
def cli(results, candidates, source1, source2, source3):
    """Validate challenge submission files before final upload."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    ok = validate_submission(results, candidates, source1, source2, source3)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    cli()
