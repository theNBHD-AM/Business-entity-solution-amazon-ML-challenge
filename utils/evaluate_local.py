"""
Quick standalone F_0.5 evaluator against a labeled val set.

Usage:
    python utils/evaluate_local.py \
        --results outputs/matching_results.tsv \
        --labels data/raw/train_labels.tsv \
        [--source1 data/raw/source1_train.tsv]
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import polars as pl
from loguru import logger


def f_beta(tp: int, fp: int, fn: int, beta: float = 0.5) -> float:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    d = (1 + beta**2) * prec + beta**2 * rec
    return (1 + beta**2) * prec * rec / d if d > 0 else 0.0


def per_entity_f05(true_set: set, pred_set: set) -> float:
    if not true_set and not pred_set:
        return 1.0
    if not true_set:
        return 0.0
    if not pred_set:
        return 0.0
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    return f_beta(tp, fp, fn, beta=0.5)


@click.command()
@click.option("--results", "-r", required=True)
@click.option("--labels", "-l", required=True)
@click.option("--source1", "-s", default="", help="Optional: to validate entity coverage")
@click.option("--country-col", default="country", help="Country column name in source1")
def main(results, labels, source1, country_col):
    """Compute macro-averaged F_0.5 against ground-truth labels."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")

    # Load labels
    label_df = pl.read_csv(labels, separator="\t", has_header=True, infer_schema_length=5000)
    ground_truth: dict[str, set] = {}
    for row in label_df.iter_rows(named=True):
        s1 = str(
            row.get("source1_entity_id")
            or row.get("source1_id")
            or row.get("entity_id")
            or ""
        )
        raw = str(
            row.get("matched_entity_ids")
            or row.get("matched_ids")
            or row.get("match_ids")
            or ""
        ).strip()
        if raw and raw not in ("", "nan", "None"):
            delims = "," if "," in raw else "|"
            ground_truth[s1] = {m.strip() for m in raw.split(delims) if m.strip()}
        else:
            ground_truth[s1] = set()

    # Load predictions
    pred_df = pl.read_csv(results, separator="\t", has_header=True,
                          dtype={"source1_id": pl.Utf8, "matched_ids": pl.Utf8})
    predictions: dict[str, set] = {}
    for row in pred_df.iter_rows(named=True):
        s1 = str(row.get("source1_id") or "")
        raw = str(row.get("matched_ids") or "").strip()
        predictions[s1] = {m.strip() for m in raw.split("|") if m.strip()} if raw not in ("", "nan", "None") else set()

    # Country map (optional)
    country_map: dict[str, str] = {}
    if source1 and Path(source1).exists():
        s1_df = pl.read_csv(source1, separator="\t", has_header=True,
                            dtype={"entity_id": pl.Utf8})
        if country_col in s1_df.columns:
            for row in s1_df.iter_rows(named=True):
                country_map[str(row["entity_id"])] = str(row.get(country_col) or "unknown")

    # Compute per-entity scores
    scores: list[float] = []
    country_scores: dict[str, list[float]] = {}
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        score = per_entity_f05(true_set, pred_set)
        scores.append(score)
        country = country_map.get(s1_id, "unknown")
        country_scores.setdefault(country, []).append(score)

    import numpy as np
    macro_f05 = float(np.mean(scores)) if scores else 0.0
    n_singletons = sum(1 for v in ground_truth.values() if not v)
    n_sing_correct = sum(
        1 for s1, gt in ground_truth.items()
        if not gt and not predictions.get(s1, set())
    )

    logger.info(f"\n{'='*55}")
    logger.info(f"  Macro F_0.5        : {macro_f05:.4f}")
    logger.info(f"  Entities evaluated : {len(scores):,}")
    logger.info(f"  Singletons         : {n_singletons:,} ({n_sing_correct} correct)")
    if country_scores:
        logger.info("  Per-country F_0.5:")
        for country in sorted(country_scores):
            cs = float(np.mean(country_scores[country]))
            logger.info(f"    [{country}] {cs:.4f}  (n={len(country_scores[country]):,})")
    logger.info(f"{'='*55}")


if __name__ == "__main__":
    main()
