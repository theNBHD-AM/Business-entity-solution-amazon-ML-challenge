"""
Local F_0.5 Evaluation Script.

Replicates the EXACT challenge metric:
  - Per Source 1 entity: compute F_0.5 between predicted matches and ground truth.
  - Correct empty prediction for a singleton entity = 1.0 (not 0.0).
  - Any false match on a true singleton = 0.0 for that entity.
  - Macro-average across ALL Source 1 entities (including singletons).

This is the canonical eval script for every iteration.
Do NOT rely solely on leaderboard submissions.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger


def f_beta_score(
    y_true_set: set[str],
    y_pred_set: set[str],
    beta: float = 0.5,
) -> float:
    """
    Compute F_beta for a single entity.

    Args:
        y_true_set: ground-truth matched IDs (empty = singleton).
        y_pred_set: predicted matched IDs.
        beta: F-beta parameter (0.5 = precision-weighted).

    Returns:
        F_beta score in [0, 1].
    """
    # Singleton handling: empty prediction on true singleton = 1.0
    if not y_true_set and not y_pred_set:
        return 1.0
    # False match on singleton = 0.0
    if not y_true_set and y_pred_set:
        return 0.0
    # No prediction on a true match entity
    if y_true_set and not y_pred_set:
        return 0.0

    tp = len(y_true_set & y_pred_set)
    fp = len(y_pred_set - y_true_set)
    fn = len(y_true_set - y_pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    denom = (1 + beta**2) * precision + beta**2 * recall
    if denom == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / denom


class EntityResolutionMetrics:
    """
    Computes macro-averaged F_0.5 across all Source 1 entities.
    Includes per-country breakdown and singleton analysis.
    """

    def __init__(self, beta: float = 0.5) -> None:
        self.beta = beta

    def evaluate(
        self,
        predictions: dict[str, set[str]],
        ground_truth: dict[str, list[str]],
        country_map: Optional[dict[str, str]] = None,
        verbose: bool = True,
    ) -> dict:
        """
        Compute macro-averaged F_beta.

        Args:
            predictions: {source1_id: set_of_predicted_match_ids}
            ground_truth: {source1_id: list_of_true_match_ids}
            country_map: optional {source1_id: country} for per-country breakdown
            verbose: whether to log results

        Returns:
            dict with keys: macro_f05, per_entity_scores, n_singletons,
                           n_singleton_correct, country_breakdown
        """
        per_entity: dict[str, float] = {}
        country_scores: dict[str, list[float]] = {}

        for s1_id, true_list in ground_truth.items():
            true_set = set(true_list) if true_list else set()
            pred_set = predictions.get(s1_id, set())

            score = f_beta_score(true_set, pred_set, beta=self.beta)
            per_entity[s1_id] = score

            # Country breakdown
            if country_map:
                country = country_map.get(s1_id, "unknown")
                if country not in country_scores:
                    country_scores[country] = []
                country_scores[country].append(score)

        macro_f05 = float(np.mean(list(per_entity.values()))) if per_entity else 0.0

        # Singleton analysis
        singletons = {k: v for k, v in ground_truth.items() if not v}
        singleton_scores = [per_entity.get(k, 0.0) for k in singletons]
        n_singleton_correct = sum(1 for s in singleton_scores if s == 1.0)

        country_breakdown = {
            country: float(np.mean(scores))
            for country, scores in country_scores.items()
        }

        result = {
            "macro_f05": macro_f05,
            "n_entities": len(per_entity),
            "n_singletons": len(singletons),
            "n_singleton_correct": n_singleton_correct,
            "singleton_accuracy": n_singleton_correct / max(len(singletons), 1),
            "per_entity_scores": per_entity,
            "country_breakdown": country_breakdown,
        }

        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"Macro F_0.5: {macro_f05:.4f}")
            logger.info(
                f"Singletons: {len(singletons)} total, "
                f"{n_singleton_correct} correct ({result['singleton_accuracy']:.1%})"
            )
            if country_breakdown:
                for country, score in sorted(country_breakdown.items()):
                    logger.info(f"  [{country}] F_0.5 = {score:.4f}")
            logger.info(f"{'='*60}")

        return result

    def compute_blocking_recall(
        self,
        candidates_df,  # polars DataFrame with source1_id, candidate_id
        ground_truth: dict[str, list[str]],
    ) -> dict:
        """
        Measure what fraction of true matches appear in the candidate set.
        This is the recall CEILING for the entire pipeline.
        """
        if hasattr(candidates_df, 'iter_rows'):
            cand_set: dict[str, set[str]] = {}
            for row in candidates_df.iter_rows(named=True):
                s1 = row["source1_id"]
                cand = row["candidate_id"]
                if s1 not in cand_set:
                    cand_set[s1] = set()
                cand_set[s1].add(cand)
        else:
            cand_set = candidates_df

        total_true = 0
        total_in_candidates = 0
        missed: list[tuple] = []

        for s1_id, true_matches in ground_truth.items():
            for tm in true_matches:
                total_true += 1
                if tm in cand_set.get(s1_id, set()):
                    total_in_candidates += 1
                else:
                    missed.append((s1_id, tm))

        blocking_recall = total_in_candidates / max(total_true, 1)
        result = {
            "blocking_recall": blocking_recall,
            "total_true_matches": total_true,
            "found_in_candidates": total_in_candidates,
            "missed_pairs": missed[:20],  # sample of missed pairs
        }
        logger.info(
            f"Blocking Recall: {blocking_recall:.4f} "
            f"({total_in_candidates}/{total_true} true matches in candidates)"
        )
        if missed[:5]:
            logger.info(f"  Sample missed pairs: {missed[:5]}")
        return result
