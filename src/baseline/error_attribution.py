"""Stage-level error-attribution report (issue #7 decision).

For every held-out validation entity, classify its outcome into exactly one
of TP / FN-blocking / FN-matching / FP-matching / TN, using this priority
(an entity can technically exhibit more than one symptom at once - e.g.
missing one true match while also keeping a spurious one - so a fixed
priority order is used to pick a single label, always leading with the
worst / most actionable problem):

  1. Singleton (no true matches):
       TN            - predicted empty (correct)
       FP-matching   - predicted at least one match (incorrect)
  2. Has true matches:
       FN-blocking   - at least one true match never reached candidate_pairs
                       (unrecoverable by the matching stage - worst failure)
       TP            - predicted set exactly equals the true set
       FN-matching   - every true match was a candidate, but at least one
                       got dropped by the matching stage
       FP-matching   - all true matches were kept, but so was a spurious one
"""
import pandas as pd

OUTCOME_ORDER = ["TP", "FN-blocking", "FN-matching", "FP-matching", "TN"]


def classify_entity(true_ids: set, cand_ids: set, pred_ids: set) -> str:
    if not true_ids:
        return "TN" if not pred_ids else "FP-matching"
    if true_ids - cand_ids:
        return "FN-blocking"
    if pred_ids == true_ids:
        return "TP"
    if true_ids - pred_ids:
        return "FN-matching"
    if pred_ids - true_ids:
        return "FP-matching"
    return "TP"


def build_report(val_entities: pd.DataFrame, true_map: dict, cand_map: dict,
                  pred_map: dict) -> pd.DataFrame:
    """One row per held-out entity: entity_id, country, outcome."""
    rows = []
    for eid, country in zip(val_entities["entity_id"], val_entities["country"]):
        outcome = classify_entity(
            true_map.get(eid, set()), cand_map.get(eid, set()), pred_map.get(eid, set())
        )
        rows.append((eid, country, outcome))
    return pd.DataFrame(rows, columns=["source1_entity_id", "country", "outcome"])


def rollup_by_country(report_df: pd.DataFrame) -> pd.DataFrame:
    counts = report_df.groupby(["country", "outcome"]).size().unstack(fill_value=0)
    for col in OUTCOME_ORDER:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[OUTCOME_ORDER]
    counts["total"] = counts.sum(axis=1)
    rates = counts[OUTCOME_ORDER].div(counts["total"], axis=0).add_suffix("_rate")
    return pd.concat([counts, rates], axis=1).reset_index()


def rollup_overall(report_df: pd.DataFrame) -> pd.Series:
    counts = report_df["outcome"].value_counts()
    counts = counts.reindex(OUTCOME_ORDER, fill_value=0)
    total = counts.sum()
    rates = (counts / total).add_suffix("_rate") if total else counts
    return pd.concat([counts, pd.Series({"total": total}), rates])
