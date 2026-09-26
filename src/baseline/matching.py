"""Matching model: logistic regression over the similarity features, with
its decision threshold tuned on the held-out validation split to maximize
macro F_0.5 (precision-weighted 2x over recall - so the search is biased
toward higher thresholds when the curve is flat, since false merges hurt
more than missed matches and singletons need a correct empty prediction).
"""
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from .features import FEATURE_COLUMNS
from .scoring import macro_f_beta

DEFAULT_THRESHOLD_GRID = np.round(np.arange(0.30, 0.96, 0.025), 4)


def label_candidates(features_df: pl.DataFrame, true_map: dict) -> pl.DataFrame:
    sids = features_df["source1_entity_id"].to_list()
    cids = features_df["candidate_entity_id"].to_list()
    labels = [1 if cid in true_map.get(sid, ()) else 0 for sid, cid in zip(sids, cids)]
    return features_df.with_columns(pl.Series("label", labels))


def train_model(labeled_features: pl.DataFrame) -> LogisticRegression:
    X = labeled_features.select(FEATURE_COLUMNS).to_numpy()
    y = labeled_features["label"].to_numpy()
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, y)
    return model


def score_candidates(model: LogisticRegression, features_df: pl.DataFrame) -> pl.DataFrame:
    X = features_df.select(FEATURE_COLUMNS).to_numpy()
    scores = model.predict_proba(X)[:, 1]
    return features_df.with_columns(pl.Series("score", scores))


def matches_at_threshold(scored_df: pl.DataFrame, threshold: float) -> dict:
    """{source1_entity_id: set(candidate_entity_id)} for score >= threshold."""
    keep = scored_df.filter(pl.col("score") >= threshold)
    out = {}
    for sid, cid in zip(keep["source1_entity_id"].to_list(), keep["candidate_entity_id"].to_list()):
        out.setdefault(sid, set()).add(cid)
    return out


def tune_threshold(scored_val_df: pl.DataFrame, true_map: dict, val_ids,
                    thresholds=DEFAULT_THRESHOLD_GRID):
    """Grid-search the threshold maximizing macro F_0.5 on the held-out split."""
    best_threshold, best_f = None, -1.0
    curve = []
    for t in thresholds:
        pred_map = matches_at_threshold(scored_val_df, t)
        f = macro_f_beta(true_map, pred_map, val_ids, beta=0.5)
        curve.append((float(t), f))
        if f > best_f:
            best_f, best_threshold = f, float(t)
    return best_threshold, best_f, curve
