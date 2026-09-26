import lightgbm as lgb
import numpy as np
import polars as pl

from business_entity_resolution.features import FEATURE_COLUMNS


def build_labels(feature_df: pl.DataFrame, truth: dict[str, set[str]]) -> np.ndarray:
    s1_ids = feature_df["source1_entity_id"].to_list()
    cand_ids = feature_df["candidate_entity_id"].to_list()
    y = np.zeros(len(s1_ids), dtype=np.int8)
    for i in range(len(s1_ids)):
        if cand_ids[i] in truth.get(s1_ids[i], ()):
            y[i] = 1
    return y


def train_model(feature_df: pl.DataFrame, y: np.ndarray) -> lgb.LGBMClassifier:
    X = feature_df.select(FEATURE_COLUMNS).to_numpy()
    model = lgb.LGBMClassifier(
        n_estimators=400,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        is_unbalance=True,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X, y)
    return model


def predict_proba(model: lgb.LGBMClassifier, feature_df: pl.DataFrame) -> np.ndarray:
    X = feature_df.select(FEATURE_COLUMNS).to_numpy()
    return model.predict_proba(X)[:, 1]
