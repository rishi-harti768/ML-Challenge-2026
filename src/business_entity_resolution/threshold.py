import numpy as np
import polars as pl


def sweep_threshold(
    feature_df: pl.DataFrame,
    proba: np.ndarray,
    label: np.ndarray,
    truth: dict[str, set[str]],
    eval_ids: set[str],
    lo: float = 0.05,
    hi: float = 0.95,
    step: float = 0.01,
) -> tuple[float, float]:
    base = feature_df.select("source1_entity_id").with_columns(
        pl.Series("proba", proba),
        pl.Series("label", label),
    )
    true_counts = {sid: len(truth.get(sid, ())) for sid in eval_ids}
    n_eval = len(eval_ids)

    best_t, best_score = 0.5, -1.0
    for t in np.arange(lo, hi + step / 2, step):
        agg = (
            base.filter(pl.col("proba") >= t)
            .group_by("source1_entity_id")
            .agg(pl.len().alias("pred_count"), pl.col("label").sum().alias("tp_count"))
        )
        agg_map = {
            row["source1_entity_id"]: (row["pred_count"], row["tp_count"])
            for row in agg.iter_rows(named=True)
        }
        total = 0.0
        for sid in eval_ids:
            true_n = true_counts[sid]
            pred_n, tp_n = agg_map.get(sid, (0, 0))
            if true_n == 0 and pred_n == 0:
                total += 1.0
            elif true_n == 0 or pred_n == 0 or tp_n == 0:
                total += 0.0
            else:
                precision = tp_n / pred_n
                recall = tp_n / true_n
                total += (1.25 * precision * recall) / (0.25 * precision + recall)
        score = total / n_eval
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t, best_score
