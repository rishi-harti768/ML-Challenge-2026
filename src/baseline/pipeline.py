"""CLI entry points tying the baseline stages together.

    python -m src.baseline.pipeline validate   # held-out F_0.5 + error attribution
    python -m src.baseline.pipeline test        # full test-set submission files

Both commands process Source-1 in batches against one candidate source
file (Source-2, then Source-3) at a time, spilling each batch's candidate
pairs + features to a parquet shard under a scratch directory, rather than
holding the full ~26M-row dataset (with per-row token-list columns) in
memory at once - this machine has only a few GB of free RAM, well under
what an eager pandas/polars join across the full dataset would need.
"""
import argparse
import gc
import glob
import json
import os
import shutil
import time

import joblib
import pandas as pd
import polars as pl

from . import blocking, data, error_attribution, features, matching, split

DEFAULT_BATCH_SIZE = 100_000
DEFAULT_MODEL_PATH = "output/model.joblib"


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _build_pairs_and_features(s1_df: pl.DataFrame, source_paths, tmp_dir: str,
                               max_df: int, max_candidates_per_entity: int,
                               batch_size: int = DEFAULT_BATCH_SIZE):
    """Block + featurize s1_df against each candidate source, batched and spilled to disk.

    Returns (candidate_pairs, feats) as polars DataFrames, assembled by
    lazily scanning the written parquet shards (bounded-memory streaming
    aggregation) rather than an eager concat of everything at once.
    """
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    n_total = s1_df.height
    shard_idx = 0
    for source_path in source_paths:
        _log(f"  loading + normalizing candidate source: {source_path}")
        shard_dir = os.path.join(tmp_dir, f"norm_{os.path.basename(source_path)}")
        cand_shard_files = data.normalize_source_to_shards(source_path, shard_dir)
        cand_lazy = data.scan_normalized(cand_shard_files)
        _log(f"  candidate source normalized ({len(cand_shard_files)} shards) - building blocking index")
        index = blocking.build_candidate_index(cand_lazy, max_df=max_df)
        _log(f"  blocking index built - name={index['name'].height:,} prefix={index['prefix'].height:,} "
             f"addr={index['addr'].height:,} postal={index['postal'].height:,}")

        for start in range(0, n_total, batch_size):
            batch = s1_df.slice(start, batch_size)
            verbose = start == 0
            if verbose:
                _log(f"  batch {start:,}-{min(start + batch_size, n_total):,} of {n_total:,} "
                     f"(source {os.path.basename(source_path)})")
            cp = blocking.query_candidates(
                batch, index, max_candidates_per_entity=max_candidates_per_entity, verbose=verbose
            )
            if cp.height == 0:
                continue
            fe = features.compute_features(cp, batch, cand_lazy)
            cp.write_parquet(os.path.join(tmp_dir, f"cp_{shard_idx:05d}.parquet"))
            fe.write_parquet(os.path.join(tmp_dir, f"fe_{shard_idx:05d}.parquet"))
            shard_idx += 1
            del cp, fe

        del index
        gc.collect()
        shutil.rmtree(shard_dir, ignore_errors=True)
        _log(f"  finished candidate source: {source_path}")

    cp_files = sorted(glob.glob(os.path.join(tmp_dir, "cp_*.parquet")))
    fe_files = sorted(glob.glob(os.path.join(tmp_dir, "fe_*.parquet")))
    if not cp_files:
        empty_cp = pl.DataFrame({"source1_entity_id": [], "candidate_entity_id": []},
                                 schema={"source1_entity_id": pl.Utf8, "candidate_entity_id": pl.Utf8})
        empty_fe = pl.DataFrame(
            {"source1_entity_id": [], "candidate_entity_id": [], **{c: [] for c in features.FEATURE_COLUMNS}}
        )
        return empty_cp, empty_fe

    candidate_pairs = pl.scan_parquet(cp_files).collect(engine="streaming")
    feats = pl.scan_parquet(fe_files).collect(engine="streaming")
    return candidate_pairs, feats


def run_validation(args):
    _log("loading train_source1 + ground truth")
    s1_df = data.normalize_source_to_parquet(args.source1, args.tmp_dir + "_s1norm")
    true_map = data.load_ground_truth(args.ground_truth)

    _log("building held-out split (issue #2 decision)")
    train_ids_df, val_ids_df = split.load_or_make_split(
        args.source1, args.split_cache_dir, seed=args.seed, holdout_frac=args.holdout_frac
    )
    train_ids = list(train_ids_df["entity_id"])
    val_ids = list(val_ids_df["entity_id"])
    _log(f"train entities: {len(train_ids):,}  val entities: {len(val_ids):,}")

    _log("blocking + features: batched over source2 then source3")
    candidate_pairs, feats = _build_pairs_and_features(
        s1_df, [args.source2, args.source3], args.tmp_dir,
        max_df=args.max_df, max_candidates_per_entity=args.max_candidates_per_entity,
        batch_size=args.batch_size,
    )
    cand_map = data.candidate_pairs_to_map(candidate_pairs)
    del candidate_pairs
    gc.collect()

    train_id_set, val_id_set = set(train_ids), set(val_ids)
    train_feats = feats.filter(pl.col("source1_entity_id").is_in(train_ids))
    val_feats = feats.filter(pl.col("source1_entity_id").is_in(val_ids))
    del feats
    _log(f"train candidate pairs: {len(train_feats):,}  val candidate pairs: {len(val_feats):,}")

    _log("training matching model (logistic regression) on the train split")
    labeled_train = matching.label_candidates(train_feats, true_map)
    _log(f"train split positive rate: {labeled_train['label'].mean():.4f}")
    model = matching.train_model(labeled_train)

    _log("scoring + tuning threshold on the held-out val split")
    scored_val = matching.score_candidates(model, val_feats)
    best_threshold, best_f, curve = matching.tune_threshold(scored_val, true_map, val_id_set)
    _log(f"best threshold: {best_threshold}  held-out macro F_0.5: {best_f:.4f}")

    # Persist the model + threshold so `test` can reuse them directly instead
    # of re-blocking the training data from scratch to retrain (that would
    # roughly double full-scale wall-clock time for no real accuracy gain -
    # this is a 5-feature logistic regression, the extra 15% of training
    # rows held out here does not meaningfully change it).
    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    joblib.dump({"model": model, "threshold": best_threshold}, args.model_path)
    _log(f"saved model + threshold to {args.model_path}")

    pred_map = matching.matches_at_threshold(scored_val, best_threshold)

    _log("computing stage-level error attribution (issue #7 decision)")
    report_df = error_attribution.build_report(val_ids_df, true_map, cand_map, pred_map)
    by_country = error_attribution.rollup_by_country(report_df)
    overall = error_attribution.rollup_overall(report_df)

    os.makedirs(os.path.dirname(args.error_report), exist_ok=True)
    report_df.to_csv(args.error_report, sep="\t", index=False)
    by_country_path = args.error_report.replace(".tsv", "_by_country.tsv")
    by_country.to_csv(by_country_path, sep="\t", index=False)

    print("\n=== Held-out validation summary ===")
    print(f"Entities: {len(val_id_set):,}  Threshold: {best_threshold}  Macro F_0.5: {best_f:.4f}")
    print("\nOverall outcome breakdown:")
    print(overall.to_string())
    print("\nBy-country outcome breakdown:")
    print(by_country.to_string(index=False))

    n_nonsingleton = sum(1 for eid in val_id_set if true_map.get(eid))
    fn_blocking_among_nonsingleton = (
        report_df[report_df["outcome"] == "FN-blocking"].shape[0] / n_nonsingleton
        if n_nonsingleton else 0.0
    )
    print(f"\nFN-blocking rate among non-singleton val entities: {fn_blocking_among_nonsingleton:.4f}")

    summary = {
        "n_val_entities": len(val_id_set),
        "n_train_entities": len(train_id_set),
        "threshold": best_threshold,
        "macro_f0.5": best_f,
        "threshold_curve": curve,
        "fn_blocking_rate_among_nonsingleton": fn_blocking_among_nonsingleton,
        "overall_outcome_counts": overall.to_dict(),
    }
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    _log(f"wrote {args.error_report}, {by_country_path}, {args.summary_json}")
    return summary


def run_test(args):
    _log(f"loading trained model + tuned threshold from {args.model_path} "
         "(reuses the model/threshold produced by `validate` - retraining on train data "
         "again here would just redo that run's blocking pass for no real benefit)")
    if not os.path.isfile(args.model_path) and args.threshold is None:
        raise SystemExit(
            f"{args.model_path} not found and no --threshold given. Run "
            "`python -m src.baseline.pipeline validate` first (it saves the model there), "
            "or pass --model-path/--threshold explicitly."
        )
    saved = joblib.load(args.model_path) if os.path.isfile(args.model_path) else {}
    model = saved.get("model")
    threshold = args.threshold if args.threshold is not None else saved.get("threshold")
    if model is None:
        raise SystemExit(f"{args.model_path} has no 'model' entry - rerun `validate` to regenerate it.")
    _log(f"using tuned threshold: {threshold}")

    _log("loading + normalizing test_source1 (reference)")
    s1_test_df = data.normalize_source_to_parquet(args.test_source1, args.tmp_dir + "_s1norm_test")
    _log(f"test source1 entities: {len(s1_test_df):,}")
    _log(f"test countries seen: {sorted(s1_test_df['country'].unique().to_list())}")

    _log("blocking + features on the real test set (batched over source2 then source3)")
    test_candidate_pairs, test_feats = _build_pairs_and_features(
        s1_test_df, [args.test_source2, args.test_source3], os.path.join(args.tmp_dir, "test"),
        max_df=args.max_df, max_candidates_per_entity=args.max_candidates_per_entity,
        batch_size=args.batch_size,
    )

    _log(f"writing {args.candidate_out}")
    all_test_ids = s1_test_df["entity_id"].to_list()
    candidate_grouped = blocking.candidates_to_grouped(test_candidate_pairs, all_test_ids)
    os.makedirs(os.path.dirname(args.candidate_out), exist_ok=True)
    candidate_grouped.write_csv(args.candidate_out, separator="\t", quote_style="never")
    del test_candidate_pairs, candidate_grouped
    gc.collect()

    _log("scoring test candidate pairs with the final model")
    scored_test = matching.score_candidates(model, test_feats)
    pred_map = matching.matches_at_threshold(scored_test, threshold)

    _log(f"writing {args.matching_out}")
    match_rows = [(eid, ",".join(sorted(pred_map.get(eid, set())))) for eid in all_test_ids]
    matching_df = pl.DataFrame(
        {"source1_entity_id": [r[0] for r in match_rows],
         "matched_entity_ids": [r[1] for r in match_rows]}
    )
    os.makedirs(os.path.dirname(args.matching_out), exist_ok=True)
    matching_df.write_csv(args.matching_out, separator="\t", quote_style="never")

    n_matched = sum(1 for _, ids in match_rows if ids)
    _log(f"done. {n_matched:,}/{len(all_test_ids):,} test entities predicted with >=1 match")

    shutil.rmtree(args.tmp_dir, ignore_errors=True)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="held-out F_0.5 + error attribution on train data")
    v.add_argument("--source1", default="dataset/train/train_source1.tsv")
    v.add_argument("--source2", default="dataset/train/train_source2.tsv")
    v.add_argument("--source3", default="dataset/train/train_source3.tsv")
    v.add_argument("--ground-truth", default="dataset/train/train_ground_truth.tsv")
    v.add_argument("--split-cache-dir", default="output/splits")
    v.add_argument("--seed", type=int, default=split.DEFAULT_SEED)
    v.add_argument("--holdout-frac", type=float, default=split.DEFAULT_HOLDOUT_FRAC)
    v.add_argument("--max-df", type=int, default=blocking.DEFAULT_MAX_DF)
    v.add_argument("--max-candidates-per-entity", type=int,
                    default=blocking.DEFAULT_MAX_CANDIDATES_PER_ENTITY)
    v.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    v.add_argument("--tmp-dir", default="output/_tmp_shards_validate")
    v.add_argument("--error-report", default="output/diagnostics/error_attribution.tsv")
    v.add_argument("--summary-json", default="output/diagnostics/validation_summary.json")
    v.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                    help="where to save the trained model + tuned threshold, for `test` to reuse")
    v.set_defaults(func=run_validation)

    t = sub.add_parser(
        "test", help="run the real test set through the model/threshold `validate` produced"
    )
    t.add_argument("--test-source1", default="dataset/test/test_source1.tsv")
    t.add_argument("--test-source2", default="dataset/test/test_source2.tsv")
    t.add_argument("--test-source3", default="dataset/test/test_source3.tsv")
    t.add_argument("--max-df", type=int, default=blocking.DEFAULT_MAX_DF)
    t.add_argument("--max-candidates-per-entity", type=int,
                    default=blocking.DEFAULT_MAX_CANDIDATES_PER_ENTITY)
    t.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    t.add_argument("--tmp-dir", default="output/_tmp_shards_test")
    t.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                    help="model + threshold saved by `validate` (see --model-path there)")
    t.add_argument("--threshold", type=float, default=None,
                    help="override the threshold saved alongside the model")
    t.add_argument("--candidate-out", default="output/candidate_pairs.tsv")
    t.add_argument("--matching-out", default="output/matching_results.tsv")
    t.set_defaults(func=run_test)

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
