import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import polars as pl

from business_entity_resolution.blocking import add_blocking_columns, build_stopgrams, generate_candidates
from business_entity_resolution.features import compute_pair_features
from business_entity_resolution.io_utils import read_ground_truth, read_source
from business_entity_resolution.modeling import build_labels, predict_proba, train_model
from business_entity_resolution.scoring import macro_f_beta, parse_matched_ids
from business_entity_resolution.splitting import country_holdout_split, random_holdout_split
from business_entity_resolution.threshold import sweep_threshold


_T0 = time.time()


def _log(msg: str) -> None:
    print(f"[{time.time() - _T0:8.1f}s] {msg}", flush=True)
    sys.stdout.flush()


def build_cand_frame(s2: pl.DataFrame, s3: pl.DataFrame) -> pl.DataFrame:
    return pl.concat([s2, s3], how="vertical")


def truth_dict(ground_truth: pl.DataFrame) -> dict[str, set[str]]:
    return {
        row["source1_entity_id"]: parse_matched_ids(row["matched_entity_ids"])
        for row in ground_truth.iter_rows(named=True)
    }


def candidate_recall(candidates: pl.DataFrame, truth: dict[str, set[str]]) -> float:
    s1_ids = []
    c_ids = []
    for sid, ids in truth.items():
        for cid in ids:
            s1_ids.append(sid)
            c_ids.append(cid)
    if not s1_ids:
        return 1.0
    truth_df = pl.DataFrame({"source1_entity_id": s1_ids, "candidate_entity_id": c_ids})
    found = truth_df.join(
        candidates.select("source1_entity_id", "candidate_entity_id"),
        on=["source1_entity_id", "candidate_entity_id"],
        how="semi",
    )
    return found.height / truth_df.height


def preds_to_dict(feature_df: pl.DataFrame, proba, threshold: float) -> dict[str, set[str]]:
    preds: dict[str, set[str]] = {}
    s1_ids = feature_df["source1_entity_id"].to_list()
    cand_ids = feature_df["candidate_entity_id"].to_list()
    for sid, cid, p in zip(s1_ids, cand_ids, proba):
        if p >= threshold:
            preds.setdefault(sid, set()).add(cid)
    return preds


def write_id_list_tsv(path: Path, id_col: str, list_col: str, all_ids: list[str], mapping: dict[str, set[str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{id_col}\t{list_col}\n")
        for sid in all_ids:
            ids = mapping.get(sid, set())
            f.write(f"{sid}\t{','.join(sorted(ids))}\n")


def cmd_train(args):
    train_dir = Path(args.train_dir)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    _log("reading source1...")
    s1 = read_source(train_dir / "train_source1.tsv")
    _log(f"source1: {s1.height} rows. reading source2...")
    s2 = read_source(train_dir / "train_source2.tsv")
    _log(f"source2: {s2.height} rows. reading source3...")
    s3 = read_source(train_dir / "train_source3.tsv")
    _log(f"source3: {s3.height} rows. reading ground truth...")
    gt = read_ground_truth(train_dir / "train_ground_truth.tsv")
    truth = truth_dict(gt)

    cand = build_cand_frame(s2, s3)
    _log("computing stopgrams...")
    name_stopgrams = build_stopgrams(cand["name_key"].to_list())
    addr_stopgrams = build_stopgrams(cand["addr_norm"].to_list())
    _log("computing blocking columns...")
    s1 = add_blocking_columns(s1, name_stopgrams, addr_stopgrams)
    cand = add_blocking_columns(cand, name_stopgrams, addr_stopgrams)

    _log("generating candidates...")
    candidates = generate_candidates(s1, cand)
    _log(f"candidates: {candidates.height} rows")
    recall = candidate_recall(candidates, truth)
    _log(f"candidate recall@blocking: {recall:.4f}")

    _log("computing features...")
    feature_df = compute_pair_features(candidates, s1, cand)
    labels = build_labels(feature_df, truth)
    feature_df = feature_df.with_columns(pl.Series("label", labels))
    _log("features done")

    metrics = {"candidate_recall": recall, "n_candidates": feature_df.height}

    for split_name, held_out_country in [("random_holdout", None), ("country_holdout", "India")]:
        if held_out_country is None:
            train_ids, val_ids = random_holdout_split(s1, frac=0.2, seed=0)
        else:
            train_ids, val_ids = country_holdout_split(s1, held_out_country)

        train_mask = feature_df["source1_entity_id"].is_in(list(train_ids))
        val_mask = feature_df["source1_entity_id"].is_in(list(val_ids))
        train_df = feature_df.filter(train_mask)
        val_df = feature_df.filter(val_mask)

        _log(f"[{split_name}] training on {train_df.height} rows...")
        model = train_model(train_df, train_df["label"].to_numpy())
        proba = predict_proba(model, val_df)
        best_t, best_score = sweep_threshold(
            val_df, proba, val_df["label"].to_numpy(), truth, val_ids
        )
        _log(f"[{split_name}] best_threshold={best_t:.2f} macro_f0.5={best_score:.4f}")
        metrics[split_name] = {"threshold": best_t, "macro_f0.5": best_score, "n_val_entities": len(val_ids)}

    _log("training final model on full training data...")
    final_model = train_model(feature_df, feature_df["label"].to_numpy())
    final_threshold = metrics["country_holdout"]["threshold"]

    joblib.dump(final_model, artifacts_dir / "model.joblib")
    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump({**metrics, "final_threshold": final_threshold}, f, indent=2)
    _log(f"saved model + metrics to {artifacts_dir}")


def cmd_predict(args):
    test_dir = Path(args.test_dir)
    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(args.output_dir)

    _log("reading source1...")
    s1 = read_source(test_dir / "test_source1.tsv")
    _log(f"source1: {s1.height} rows. reading source2...")
    s2 = read_source(test_dir / "test_source2.tsv")
    _log(f"source2: {s2.height} rows. reading source3...")
    s3 = read_source(test_dir / "test_source3.tsv")
    _log(f"source3: {s3.height} rows")

    cand = build_cand_frame(s2, s3)
    _log("computing stopgrams...")
    name_stopgrams = build_stopgrams(cand["name_key"].to_list())
    addr_stopgrams = build_stopgrams(cand["addr_norm"].to_list())
    _log("computing blocking columns...")
    s1 = add_blocking_columns(s1, name_stopgrams, addr_stopgrams)
    cand = add_blocking_columns(cand, name_stopgrams, addr_stopgrams)

    _log("generating candidates...")
    candidates = generate_candidates(s1, cand)
    _log(f"candidates: {candidates.height} rows")

    _log("computing features...")
    feature_df = compute_pair_features(candidates, s1, cand)
    _log("features done")

    model = joblib.load(artifacts_dir / "model.joblib")
    with open(artifacts_dir / "metrics.json") as f:
        metrics = json.load(f)
    threshold = metrics["final_threshold"]

    _log("scoring candidates...")
    proba = predict_proba(model, feature_df)
    matches = preds_to_dict(feature_df, proba, threshold)

    all_s1_ids = s1["entity_id"].to_list()

    cand_map: dict[str, set[str]] = {}
    for sid, cid in zip(
        candidates["source1_entity_id"].to_list(), candidates["candidate_entity_id"].to_list()
    ):
        cand_map.setdefault(sid, set()).add(cid)

    write_id_list_tsv(
        output_dir / "candidate_pairs.tsv", "source1_entity_id", "candidate_entity_ids", all_s1_ids, cand_map
    )
    write_id_list_tsv(
        output_dir / "matching_results.tsv", "source1_entity_id", "matched_entity_ids", all_s1_ids, matches
    )
    _log(f"wrote outputs to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--train-dir", default="dataset/train")
    p_train.add_argument("--artifacts-dir", default="artifacts")
    p_train.set_defaults(func=cmd_train)

    p_predict = sub.add_parser("predict")
    p_predict.add_argument("--test-dir", default="dataset/test")
    p_predict.add_argument("--artifacts-dir", default="artifacts")
    p_predict.add_argument("--output-dir", default="output")
    p_predict.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
