"""Held-out validation split (issue #2 decision).

Stratified ~85/15 hold-out on Source-1 (reference) entities, stratified by
`country`, fixed random seed, split at the entity level so a reference
entity's full ground-truth match list travels with it into whichever split
it lands in. Reusable by any pipeline ticket (this baseline, and later the
hybrid architecture in issue #11) so comparisons are apples-to-apples.
"""
import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_SEED = 42
DEFAULT_HOLDOUT_FRAC = 0.15


def make_split(source1_df: pd.DataFrame, seed: int = DEFAULT_SEED,
                holdout_frac: float = DEFAULT_HOLDOUT_FRAC):
    """Split Source-1 entities into (train_df, val_df), stratified by country.

    Entity-level split: every row of `source1_df` is a distinct reference
    entity, so this returns disjoint row subsets. Ground truth is not
    touched here - callers join on `entity_id` afterwards, which is what
    makes a reference entity's match list "travel with it" automatically.
    """
    train_df, val_df = train_test_split(
        source1_df,
        test_size=holdout_frac,
        random_state=seed,
        stratify=source1_df["country"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def load_or_make_split(source1_path: str, cache_dir: str,
                        seed: int = DEFAULT_SEED,
                        holdout_frac: float = DEFAULT_HOLDOUT_FRAC):
    """Cached wrapper: writes/reads train_ids.txt + val_ids.txt under cache_dir.

    The split itself is fully deterministic (fixed seed) so caching is a
    pure speed optimization, never a correctness dependency - deleting the
    cache and rerunning reproduces byte-identical id sets.
    """
    train_ids_path = os.path.join(cache_dir, "train_ids.txt")
    val_ids_path = os.path.join(cache_dir, "val_ids.txt")
    # Only entity_id + country are needed for the split itself - skipping
    # business_name/business_address avoids holding a second, redundant
    # full-text copy of Source-1 in memory alongside the caller's own
    # (already normalized) copy.
    source1_df = pd.read_csv(source1_path, sep="\t", dtype=str, usecols=["entity_id", "country"])

    if os.path.isfile(train_ids_path) and os.path.isfile(val_ids_path):
        with open(train_ids_path, encoding="utf-8") as f:
            train_ids = set(line.strip() for line in f if line.strip())
        with open(val_ids_path, encoding="utf-8") as f:
            val_ids = set(line.strip() for line in f if line.strip())
        train_df = source1_df[source1_df["entity_id"].isin(train_ids)].reset_index(drop=True)
        val_df = source1_df[source1_df["entity_id"].isin(val_ids)].reset_index(drop=True)
        return train_df, val_df

    train_df, val_df = make_split(source1_df, seed=seed, holdout_frac=holdout_frac)
    os.makedirs(cache_dir, exist_ok=True)
    with open(train_ids_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_df["entity_id"]))
    with open(val_ids_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_df["entity_id"]))
    return train_df, val_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source1", default="dataset/train/train_source1.tsv")
    parser.add_argument("--cache-dir", default="output/splits")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--holdout-frac", type=float, default=DEFAULT_HOLDOUT_FRAC)
    args = parser.parse_args()

    train_df, val_df = load_or_make_split(
        args.source1, args.cache_dir, seed=args.seed, holdout_frac=args.holdout_frac
    )
    print(f"train: {len(train_df)} entities, val: {len(val_df)} entities")
    print("train country counts:")
    print(train_df["country"].value_counts())
    print("val country counts:")
    print(val_df["country"].value_counts())


if __name__ == "__main__":
    main()
