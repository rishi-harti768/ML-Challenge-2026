"""Similarity features on surviving candidate pairs.

Fast classical baseline: Jaccard token overlap (vectorized via polars list
set-operations) + rapidfuzz edit-distance ratio on normalized name and
address, plus a postal-match flag. Combined downstream (matching.py) into
a single match score via logistic regression - kept simple and fast on
purpose (this is the disposable baseline, not the hybrid embedding
architecture from issue #8/#11).
"""
import polars as pl
from rapidfuzz import fuzz

FEATURE_COLUMNS = [
    "name_jaccard", "name_ratio", "addr_jaccard", "addr_ratio", "postal_match",
]

_LOOKUP_COLS = ["entity_id", "name_norm", "addr_norm", "name_tok", "addr_tok", "postal"]


def compute_features(candidate_pairs: pl.DataFrame, s1_lookup: pl.DataFrame,
                      cand_lookup) -> pl.DataFrame:
    """Join candidate pairs against entity lookups and compute similarity features.

    `s1_lookup` carries the normalized helper columns (name_norm, addr_norm,
    name_tok, addr_tok, postal) from data.add_normalized_columns, indexed
    implicitly by entity_id - it's the (small) Source-1 batch, held eagerly.

    `cand_lookup` may be the same kind of eager DataFrame (small/sample
    runs) or a `pl.LazyFrame` scanning the candidate source's parquet
    shards (`data.scan_normalized`) - in the latter case the join stays
    lazy and only `candidate_pairs`-sized output is ever materialized, so
    the full multi-million-row candidate table is never held in memory at
    once just to answer one batch's lookups.
    """
    s1_lazy = s1_lookup.lazy() if isinstance(s1_lookup, pl.DataFrame) else s1_lookup
    cand_lazy = cand_lookup.lazy() if isinstance(cand_lookup, pl.DataFrame) else cand_lookup

    s1_cols = s1_lazy.select(_LOOKUP_COLS).rename({
        "entity_id": "source1_entity_id",
        "name_norm": "name_norm_s1", "addr_norm": "addr_norm_s1",
        "name_tok": "name_tok_s1", "addr_tok": "addr_tok_s1", "postal": "postal_s1",
    })
    cand_cols = cand_lazy.select(_LOOKUP_COLS).rename({
        "entity_id": "candidate_entity_id",
        "name_norm": "name_norm_cand", "addr_norm": "addr_norm_cand",
        "name_tok": "name_tok_cand", "addr_tok": "addr_tok_cand", "postal": "postal_cand",
    })

    df = candidate_pairs.lazy().join(s1_cols, on="source1_entity_id", how="left")
    df = df.join(cand_cols, on="candidate_entity_id", how="left")

    df = df.with_columns([
        pl.col("name_tok_s1").list.set_intersection(pl.col("name_tok_cand")).list.len().alias("__name_inter"),
        pl.col("name_tok_s1").list.set_union(pl.col("name_tok_cand")).list.len().alias("__name_union"),
        pl.col("addr_tok_s1").list.set_intersection(pl.col("addr_tok_cand")).list.len().alias("__addr_inter"),
        pl.col("addr_tok_s1").list.set_union(pl.col("addr_tok_cand")).list.len().alias("__addr_union"),
    ])
    df = df.with_columns([
        pl.when(pl.col("__name_union") > 0)
          .then(pl.col("__name_inter") / pl.col("__name_union")).otherwise(0.0).alias("name_jaccard"),
        pl.when(pl.col("__addr_union") > 0)
          .then(pl.col("__addr_inter") / pl.col("__addr_union")).otherwise(0.0).alias("addr_jaccard"),
        ((pl.col("postal_s1") != "") & (pl.col("postal_s1") == pl.col("postal_cand")))
          .cast(pl.Float64).alias("postal_match"),
    ])

    # candidate_pairs (the left/driving side of both joins) is already
    # small - batch-sized, at most batch_size*max_candidates_per_entity
    # rows - so this collect only ever materializes a batch-sized result,
    # even though cand_cols may be lazily scanning a multi-million-row
    # candidate source.
    df = df.collect(engine="streaming")

    name_ratio = [
        fuzz.ratio(a, b) / 100.0
        for a, b in zip(df["name_norm_s1"].to_list(), df["name_norm_cand"].to_list())
    ]
    addr_ratio = [
        fuzz.ratio(a, b) / 100.0
        for a, b in zip(df["addr_norm_s1"].to_list(), df["addr_norm_cand"].to_list())
    ]
    df = df.with_columns([
        pl.Series("name_ratio", name_ratio),
        pl.Series("addr_ratio", addr_ratio),
    ])

    keep = ["source1_entity_id", "candidate_entity_id"] + FEATURE_COLUMNS
    return df.select(keep)
