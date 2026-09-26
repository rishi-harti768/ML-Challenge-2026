import random
import zlib
from collections import Counter
from functools import partial

import polars as pl

from business_entity_resolution.parallel import parallel_map

ANCHOR_N = 5
ANCHOR_K = 3
MAX_BUCKET_SIDE = 60
STOPGRAM_SAMPLE_SIZE = 300_000
STOPGRAM_TOP_K = 3000


def _ngrams(s: str, n: int = ANCHOR_N) -> set[str]:
    padded = f" {s} "
    if len(padded) < n:
        return {padded}
    return {padded[i : i + n] for i in range(len(padded) - n + 1)}


def build_stopgrams(
    texts: list[str],
    sample_size: int = STOPGRAM_SAMPLE_SIZE,
    top_k: int = STOPGRAM_TOP_K,
    n: int = ANCHOR_N,
) -> frozenset[str]:
    if len(texts) > sample_size:
        rng = random.Random(0)
        sample = rng.sample(texts, sample_size)
    else:
        sample = texts
    freq: Counter[str] = Counter()
    for s in sample:
        freq.update(_ngrams(s, n))
    return frozenset(g for g, _ in freq.most_common(top_k))


def anchor_keys(
    s: str, stopgrams: frozenset[str] = frozenset(), k: int = ANCHOR_K, n: int = ANCHOR_N
) -> list[str]:
    grams = _ngrams(s, n) - stopgrams
    if not grams:
        grams = _ngrams(s, n)
    ranked = sorted(grams, key=lambda g: zlib.crc32(g.encode("utf-8")))
    return ranked[:k]


def add_blocking_columns(
    df: pl.DataFrame,
    name_stopgrams: frozenset[str] = frozenset(),
    addr_stopgrams: frozenset[str] = frozenset(),
) -> pl.DataFrame:
    name_anchors = parallel_map(partial(anchor_keys, stopgrams=name_stopgrams), df["name_key"].to_list())
    addr_anchors = parallel_map(partial(anchor_keys, stopgrams=addr_stopgrams), df["addr_norm"].to_list())
    return df.with_columns(
        pl.Series("anchor_keys", name_anchors, dtype=pl.List(pl.Utf8)),
        pl.Series("addr_anchor_keys", addr_anchors, dtype=pl.List(pl.Utf8)),
    )


def _prep_side(df: pl.DataFrame, id_col: str) -> pl.DataFrame:
    return df.select(
        pl.col("entity_id").alias(id_col),
        "country",
        "name_key",
        "addr_first_token",
        "anchor_keys",
        "addr_anchor_keys",
    )


def _cap_bucket_side(df: pl.DataFrame, key_cols: list[str], max_size: int, seed: int) -> pl.DataFrame:
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=seed)
    return (
        shuffled.with_columns(pl.int_range(pl.len()).over(key_cols).alias("_rank"))
        .filter(pl.col("_rank") < max_size)
        .drop("_rank")
    )


def _bounded_pairs(
    left: pl.DataFrame,
    right: pl.DataFrame,
    key_cols: list[str],
    left_id: str,
    right_id: str,
    max_bucket_side: int = MAX_BUCKET_SIDE,
) -> pl.DataFrame:
    l2 = _cap_bucket_side(left.select(key_cols + [left_id]), key_cols, max_bucket_side, seed=1)
    r2 = _cap_bucket_side(right.select(key_cols + [right_id]), key_cols, max_bucket_side, seed=2)
    return l2.join(r2, on=key_cols, how="inner").select(left_id, right_id)


def generate_candidates(s1: pl.DataFrame, cand: pl.DataFrame) -> pl.DataFrame:
    left = _prep_side(s1, "source1_entity_id")
    right = _prep_side(cand, "candidate_entity_id")

    exact = _bounded_pairs(
        left.filter(pl.col("name_key") != ""),
        right.filter(pl.col("name_key") != ""),
        ["country", "name_key"],
        "source1_entity_id",
        "candidate_entity_id",
    ).with_columns(pl.lit(True).alias("from_exact_key"))

    left_anchor = left.select("source1_entity_id", "country", "anchor_keys").explode("anchor_keys")
    right_anchor = right.select("candidate_entity_id", "country", "anchor_keys").explode("anchor_keys")
    anchor = (
        _bounded_pairs(
            left_anchor, right_anchor, ["country", "anchor_keys"], "source1_entity_id", "candidate_entity_id"
        )
        .unique()
        .with_columns(pl.lit(True).alias("from_anchor"))
    )

    left_addr_anchor = left.select("source1_entity_id", "country", "addr_anchor_keys").explode(
        "addr_anchor_keys"
    )
    right_addr_anchor = right.select("candidate_entity_id", "country", "addr_anchor_keys").explode(
        "addr_anchor_keys"
    )
    addr_anchor = (
        _bounded_pairs(
            left_addr_anchor,
            right_addr_anchor,
            ["country", "addr_anchor_keys"],
            "source1_entity_id",
            "candidate_entity_id",
        )
        .unique()
        .with_columns(pl.lit(True).alias("from_addr_anchor"))
    )

    addr = left.filter(pl.col("addr_first_token") != "").select(
        "source1_entity_id", "country", "addr_first_token"
    )
    addr_r = right.filter(pl.col("addr_first_token") != "").select(
        "candidate_entity_id", "country", "addr_first_token"
    )
    addr_pairs = _bounded_pairs(
        addr, addr_r, ["country", "addr_first_token"], "source1_entity_id", "candidate_entity_id"
    ).with_columns(pl.lit(True).alias("from_addr_key"))

    id_cols = ["source1_entity_id", "candidate_entity_id"]
    tagged = pl.concat(
        [
            exact.select(*id_cols, pl.lit("exact").alias("src")),
            anchor.select(*id_cols, pl.lit("anchor").alias("src")),
            addr_anchor.select(*id_cols, pl.lit("addr_anchor").alias("src")),
            addr_pairs.select(*id_cols, pl.lit("addr_key").alias("src")),
        ],
        how="vertical",
    )
    merged = tagged.group_by(id_cols).agg(
        (pl.col("src") == "exact").any().alias("from_exact_key"),
        (pl.col("src") == "anchor").any().alias("from_anchor"),
        (pl.col("src") == "addr_anchor").any().alias("from_addr_anchor"),
        (pl.col("src") == "addr_key").any().alias("from_addr_key"),
    )
    return merged
