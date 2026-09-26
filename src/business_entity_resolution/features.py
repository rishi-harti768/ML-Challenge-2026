import numpy as np
import polars as pl
from joblib import Parallel, delayed
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from business_entity_resolution.parallel import default_n_jobs

FEATURE_COLUMNS = [
    "name_jaro_winkler",
    "name_levenshtein_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaccard",
    "name_common_token_ratio",
    "name_length_ratio",
    "name_first_letter_match",
    "addr_jaro_winkler",
    "addr_levenshtein_ratio",
    "addr_jaccard",
    "addr_common_token_ratio",
    "addr_length_ratio",
    "addr_numeric_overlap",
    "blocking_agreement_count",
    "competition_density",
    "name_score_rank",
]

_RAW_FEATURE_NAMES = FEATURE_COLUMNS[:14]


def _token_jaccard(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _common_token_ratio(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def _length_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def _numeric_overlap(a: str, b: str) -> float:
    ta = set(a.split("|")) if a else set()
    tb = set(b.split("|")) if b else set()
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def _score_chunk(s1_name, c_name, s1_addr, c_addr, s1_num, c_num) -> np.ndarray:
    n = len(s1_name)
    out = np.empty((n, len(_RAW_FEATURE_NAMES)), dtype=np.float32)

    for i in range(n):
        a, b = s1_name[i] or "", c_name[i] or ""
        out[i, 0] = JaroWinkler.normalized_similarity(a, b)
        out[i, 1] = fuzz.ratio(a, b) / 100.0
        out[i, 2] = fuzz.token_sort_ratio(a, b) / 100.0
        out[i, 3] = fuzz.token_set_ratio(a, b) / 100.0
        out[i, 4] = _token_jaccard(a, b)
        out[i, 5] = _common_token_ratio(a, b)
        out[i, 6] = _length_ratio(a, b)
        out[i, 7] = 1.0 if a and b and a[0] == b[0] else 0.0

        aa, bb = s1_addr[i] or "", c_addr[i] or ""
        out[i, 8] = JaroWinkler.normalized_similarity(aa, bb)
        out[i, 9] = fuzz.ratio(aa, bb) / 100.0
        out[i, 10] = _token_jaccard(aa, bb)
        out[i, 11] = _common_token_ratio(aa, bb)
        out[i, 12] = _length_ratio(aa, bb)
        out[i, 13] = _numeric_overlap(s1_num[i] or "", c_num[i] or "")

    return out


def _join_sides(candidates: pl.DataFrame, s1: pl.DataFrame, cand: pl.DataFrame) -> pl.DataFrame:
    s1_small = s1.select(
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("name_norm").alias("s1_name"),
        pl.col("addr_norm").alias("s1_addr"),
        pl.col("addr_numeric_tokens").alias("s1_num"),
    )
    cand_small = cand.select(
        pl.col("entity_id").alias("candidate_entity_id"),
        pl.col("name_norm").alias("c_name"),
        pl.col("addr_norm").alias("c_addr"),
        pl.col("addr_numeric_tokens").alias("c_num"),
    )
    return candidates.join(s1_small, on="source1_entity_id", how="left").join(
        cand_small, on="candidate_entity_id", how="left"
    )


def compute_pair_features(
    candidates: pl.DataFrame, s1: pl.DataFrame, cand: pl.DataFrame, n_jobs: int = -1
) -> pl.DataFrame:
    df = _join_sides(candidates, s1, cand)

    s1_name = df["s1_name"].to_list()
    c_name = df["c_name"].to_list()
    s1_addr = df["s1_addr"].to_list()
    c_addr = df["c_addr"].to_list()
    s1_num = df["s1_num"].to_list()
    c_num = df["c_num"].to_list()

    n = len(df)
    if n_jobs == -1:
        n_jobs = default_n_jobs()
    n_jobs = max(1, min(n_jobs, max(1, n // 5000) or 1))
    chunk_size = -(-n // n_jobs) if n_jobs else n or 1
    chunk_size = max(chunk_size, 1)

    bounds = list(range(0, max(n, 1), chunk_size))
    if not bounds:
        bounds = [0]

    chunks = Parallel(n_jobs=n_jobs)(
        delayed(_score_chunk)(
            s1_name[i : i + chunk_size],
            c_name[i : i + chunk_size],
            s1_addr[i : i + chunk_size],
            c_addr[i : i + chunk_size],
            s1_num[i : i + chunk_size],
            c_num[i : i + chunk_size],
        )
        for i in bounds
    )
    scores = np.concatenate(chunks, axis=0) if chunks else np.empty((0, len(_RAW_FEATURE_NAMES)), dtype=np.float32)

    df = df.with_columns(
        [pl.Series(name, scores[:, idx]) for idx, name in enumerate(_RAW_FEATURE_NAMES)]
    )

    df = df.with_columns(
        (
            pl.col("from_exact_key").cast(pl.Int8)
            + pl.col("from_anchor").cast(pl.Int8)
            + pl.col("from_addr_anchor").cast(pl.Int8)
            + pl.col("from_addr_key").cast(pl.Int8)
        ).alias("blocking_agreement_count"),
        (0.6 * pl.col("name_jaro_winkler") + 0.4 * pl.col("name_token_set_ratio")).alias("_name_score"),
    )
    df = df.with_columns(
        pl.len().over("source1_entity_id").alias("competition_density"),
        pl.col("_name_score").rank(method="ordinal", descending=True).over("source1_entity_id").alias(
            "name_score_rank"
        ),
    )
    return df.drop("_name_score")
