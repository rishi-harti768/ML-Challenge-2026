"""Loading + normalizing the tab-separated source files.

Uses polars (Arrow-backed columnar memory) rather than pandas: this
machine has only a few GB of free RAM, and the ~26M-row combined dataset
with per-row token-list columns does not fit comfortably in pandas'
per-object string/list overhead. Polars keeps strings and list columns in
contiguous Arrow buffers, which is dramatically lighter at this scale.
"""
import glob
import os

import polars as pl

from .normalize import (
    address_tokens_expr, extract_postal_expr, name_prefixes_expr, name_tokens_expr, normalize_text_expr,
)

DEFAULT_NORMALIZE_CHUNK_SIZE = 250_000


def load_source(path: str) -> pl.DataFrame:
    return pl.read_csv(
        path, separator="\t", infer_schema_length=0, schema_overrides=None,
    ).with_columns(pl.all().fill_null(""))


def add_normalized_columns(df: pl.DataFrame, keep_raw: bool = False) -> pl.DataFrame:
    """Add name_norm/addr_norm/name_tok/addr_tok/name_prefix/postal columns.

    Drops the original `business_name`/`business_address` text columns by
    default once their normalized derivatives are computed - at this
    dataset's scale (~26M rows total) keeping both raw and normalized text
    around roughly doubles memory for no downstream benefit (nothing after
    normalization needs the raw text again; output files only ever
    reference entity_id).
    """
    df = df.with_columns([
        normalize_text_expr("business_name").alias("name_norm"),
        normalize_text_expr("business_address").alias("addr_norm"),
        extract_postal_expr("business_address").alias("postal"),
    ])
    df = df.with_columns([
        name_tokens_expr(pl.col("name_norm")).alias("name_tok"),
        address_tokens_expr(pl.col("addr_norm")).alias("addr_tok"),
    ])
    df = df.with_columns(
        name_prefixes_expr(pl.col("name_tok")).alias("name_prefix")
    )
    if not keep_raw:
        df = df.drop(["business_name", "business_address"])
    return df


def load_candidates(source2_path: str, source3_path: str) -> pl.DataFrame:
    """Source 2 + Source 3 records concatenated, normalized, ready for blocking."""
    s2 = load_source(source2_path)
    s3 = load_source(source3_path)
    cand = pl.concat([s2, s3], how="vertical")
    return add_normalized_columns(cand)


def normalize_source_to_shards(path: str, shard_dir: str,
                                chunk_size: int = DEFAULT_NORMALIZE_CHUNK_SIZE) -> list:
    """Read + normalize a source file in bounded-size chunks, writing parquet shards.

    `map_elements` over an entire multi-million-row column transiently
    holds every row's Python-object result (lists of token strings) in
    memory at once before polars can pack it into a compact Arrow column -
    on this machine (a few GB of free RAM) that transient spike alone was
    enough to push the process into heavy paging/thrashing for a single
    ~5M-row source file. Chunking the read+normalize+write bounds that
    spike to one chunk at a time. Returns the sorted list of shard paths -
    callers that can afford to hold the full table (the much smaller
    Source-1 side) pass this to `pl.read_parquet`; callers working with the
    large candidate sources instead scan the shards lazily
    (`scan_normalized`) so the ~5M-row table is never fully materialized in
    Python-managed memory at once.
    """
    if os.path.isdir(shard_dir):
        for f in glob.glob(os.path.join(shard_dir, "*.parquet")):
            os.remove(f)
    os.makedirs(shard_dir, exist_ok=True)

    reader = pl.read_csv_batched(
        path, separator="\t", infer_schema_length=0, batch_size=chunk_size,
    )
    shard_idx = 0
    while True:
        batches = reader.next_batches(1)
        if not batches:
            break
        chunk = batches[0].with_columns(pl.all().fill_null(""))
        chunk = add_normalized_columns(chunk)
        chunk.write_parquet(os.path.join(shard_dir, f"norm_{shard_idx:05d}.parquet"))
        shard_idx += 1

    return sorted(glob.glob(os.path.join(shard_dir, "*.parquet")))


def normalize_source_to_parquet(path: str, shard_dir: str,
                                 chunk_size: int = DEFAULT_NORMALIZE_CHUNK_SIZE) -> pl.DataFrame:
    """Like `normalize_source_to_shards`, but materializes the full table.

    Only for the Source-1 side (~2.2M rows train / ~1.7M test) - small
    enough to hold eagerly, and needed as an eager DataFrame so callers can
    `.slice()` it into batches.
    """
    shard_files = normalize_source_to_shards(path, shard_dir, chunk_size=chunk_size)
    return pl.read_parquet(shard_files)


def scan_normalized(shard_files: list) -> pl.LazyFrame:
    """Lazily scan normalized candidate shards without materializing them."""
    return pl.scan_parquet(shard_files)


def load_ground_truth(path: str) -> dict:
    """{source1_entity_id: set(matched_entity_ids)} - empty string means singleton."""
    gt = pl.read_csv(path, separator="\t", infer_schema_length=0).with_columns(pl.all().fill_null(""))
    out = {}
    for sid, ids in zip(gt["source1_entity_id"].to_list(), gt["matched_entity_ids"].to_list()):
        out[sid] = set(ids.split(",")) if ids else set()
    return out


def candidate_pairs_to_map(candidate_pairs: pl.DataFrame) -> dict:
    """{source1_entity_id: set(candidate_entity_id)}"""
    out = {}
    for sid, cid in zip(
        candidate_pairs["source1_entity_id"].to_list(), candidate_pairs["candidate_entity_id"].to_list()
    ):
        out.setdefault(sid, set()).add(cid)
    return out
