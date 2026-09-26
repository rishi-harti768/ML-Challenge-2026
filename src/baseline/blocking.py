"""Multi-key inverted-index blocking / candidate generation.

Keeps candidate-set sizes manageable at ~2.2M x (5M+5.3M) scale by:

  * Multiple OR'd blocking keys, per the ticket's ask:
      1. (country, significant name token)   -- shared-token blocking
      2. (country, name-token prefix)        -- cheap typo/OCR tolerance
         (e.g. "Centre" vs a corrupted "C?ntre" still shares "cent")
      3. (country, significant address token) -- city/street-fragment
         blocking, cheaply parsed from the free-text address (no
         geocoding, no external data - just stopword-filtered tokens)
      4. (country, postal code) -- regex digit-run extracted from the
         address, when present
  * Document-frequency pruning: any key whose candidate-side posting list
    is bigger than `max_df` is dropped before the join, so one generic
    token/word can't blow the join up into a near-cross-product ("block
    purging", the standard fix for token blocking's high-recall/low-
    reduction-ratio failure mode noted in the issue #8 research doc).
  * A per-entity cap (`max_candidates_per_entity`), keeping the
    candidates that matched the most distinct blocking keys when an
    entity's union still comes out large.

Implemented with polars (Arrow-backed) explode + join (a vectorized hash
join, not a python loop over the full source x source cross-product).

The index-building step (`build_candidate_index`) is separated from the
query step (`query_candidates`) so a caller processing Source-1 in
batches (see pipeline.py) builds the candidate-side inverted index once
per candidate source file and reuses it across every batch, instead of
re-tokenizing/re-pruning the multi-million-row candidate pool on every
batch.
"""
import polars as pl

DEFAULT_MAX_DF = 2000
DEFAULT_MAX_CANDIDATES_PER_ENTITY = 50


def _exploded_token_frame(df, id_col: str, token_col: str, out_id_name: str):
    """entity_id/country/token long frame, one row per (entity, token).

    Works on either an eager DataFrame or a LazyFrame - the candidate-side
    index builder passes a LazyFrame (scanned straight from parquet shards)
    so this explode never materializes the full candidate table.
    """
    sub = df.select([id_col, "country", token_col]).explode(token_col)
    sub = sub.rename({id_col: out_id_name, token_col: "token"})
    return sub.filter(pl.col("token") != "")


def _exploded_postal_frame(df, id_col: str, out_id_name: str):
    sub = df.select([id_col, "country", "postal"]).rename({id_col: out_id_name})
    return sub.filter(pl.col("postal") != "")


def _prune_high_df(cand_long, key_cols: list, max_df: int):
    """Drop keys whose candidate-side posting list exceeds max_df."""
    counts = cand_long.group_by(key_cols).agg(pl.len().alias("__n"))
    keep_keys = counts.filter(pl.col("__n") <= max_df).select(key_cols)
    return cand_long.join(keep_keys, on=key_cols, how="semi")


def _join_on_key(s1_long: pl.DataFrame, cand_long: pl.DataFrame, key_cols: list,
                  key_type: str) -> pl.DataFrame:
    merged = s1_long.join(cand_long, on=key_cols, how="inner")
    merged = merged.select(["entity_id_s1", "entity_id_cand"]).unique()
    return merged.with_columns(pl.lit(key_type).alias("key_type"))


def build_candidate_index(cand_lazy: pl.LazyFrame, max_df: int = DEFAULT_MAX_DF) -> dict:
    """Pre-tokenize + document-frequency-prune the candidate pool once.

    `cand_lazy` is a LazyFrame (e.g. `data.scan_normalized(shard_files)`) so
    the (potentially ~5M-row) candidate table is never fully materialized -
    only the resulting long (entity_id_cand, country, token) frames are
    collected, which are far more compact than the raw table (no text
    columns, no per-record list columns) and safe to hold in memory for the
    lifetime of this candidate source's batches.
    """
    cand_name = _exploded_token_frame(cand_lazy, "entity_id", "name_tok", "entity_id_cand")
    cand_name = _prune_high_df(cand_name, ["country", "token"], max_df)

    cand_prefix = _exploded_token_frame(cand_lazy, "entity_id", "name_prefix", "entity_id_cand")
    cand_prefix = _prune_high_df(cand_prefix, ["country", "token"], max_df)

    cand_addr = _exploded_token_frame(cand_lazy, "entity_id", "addr_tok", "entity_id_cand")
    cand_addr = _prune_high_df(cand_addr, ["country", "token"], max_df)

    cand_postal = _exploded_postal_frame(cand_lazy, "entity_id", "entity_id_cand")
    cand_postal = _prune_high_df(cand_postal, ["country", "postal"], max_df)

    return {
        "name": cand_name.collect(engine="streaming"),
        "prefix": cand_prefix.collect(engine="streaming"),
        "addr": cand_addr.collect(engine="streaming"),
        "postal": cand_postal.collect(engine="streaming"),
    }


def query_candidates(s1_df: pl.DataFrame, index: dict,
                      max_candidates_per_entity: int = DEFAULT_MAX_CANDIDATES_PER_ENTITY,
                      verbose: bool = True) -> pl.DataFrame:
    """Return candidate pairs for this Source-1 batch against a prebuilt index."""
    s1_name = _exploded_token_frame(s1_df, "entity_id", "name_tok", "entity_id_s1")
    s1_prefix = _exploded_token_frame(s1_df, "entity_id", "name_prefix", "entity_id_s1")
    s1_addr = _exploded_token_frame(s1_df, "entity_id", "addr_tok", "entity_id_s1")
    s1_postal = _exploded_postal_frame(s1_df, "entity_id", "entity_id_s1")

    name_pairs = _join_on_key(s1_name, index["name"], ["country", "token"], "name_token")
    prefix_pairs = _join_on_key(s1_prefix, index["prefix"], ["country", "token"], "name_prefix")
    addr_pairs = _join_on_key(s1_addr, index["addr"], ["country", "token"], "addr_token")
    postal_pairs = _join_on_key(s1_postal, index["postal"], ["country", "postal"], "postal")

    if verbose:
        print(f"    name-token candidate pairs (raw):   {name_pairs.height:,}")
        print(f"    name-prefix candidate pairs (raw):  {prefix_pairs.height:,}")
        print(f"    addr-token candidate pairs (raw):   {addr_pairs.height:,}")
        print(f"    postal candidate pairs (raw):       {postal_pairs.height:,}")

    all_pairs = pl.concat([name_pairs, prefix_pairs, addr_pairs, postal_pairs], how="vertical")

    hits = (
        all_pairs.group_by(["entity_id_s1", "entity_id_cand"])
        .agg(pl.col("key_type").n_unique().alias("n_key_hits"))
    )
    if verbose:
        print(f"    union candidate pairs (deduped):    {hits.height:,}")

    if max_candidates_per_entity is not None and hits.height > 0:
        hits = hits.sort(["entity_id_s1", "n_key_hits"], descending=[False, True])
        hits = hits.group_by("entity_id_s1", maintain_order=True).head(max_candidates_per_entity)
        if verbose:
            print(f"    candidate pairs after per-entity cap ({max_candidates_per_entity}): {hits.height:,}")

    result = hits.rename(
        {"entity_id_s1": "source1_entity_id", "entity_id_cand": "candidate_entity_id"}
    ).select(["source1_entity_id", "candidate_entity_id"])
    return result


def generate_candidates(s1_df: pl.DataFrame, cand_df: pl.DataFrame,
                         max_df: int = DEFAULT_MAX_DF,
                         max_candidates_per_entity: int = DEFAULT_MAX_CANDIDATES_PER_ENTITY,
                         verbose: bool = True) -> pl.DataFrame:
    """Single-shot convenience wrapper (build index + query) for small/sample runs.

    `s1_df` and `cand_df` must already carry the normalized helper columns
    from :func:`baseline.data.add_normalized_columns` (name_tok, addr_tok,
    name_prefix, postal, country). For the full-scale train/test runs,
    pipeline.py calls `build_candidate_index` once per candidate source and
    `query_candidates` once per Source-1 batch instead, to bound memory.
    """
    index = build_candidate_index(cand_df.lazy(), max_df=max_df)
    return query_candidates(s1_df, index, max_candidates_per_entity=max_candidates_per_entity,
                             verbose=verbose)


def candidates_to_grouped(candidate_pairs: pl.DataFrame, all_s1_ids) -> pl.DataFrame:
    """Roll candidate pairs up into one row per Source-1 id (id, comma-list).

    Every id in `all_s1_ids` gets exactly one row, empty string when no
    candidates were found - matches the required output-file shape.
    """
    grouped = (
        candidate_pairs.sort("candidate_entity_id")
        .group_by("source1_entity_id")
        .agg(pl.col("candidate_entity_id").unique().sort().str.concat(",").alias("candidate_entity_ids"))
    )
    all_ids_df = pl.DataFrame({"source1_entity_id": list(all_s1_ids)})
    out = all_ids_df.join(grouped, on="source1_entity_id", how="left")
    out = out.with_columns(pl.col("candidate_entity_ids").fill_null(""))
    return out
