from pathlib import Path

import polars as pl

from business_entity_resolution.normalize import (
    address_numeric_tokens,
    name_block_key,
    normalize_address,
    normalize_name,
)
from business_entity_resolution.parallel import parallel_map

SOURCE_SCHEMA = {
    "entity_id": pl.Utf8,
    "business_name": pl.Utf8,
    "business_address": pl.Utf8,
    "country": pl.Utf8,
}


def _first_token(s: str) -> str:
    return s.split()[0] if s else ""


def _joined_numeric_tokens(s: str) -> str:
    return "|".join(sorted(address_numeric_tokens(s)))


def read_source(path: str | Path) -> pl.DataFrame:
    df = pl.read_csv(
        path,
        separator="\t",
        schema_overrides=SOURCE_SCHEMA,
        null_values=[""],
        quote_char=None,
    )
    df = df.with_columns(
        pl.col("business_name").fill_null("").alias("business_name"),
        pl.col("business_address").fill_null("").alias("business_address"),
    )

    names = df["business_name"].to_list()
    addrs = df["business_address"].to_list()

    name_norm_list = parallel_map(normalize_name, names)
    addr_norm_list = parallel_map(normalize_address, addrs)
    name_key_list = parallel_map(name_block_key, name_norm_list)
    addr_first_list = parallel_map(_first_token, addr_norm_list)
    addr_num_list = parallel_map(_joined_numeric_tokens, addr_norm_list)

    df = df.with_columns(
        pl.Series("name_norm", name_norm_list),
        pl.Series("addr_norm", addr_norm_list),
        pl.Series("name_key", name_key_list),
        pl.Series("addr_first_token", addr_first_list),
        pl.Series("addr_numeric_tokens", addr_num_list),
    )
    return df


def read_ground_truth(path: str | Path) -> pl.DataFrame:
    return pl.read_csv(
        path,
        separator="\t",
        schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8},
        null_values=[""],
        quote_char=None,
    ).with_columns(pl.col("matched_entity_ids").fill_null(""))
