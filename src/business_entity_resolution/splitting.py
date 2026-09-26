import random

import polars as pl


def random_holdout_split(s1: pl.DataFrame, frac: float = 0.2, seed: int = 0) -> tuple[set[str], set[str]]:
    ids = s1["entity_id"].to_list()
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * (1 - frac))
    return set(shuffled[:cut]), set(shuffled[cut:])


def country_holdout_split(s1: pl.DataFrame, held_out_country: str) -> tuple[set[str], set[str]]:
    train_ids = set(s1.filter(pl.col("country") != held_out_country)["entity_id"].to_list())
    val_ids = set(s1.filter(pl.col("country") == held_out_country)["entity_id"].to_list())
    return train_ids, val_ids
