"""Cheap, vectorizable normalization for business names and addresses.

No external lookups (geocoding, dictionaries, etc.) are used anywhere here -
everything is a fixed, hand-written rule over the string itself, per the
challenge's fair-play constraint.
"""
import re

import polars as pl

# Common legal-entity suffixes / abbreviations seen across US + India naming
# conventions (README explicitly calls out Inc/Corp/Ltd/Pvt-style variation).
# Stripped from name tokens so "Foo Inc" and "Foo Corporation" block/compare
# the same.
LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "llc", "llp", "lp", "plc",
    "pvt", "private", "pte",
    "gmbh", "sa", "srl", "sarl", "bv", "nv", "ag",
    "group", "holdings", "enterprises", "enterprise",
}

# Small stopword list for business names (connective words that add no
# discriminative signal to a blocking key or a token-overlap feature).
NAME_STOPWORDS = {"the", "and", "of", "for", "a", "an", "&"}

# Address stopwords: generic road/unit vocabulary + directionals that are
# common enough to be useless (and dangerous) as blocking keys.
ADDRESS_STOPWORDS = {
    "street", "st", "road", "rd", "avenue", "ave", "drive", "dr",
    "lane", "ln", "boulevard", "blvd", "court", "ct", "circle", "cir",
    "way", "place", "pl", "terrace", "ter", "highway", "hwy",
    "north", "south", "east", "west", "n", "s", "e", "w",
    "unit", "apt", "apartment", "suite", "ste", "floor", "fl", "near",
    "po", "box", "building", "block", "no", "number",
}

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_POSTAL_RE = re.compile(r"\b\d{5,6}\b")


def normalize_text(s: str) -> str:
    """Lowercase, expand '&', strip punctuation, collapse whitespace.

    Unicode-aware (``\\w`` matches Devanagari letters too), so this is a
    no-op on Hindi-script text beyond whitespace/punctuation cleanup - we
    never transliterate (that would need an external resource).
    """
    if not isinstance(s, str) or not s:
        return ""
    s = s.lower()
    s = s.replace("&", " and ")
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def name_tokens(s: str) -> list:
    """Normalized, suffix/stopword-stripped significant tokens of a name."""
    norm = normalize_text(s)
    if not norm:
        return []
    out = []
    for tok in norm.split(" "):
        if len(tok) < 2:
            continue
        if tok in LEGAL_SUFFIXES or tok in NAME_STOPWORDS:
            continue
        out.append(tok)
    return out


def address_tokens(s: str) -> list:
    """Normalized, stopword-stripped significant address tokens.

    Purely-numeric tokens are dropped here (postal codes are handled
    separately by :func:`extract_postal`, house numbers are too noisy to be
    a useful blocking key on their own).
    """
    norm = normalize_text(s)
    if not norm:
        return []
    out = []
    for tok in norm.split(" "):
        if len(tok) < 3:
            continue
        if tok.isdigit():
            continue
        if tok in ADDRESS_STOPWORDS:
            continue
        out.append(tok)
    return out


def name_prefixes(tokens: list, prefix_len: int = 4) -> list:
    """First-`prefix_len`-chars of each significant name token (deduped).

    A cheap, typo-tolerant extra blocking key: token blocking alone misses
    pairs where a single character is corrupted/substituted mid-token (a
    real pattern in this data, e.g. "Centre" -> "C?ntre", "S" -> "5"
    look-alike OCR-style typos) but usually leaves the token's first few
    characters intact. No external phonetic library needed - just a
    prefix truncation, still O(1) per token.
    """
    out = []
    seen = set()
    for tok in tokens:
        if len(tok) < prefix_len:
            continue
        p = tok[:prefix_len]
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def extract_postal(s: str) -> str:
    """First 5-6 digit run in the raw address string, or '' if none.

    Covers US 5-digit ZIP, India 6-digit PIN, and France's 5-digit postal
    code - a plain digit-run regex, no country-specific branching, so it
    generalizes to the unseen France label without hardcoding.
    """
    if not isinstance(s, str) or not s:
        return ""
    m = _POSTAL_RE.search(s)
    return m.group(0) if m else ""


def normalize_text_expr(col: str) -> pl.Expr:
    """Vectorized (Rust-native) equivalent of :func:`normalize_text`.

    Runs as a polars string expression instead of a per-row Python
    callback - on the ~5-10M row source files here, that is the difference
    between tens of seconds and several minutes, since every row-wise
    Python function call (even via `map_elements`) pays Python call/object
    overhead that a native expression does not.
    """
    return (
        pl.col(col)
        .str.to_lowercase()
        .str.replace_all("&", " and ", literal=True)
        .str.replace_all(r"[^\w\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def name_tokens_expr(norm_expr: pl.Expr) -> pl.Expr:
    """Vectorized equivalent of :func:`name_tokens`, applied to an already-
    normalized text expression (see :func:`normalize_text_expr`)."""
    exclude = sorted(LEGAL_SUFFIXES | NAME_STOPWORDS)
    return norm_expr.str.split(" ").list.eval(
        pl.element().filter(
            (pl.element().str.len_chars() >= 2) & (~pl.element().is_in(exclude))
        )
    )


def address_tokens_expr(norm_expr: pl.Expr) -> pl.Expr:
    """Vectorized equivalent of :func:`address_tokens`."""
    exclude = sorted(ADDRESS_STOPWORDS)
    return norm_expr.str.split(" ").list.eval(
        pl.element().filter(
            (pl.element().str.len_chars() >= 3)
            & (~pl.element().str.contains(r"^\d+$"))
            & (~pl.element().is_in(exclude))
        )
    )


def name_prefixes_expr(name_tok_expr: pl.Expr, prefix_len: int = 4) -> pl.Expr:
    """Vectorized equivalent of :func:`name_prefixes`."""
    return (
        name_tok_expr
        .list.eval(pl.element().filter(pl.element().str.len_chars() >= prefix_len).str.slice(0, prefix_len))
        .list.unique()
    )


def extract_postal_expr(col: str) -> pl.Expr:
    """Vectorized equivalent of :func:`extract_postal`."""
    return pl.col(col).str.extract(r"\b(\d{5,6})\b", 1).fill_null("")
