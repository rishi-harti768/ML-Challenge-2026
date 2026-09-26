import re
import unicodedata

_SUFFIX_CANON = {
    "corp": "corp", "corporation": "corp",
    "inc": "inc", "incorporated": "inc",
    "ltd": "ltd", "limited": "ltd",
    "llc": "llc", "llp": "llp", "plc": "plc",
    "co": "co", "company": "co", "companies": "co",
    "pvt": "pvt", "private": "pvt",
    "gmbh": "gmbh",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl",
    "and": "and", "&": "and",
}

_ADDR_ABBR = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
    "apt": "apartment", "bldg": "building", "hwy": "highway",
    "pkwy": "parkway", "sq": "square", "ter": "terrace", "pl": "place",
}

_WORD_RE = re.compile(r"[\w]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text or "").lower()
    return _WORD_RE.findall(text)


def normalize_name(name: str) -> str:
    tokens = [_SUFFIX_CANON.get(t, t) for t in _tokenize(name)]
    return " ".join(tokens)


def name_block_key(name_norm: str) -> str:
    return " ".join(sorted(name_norm.split()))


def normalize_address(address: str) -> str:
    tokens = [_ADDR_ABBR.get(t, t) for t in _tokenize(address)]
    return " ".join(tokens)


def address_numeric_tokens(addr_norm: str) -> frozenset[str]:
    return frozenset(t for t in addr_norm.split() if any(c.isdigit() for c in t))
