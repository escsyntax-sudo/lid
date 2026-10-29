from __future__ import annotations

import hashlib
import math
import re
import struct
from typing import Dict, Iterable, List, Sequence, Tuple

MODEL_NAME = "hashed-tfidf-v1"
DEFAULT_DIM = 512

TOKEN_RE = re.compile(r"\w+", re.UNICODE)

STOPWORDS = frozenset(
    """
    a an and are as at be been being but by can could did do does doing for from had has have
    having he her here hers herself him himself his how i if in into is it its itself just me
    more most my myself no nor not now of off on once only or other our ours ourselves out over
    own same she should so some such than that the their theirs them themselves then there these
    they this those through to too under until up very was we were what when where which while
    who whom why will with would you your yours yourself yourselves
    """.split()
)


def tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for match in TOKEN_RE.finditer(text.lower()):
        token = match.group(0).strip("_")
        if not token:
            continue
        if token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def term_counts(text: str, heading: str = "", heading_boost: int = 2) -> Dict[str, int]:
    counts: Dict[str, int] = {}

    def add(tokens: Iterable[str], weight: int) -> None:
        previous = None
        for token in tokens:
            counts[token] = counts.get(token, 0) + weight
            if previous is not None:
                bigram = previous + " " + token
                counts[bigram] = counts.get(bigram, 0) + weight
            previous = token

    add(tokenize(heading), heading_boost)
    add(tokenize(text), 1)
    return counts


def _hash_slot(term: str, dim: int) -> Tuple[int, float]:
    digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
    number = int.from_bytes(digest, "big")
    slot = number % dim
    sign = 1.0 if (number >> 63) & 1 else -1.0
    return slot, sign


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    sumprod = getattr(math, "sumprod", None)
    if sumprod is not None:
        return float(sumprod(left, right))
    total = 0.0
    for a, b in zip(left, right):
        total += a * b
    return total


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    return _dot(left, right)


class TfidfModel:
    """Feature hashing with corpus IDF and L2 normalisation."""

    name = MODEL_NAME

    def __init__(self, dim: int = DEFAULT_DIM):
        if dim < 16:
            raise ValueError("dimension must be at least 16")
        self.dim = dim
        self.document_count = 0
        self.idf: Dict[str, float] = {}

    def fit(self, term_dicts: Iterable[Dict[str, int]]) -> "TfidfModel":
        document_frequency: Dict[str, int] = {}
        count = 0
        for terms in term_dicts:
            count += 1
            for term in terms:
                document_frequency[term] = document_frequency.get(term, 0) + 1
        self.document_count = count
        self.idf = {
            term: math.log((count + 1.0) / (frequency + 1.0)) + 1.0
            for term, frequency in document_frequency.items()
        }
        return self

    def _default_idf(self) -> float:
        return math.log(self.document_count + 1.0) + 1.0

    def vector(self, terms: Dict[str, int]) -> List[float]:
        vector = [0.0] * self.dim
        for term, count in terms.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            weight = 1.0 + math.log(count) if count > 0 else 0.0
            weight *= idf
            slot, sign = _hash_slot(term, self.dim)
            vector[slot] += sign * weight
        norm = math.sqrt(_dot(vector, vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        return vector

    def vector_from_text(self, text: str, heading: str = "") -> List[float]:
        return self.vector(term_counts(text, heading))

    def to_bytes(self, vector: Sequence[float]) -> bytes:
        return struct.pack(f"<{self.dim}f", *vector)

    def from_bytes(self, blob: bytes) -> Tuple[float, ...]:
        expected = self.dim * 4
        if len(blob) != expected:
            raise ValueError(f"vector has {len(blob)} bytes, expected {expected}")
        return struct.unpack(f"<{self.dim}f", blob)

    def similarity(self, query: Sequence[float], blob: bytes) -> float:
        return _dot(query, self.from_bytes(blob))
