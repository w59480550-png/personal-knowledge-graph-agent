"""Small, dependency-free text utilities shared by extraction and retrieval."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from typing import Iterable, Sequence

_CJK = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_WORD_RE = re.compile(rf"[A-Za-z][A-Za-z0-9_+#.-]*|\d+(?:\.\d+)?|[{_CJK}]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])|\n{2,}")
_SPACE_RE = re.compile(r"[ \t\u3000]+")
_PUNCT_RE = re.compile(rf"[^\w{_CJK}#._-]+", re.UNICODE)

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are",
    "was", "were", "be", "been", "at", "by", "from", "as", "it", "its", "this", "that",
    "these", "those", "we", "you", "i", "he", "she", "they", "them", "his", "her",
    "的", "了", "是", "在", "和", "与", "及", "有", "我", "你", "他", "她", "它", "们",
    "这", "那", "一个", "一些", "以及", "并", "被", "把", "对", "中", "上", "下", "也",
    "都", "就", "而", "但", "如果", "因为", "所以", "可以", "这个", "那个", "什么", "如何",
}


def clean_whitespace(text: str) -> str:
    return _SPACE_RE.sub(" ", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def normalize_text(text: str) -> str:
    """NFKC + lowercase + collapsed whitespace; used for matching keys."""
    return clean_whitespace(unicodedata.normalize("NFKC", text)).lower()


def normalize_name(name: str) -> str:
    """Canonical comparison key for entity names (drops punctuation/spacing)."""
    name = unicodedata.normalize("NFKC", clean_whitespace(name)).lower()
    return _PUNCT_RE.sub("", name)


def stable_id(prefix: str, key: str) -> str:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=10).hexdigest()
    return f"{prefix}_{digest}"


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def tokenize(text: str) -> list[str]:
    """CJK unigrams + bigrams and latin word tokens.

    Bigrams give enough specificity for Chinese BM25 without a segmenter
    dependency; unigrams keep recall for rare single-character terms.
    """
    tokens: list[str] = []
    raw = _WORD_RE.findall(unicodedata.normalize("NFKC", text))
    cjk_run: list[str] = []

    def flush() -> None:
        if not cjk_run:
            return
        tokens.extend(cjk_run)
        tokens.extend(a + b for a, b in zip(cjk_run, cjk_run[1:]))
        cjk_run.clear()

    for token in raw:
        if len(token) == 1 and re.match(rf"[{_CJK}]", token):
            cjk_run.append(token)
            continue
        flush()
        low = token.lower()
        if low not in STOPWORDS:
            tokens.append(low)
    flush()
    return [t for t in tokens if len(t) > 0 and t not in STOPWORDS]


def term_counts(text: str) -> Counter:
    return Counter(tokenize(text))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [v / norm for v in vector]


def bm25_scores(
    query_terms: Iterable[str],
    docs: Sequence[tuple[str, str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, float]:
    """Okapi BM25 over ``docs`` = [(doc_id, text)].

    Returns only non-zero scores. Personal knowledge bases are small enough to
    score in memory, which keeps both storage backends equally capable.
    """
    terms = [t for t in dict.fromkeys(query_terms) if t]
    if not terms or not docs:
        return {}
    counts = {doc_id: term_counts(text) for doc_id, text in docs}
    lengths = {doc_id: sum(c.values()) for doc_id, c in counts.items()}
    avg_len = sum(lengths.values()) / max(len(lengths), 1) or 1.0
    doc_freq = {term: sum(1 for c in counts.values() if term in c) for term in terms}
    n_docs = len(docs)
    scores: dict[str, float] = {}
    for doc_id, c in counts.items():
        length = lengths[doc_id]
        score = 0.0
        for term in terms:
            tf = c.get(term, 0)
            if not tf:
                continue
            df = doc_freq[term]
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * length / avg_len))
        if score > 0:
            scores[doc_id] = score
    return scores


def slugify(title: str, max_len: int = 40) -> str:
    ascii_part = re.sub(r"[^a-zA-Z0-9]+", "-", unicodedata.normalize("NFKD", title)).strip("-").lower()
    if ascii_part:
        return ascii_part[:max_len]
    return hashlib.blake2b(title.encode("utf-8"), digest_size=4).hexdigest()


def truncate(text: str, limit: int, suffix: str = "…") -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))].rstrip() + suffix
