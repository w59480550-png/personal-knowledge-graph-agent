"""Embedding backends: hosted OpenAI-compatible embeddings, or a deterministic
local hashing embedder that keeps the whole project usable offline."""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from typing import Protocol, Sequence

from .config import Settings
from .llm import LLMClient
from .text import l2_normalize, tokenize


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@lru_cache(maxsize=65536)
def _bucket(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest[:8], "big")
    return value % dim, 1.0 if (value >> 63) & 1 == 0 else -1.0


class HashEmbedder:
    """Hashing (a.k.a. feature-hashing) bag-of-ngrams embedder.

    Not a semantic model, but it is deterministic, dependency free and
    surprisingly effective for personal notes because CJK bigrams + latin
    word tokens capture most of the meaning-bearing terms.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = max(64, int(dim))
        self.name = f"hash-ngram-{self.dim}"

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        counts: dict[str, int] = {}
        for token in tokenize(text):
            counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            index, sign = _bucket(token, self.dim)
            weight = (1.0 + math.log(count)) * (1.0 + min(len(token), 6) / 6)
            vector[index] += sign * weight
        return l2_normalize(vector)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class OpenAIEmbedder:
    """Embeddings from any OpenAI-compatible ``/embeddings`` endpoint."""

    def __init__(self, client: LLMClient, model: str) -> None:
        self.client = client
        self.model = model
        self.name = f"openai:{model}"
        self.dim = 0  # discovered from the first response
        self._expected_dim = client.settings.embedding_dim

    def _check(self, vectors: list[list[float]]) -> list[list[float]]:
        if not vectors:
            return vectors
        dim = len(vectors[0])
        if self.dim == 0:
            self.dim = dim
        elif dim != self.dim:
            raise ValueError(f"embedding dimension changed from {self.dim} to {dim}")
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.client.embed(list(texts), model=self.model)
        return [l2_normalize(v) for v in self._check(vectors)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embedder(settings: Settings, client: LLMClient | None = None) -> Embedder:
    if settings.use_openai_embeddings:
        if not settings.llm_enabled:
            raise ValueError(
                "KGNOTES_EMBEDDING_PROVIDER=openai requires LLM_API_KEY "
                "(or use KGNOTES_EMBEDDING_PROVIDER=hash for offline mode)"
            )
        return OpenAIEmbedder(client or LLMClient(settings), settings.embedding_model)
    return HashEmbedder(settings.embedding_dim)
