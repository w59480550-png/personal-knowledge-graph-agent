from __future__ import annotations

import pytest

from kgnotes.embeddings import HashEmbedder, build_embedder
from kgnotes.text import l2_normalize
from kgnotes.vector import SqliteVectorIndex, VectorIndexMismatch


def test_hash_embedder_is_deterministic_and_normalised():
    embedder = HashEmbedder(128)
    a = embedder.embed_query("知识图谱与向量检索")
    b = embedder.embed_query("知识图谱与向量检索")
    assert a == b
    assert len(a) == 128
    assert pytest.approx(sum(x * x for x in a), rel=1e-6) == 1.0


def test_hash_embedder_ranks_related_text_higher():
    embedder = HashEmbedder(256)
    query = embedder.embed_query("Neo4j 图数据库")
    related = embedder.embed_query("Neo4j 是一个图数据库，支持 Cypher。")
    unrelated = embedder.embed_query("今天午餐吃了牛肉面")
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731
    assert dot(query, related) > dot(query, unrelated)


def test_sqlite_vector_index_add_search_delete(tmp_path):
    index = SqliteVectorIndex(tmp_path / "v.sqlite", "test", dim=4)
    index.add([
        ("a", l2_normalize([1, 0, 0, 0])),
        ("b", l2_normalize([0, 1, 0, 0])),
        ("c", l2_normalize([0.9, 0.1, 0, 0])),
    ])
    assert index.count() == 3
    hits = index.search([1, 0, 0, 0], limit=2)
    assert [hit[0] for hit in hits] == ["a", "c"]
    assert index.get("a") is not None
    assert index.delete(["a"]) == 1
    assert index.count() == 2
    assert [hit[0] for hit in index.search([1, 0, 0, 0], limit=2)] == ["c"]
    index.close()


def test_sqlite_vector_index_persists(tmp_path):
    path = tmp_path / "v.sqlite"
    index = SqliteVectorIndex(path, "test", dim=3)
    index.add([("a", [1.0, 0.0, 0.0])])
    index.close()
    reopened = SqliteVectorIndex(path, "test", dim=3)
    assert reopened.count() == 1
    assert reopened.search([1.0, 0.0, 0.0], limit=1)[0][0] == "a"
    reopened.close()


def test_dimension_mismatch_raises(tmp_path):
    index = SqliteVectorIndex(tmp_path / "v.sqlite", "test", dim=3)
    with pytest.raises(VectorIndexMismatch):
        index.add([("a", [1.0, 0.0])])
    index.close()


def test_build_embedder_offline(settings):
    embedder = build_embedder(settings)
    assert isinstance(embedder, HashEmbedder)
    assert embedder.dim == settings.embedding_dim
