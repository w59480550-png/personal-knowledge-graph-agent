from __future__ import annotations

from kgnotes.graph.sqlite_store import SQLiteGraphStore
from kgnotes.models import Entity, Relation
from kgnotes.notes import Note, NoteLoader


def _note_and_chunks(tmp_path, title="测试笔记", content="Neo4j 是一个图数据库。", filename="a.md"):
    loader = NoteLoader()
    note = loader.from_text(content, title=title, path=filename, tags=["kg"])
    return note, loader.chunk(note)


def test_note_chunk_roundtrip(tmp_path):
    store = SQLiteGraphStore(tmp_path / "g.sqlite")
    note, chunks = _note_and_chunks(tmp_path)
    assert store.upsert_note(note, chunks) is True
    assert store.upsert_note(note, chunks) is False  # update, not create

    loaded = store.get_note(note.id)
    assert loaded is not None and loaded.title == note.title and loaded.tags == ["kg"]
    pairs = store.iter_chunks()
    assert len(pairs) == len(chunks)
    chunk, parent = pairs[0]
    assert parent.id == note.id and chunk.text.startswith("Neo4j")
    assert store.get_chunk(chunk.id) is not None
    store.close()


def test_entity_resolution_merges_alias_and_upgrades_type(tmp_path):
    store = SQLiteGraphStore(tmp_path / "g.sqlite")

    async_entity = store.upsert_entity(Entity(name="Neo4j", type="TAG", note_ids=["n1"]))
    upgraded = store.upsert_entity(Entity(name="Neo4j", type="TECHNOLOGY", note_ids=["n2"]))
    assert upgraded.id == async_entity.id
    assert upgraded.type == "TECHNOLOGY"
    assert set(upgraded.note_ids) == {"n1", "n2"}

    project = store.upsert_entity(
        Entity(name="知识图谱助手项目", type="PROJECT", aliases=["kgnotes 项目"])
    )
    by_alias = store.upsert_entity(Entity(name="kgnotes 项目", type="CONCEPT"))
    assert by_alias.id == project.id
    assert "kgnotes 项目" in by_alias.aliases
    assert store.stats().entities == 2
    assert store.get_entity(name="neo4j").id == async_entity.id
    assert store.match_entities(["Neo4j"])[0].id == async_entity.id
    store.close()


def test_display_name_prefers_capitalisation(tmp_path):
    store = SQLiteGraphStore(tmp_path / "g.sqlite")
    store.upsert_entity(Entity(name="graphrag", type="TAG"))
    merged = store.upsert_entity(Entity(name="GraphRAG", type="CONCEPT"))
    assert merged.name == "GraphRAG"
    assert merged.type == "CONCEPT"
    assert store.get_entity(name="graphrag").id == merged.id
    store.close()


def test_relations_are_idempotent_and_traversable(tmp_path):
    store = SQLiteGraphStore(tmp_path / "g.sqlite")
    note, chunks = _note_and_chunks(tmp_path)
    store.upsert_note(note, chunks)

    a = store.upsert_entity(Entity(name="A", type="TECHNOLOGY"))
    b = store.upsert_entity(Entity(name="B", type="TECHNOLOGY"))
    c = store.upsert_entity(Entity(name="C", type="TECHNOLOGY"))
    rel = Relation(
        source_id=a.id, target_id=b.id, type="USES", source_name="A", target_name="B",
        note_id=note.id, chunk_id=chunks[0].id, confidence=0.7, evidence="A 使用 B。",
    )
    store.upsert_relation(rel)
    store.upsert_relation(rel)
    assert store.stats().relations == 1

    store.upsert_relation(Relation(
        source_id=b.id, target_id=c.id, type="USES", source_name="B", target_name="C",
        note_id=note.id, confidence=0.6,
    ))
    entities, relations = store.neighbors(a.id, depth=2, limit=10)
    names = {entity.name for entity in entities}
    assert {"A", "B", "C"} <= names
    assert len(relations) == 2
    assert {r.type for r in relations} == {"USES"}
    store.close()


def test_delete_note_cascades_but_keeps_shared_entities(tmp_path):
    store = SQLiteGraphStore(tmp_path / "g.sqlite")
    note, chunks = _note_and_chunks(tmp_path)
    store.upsert_note(note, chunks)
    entity = store.upsert_entity(Entity(name="Neo4j", type="TECHNOLOGY", note_ids=[note.id]))
    other = store.upsert_entity(Entity(name="Cypher", type="TECHNOLOGY", note_ids=[note.id]))
    store.add_mentions(note.id, chunks[0].id, [entity.id, other.id])
    store.upsert_relation(Relation(
        source_id=entity.id, target_id=other.id, type="USES", note_id=note.id,
    ))

    assert store.delete_note(note.id) is True
    assert store.get_note(note.id) is None
    assert store.iter_chunks() == []
    assert store.stats().relations == 0
    assert store.stats().entities == 2  # entities are global, not note-owned
    assert store.delete_note("missing") is False
    store.close()


def test_upsert_relation_requires_endpoints(tmp_path):
    store = SQLiteGraphStore(tmp_path / "g.sqlite")
    try:
        store.upsert_relation(Relation(source_id="x", target_id="y", type="USES"))
    except ValueError as exc:
        assert "endpoints" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
    store.close()
