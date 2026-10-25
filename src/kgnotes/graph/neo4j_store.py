"""Neo4j graph store.

Graph model (reified relations keep multi-note provenance):

    (:Note)-[:HAS_CHUNK]->(:Chunk)
    (:Entity)-[:MENTIONED_IN {chunk_id}]->(:Note)
    (:Relation)-[:FROM]->(:Entity)
    (:Relation)-[:TO]->(:Entity)
    (:Relation)-[:EVIDENCE_OF]->(:Chunk)

The reified ``Relation`` node stores ``type`` as a property, which keeps
Cypher injection-free and lets one entity pair carry several relation types.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any, Sequence

from ..config import Settings
from ..models import Entity, Relation, merge_entity_type, prefer_display_name
from ..notes import Chunk, Note
from ..text import normalize_name
from .base import GraphStats

log = logging.getLogger(__name__)

_SCHEMA = [
    "CREATE CONSTRAINT note_id IF NOT EXISTS FOR (n:Note) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT relation_id IF NOT EXISTS FOR (r:Relation) REQUIRE r.id IS UNIQUE",
    "CREATE INDEX entity_name_key IF NOT EXISTS FOR (e:Entity) ON (e.name_key)",
    "CREATE INDEX relation_note IF NOT EXISTS FOR (r:Relation) ON (r.note_id)",
    "CREATE INDEX chunk_note IF NOT EXISTS FOR (c:Chunk) ON (c.note_id)",
]


class Neo4jGraphStore:
    backend = "neo4j"

    def __init__(self, settings: Settings) -> None:
        try:
            from neo4j import GraphDatabase  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "Neo4j backend requested but the driver is missing. "
                "Install with: pip install 'kgnotes[neo4j]'"
            ) from exc
        self.settings = settings
        self._driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
        self._driver.verify_connectivity()
        for statement in _SCHEMA:
            self._write(statement)

    # -- helpers ------------------------------------------------------------
    def _run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        with self._driver.session(database=self.settings.neo4j_database) as session:
            return [record.data() for record in session.run(query, **params)]

    def _write(self, query: str, **params: Any) -> list[dict[str, Any]]:
        with self._driver.session(database=self.settings.neo4j_database) as session:
            result = session.run(query, **params)
            records = [record.data() for record in result]
            result.consume()
            return records

    # -- notes / chunks -----------------------------------------------------
    def upsert_note(self, note: Note, chunks: Sequence[Chunk]) -> bool:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        existing = self._run("MATCH (n:Note {id:$id}) RETURN n.id AS id", id=note.id)
        self._write(
            """
            MERGE (n:Note {id:$id})
            SET n.title=$title, n.path=$path, n.content=$content, n.content_hash=$hash,
                n.tags=$tags, n.aliases=$aliases, n.metadata=$metadata,
                n.updated_at=$now, n.created_at=coalesce(n.created_at, $created_at)
            """,
            id=note.id, title=note.title, path=note.path, content=note.content,
            hash=note.content_hash, tags=note.tags, aliases=note.aliases,
            metadata=json.dumps(note.metadata, ensure_ascii=False, default=str),
            now=now, created_at=note.created_at,
        )
        self._write("MATCH (n:Note {id:$id})-[:HAS_CHUNK]->(c:Chunk) DETACH DELETE c", id=note.id)
        self._write("MATCH (n:Note {id:$id})<-[m:MENTIONED_IN]-(:Entity) DELETE m", id=note.id)
        self._write("MATCH (r:Relation {note_id:$id}) DETACH DELETE r", id=note.id)
        if chunks:
            self._write(
                """
                MATCH (n:Note {id:$note_id})
                UNWIND $chunks AS chunk
                CREATE (c:Chunk {id:chunk.id, note_id:chunk.note_id, seq:chunk.seq,
                                 heading:chunk.heading, text:chunk.text})
                CREATE (n)-[:HAS_CHUNK]->(c)
                """,
                note_id=note.id,
                chunks=[c.as_dict() for c in chunks],
            )
        return not existing

    def delete_note(self, note_id: str) -> bool:
        existed = bool(self._run("MATCH (n:Note {id:$id}) RETURN n.id AS id", id=note_id))
        self._write(
            "MATCH (n:Note {id:$id})-[:HAS_CHUNK]->(c:Chunk) DETACH DELETE c",
            id=note_id,
        )
        self._write("MATCH (n:Note {id:$id}) DETACH DELETE n", id=note_id)
        self._write("MATCH (r:Relation {note_id:$id}) DETACH DELETE r", id=note_id)
        return existed

    def get_note(self, note_id: str) -> Note | None:
        rows = self._run("MATCH (n:Note {id:$id}) RETURN n AS n", id=note_id)
        return self._row_to_note(rows[0]["n"]) if rows else None

    def list_notes(self, limit: int = 100, offset: int = 0) -> list[Note]:
        rows = self._run(
            "MATCH (n:Note) RETURN n AS n ORDER BY n.updated_at DESC, n.title SKIP $offset LIMIT $limit",
            offset=offset, limit=limit,
        )
        return [self._row_to_note(r["n"]) for r in rows]

    def get_chunk(self, chunk_id: str) -> tuple[Chunk, Note] | None:
        rows = self._run(
            """
            MATCH (n:Note)-[:HAS_CHUNK]->(c:Chunk {id:$id})
            RETURN c AS c, n AS n
            """,
            id=chunk_id,
        )
        if not rows:
            return None
        return self._row_to_chunk(rows[0]["c"]), self._row_to_note(rows[0]["n"])

    def iter_chunks(self) -> list[tuple[Chunk, Note]]:
        rows = self._run(
            """
            MATCH (n:Note)-[:HAS_CHUNK]->(c:Chunk)
            RETURN c AS c, n AS n ORDER BY c.note_id, c.seq
            """
        )
        return [(self._row_to_chunk(r["c"]), self._row_to_note(r["n"])) for r in rows]

    def add_mentions(self, note_id: str, chunk_id: str, entity_ids: Sequence[str]) -> None:
        if not entity_ids:
            return
        self._write(
            """
            MATCH (n:Note {id:$note_id})
            UNWIND $entity_ids AS eid
            MATCH (e:Entity {id:eid})
            MERGE (e)-[m:MENTIONED_IN {chunk_id:$chunk_id}]->(n)
            """,
            note_id=note_id, chunk_id=chunk_id, entity_ids=list(dict.fromkeys(entity_ids)),
        )

    # -- entities -----------------------------------------------------------
    def _find_entity(self, name: str, entity_type: str = "") -> Entity | None:
        key = normalize_name(name)
        if not key:
            return None
        rows = self._run(
            """
            MATCH (e:Entity)
            WHERE e.name_key = $key OR $key IN coalesce(e.alias_keys, [])
            RETURN e AS e,
                   CASE WHEN e.name_key = $key THEN 0 ELSE 1 END AS name_match,
                   CASE WHEN e.type = $type THEN 0 ELSE 1 END AS type_match
            ORDER BY name_match, type_match, e.created_at
            LIMIT 1
            """,
            key=key, type=entity_type,
        )
        return self._row_to_entity(rows[0]["e"]) if rows else None

    def upsert_entity(self, entity: Entity) -> Entity:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        existing = self._find_entity(entity.name, entity.type)
        if existing is None:
            self._write(
                """
                CREATE (e:Entity {id:$id, name:$name, name_key:$name_key, type:$type,
                                  description:$description, aliases:$aliases, alias_keys:$alias_keys,
                                  note_ids:$note_ids, created_at:$now, updated_at:$now})
                """,
                id=entity.id, name=entity.name, name_key=entity.key, type=entity.type,
                description=entity.description, aliases=entity.aliases,
                alias_keys=[normalize_name(a) for a in entity.aliases if normalize_name(a)],
                note_ids=entity.note_ids, now=now,
            )
            return Entity(
                name=entity.name, type=entity.type, description=entity.description,
                aliases=list(entity.aliases), note_ids=list(entity.note_ids), id=entity.id,
            )
        display_name = prefer_display_name(existing.name, entity.name)
        aliases = list(existing.aliases)
        if display_name != existing.name and existing.name not in aliases:
            aliases.append(existing.name)
        for alias in [entity.name, *entity.aliases]:
            if normalize_name(alias) != normalize_name(display_name) and alias not in aliases:
                aliases.append(alias)
        aliases = [a for a in aliases if normalize_name(a) != normalize_name(display_name)]
        note_ids = list(dict.fromkeys(existing.note_ids + entity.note_ids))
        description = existing.description if len(existing.description) >= len(entity.description) else entity.description
        entity_type = merge_entity_type(existing.type, entity.type)
        self._write(
            """
            MATCH (e:Entity {id:$id})
            SET e.name=$name, e.name_key=$name_key, e.type=$type, e.description=$description,
                e.aliases=$aliases, e.alias_keys=$alias_keys, e.note_ids=$note_ids, e.updated_at=$now
            """,
            id=existing.id, name=display_name, name_key=normalize_name(display_name),
            type=entity_type, description=description, aliases=aliases,
            alias_keys=[normalize_name(a) for a in [display_name, *aliases] if normalize_name(a)],
            note_ids=note_ids, now=now,
        )
        return Entity(
            name=display_name, type=entity_type, description=description,
            aliases=aliases, note_ids=note_ids, id=existing.id,
        )


    def upsert_relation(self, relation: Relation) -> Relation:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        endpoints = self._run(
            "MATCH (e:Entity) WHERE e.id IN $ids RETURN e.id AS id",
            ids=[relation.source_id, relation.target_id],
        )
        if len({row["id"] for row in endpoints}) < 2:
            raise ValueError("relation endpoints not found")
        self._write(
            """
            MERGE (r:Relation {id:$id})
            SET r.type=$type, r.note_id=$note_id, r.chunk_id=$chunk_id, r.created_at=coalesce(r.created_at, $now),
                r.confidence=CASE WHEN r.confidence IS NULL THEN $confidence
                                  ELSE CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END END,
                r.evidence=CASE WHEN size($evidence) > size(coalesce(r.evidence, '')) THEN $evidence ELSE coalesce(r.evidence, '') END,
                r.description=CASE WHEN size($description) > size(coalesce(r.description, '')) THEN $description ELSE coalesce(r.description, '') END
            WITH r
            MATCH (s:Entity {id:$source_id})
            MATCH (t:Entity {id:$target_id})
            MERGE (r)-[:FROM]->(s)
            MERGE (r)-[:TO]->(t)
            """,
            id=relation.id, type=relation.type, note_id=relation.note_id,
            chunk_id=relation.chunk_id, now=now, confidence=relation.confidence,
            evidence=relation.evidence, description=relation.description,
            source_id=relation.source_id, target_id=relation.target_id,
        )
        if relation.chunk_id:
            self._write(
                """
                MATCH (r:Relation {id:$id}), (c:Chunk {id:$chunk_id})
                MERGE (r)-[:EVIDENCE_OF]->(c)
                """,
                id=relation.id, chunk_id=relation.chunk_id,
            )
        return relation

    def get_entity(self, entity_id: str | None = None, name: str | None = None) -> Entity | None:
        if entity_id:
            rows = self._run("MATCH (e:Entity {id:$id}) RETURN e AS e", id=entity_id)
            return self._row_to_entity(rows[0]["e"]) if rows else None
        if name:
            return self._find_entity(name)
        return None

    def match_entities(self, texts: Sequence[str]) -> list[Entity]:
        found: dict[str, Entity] = {}
        for text in texts:
            entity = self._find_entity(text.strip()) if text and text.strip() else None
            if entity:
                found[entity.id] = entity
        return list(found.values())

    def list_entities(
        self, query: str | None = None, entity_type: str | None = None, limit: int = 100
    ) -> list[Entity]:
        rows = self._run(
            """
            MATCH (e:Entity)
            WHERE ($query IS NULL
                   OR toLower(e.name) CONTAINS toLower($query)
                   OR e.name_key CONTAINS $key
                   OR toLower(coalesce(e.description, '')) CONTAINS toLower($query)
                   OR ANY(a IN coalesce(e.aliases, []) WHERE toLower(a) CONTAINS toLower($query)))
              AND ($type IS NULL OR e.type = $type)
            RETURN e AS e
            ORDER BY e.name
            LIMIT $limit
            """,
            query=query, key=normalize_name(query or ""), type=entity_type, limit=max(1, limit),
        )
        return [self._row_to_entity(row["e"]) for row in rows]

    # -- graph traversal ----------------------------------------------------
    def relations_for(self, entity_ids: Sequence[str], limit: int = 100) -> list[Relation]:
        ids = list(dict.fromkeys(entity_ids))
        if not ids:
            return []
        rows = self._run(
            """
            MATCH (r:Relation)-[:FROM]->(s:Entity)
            MATCH (r)-[:TO]->(t:Entity)
            WHERE s.id IN $ids OR t.id IN $ids
            RETURN r AS r, s.id AS source_id, s.name AS source_name,
                   t.id AS target_id, t.name AS target_name
            ORDER BY r.confidence DESC
            LIMIT $limit
            """,
            ids=ids, limit=limit,
        )
        return [self._row_to_relation(row) for row in rows]

    def neighbors(self, entity_id: str, depth: int = 1, limit: int = 50) -> tuple[list[Entity], list[Relation]]:
        depth = max(1, min(int(depth), 3))
        entities: dict[str, Entity] = {}
        relations: dict[str, Relation] = {}
        frontier: set[str] = {entity_id}
        visited: set[str] = {entity_id}
        for _ in range(depth):
            rels = self.relations_for(sorted(frontier), limit=limit * 3)
            next_frontier: set[str] = set()
            for rel in rels:
                if len(relations) >= limit * 3:
                    break
                relations[rel.id] = rel
                for eid in (rel.source_id, rel.target_id):
                    if eid not in visited:
                        next_frontier.add(eid)
                    if eid not in entities and len(entities) < limit:
                        entity = self.get_entity(entity_id=eid)
                        if entity:
                            entities[eid] = entity
            visited |= next_frontier
            frontier = next_frontier
            if not frontier:
                break
        root = self.get_entity(entity_id=entity_id)
        return ([root] if root else []) + list(entities.values()), list(relations.values())

    def stats(self) -> GraphStats:
        rows = self._run(
            """
            RETURN COUNT { MATCH (n:Note) } AS notes,
                   COUNT { MATCH (c:Chunk) } AS chunks,
                   COUNT { MATCH (e:Entity) } AS entities,
                   COUNT { MATCH (r:Relation) } AS relations
            """
        )
        row = rows[0] if rows else {}
        entity_types = {
            r["type"]: r["count"]
            for r in self._run("MATCH (e:Entity) RETURN e.type AS type, count(*) AS count")
        }
        relation_types = {
            r["type"]: r["count"]
            for r in self._run("MATCH (r:Relation) RETURN r.type AS type, count(*) AS count")
        }
        return GraphStats(
            notes=row.get("notes", 0), chunks=row.get("chunks", 0),
            entities=row.get("entities", 0), relations=row.get("relations", 0),
            backend=self.backend, entity_types=entity_types, relation_types=relation_types,
        )

    def export_graph(self, limit: int = 5000) -> dict:
        entities = self._run("MATCH (e:Entity) RETURN e AS e ORDER BY e.name LIMIT $limit", limit=limit)
        relations = self._run(
            """
            MATCH (r:Relation)-[:FROM]->(s:Entity)
            MATCH (r)-[:TO]->(t:Entity)
            RETURN r AS r, s.id AS source_id, s.name AS source_name,
                   t.id AS target_id, t.name AS target_name
            LIMIT $limit
            """,
            limit=limit,
        )
        return {
            "entities": [self._row_to_entity(row["e"]).as_dict() for row in entities],
            "relations": [self._row_to_relation(row).as_dict() for row in relations],
        }

    def reset(self) -> None:
        self._write("MATCH (n) DETACH DELETE n")

    def close(self) -> None:
        self._driver.close()

    # -- mappers ------------------------------------------------------------
    @staticmethod
    def _node_to_dict(node: Any) -> dict[str, Any]:
        return dict(node)

    @classmethod
    def _row_to_note(cls, node: Any) -> Note:
        data = cls._node_to_dict(node)
        metadata = data.get("metadata") or "{}"
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        return Note(
            title=data.get("title", ""), content=data.get("content", ""),
            path=data.get("path", ""), tags=list(data.get("tags") or []),
            aliases=list(data.get("aliases") or []), metadata=metadata,
            created_at=data.get("created_at", ""), id=data.get("id", ""),
        )

    @staticmethod
    def _row_to_chunk(node: Any) -> Chunk:
        data = dict(node)
        return Chunk(
            id=data["id"], note_id=data["note_id"], seq=int(data.get("seq", 0)),
            text=data.get("text", ""), heading=data.get("heading", ""),
        )

    @staticmethod
    def _row_to_entity(node: Any) -> Entity:
        data = dict(node)
        return Entity(
            name=data.get("name", ""), type=data.get("type", "CONCEPT"),
            description=data.get("description", ""), aliases=list(data.get("aliases") or []),
            note_ids=list(data.get("note_ids") or []), id=data.get("id", ""),
        )

    @classmethod
    def _row_to_relation(cls, row: dict[str, Any]) -> Relation:
        data = cls._node_to_dict(row["r"])
        return Relation(
            source_id=row.get("source_id", data.get("source_id", "")),
            target_id=row.get("target_id", data.get("target_id", "")),
            type=data.get("type", "RELATED_TO"),
            source_name=row.get("source_name", ""), target_name=row.get("target_name", ""),
            description=data.get("description", ""), evidence=data.get("evidence", ""),
            confidence=float(data.get("confidence") or 0.6), note_id=data.get("note_id", ""),
            chunk_id=data.get("chunk_id", ""),
        )
