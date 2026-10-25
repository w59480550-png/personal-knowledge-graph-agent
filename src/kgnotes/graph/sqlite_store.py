"""SQLite-backed graph store: the zero-setup default.

Its schema mirrors the Neo4j property-graph model (Note, Chunk, Entity,
Relation) so switching backends does not change retrieval behaviour.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
import threading
from pathlib import Path
from typing import Sequence

from ..models import Entity, Relation, merge_entity_type, prefer_display_name
from ..notes import Chunk, Note
from ..text import normalize_name
from .base import GraphStats

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    path TEXT DEFAULT '',
    content TEXT NOT NULL,
    content_hash TEXT DEFAULT '',
    tags TEXT DEFAULT '[]',
    aliases TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    created_at TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    note_id TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    heading TEXT DEFAULT '',
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_note ON chunks(note_id, seq);
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    name_key TEXT NOT NULL,
    type TEXT NOT NULL,
    description TEXT DEFAULT '',
    aliases TEXT DEFAULT '[]',
    note_ids TEXT DEFAULT '[]',
    created_at TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_entities_key ON entities(name_key);
CREATE TABLE IF NOT EXISTS entity_aliases (
    alias_key TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_aliases_entity ON entity_aliases(entity_id);
CREATE TABLE IF NOT EXISTS relations (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    type TEXT NOT NULL,
    description TEXT DEFAULT '',
    evidence TEXT DEFAULT '',
    confidence REAL DEFAULT 0.6,
    note_id TEXT DEFAULT '',
    chunk_id TEXT DEFAULT '',
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_rel_source ON relations(source_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relations(target_id);
CREATE INDEX IF NOT EXISTS idx_rel_note ON relations(note_id);
CREATE TABLE IF NOT EXISTS mentions (
    note_id TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    chunk_id TEXT NOT NULL,
    entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    count INTEGER DEFAULT 1,
    PRIMARY KEY (note_id, chunk_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_mentions_entity ON mentions(entity_id);
"""


class SQLiteGraphStore:
    backend = "sqlite"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- notes / chunks -----------------------------------------------------
    def upsert_note(self, note: Note, chunks: Sequence[Chunk]) -> bool:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        with self._lock:
            exists = self._conn.execute("SELECT 1 FROM notes WHERE id=?", (note.id,)).fetchone() is not None
            self._conn.execute(
                """
                INSERT INTO notes(id,title,path,content,content_hash,tags,aliases,metadata,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title, path=excluded.path, content=excluded.content,
                    content_hash=excluded.content_hash, tags=excluded.tags,
                    aliases=excluded.aliases, metadata=excluded.metadata, updated_at=excluded.updated_at
                """,
                (
                    note.id, note.title, note.path, note.content, note.content_hash,
                    json.dumps(note.tags, ensure_ascii=False),
                    json.dumps(note.aliases, ensure_ascii=False),
                    json.dumps(note.metadata, ensure_ascii=False, default=str),
                    note.created_at, now,
                ),
            )
            self._conn.execute("DELETE FROM chunks WHERE note_id=?", (note.id,))
            self._conn.execute("DELETE FROM mentions WHERE note_id=?", (note.id,))
            self._conn.execute("DELETE FROM relations WHERE note_id=?", (note.id,))
            self._conn.executemany(
                "INSERT INTO chunks(id,note_id,seq,heading,text) VALUES (?,?,?,?,?)",
                [(c.id, c.note_id, c.seq, c.heading, c.text) for c in chunks],
            )
            self._conn.commit()
        return not exists

    def delete_note(self, note_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM notes WHERE id=?", (note_id,))
            self._conn.execute("DELETE FROM relations WHERE note_id=?", (note_id,))
            self._conn.execute("DELETE FROM mentions WHERE note_id=?", (note_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def get_note(self, note_id: str) -> Note | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM notes WHERE id=?", (note_id,)).fetchone()
        return self._row_to_note(row) if row else None

    def list_notes(self, limit: int = 100, offset: int = 0) -> list[Note]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM notes ORDER BY updated_at DESC, title LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
        return [self._row_to_note(row) for row in rows]

    def get_chunk(self, chunk_id: str) -> tuple[Chunk, Note] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM chunks WHERE id=?", (chunk_id,)).fetchone()
        if not row:
            return None
        note = self.get_note(row["note_id"])
        return (self._row_to_chunk(row), note) if note else None

    def iter_chunks(self) -> list[tuple[Chunk, Note]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM chunks ORDER BY note_id, seq").fetchall()
        notes: dict[str, Note] = {}
        out: list[tuple[Chunk, Note]] = []
        for row in rows:
            note = notes.get(row["note_id"])
            if note is None:
                note = self.get_note(row["note_id"])
                if note is None:
                    continue
                notes[note.id] = note
            out.append((self._row_to_chunk(row), note))
        return out

    def add_mentions(self, note_id: str, chunk_id: str, entity_ids: Sequence[str]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT OR IGNORE INTO mentions(note_id,chunk_id,entity_id,count) VALUES (?,?,?,1)",
                [(note_id, chunk_id, eid) for eid in dict.fromkeys(entity_ids)],
            )
            self._conn.commit()

    # -- entities -----------------------------------------------------------
    def _find_entity_row(self, name: str, entity_type: str = "") -> sqlite3.Row | None:
        key = normalize_name(name)
        if not key:
            return None
        with self._lock:
            return self._conn.execute(
                """
                SELECT e.*,
                       CASE WHEN e.name_key = :key THEN 1 ELSE 0 END AS name_match,
                       CASE WHEN e.type = :type THEN 1 ELSE 0 END AS type_match
                FROM entities e
                LEFT JOIN entity_aliases a ON a.entity_id = e.id
                WHERE e.name_key = :key OR a.alias_key = :key
                ORDER BY name_match DESC, type_match DESC, e.created_at
                LIMIT 1
                """,
                {"key": key, "type": entity_type},
            ).fetchone()

    def upsert_entity(self, entity: Entity) -> Entity:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        with self._lock:
            row = self._find_entity_row(entity.name, entity.type)
            if row is None:
                self._conn.execute(
                    """
                    INSERT INTO entities(id,name,name_key,type,description,aliases,note_ids,created_at,updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        entity.id, entity.name, entity.key, entity.type, entity.description,
                        json.dumps(entity.aliases, ensure_ascii=False),
                        json.dumps(entity.note_ids, ensure_ascii=False), now, now,
                    ),
                )
                resolved = Entity(
                    name=entity.name, type=entity.type, description=entity.description,
                    aliases=list(entity.aliases), note_ids=list(entity.note_ids), id=entity.id,
                )
            else:
                aliases = list(json.loads(row["aliases"] or "[]"))
                display_name = prefer_display_name(row["name"], entity.name)
                if display_name != row["name"] and row["name"] not in aliases:
                    aliases.append(row["name"])
                for alias in [entity.name, *entity.aliases]:
                    if normalize_name(alias) != normalize_name(display_name) and alias not in aliases:
                        aliases.append(alias)
                aliases = [
                    alias for alias in aliases
                    if normalize_name(alias) != normalize_name(display_name)
                ]
                note_ids = list(dict.fromkeys(json.loads(row["note_ids"] or "[]") + entity.note_ids))
                description = row["description"] or entity.description
                if len(entity.description) > len(description):
                    description = entity.description
                entity_type = merge_entity_type(row["type"], entity.type)
                self._conn.execute(
                    """
                    UPDATE entities SET name=?, name_key=?, type=?, description=?, aliases=?, note_ids=?, updated_at=?
                    WHERE id=?
                    """,
                    (display_name, normalize_name(display_name), entity_type, description,
                     json.dumps(aliases, ensure_ascii=False),
                     json.dumps(note_ids, ensure_ascii=False), now, row["id"]),
                )
                resolved = Entity(
                    name=display_name, type=entity_type, description=description,
                    aliases=aliases, note_ids=note_ids, id=row["id"],
                )
            for alias in [resolved.name, *resolved.aliases]:
                alias_key = normalize_name(alias)
                if alias_key:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO entity_aliases(alias_key, entity_id) VALUES (?,?)",
                        (alias_key, resolved.id),
                    )
            self._conn.commit()
        return resolved


    def upsert_relation(self, relation: Relation) -> Relation:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        with self._lock:
            missing = [
                eid for eid in (relation.source_id, relation.target_id)
                if not self._conn.execute("SELECT 1 FROM entities WHERE id=?", (eid,)).fetchone()
            ]
            if missing:
                raise ValueError(f"relation endpoints not found: {missing}")
            self._conn.execute(
                """
                INSERT INTO relations(id,source_id,target_id,type,description,evidence,confidence,note_id,chunk_id,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    confidence=MAX(confidence, excluded.confidence),
                    evidence=CASE WHEN LENGTH(excluded.evidence) > LENGTH(evidence)
                                  THEN excluded.evidence ELSE evidence END,
                    description=CASE WHEN LENGTH(excluded.description) > LENGTH(description)
                                     THEN excluded.description ELSE description END
                """,
                (
                    relation.id, relation.source_id, relation.target_id, relation.type,
                    relation.description, relation.evidence, relation.confidence,
                    relation.note_id, relation.chunk_id, now,
                ),
            )
            self._conn.commit()
        return relation

    def get_entity(self, entity_id: str | None = None, name: str | None = None) -> Entity | None:
        if entity_id:
            with self._lock:
                row = self._conn.execute("SELECT * FROM entities WHERE id=?", (entity_id,)).fetchone()
            return self._row_to_entity(row) if row else None
        if name:
            row = self._find_entity_row(name)
            return self._row_to_entity(row) if row else None
        return None

    def match_entities(self, texts: Sequence[str]) -> list[Entity]:
        found: dict[str, Entity] = {}
        for text in texts:
            if not text or not text.strip():
                continue
            row = self._find_entity_row(text.strip())
            if row:
                entity = self._row_to_entity(row)
                found[entity.id] = entity
        return list(found.values())

    def list_entities(
        self, query: str | None = None, entity_type: str | None = None, limit: int = 100
    ) -> list[Entity]:
        sql = "SELECT * FROM entities"
        clauses: list[str] = []
        params: list[object] = []
        if query:
            clauses.append("(name LIKE ? OR name_key LIKE ? OR description LIKE ? OR aliases LIKE ?)")
            like = f"%{query.strip()}%"
            params.extend([like, f"%{normalize_name(query)}%", like, like])
        if entity_type:
            clauses.append("type = ?")
            params.append(entity_type.upper())
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY name LIMIT ?"
        params.append(max(1, limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_entity(row) for row in rows]

    # -- graph traversal ----------------------------------------------------
    def _relations_touching(self, entity_ids: Sequence[str], limit: int = 200) -> list[Relation]:
        ids = list(dict.fromkeys(entity_ids))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        sql = f"""
            SELECT r.*, s.name AS source_name, t.name AS target_name
            FROM relations r
            JOIN entities s ON s.id = r.source_id
            JOIN entities t ON t.id = r.target_id
            WHERE r.source_id IN ({placeholders}) OR r.target_id IN ({placeholders})
            ORDER BY r.confidence DESC
            LIMIT ?
        """
        with self._lock:
            rows = self._conn.execute(sql, [*ids, *ids, limit]).fetchall()
        return [self._row_to_relation(row) for row in rows]

    def relations_for(self, entity_ids: Sequence[str], limit: int = 100) -> list[Relation]:
        return self._relations_touching(entity_ids, limit)

    def neighbors(self, entity_id: str, depth: int = 1, limit: int = 50) -> tuple[list[Entity], list[Relation]]:
        depth = max(1, min(int(depth), 3))
        entities: dict[str, Entity] = {}
        relations: dict[str, Relation] = {}
        frontier = {entity_id}
        visited = {entity_id}
        for _ in range(depth):
            rels = self._relations_touching(sorted(frontier), limit=limit * 3)
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
        with self._lock:
            notes = self._conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
            chunks = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            entities = self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            relations = self._conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
            entity_types = dict(self._conn.execute("SELECT type, COUNT(*) FROM entities GROUP BY type"))
            relation_types = dict(self._conn.execute("SELECT type, COUNT(*) FROM relations GROUP BY type"))
        return GraphStats(
            notes=notes, chunks=chunks, entities=entities, relations=relations,
            backend=self.backend, entity_types=entity_types, relation_types=relation_types,
        )

    def export_graph(self, limit: int = 5000) -> dict:
        with self._lock:
            entity_rows = self._conn.execute("SELECT * FROM entities LIMIT ?", (limit,)).fetchall()
            relation_rows = self._conn.execute(
                """
                SELECT r.*, s.name AS source_name, t.name AS target_name
                FROM relations r JOIN entities s ON s.id=r.source_id
                JOIN entities t ON t.id=r.target_id LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return {
            "entities": [self._row_to_entity(row).as_dict() for row in entity_rows],
            "relations": [self._row_to_relation(row).as_dict() for row in relation_rows],
        }

    def reset(self) -> None:
        with self._lock:
            for table in ("mentions", "relations", "entity_aliases", "chunks", "entities", "notes"):
                self._conn.execute(f"DELETE FROM {table}")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- row mappers --------------------------------------------------------
    @staticmethod
    def _row_to_note(row: sqlite3.Row) -> Note:
        return Note(
            title=row["title"], content=row["content"], path=row["path"] or "",
            tags=json.loads(row["tags"] or "[]"), aliases=json.loads(row["aliases"] or "[]"),
            metadata=json.loads(row["metadata"] or "{}"), created_at=row["created_at"] or "", id=row["id"],
        )

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(id=row["id"], note_id=row["note_id"], seq=row["seq"], text=row["text"], heading=row["heading"] or "")

    @staticmethod
    def _row_to_entity(row: sqlite3.Row) -> Entity:
        return Entity(
            name=row["name"], type=row["type"], description=row["description"] or "",
            aliases=json.loads(row["aliases"] or "[]"), note_ids=json.loads(row["note_ids"] or "[]"),
            id=row["id"],
        )

    @staticmethod
    def _row_to_relation(row: sqlite3.Row) -> Relation:
        keys = row.keys()
        return Relation(
            source_id=row["source_id"], target_id=row["target_id"], type=row["type"],
            source_name=row["source_name"] if "source_name" in keys else "",
            target_name=row["target_name"] if "target_name" in keys else "",
            description=row["description"] or "", evidence=row["evidence"] or "",
            confidence=row["confidence"] or 0.6, note_id=row["note_id"] or "",
            chunk_id=row["chunk_id"] or "",
        )
