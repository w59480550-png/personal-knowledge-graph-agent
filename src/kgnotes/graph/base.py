"""Storage abstraction shared by the SQLite (offline) and Neo4j backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from ..models import Entity, Relation
from ..notes import Chunk, Note


@dataclass
class GraphStats:
    notes: int = 0
    chunks: int = 0
    entities: int = 0
    relations: int = 0
    backend: str = "sqlite"
    entity_types: dict[str, int] = field(default_factory=dict)
    relation_types: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "backend": self.backend,
            "notes": self.notes,
            "chunks": self.chunks,
            "entities": self.entities,
            "relations": self.relations,
            "entity_types": dict(sorted(self.entity_types.items(), key=lambda kv: -kv[1])),
            "relation_types": dict(sorted(self.relation_types.items(), key=lambda kv: -kv[1])),
        }


class GraphStore(Protocol):
    backend: str

    def upsert_note(self, note: Note, chunks: Sequence[Chunk]) -> bool: ...

    def delete_note(self, note_id: str) -> bool: ...

    def get_note(self, note_id: str) -> Note | None: ...

    def list_notes(self, limit: int = 100, offset: int = 0) -> list[Note]: ...

    def get_chunk(self, chunk_id: str) -> tuple[Chunk, Note] | None: ...

    def iter_chunks(self) -> list[tuple[Chunk, Note]]: ...

    def upsert_entity(self, entity: Entity) -> Entity: ...

    def upsert_relation(self, relation: Relation) -> Relation: ...

    def get_entity(self, entity_id: str | None = None, name: str | None = None) -> Entity | None: ...

    def match_entities(self, texts: Sequence[str]) -> list[Entity]: ...

    def list_entities(
        self, query: str | None = None, entity_type: str | None = None, limit: int = 100
    ) -> list[Entity]: ...

    def neighbors(self, entity_id: str, depth: int = 1, limit: int = 50) -> tuple[list[Entity], list[Relation]]: ...

    def relations_for(self, entity_ids: Sequence[str], limit: int = 100) -> list[Relation]: ...

    def stats(self) -> GraphStats: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...
