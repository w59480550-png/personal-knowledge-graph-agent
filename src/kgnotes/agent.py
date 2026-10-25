"""Top-level orchestration: ingest notes, maintain graph + vectors, answer."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import Settings
from .embeddings import Embedder, build_embedder
from .extraction.pipeline import ExtractorPipeline
from .graph.neo4j_store import Neo4jGraphStore
from .graph.sqlite_store import SQLiteGraphStore
from .llm import LLMClient
from .models import Entity, Extraction, Relation
from .notes import Chunk, Note, NoteLoader
from .qa import QAService
from .retrieval import HybridRetriever, RetrievalResult
from .sessions import SessionStore
from .text import normalize_name, truncate
from .vector import build_vector_index

log = logging.getLogger(__name__)


def build_store(settings: Settings):
    if settings.graph_store == "neo4j":
        return Neo4jGraphStore(settings)
    return SQLiteGraphStore(settings.sqlite_path)


@dataclass
class IngestSummary:
    note_id: str
    title: str
    chunks: int
    entities: int
    relations: int
    extractor: str
    new_note: bool
    warnings: list[str] = field(default_factory=list)
    elapsed_ms: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "note_id": self.note_id,
            "title": self.title,
            "chunks": self.chunks,
            "entities": self.entities,
            "relations": self.relations,
            "extractor": self.extractor,
            "new_note": self.new_note,
            "warnings": self.warnings,
            "elapsed_ms": self.elapsed_ms,
        }


def _surface_in_text(text: str, surface: str) -> bool:
    needle = normalize_name(surface)
    if len(needle) < 2:
        return False
    haystack = normalize_name(text)
    if needle not in haystack:
        return False
    if re.fullmatch(r"[a-z0-9_+#.\- ]+", needle):
        return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack) is not None
    return True


class KnowledgeAgent:
    """Facade used by the CLI, HTTP API and tests."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        store=None,
        vector_index=None,
        embedder: Embedder | None = None,
        llm: LLMClient | None = None,
        sessions: SessionStore | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.settings.ensure_dirs()
        self.llm = llm or LLMClient(self.settings)
        self.embedder = embedder or build_embedder(self.settings, self.llm)
        self.store = store or build_store(self.settings)
        self.vector_index = vector_index or build_vector_index(self.settings, self.embedder)
        self.sessions = sessions or SessionStore(self.settings.data_dir / "sessions.sqlite")
        self.extractors = ExtractorPipeline(self.settings, self.llm)
        self.loader = NoteLoader(self.settings.max_chunk_chars, self.settings.chunk_overlap)
        self.retriever = HybridRetriever(self.store, self.vector_index, self.embedder, self.settings)
        self.qa = QAService(self.settings, self.retriever, self.sessions, self.llm)

    # -- ingest -------------------------------------------------------------
    def ingest_note(self, note: Note) -> IngestSummary:
        started = time.perf_counter()
        chunks = self.loader.chunk(note)
        extraction = self.extractors.extract_note(note, chunks)
        created = self.store.upsert_note(note, chunks)
        warnings: list[str] = []

        key_to_id: dict[str, str] = {}
        for entity in extraction.entities:
            entity.note_ids = list(dict.fromkeys(entity.note_ids + [note.id]))
            resolved = self.store.upsert_entity(entity)
            key_to_id[entity.key] = resolved.id

        relations = self._persist_relations(extraction, note, key_to_id)
        self._persist_mentions(note, chunks, extraction, key_to_id)

        try:
            vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
            self.vector_index.add([(chunk.id, vec) for chunk, vec in zip(chunks, vectors)])
        except Exception as exc:  # noqa: BLE001 - graph stays usable without vectors
            log.warning("embedding failed for note %s: %s", note.id, exc)
            warnings.append(f"embedding failed: {exc}")

        return IngestSummary(
            note_id=note.id,
            title=note.title,
            chunks=len(chunks),
            entities=len({e.id for e in extraction.entities}),
            relations=len(relations),
            extractor=self.extractors.name,
            new_note=created,
            warnings=warnings,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    def ingest_text(
        self,
        content: str,
        title: str | None = None,
        path: str = "",
        tags: Iterable[str] | None = None,
        aliases: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestSummary:
        note = self.loader.from_text(
            content, title=title, path=path, tags=tags, aliases=aliases, metadata=metadata
        )
        return self.ingest_note(note)

    def ingest_file(self, path: str | Path, root: str | Path | None = None) -> IngestSummary:
        return self.ingest_note(self.loader.from_file(path, root=root))

    def ingest_dir(self, directory: str | Path) -> list[IngestSummary]:
        return [self.ingest_note(note) for note in self.loader.load_dir(directory)]

    def delete_note(self, note_id: str) -> bool:
        chunks = [chunk.id for chunk, _ in self.store.iter_chunks() if chunk.note_id == note_id]
        deleted = self.store.delete_note(note_id)
        if deleted:
            self.vector_index.delete(chunks)
        return deleted


    def _persist_relations(
        self, extraction: Extraction, note: Note, key_to_id: dict[str, str]
    ) -> list[Relation]:
        saved: list[Relation] = []
        for draft in extraction.relations:
            source_id = key_to_id.get(normalize_name(draft.source))
            target_id = key_to_id.get(normalize_name(draft.target))
            if not source_id or not target_id or source_id == target_id:
                continue
            relation = Relation(
                source_id=source_id,
                target_id=target_id,
                type=draft.type,
                source_name=draft.source,
                target_name=draft.target,
                description=draft.description,
                evidence=draft.evidence,
                confidence=draft.confidence,
                note_id=note.id,
                chunk_id=draft.chunk_id,
            )
            try:
                self.store.upsert_relation(relation)
                saved.append(relation)
            except ValueError as exc:
                log.warning("skipping relation %s -> %s: %s", draft.source, draft.target, exc)
        return saved

    def _persist_mentions(
        self,
        note: Note,
        chunks: list[Chunk],
        extraction: Extraction,
        key_to_id: dict[str, str],
    ) -> None:
        by_key = {normalize_name(e.name): e for e in extraction.entities}
        for chunk in chunks:
            ids: list[str] = []
            for key, entity_id in key_to_id.items():
                entity = by_key.get(key)
                surfaces = [entity.name, *entity.aliases] if entity else [key]
                if any(_surface_in_text(chunk.text, surface) for surface in surfaces):
                    ids.append(entity_id)
            if ids:
                self.store.add_mentions(note.id, chunk.id, ids)

    # -- queries ------------------------------------------------------------
    def search(self, query: str, top_k: int | None = None) -> RetrievalResult:
        return self.retriever.retrieve(query, top_k=top_k)

    def ask(self, question: str, session_id: str | None = None, top_k: int | None = None):
        return self.qa.ask(question, session_id=session_id, top_k=top_k)

    def graph(self, entity: str, depth: int = 1, limit: int | None = None) -> dict[str, Any]:
        found = self.store.get_entity(entity_id=entity) or self.store.get_entity(name=entity)
        if not found:
            return {"entity": None, "entities": [], "relations": []}
        entities, relations = self.store.neighbors(
            found.id, depth=depth, limit=limit or self.settings.graph_neighbor_limit
        )
        return {
            "entity": found.as_dict(),
            "entities": [e.as_dict() for e in entities],
            "relations": [r.as_dict() for r in relations],
        }

    def stats(self) -> dict[str, Any]:
        graph = self.store.stats()
        try:
            vectors = self.vector_index.count()
        except Exception:  # noqa: BLE001
            vectors = None
        return {
            "graph": graph.as_dict(),
            "vectors": {
                "count": vectors,
                "model": self.embedder.name,
                "backend": getattr(self.vector_index, "backend", ""),
            },
            "extractor": self.extractors.name,
            "llm_enabled": self.llm.enabled,
            "llm_model": self.settings.llm_model if self.llm.enabled else "",
            "sessions": len(self.sessions.list_sessions(limit=1000)),
        }

    # -- maintenance --------------------------------------------------------
    def reindex(self) -> dict[str, Any]:
        """Re-embed all chunks (after changing embedding model/dimension)."""
        pairs = self.store.iter_chunks()
        self.vector_index.reset()
        if not pairs:
            return {"chunks": 0, "model": self.embedder.name}
        batch_size = 64
        added = 0
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            vectors = self.embedder.embed_documents([chunk.text for chunk, _ in batch])
            added += self.vector_index.add([(chunk.id, vec) for (chunk, _), vec in zip(batch, vectors)])
        return {"chunks": added, "model": self.embedder.name}

    def export_graph(self, path: str | Path | None = None) -> dict[str, Any]:
        data = self.store.export_graph()
        if path:
            import json

            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data

    def reset(self, include_sessions: bool = True) -> None:
        self.store.reset()
        self.vector_index.reset()
        if include_sessions:
            self.sessions.reset()

    def close(self) -> None:
        for resource in (self.vector_index, self.store, self.sessions):
            try:
                resource.close()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self) -> "KnowledgeAgent":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

