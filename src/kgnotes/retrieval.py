"""Hybrid retrieval: dense chunk vectors + BM25 + graph neighbourhoods.

Ranking uses Reciprocal Rank Fusion (RRF) and adds a small boost for chunks
that are evidence of graph relations touching entities found in the question.
This keeps answers grounded even when note wording differs from the question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import Settings
from .embeddings import Embedder
from .models import Citation, Entity, Relation
from .notes import Chunk, Note
from .text import bm25_scores, normalize_name, tokenize, truncate

RRF_K = 60


@dataclass
class RetrievalResult:
    citations: list[Citation] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


class HybridRetriever:
    def __init__(self, store, vector_index, embedder: Embedder, settings: Settings) -> None:
        self.store = store
        self.vector_index = vector_index
        self.embedder = embedder
        self.settings = settings

    # -- entity linking -----------------------------------------------------
    def link_entities(self, text: str, limit: int = 8) -> list[Entity]:
        if not text.strip():
            return []
        normalized = normalize_name(text)
        matches: list[tuple[int, Entity]] = []
        for entity in self.store.list_entities(limit=3000):
            for surface in [entity.name, *entity.aliases]:
                if self._contains(normalized, surface):
                    # Longer names are more specific; NOTE titles are useful too.
                    matches.append((len(normalize_name(surface)), entity))
                    break
        matches.sort(key=lambda pair: (-pair[0], pair[1].type != "NOTE", pair[1].name))
        seen: set[str] = set()
        out: list[Entity] = []
        for _, entity in matches:
            if entity.id not in seen:
                seen.add(entity.id)
                out.append(entity)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _contains(haystack: str, surface: str) -> bool:
        needle = normalize_name(surface)
        if len(needle) < 2 or needle not in haystack:
            return False
        if re.fullmatch(r"[a-z0-9_+#.\- ]+", needle):
            return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack) is not None
        return True

    # -- retrieval ----------------------------------------------------------
    def retrieve(self, query: str, top_k: int | None = None, depth: int | None = None) -> RetrievalResult:
        top_k = max(1, top_k or self.settings.top_k)
        depth = depth or self.settings.graph_depth
        chunk_pairs: list[tuple[Chunk, Note]] = self.store.iter_chunks()
        if not chunk_pairs:
            return RetrievalResult(stats={"chunks": 0})

        by_id = {chunk.id: (chunk, note) for chunk, note in chunk_pairs}
        entities = self.link_entities(query)
        relations: list[Relation] = []
        if entities:
            relations = self.store.relations_for([e.id for e in entities], limit=60)
            for entity in entities[:3]:
                _, extra = self.store.neighbors(
                    entity.id, depth=depth, limit=self.settings.graph_neighbor_limit
                )
                relations.extend(extra)
        unique_relations: dict[str, Relation] = {}
        for rel in relations:
            existing = unique_relations.get(rel.id)
            if existing is None or rel.confidence > existing.confidence:
                unique_relations[rel.id] = rel
        relations = sorted(
            unique_relations.values(),
            key=lambda r: (r.type in {"TAGGED_WITH", "LINKS_TO"}, -r.confidence),
        )

        scores: dict[str, float] = {}
        reasons: dict[str, set[str]] = {}
        embedding_error = ""
        try:
            query_vector = self.embedder.embed_query(query)
            vector_scores = dict(self.vector_index.search(query_vector, limit=top_k * 4))
        except Exception as exc:  # noqa: BLE001 - keyword/graph retrieval still works
            vector_scores = {}
            embedding_error = str(exc)

        docs = [
            (chunk.id, f"{note.title}\n{chunk.heading}\n{chunk.text}") for chunk, note in chunk_pairs
        ]
        keyword_scores = bm25_scores(tokenize(query), docs)

        for rank, (chunk_id, _) in enumerate(
            sorted(vector_scores.items(), key=lambda pair: (-pair[1], pair[0]))
        ):
            if chunk_id in by_id:
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
                reasons.setdefault(chunk_id, set()).add("vector")
        for rank, (chunk_id, _) in enumerate(
            sorted(keyword_scores.items(), key=lambda pair: (-pair[1], pair[0]))
        ):
            if chunk_id in by_id:
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
                reasons.setdefault(chunk_id, set()).add("keyword")

        # Graph evidence boost / graph-only candidates.
        for rel in relations:
            if rel.chunk_id and rel.chunk_id in by_id:
                scores[rel.chunk_id] = scores.get(rel.chunk_id, 0.0) + 0.02 + 0.01 * rel.confidence
                reasons.setdefault(rel.chunk_id, set()).add("graph")

        # Mention boost: chunks that literally contain a linked entity name.
        if entities:
            needles = [normalize_name(e.name) for e in entities if len(normalize_name(e.name)) >= 2]
            if needles:
                for chunk, note in chunk_pairs:
                    haystack = normalize_name(f"{note.title}\n{chunk.text}")
                    if any(needle in haystack for needle in needles):
                        scores[chunk.id] = scores.get(chunk.id, 0.0) + 0.015
                        reasons.setdefault(chunk.id, set()).add("entity")

        if not scores and entities:
            wanted_notes = {nid for entity in entities for nid in entity.note_ids}
            for chunk, note in chunk_pairs:
                if note.id in wanted_notes:
                    scores[chunk.id] = 0.001
                    reasons.setdefault(chunk.id, set()).add("entity-note")

        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:top_k]
        relation_labels: dict[str, list[str]] = {}
        for rel in relations:
            label = f"{rel.source_name} --{rel.type}--> {rel.target_name}"
            relation_labels.setdefault(rel.chunk_id, []).append(label)

        citations: list[Citation] = []
        for index, (chunk_id, score) in enumerate(ranked, start=1):
            chunk, note = by_id[chunk_id]
            citations.append(
                Citation(
                    index=index,
                    note_id=note.id,
                    note_title=note.title,
                    chunk_id=chunk.id,
                    chunk_index=chunk.seq,
                    text=truncate(chunk.text, 900),
                    reason="+".join(sorted(reasons.get(chunk_id, {"hybrid"}))),
                    score=score,
                    relations=(relation_labels.get(chunk_id) or [])[:6],
                )
            )

        try:
            indexed_vectors = self.vector_index.count()
        except Exception:  # noqa: BLE001
            indexed_vectors = None
        stats = {
            "chunks": len(chunk_pairs),
            "vector_hits": len(vector_scores),
            "keyword_hits": len(keyword_scores),
            "graph_relations": len(relations),
            "entities": len(entities),
            "embedding_error": embedding_error,
            "index_backend": getattr(self.vector_index, "backend", ""),
            "indexed_vectors": indexed_vectors,
        }
        return RetrievalResult(
            citations=citations, entities=entities, relations=relations, stats=stats
        )

