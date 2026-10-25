"""Extractor selection + per-note extraction orchestration."""

from __future__ import annotations

import logging

from ..config import Settings
from ..llm import LLMClient, LLMError
from ..models import Extraction
from ..notes import Chunk, Note
from .heuristic import HeuristicExtractor
from .llm import LLMExtractor

log = logging.getLogger(__name__)


class ExtractorPipeline:
    """Runs the configured extractor, falling back to rules on LLM failure.

    ``KGNOTES_EXTRACTOR``:
      * ``heuristic`` - offline only
      * ``llm``       - LLM only (errors propagate)
      * ``auto`` (default) - LLM when configured, heuristic otherwise; an LLM
        failure for a chunk is logged and that chunk falls back to rules.
    """

    def __init__(self, settings: Settings, client: LLMClient | None = None) -> None:
        self.settings = settings
        self.mode = settings.extractor
        self.heuristic = HeuristicExtractor(max_triples=settings.max_triples_per_chunk)
        self.llm: LLMExtractor | None = None
        if client is not None and client.enabled:
            self.llm = LLMExtractor(client, max_triples=settings.max_triples_per_chunk)

    @property
    def name(self) -> str:
        if self.mode == "llm":
            return "llm"
        if self.mode == "heuristic" or self.llm is None:
            return "heuristic"
        return "llm+heuristic-fallback"

    def note_graph(self, note: Note) -> Extraction:
        return self.heuristic.note_graph(note)

    def extract(self, note: Note, chunk: Chunk) -> Extraction:
        use_llm = self.mode == "llm" or (self.mode == "auto" and self.llm is not None)
        if use_llm and self.llm is not None:
            try:
                extraction = self.llm.extract(note, chunk)
                # Merge structural/markup entities the LLM may have skipped.
                extra = self.heuristic.extract(note, chunk)
                structural = [e for e in extra.entities if e.type in {"NOTE", "TAG"}]
                extraction.entities.extend(structural)
                return extraction
            except LLMError as exc:
                if self.mode == "llm":
                    raise
                log.warning("LLM extraction failed for chunk %s: %s; using rules", chunk.id, exc)
        return self.heuristic.extract(note, chunk)

    def extract_note(self, note: Note, chunks: list[Chunk]) -> Extraction:
        combined = self.note_graph(note)
        for chunk in chunks:
            combined.merge(self.extract(note, chunk))
        return combined
