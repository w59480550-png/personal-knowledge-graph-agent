"""LLM-based extraction with strict JSON output and heuristic fallback hooks."""

from __future__ import annotations

import logging

from ..llm import LLMClient, LLMError, extract_json_object
from ..models import Entity, Extraction, RelationDraft, normalize_entity_type
from ..notes import Chunk, Note
from ..text import truncate
from .heuristic import HeuristicExtractor, infer_entity_type

log = logging.getLogger(__name__)

_SYSTEM = """你是一个知识图谱信息抽取引擎。你的输出必须是合法 JSON，且只包含笔记片段中明确陈述的事实。"""

_PROMPT = """请从下面的笔记片段中抽取实体与关系。

实体类型只能是：PERSON, ORGANIZATION, PROJECT, TECHNOLOGY, LOCATION, EVENT, CONCEPT
关系类型使用大写下划线命名，优先使用：IS_A, PART_OF, USES, WORKS_AT, FOUNDED,
AUTHORED, LOCATED_IN, RELATED_TO, SUPPORTS, COLLABORATES_WITH。

要求：
1. 实体名称保持原文（中文或英文），不要把整句话当作实体。
2. 关系必须能从原文直接得到，evidence 必须是原文中的一句原话（可截断）。
3. 不要抽取“我/你/它/这个”等代词，不要编造别名。
4. confidence 取 0~1，明确陈述用 0.8 以上，推测性内容不要输出。

只输出如下 JSON（不要 markdown 代码块）：
{{"entities": [{{"name": "...", "type": "TECHNOLOGY", "description": "...", "aliases": []}}],
  "relations": [{{"source": "...", "target": "...", "type": "USES",
                 "description": "...", "evidence": "...", "confidence": 0.9}}]}}

笔记标题：{title}
笔记标签：{tags}
片段（第 {seq} 段，章节：{heading}）：
\"\"\"
{text}
\"\"\"
"""


class LLMExtractor:
    name = "llm"

    def __init__(self, client: LLMClient, max_triples: int = 20, max_entities: int = 80) -> None:
        self.client = client
        self.max_triples = max_triples
        self.max_entities = max_entities
        self._note_level = HeuristicExtractor(max_triples=max_triples)

    def note_graph(self, note: Note) -> Extraction:
        """Tags / wikilinks are structural; extract them without an LLM call."""
        return self._note_level.note_graph(note)

    def extract(self, note: Note, chunk: Chunk) -> Extraction:
        if not self.client.enabled:
            raise LLMError("LLM is not configured")
        prompt = _PROMPT.format(
            title=note.title,
            tags=", ".join(note.all_tags()) or "-",
            seq=chunk.seq,
            heading=chunk.heading or "-",
            text=truncate(chunk.text, 4000),
        )
        raw = self.client.chat(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.0,
            json_mode=True,
        )
        data = extract_json_object(raw)
        if not data:
            raise LLMError(f"extractor returned non-JSON output: {raw[:200]!r}")
        return self._to_extraction(data, note, chunk)

    def _to_extraction(self, data: dict, note: Note, chunk: Chunk) -> Extraction:
        extraction = Extraction(extractor=self.name)
        aliases_by_key: dict[str, list[str]] = {}

        for item in data.get("entities") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not (2 <= len(name) <= 60):
                continue
            aliases = [str(a).strip() for a in (item.get("aliases") or []) if str(a).strip()]
            raw_type = str(item.get("type") or "").strip()
            entity = Entity(
                name=name,
                type=normalize_entity_type(raw_type) if raw_type else infer_entity_type(name),
                description=str(item.get("description") or "")[:240],
                aliases=aliases,
                note_ids=[note.id],
            )
            aliases_by_key.setdefault(entity.key, [])
            aliases_by_key[entity.key].extend(aliases)
            self._add_entity(extraction, entity)

        for item in data.get("relations") or []:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or "").strip()
            target = str(item.get("target") or "").strip()
            if not source or not target or source == target:
                continue
            try:
                confidence = float(item.get("confidence", 0.7))
            except (TypeError, ValueError):
                confidence = 0.7
            for name in (source, target):
                if not any(e.key == self._key(name) for e in extraction.entities):
                    self._add_entity(extraction, Entity(
                        name=name, type=infer_entity_type(name, chunk.text),
                        aliases=aliases_by_key.get(self._key(name), []), note_ids=[note.id],
                    ))
            extraction.relations.append(
                RelationDraft(
                    source=source, target=target,
                    type=str(item.get("type") or "RELATED_TO"),
                    description=str(item.get("description") or "")[:240],
                    evidence=str(item.get("evidence") or "")[:400],
                    confidence=confidence, chunk_id=chunk.id,
                )
            )
            if len(extraction.relations) >= self.max_triples:
                break
        return extraction

    @staticmethod
    def _key(name: str) -> str:
        from ..text import normalize_name  # local import keeps module import cheap

        return normalize_name(name)

    def _add_entity(self, extraction: Extraction, entity: Entity) -> None:
        for existing in extraction.entities:
            if existing.key == entity.key:
                existing.aliases = list(dict.fromkeys(existing.aliases + entity.aliases))
                if not existing.description and entity.description:
                    existing.description = entity.description
                return
        if len(extraction.entities) < self.max_entities:
            extraction.entities.append(entity)
