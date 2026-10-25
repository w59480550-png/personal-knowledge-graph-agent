"""Domain models shared across extraction, storage, retrieval and API layers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .text import normalize_name, stable_id

ENTITY_TYPES = (
    "PERSON",
    "ORGANIZATION",
    "PROJECT",
    "TECHNOLOGY",
    "LOCATION",
    "EVENT",
    "CONCEPT",
    "TAG",
    "NOTE",
)

_TYPE_ALIASES = {
    "人": "PERSON", "人物": "PERSON", "person": "PERSON", "people": "PERSON",
    "组织": "ORGANIZATION", "公司": "ORGANIZATION", "机构": "ORGANIZATION",
    "organization": "ORGANIZATION", "company": "ORGANIZATION", "org": "ORGANIZATION",
    "项目": "PROJECT", "product": "PROJECT", "project": "PROJECT",
    "技术": "TECHNOLOGY", "工具": "TECHNOLOGY", "technology": "TECHNOLOGY",
    "tool": "TECHNOLOGY", "framework": "TECHNOLOGY", "language": "TECHNOLOGY",
    "地点": "LOCATION", "位置": "LOCATION", "城市": "LOCATION",
    "location": "LOCATION", "place": "LOCATION", "city": "LOCATION", "country": "LOCATION",
    "事件": "EVENT", "event": "EVENT",
    "概念": "CONCEPT", "主题": "CONCEPT", "concept": "CONCEPT", "topic": "CONCEPT",
    "other": "CONCEPT", "misc": "CONCEPT",
    "tag": "TAG", "标签": "TAG",
    "note": "NOTE", "笔记": "NOTE", "document": "NOTE",
}

_RELATION_TYPES = {
    "IS_A": {"is a", "isa", "is an", "type of", "kind of", "是", "属于", "是一种", "instance of"},
    "PART_OF": {"part of", "belongs to", "belong to", "包含于", "属于", "组成", "子集"},
    "USES": {"uses", "use", "using", "depends on", "built with", "使用", "依赖", "基于", "采用"},
    "WORKS_AT": {"works at", "work at", "worked at", "employed by", "就职于", "工作于", "任职于"},
    "FOUNDED": {"founded", "founder of", "created", "created by", "创立", "创立了", "创建", "创建了", "创始人"},
    "AUTHORED": {"authored", "author of", "written by", "wrote", "作者", "编写", "撰写了"},
    "LOCATED_IN": {"located in", "based in", "lives in", "位于", "总部在", "住在"},
    "MENTIONS": {"mentions", "mentioned in", "提到", "提及"},
    "LINKS_TO": {"links to", "references", "引用", "链接到", "关联"},
    "TAGGED_WITH": {"tagged with", "has tag", "标签为"},
    "RELATED_TO": {"related to", "related", "associated with", "相关", "有关"},
}

_REL_LOOKUP: dict[str, str] = {}
for _canonical, _phrases in _RELATION_TYPES.items():
    _REL_LOOKUP[_canonical.lower()] = _canonical
    for _phrase in _phrases:
        _REL_LOOKUP[_phrase.lower()] = _canonical

_REL_RE = re.compile(r"[^A-Za-z0-9_\u4e00-\u9fff]+")

_TYPE_RANK = {
    "TAG": 0, "NOTE": 1, "CONCEPT": 2, "EVENT": 3, "PROJECT": 4,
    "TECHNOLOGY": 5, "LOCATION": 6, "ORGANIZATION": 7, "PERSON": 8,
}


def merge_entity_type(existing: str, incoming: str) -> str:
    """Keep the most informative type; never demote NOTE/PERSON/ORG to TAG."""
    existing, incoming = normalize_entity_type(existing), normalize_entity_type(incoming)
    if existing == incoming:
        return existing
    if existing == "NOTE" or incoming == "NOTE":
        return existing if existing == "NOTE" else incoming
    if existing in {"TAG", "CONCEPT"} and _TYPE_RANK[incoming] > _TYPE_RANK[existing]:
        return incoming
    return existing


def prefer_display_name(existing: str, incoming: str) -> str:
    """Prefer 'Neo4j' over a lowercase tag spelling 'neo4j'."""
    if existing != incoming and existing.lower() == incoming.lower():
        if existing.islower() and not incoming.islower():
            return incoming
    return existing


def normalize_entity_type(raw: str | None) -> str:
    if not raw:
        return "CONCEPT"
    key = str(raw).strip()
    upper = key.upper().replace(" ", "_")
    if upper in ENTITY_TYPES:
        return upper
    return _TYPE_ALIASES.get(key.lower(), _TYPE_ALIASES.get(key, "CONCEPT"))


def normalize_relation_type(raw: str | None) -> str:
    if not raw:
        return "RELATED_TO"
    key = str(raw).strip().lower()
    if key in _REL_LOOKUP:
        return _REL_LOOKUP[key]
    cleaned = _REL_RE.sub("_", str(raw).strip()).strip("_").upper()
    if not cleaned:
        return "RELATED_TO"
    if cleaned[0].isdigit():
        cleaned = f"REL_{cleaned}"
    return cleaned[:48]



@dataclass
class Entity:
    name: str
    type: str = "CONCEPT"
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    note_ids: list[str] = field(default_factory=list)
    id: str = ""

    def __post_init__(self) -> None:
        self.name = self.name.strip()
        self.type = normalize_entity_type(self.type)
        self.aliases = [a.strip() for a in self.aliases if a and a.strip()]
        if not self.id:
            self.id = stable_id("ent", f"{self.type}:{normalize_name(self.name)}")

    @property
    def key(self) -> str:
        return normalize_name(self.name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "aliases": list(self.aliases),
            "note_ids": list(self.note_ids),
        }


@dataclass
class RelationDraft:
    """A relation as produced by an extractor (entities referenced by name)."""

    source: str
    target: str
    type: str = "RELATED_TO"
    description: str = ""
    evidence: str = ""
    confidence: float = 0.6
    chunk_id: str = ""

    def __post_init__(self) -> None:
        self.source = self.source.strip()
        self.target = self.target.strip()
        self.type = normalize_relation_type(self.type)
        self.confidence = max(0.0, min(1.0, float(self.confidence)))


@dataclass
class Relation:
    """A persisted, resolved relation between two entity ids."""

    source_id: str
    target_id: str
    type: str
    source_name: str = ""
    target_name: str = ""
    description: str = ""
    evidence: str = ""
    confidence: float = 0.6
    note_id: str = ""
    chunk_id: str = ""

    def __post_init__(self) -> None:
        self.type = normalize_relation_type(self.type)
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    @property
    def id(self) -> str:
        return stable_id("rel", f"{self.source_id}|{self.type}|{self.target_id}|{self.note_id}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "source": self.source_name,
            "target_id": self.target_id,
            "target": self.target_name,
            "type": self.type,
            "description": self.description,
            "evidence": self.evidence,
            "confidence": round(self.confidence, 3),
            "note_id": self.note_id,
            "chunk_id": self.chunk_id,
        }


@dataclass
class Extraction:
    entities: list[Entity] = field(default_factory=list)
    relations: list[RelationDraft] = field(default_factory=list)
    extractor: str = "heuristic"

    def merge(self, other: "Extraction") -> "Extraction":
        by_key: dict[tuple[str, str], Entity] = {(e.type, e.key): e for e in self.entities}
        for entity in other.entities:
            key = (entity.type, entity.key)
            existing = by_key.get(key)
            if existing:
                existing.aliases = list(dict.fromkeys(existing.aliases + entity.aliases))
                if not existing.description and entity.description:
                    existing.description = entity.description
            else:
                by_key[key] = entity
                self.entities.append(entity)
        seen = {(r.source, r.type, r.target, r.chunk_id) for r in self.relations}
        for rel in other.relations:
            key = (rel.source, rel.type, rel.target, rel.chunk_id)
            if key not in seen:
                self.relations.append(rel)
                seen.add(key)
        return self


@dataclass
class Citation:
    index: int
    note_id: str
    note_title: str
    chunk_id: str
    chunk_index: int
    text: str
    reason: str = "vector"
    score: float = 0.0
    relations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "note_id": self.note_id,
            "note_title": self.note_title,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "reason": self.reason,
            "score": round(self.score, 4),
            "relations": list(self.relations),
        }


@dataclass
class SearchHit:
    chunk_id: str
    score: float
    reason: str
    text: str = ""


@dataclass
class Answer:
    answer: str
    session_id: str
    question: str
    rewritten_question: str
    citations: list[Citation] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    mode: str = "extractive"
    latency_ms: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "session_id": self.session_id,
            "question": self.question,
            "rewritten_question": self.rewritten_question,
            "citations": [c.as_dict() for c in self.citations],
            "entities": self.entities,
            "relations": self.relations,
            "mode": self.mode,
            "latency_ms": self.latency_ms,
        }

