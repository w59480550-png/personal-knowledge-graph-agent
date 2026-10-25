"""Note loading: Markdown/YAML front-matter, wikilinks, tags and chunking."""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from .text import clean_whitespace, split_sentences, stable_id, truncate

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|([^\]]+))?\]\]")
_TAG_RE = re.compile(r"(?<![\w/])#([A-Za-z0-9_\u4e00-\u9fff][\w\-/\u4e00-\u9fff]{0,39})")
_SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}


@dataclass
class Chunk:
    id: str
    note_id: str
    seq: int
    text: str
    heading: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "note_id": self.note_id,
            "seq": self.seq,
            "text": self.text,
            "heading": self.heading,
        }


@dataclass
class Note:
    title: str
    content: str
    path: str = ""
    tags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    id: str = ""

    def __post_init__(self) -> None:
        self.title = clean_whitespace(self.title) or "Untitled"
        self.content = self.content.replace("\r\n", "\n").replace("\r", "\n").strip()
        self.tags = list(dict.fromkeys(t.strip().lstrip("#") for t in self.tags if t and t.strip()))
        self.aliases = list(dict.fromkeys(a.strip() for a in self.aliases if a and a.strip()))
        if not self.id:
            key = self.path or f"{self.title}:{self.content[:120]}"
            self.id = stable_id("note", key)

    @property
    def content_hash(self) -> str:
        return hashlib.blake2b(self.content.encode("utf-8"), digest_size=8).hexdigest()

    def wikilinks(self) -> list[tuple[str, str]]:
        """Return (target, label) pairs; label defaults to the target."""
        out: list[tuple[str, str]] = []
        for target, label in _WIKILINK_RE.findall(self.content):
            target = target.strip()
            if target:
                out.append((target, (label or target).strip()))
        return out

    def inline_tags(self) -> list[str]:
        return [t for t in _TAG_RE.findall(self.content)]

    def all_tags(self) -> list[str]:
        return list(dict.fromkeys(self.tags + self.inline_tags()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "path": self.path,
            "content": self.content,
            "tags": self.tags,
            "aliases": self.aliases,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "content_hash": self.content_hash,
        }


def parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return {}, raw
    if not isinstance(meta, dict):
        return {}, raw
    return meta, raw[match.end():]


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in re.split(r"[,，;；]", value) if v.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value)]


class NoteLoader:
    """Loads notes from text or a directory of Markdown files."""

    def __init__(self, max_chunk_chars: int = 700, chunk_overlap: int = 80) -> None:
        self.max_chunk_chars = max(120, max_chunk_chars)
        self.chunk_overlap = max(0, min(chunk_overlap, self.max_chunk_chars // 2))

    # -- loading ------------------------------------------------------------
    def from_text(
        self,
        content: str,
        title: str | None = None,
        path: str = "",
        tags: Iterable[str] | None = None,
        aliases: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Note:
        meta, body = parse_frontmatter(content)
        merged_meta: dict[str, Any] = {**meta, **(metadata or {})}
        resolved_title = (
            title
            or merged_meta.get("title")
            or self._title_from_body(body)
            or (Path(path).stem if path else "Untitled")
        )
        return Note(
            title=str(resolved_title),
            content=self._strip_title_heading(body, str(resolved_title)),
            path=path,
            tags=_as_list(tags) + _as_list(merged_meta.get("tags")),
            aliases=_as_list(aliases) + _as_list(merged_meta.get("aliases")),
            metadata=merged_meta,
        )

    def from_file(self, path: str | Path, root: str | Path | None = None) -> Note:
        p = Path(path)
        raw = p.read_text(encoding="utf-8")
        rel = str(p.relative_to(root)) if root else str(p)
        return self.from_text(raw, path=rel)

    def load_dir(self, directory: str | Path) -> list[Note]:
        root = Path(directory)
        files = sorted(
            f for f in root.rglob("*") if f.is_file() and f.suffix.lower() in _SUPPORTED_SUFFIXES
        )
        return [self.from_file(f, root=root) for f in files]

    @staticmethod
    def _title_from_body(body: str) -> str | None:
        for line in body.splitlines():
            match = _HEADING_RE.match(line.strip())
            if match and len(match.group(1)) == 1:
                return match.group(2).strip()
        for line in body.splitlines():
            if line.strip():
                return truncate(line.strip(), 60)
        return None

    @staticmethod
    def _strip_title_heading(body: str, title: str) -> str:
        lines = body.splitlines()
        for index, line in enumerate(lines):
            match = _HEADING_RE.match(line.strip())
            if match and len(match.group(1)) == 1 and match.group(2).strip() == title:
                return "\n".join(lines[:index] + lines[index + 1:]).strip()
            if line.strip():
                break
        return body.strip()


    # -- chunking -----------------------------------------------------------
    def chunk(self, note: Note) -> list[Chunk]:
        sections = self._sections(note.content)
        chunks: list[Chunk] = []
        for heading, body in sections:
            for piece in self._pack(body):
                text = f"{heading}\n{piece}" if heading else piece
                seq = len(chunks)
                chunks.append(
                    Chunk(
                        id=stable_id("chunk", f"{note.id}:{seq}"),
                        note_id=note.id,
                        seq=seq,
                        text=text.strip(),
                        heading=heading,
                    )
                )
        if not chunks:
            chunks.append(
                Chunk(id=stable_id("chunk", f"{note.id}:0"), note_id=note.id, seq=0, text=note.title)
            )
        return chunks

    @staticmethod
    def _sections(content: str) -> list[tuple[str, str]]:
        heading_stack: list[tuple[int, str]] = []
        sections: list[tuple[str, list[str]]] = []
        current: list[str] = []
        for line in content.splitlines():
            match = _HEADING_RE.match(line.strip())
            if match:
                if current:
                    sections.append((" > ".join(h for _, h in heading_stack), "\n".join(current).strip()))
                    current = []
                level = len(match.group(1))
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, match.group(2).strip()))
            else:
                current.append(line)
        if current:
            sections.append((" > ".join(h for _, h in heading_stack), "\n".join(current).strip()))
        return [(h, b) for h, b in sections if b]

    def _pack(self, body: str) -> list[str]:
        paragraphs = [clean_whitespace(p) for p in re.split(r"\n\s*\n", body) if p.strip()]
        units: list[str] = []
        for paragraph in paragraphs:
            if len(paragraph) <= self.max_chunk_chars:
                units.append(paragraph)
                continue
            buffer = ""
            for sentence in split_sentences(paragraph):
                if len(sentence) > self.max_chunk_chars:
                    if buffer:
                        units.append(buffer)
                        buffer = ""
                    units.extend(
                        sentence[i: i + self.max_chunk_chars]
                        for i in range(0, len(sentence), self.max_chunk_chars)
                    )
                elif len(buffer) + len(sentence) + 1 > self.max_chunk_chars:
                    units.append(buffer)
                    buffer = sentence
                else:
                    buffer = f"{buffer} {sentence}".strip()
            if buffer:
                units.append(buffer)

        chunks: list[str] = []
        current = ""
        for unit in units:
            candidate = f"{current}\n\n{unit}".strip() if current else unit
            if len(candidate) <= self.max_chunk_chars:
                current = candidate
                continue
            if current:
                chunks.append(current)
                tail = self._tail(current)
                current = f"{tail}\n\n{unit}".strip() if tail else unit
            else:
                current = unit
        if current:
            chunks.append(current)
        return chunks or ([body.strip()] if body.strip() else [])

    def _tail(self, text: str) -> str:
        if self.chunk_overlap <= 0:
            return ""
        tail = text[-self.chunk_overlap:]
        parts = split_sentences(tail)
        if len(parts) > 1:
            tail = " ".join(parts[1:])
        return tail.strip()
