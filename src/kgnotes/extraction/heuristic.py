"""Rule-based (offline) entity & relation extraction.

Pattern-oriented: powers the zero-dependency demo and acts as a safety net when
an LLM call fails. Handles mixed Chinese / English notes, wikilinks, tags,
Markdown emphasis and a small set of common relation patterns.
"""

from __future__ import annotations

import re

from ..models import Entity, Extraction, RelationDraft
from ..notes import Chunk, Note
from ..text import normalize_name, split_sentences, truncate

_BOLD_RE = re.compile(r"\*\*([^*\n]{2,40})\*\*")
_CODE_RE = re.compile(r"`([^`\n]{2,40})`")
_ALIAS_RE = re.compile(
    r"([\w\u4e00-\u9fff][\w\u4e00-\u9fff .\-]{1,30})[（(]"
    r"(?:又名|又称|简称|alias|aka)[:：]?\s*([^）)]{1,30})[）)]",
    re.I,
)
_CAPITALIZED_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[ .-][A-Z][A-Za-z0-9]*){0,3}\b")

_PERSON_HINTS = ("教授", "博士", "工程师", "创始人", "作者", "研究员", "设计师", "科学家")
_ORG_HINTS = ("公司", "大学", "学院", "实验室", "研究所", "研究院", "集团", "基金会", "协会", "团队")
_PROJECT_HINTS = ("项目", "计划", "产品", "应用")
_TECH_HINTS = (
    "框架", "语言", "系统", "工具", "平台", "数据库", "模型", "算法", "引擎", "协议",
    "库", "服务", "组件", "中间件", "sdk", "api", "cli",
)
_LOCATION_HINTS = ("市", "省", "国", "州", "区", "县", "镇", "街", "路", "总部")
_ORG_EN = ("inc", "ltd", "corp", "corporation", "university", "college", "lab", "labs", "foundation")
_TECH_EN = ("api", "sdk", "js", "py", "db", "framework", "engine", "library", "protocol", "model")

_PRONOUN_PREFIXES = ("它", "他", "她", "这", "那", "该", "其", "此", "these", "this", "that", "it", "they")

# Common Chinese surnames used as a lightweight person-name heuristic.
_SURNAMES = set(
    "王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘于蒋蔡余杜叶程苏魏吕"
    "丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤"
)
_COMMON_NON_NAMES = {"北京", "上海", "广州", "深圳", "中国", "美国", "项目", "团队", "公司", "系统", "模块"}


def _looks_like_person(name: str) -> bool:
    if not (2 <= len(name) <= 3) or name in _COMMON_NON_NAMES:
        return False
    if not all("\u4e00" <= ch <= "\u9fff" for ch in name):
        return False
    return name[0] in _SURNAMES


# (relation, regex, source_group, target_group, confidence)
_ZH_PATTERNS: list[tuple[str, re.Pattern[str], int, int, float]] = [
    ("IS_A", re.compile(r"^([^，。；：:]{2,24}?)(?:是一种|是一个|属于)([^，。；：:]{2,24})$"), 1, 2, 0.72),
    ("USES", re.compile(r"^([^，。；：:]{2,24}?)(?:使用|采用了?|基于|依赖(?:于)?)([^，。；：:]{2,24})$"), 1, 2, 0.72),
    ("LOCATED_IN", re.compile(r"^([^，。；：:]{2,24}?)(?:位于|总部位于|总部在|坐落在)([^，。；：:]{2,24})$"), 1, 2, 0.7),
    ("WORKS_AT", re.compile(r"^([^，。；：:]{2,24}?)(?:就职于|工作于|任职于|加入)([^，。；：:]{2,24})$"), 1, 2, 0.7),
    ("PART_OF", re.compile(r"^([^，。；：:]{2,24}?)包含([^，。；：:]{2,24})$"), 2, 1, 0.68),
    ("FOUNDED", re.compile(r"^([^，。；：:]{2,24}?)由([^，。；：:]{2,24}?)(?:创立|创建|创办|开发|设计)$"), 2, 1, 0.72),
    ("AUTHORED", re.compile(r"^([^，。；：:]{2,24}?)(?:的)?(?:作者|创始人)(?:是|为|：|:)([^，。；：:]{2,24})$"), 2, 1, 0.72),
    ("SUPPORTS", re.compile(r"^([^，。；：:]{2,24}?)支持([^，。；：:]{2,24})$"), 1, 2, 0.66),
    ("OWNS", re.compile(r"^([^，。；：:]{2,24}?)负责([^，。；：:]{2,24})$"), 1, 2, 0.66),
    ("LEADS", re.compile(r"^([^，。；：:]{2,24}?)是([^，。；：:]{2,24}?)的(?:负责人|项目经理|主管|组长)$"), 1, 2, 0.72),
    ("COLLABORATES_WITH", re.compile(r"^([^，。；：:]{2,24}?)(?:与|和)([^，。；：:]{2,24}?)合作$"), 1, 2, 0.66),
]

_EN_PATTERNS: list[tuple[str, re.Pattern[str], int, int, float]] = [
    ("IS_A", re.compile(r"^(.{2,60}?) is (?:a|an|the) (.{2,60}?)$", re.I), 1, 2, 0.7),
    ("USES", re.compile(r"^(.{2,60}?)(?: uses| depends on| is based on| is built with| relies on) (.{2,60}?)$", re.I), 1, 2, 0.7),
    ("LOCATED_IN", re.compile(r"^(.{2,60}?) is (?:located|based) in (.{2,60}?)$", re.I), 1, 2, 0.68),
    ("WORKS_AT", re.compile(r"^(.{2,60}?) works? (?:at|for) (.{2,60}?)$", re.I), 1, 2, 0.68),
    ("FOUNDED", re.compile(r"^(.{2,60}?) was (?:founded|created|developed) by (.{2,60}?)$", re.I), 2, 1, 0.7),
    ("AUTHORED", re.compile(r"^(.{2,60}?) was (?:written|authored) by (.{2,60}?)$", re.I), 2, 1, 0.7),
    ("PART_OF", re.compile(r"^(.{2,60}?) is part of (.{2,60}?)$", re.I), 1, 2, 0.68),
]


def infer_entity_type(name: str, context: str = "") -> str:
    lowered = name.lower()
    haystack = f"{name} {context}"
    if lowered.endswith((".js", ".py", ".ts", ".go", ".rs", ".java")) or lowered in _TECH_EN:
        return "TECHNOLOGY"
    if any(token in lowered for token in _ORG_EN) or lowered.endswith(("corp.", "inc.")):
        return "ORGANIZATION"
    if any(hint in name for hint in _ORG_HINTS):
        return "ORGANIZATION"
    if any(hint in name for hint in _PROJECT_HINTS):
        return "PROJECT"
    if any(hint in lowered for hint in _TECH_EN) or any(hint in haystack for hint in _TECH_HINTS):
        return "TECHNOLOGY"
    if _looks_like_person(name):
        return "PERSON"
    if any(hint in name for hint in _LOCATION_HINTS):
        return "LOCATION"
    if any(hint in haystack for hint in _PERSON_HINTS):
        return "PERSON"
    return "CONCEPT"


def _clean_operand(text: str) -> str:
    value = text.strip().strip("。！？!?；;，,：:、'\"“”‘’()（）[]【】《》")
    value = re.sub(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", r"\1", value)
    value = value.replace("[[", "").replace("]]", "").replace("**", "").replace("`", "")
    value = re.sub(r"\s+中的.{0,24}$", "", value)
    value = re.sub(r"^(?:并且|而且|但是|因此|所以|同时|另外|此外|例如|比如|其中)\s*", "", value)
    value = re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.I)
    value = re.sub(r"(?:的|了|着|过)$", "", value).strip()
    return re.sub(r"\s{2,}", " ", value).rstrip(".").strip()


def _valid_operand(value: str) -> bool:
    if not (2 <= len(value) <= 48):
        return False
    if value.lower().startswith(_PRONOUN_PREFIXES):
        return False
    if normalize_name(value) in {"", "它", "他", "她", "这个", "那个"}:
        return False
    if any(ch in value for ch in "\n\t"):
        return False
    return re.search(r"[\w\u4e00-\u9fff]", value) is not None


class HeuristicExtractor:
    name = "heuristic"

    def __init__(self, max_triples: int = 20, max_entities: int = 80) -> None:
        self.max_triples = max_triples
        self.max_entities = max_entities

    # -- note-level graph ---------------------------------------------------
    def note_graph(self, note: Note) -> Extraction:
        """Note title node, tags, wikilinks and declared aliases."""
        extraction = Extraction(extractor=self.name)
        note_entity = Entity(
            name=note.title, type="NOTE", description=truncate(note.content, 160),
            aliases=note.aliases, note_ids=[note.id],
        )
        self._add_entity(extraction, note_entity)
        for tag in note.all_tags():
            self._add_entity(extraction, Entity(name=tag, type="TAG", note_ids=[note.id]))
            extraction.relations.append(
                RelationDraft(source=note.title, target=tag, type="TAGGED_WITH",
                              description=f"笔记标签 #{tag}", evidence=f"#{tag}", confidence=0.95)
            )
        for target, label in note.wikilinks():
            entity = Entity(
                name=label, type=infer_entity_type(label),
                aliases=[] if label == target else [target], note_ids=[note.id],
            )
            self._add_entity(extraction, entity)
            extraction.relations.append(
                RelationDraft(source=note.title, target=label, type="LINKS_TO",
                              description="笔记中的双链引用", evidence=f"[[{target}]]", confidence=0.9)
            )
        return extraction

    # -- chunk-level extraction ---------------------------------------------
    def extract(self, note: Note, chunk: Chunk) -> Extraction:
        extraction = Extraction(extractor=self.name)
        text = chunk.text
        if chunk.heading and text.startswith(chunk.heading):
            text = text[len(chunk.heading):].lstrip("\n")
        self._collect_markup_entities(text, note, extraction)
        self._collect_aliases(text, note, extraction)

        for sentence in split_sentences(text):
            sentence = sentence.strip()
            if len(sentence) < 4:
                continue
            matched = self._match_patterns(sentence, note, chunk, extraction)
            if not matched:
                self._collect_capitalized(sentence, note, extraction)

        self._add_cooccurrence(note, chunk, extraction)
        extraction.relations = extraction.relations[: self.max_triples]
        return extraction

    def _match_patterns(
        self, sentence: str, note: Note, chunk: Chunk, extraction: Extraction
    ) -> bool:
        matched = False
        stripped = sentence.strip().rstrip("。！？!?；;，,")
        for relation, pattern, src_group, tgt_group, confidence in _ZH_PATTERNS + _EN_PATTERNS:
            match = pattern.match(stripped)
            if not match:
                continue
            source = _clean_operand(match.group(src_group))
            target = _clean_operand(match.group(tgt_group))
            if not (_valid_operand(source) and _valid_operand(target)):
                continue
            if normalize_name(source) == normalize_name(target):
                continue
            self._add_entity(extraction, Entity(
                name=source, type=infer_entity_type(source, sentence), note_ids=[note.id],
            ))
            self._add_entity(extraction, Entity(
                name=target, type=infer_entity_type(target, sentence), note_ids=[note.id],
            ))
            extraction.relations.append(
                RelationDraft(
                    source=source, target=target, type=relation,
                    description=truncate(sentence, 120), evidence=truncate(sentence, 240),
                    confidence=confidence, chunk_id=chunk.id,
                )
            )
            matched = True
        return matched

    def _collect_markup_entities(self, text: str, note: Note, extraction: Extraction) -> None:
        for name in _BOLD_RE.findall(text):
            name = _clean_operand(name)
            if _valid_operand(name):
                self._add_entity(extraction, Entity(
                    name=name, type=infer_entity_type(name, text), note_ids=[note.id],
                ))
        for name in _CODE_RE.findall(text):
            name = _clean_operand(name)
            if _valid_operand(name):
                self._add_entity(extraction, Entity(name=name, type="TECHNOLOGY", note_ids=[note.id]))

    def _collect_aliases(self, text: str, note: Note, extraction: Extraction) -> None:
        for canonical, alias in _ALIAS_RE.findall(text):
            canonical, alias = _clean_operand(canonical), _clean_operand(alias)
            if not (_valid_operand(canonical) and _valid_operand(alias)):
                continue
            self._add_entity(extraction, Entity(
                name=canonical, type=infer_entity_type(canonical, text),
                aliases=[alias], note_ids=[note.id],
            ))

    def _collect_capitalized(self, sentence: str, note: Note, extraction: Extraction) -> None:
        """Pick up English proper nouns when no relation pattern matched."""
        for match in _CAPITALIZED_RE.finditer(sentence):
            name = _clean_operand(match.group(0))
            if not _valid_operand(name) or len(name.split()) > 4:
                continue
            if match.start() == 0 and not name.isupper():
                continue  # avoid "This"/"The" at sentence start
            self._add_entity(extraction, Entity(
                name=name, type=infer_entity_type(name, sentence), note_ids=[note.id],
            ))

    def _add_cooccurrence(self, note: Note, chunk: Chunk, extraction: Extraction) -> None:
        names = [e.name for e in extraction.entities if e.type not in {"NOTE", "TAG"}]
        if len(names) < 2:
            return
        existing = {(r.source, r.target) for r in extraction.relations}
        existing |= {(r.target, r.source) for r in extraction.relations}
        added = 0
        for i, left in enumerate(names[:8]):
            for right in names[i + 1:8]:
                if added >= 4:
                    return
                if (left, right) in existing:
                    continue
                extraction.relations.append(
                    RelationDraft(
                        source=left, target=right, type="RELATED_TO",
                        description="同一段落中被共同提及",
                        evidence=truncate(chunk.text, 240), confidence=0.35,
                        chunk_id=chunk.id,
                    )
                )
                added += 1

    def _add_entity(self, extraction: Extraction, entity: Entity) -> None:
        for existing in extraction.entities:
            if existing.key == entity.key:
                existing.aliases = list(dict.fromkeys(existing.aliases + entity.aliases))
                if not existing.description and entity.description:
                    existing.description = entity.description
                return
        if len(extraction.entities) < self.max_entities:
            extraction.entities.append(entity)
