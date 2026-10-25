from __future__ import annotations

import json

from kgnotes.config import Settings
from kgnotes.extraction.heuristic import HeuristicExtractor, infer_entity_type
from kgnotes.extraction.llm import LLMExtractor
from kgnotes.extraction.pipeline import ExtractorPipeline
from kgnotes.llm import LLMError
from kgnotes.notes import NoteLoader


def _chunks(text: str):
    loader = NoteLoader(max_chunk_chars=600, chunk_overlap=40)
    note = loader.from_text(text, title="测试笔记")
    return note, loader.chunk(note)


def test_heuristic_note_graph_has_tags_and_wikilinks():
    loader = NoteLoader()
    note = loader.from_text(
        "---\ntags: [kg, neo4j]\n---\n# 测试笔记\n\n链接到 [[知识图谱助手项目]]。",
        path="demo.md",
    )
    extraction = HeuristicExtractor().note_graph(note)
    names = {entity.name for entity in extraction.entities}
    assert {"测试笔记", "kg", "neo4j", "知识图谱助手项目"} <= names
    triples = {(rel.source, rel.type, rel.target) for rel in extraction.relations}
    assert ("测试笔记", "TAGGED_WITH", "kg") in triples
    assert ("测试笔记", "LINKS_TO", "知识图谱助手项目") in triples


def test_heuristic_relations_chinese():
    note, chunks = _chunks("Neo4j 是一个图数据库。王小明就职于示例科技公司。示例科技公司位于北京。")
    extraction = HeuristicExtractor().extract(note, chunks[0])
    triples = {(rel.source, rel.type, rel.target) for rel in extraction.relations}
    assert ("Neo4j", "IS_A", "图数据库") in triples
    assert ("王小明", "WORKS_AT", "示例科技公司") in triples
    assert ("示例科技公司", "LOCATED_IN", "北京") in triples


def test_heuristic_is_deterministic_and_bounded():
    note, chunks = _chunks("A 使用 B。B 使用 C。C 使用 D。D 使用 E。")
    extractor = HeuristicExtractor(max_triples=2)
    first = extractor.extract(note, chunks[0])
    second = extractor.extract(note, chunks[0])
    assert len(first.relations) <= 2
    assert [(r.source, r.type, r.target) for r in first.relations] == [
        (r.source, r.type, r.target) for r in second.relations
    ]


def test_entity_type_inference():
    assert infer_entity_type("Neo4j", "图数据库") == "TECHNOLOGY"
    assert infer_entity_type("示例科技公司") == "ORGANIZATION"
    assert infer_entity_type("知识图谱助手项目") == "PROJECT"
    assert infer_entity_type("北京市") == "LOCATION"
    assert infer_entity_type("GraphRAG") == "CONCEPT"


class FakeLLM:
    enabled = True

    def __init__(self, payload: dict | None = None, fail: bool = False) -> None:
        self.payload = payload or {}
        self.fail = fail
        self.calls = 0

    def chat(self, messages, **kwargs) -> str:
        self.calls += 1
        if self.fail:
            raise LLMError("boom")
        return json.dumps(self.payload, ensure_ascii=False)


def test_llm_extractor_parses_json_and_resolves_missing_entities():
    note, chunks = _chunks("Neo4j 使用 Cypher。")
    fake = FakeLLM({
        "entities": [{"name": "Neo4j", "type": "TECHNOLOGY", "description": "图数据库"}],
        "relations": [
            {"source": "Neo4j", "target": "Cypher", "type": "uses",
             "evidence": "Neo4j 使用 Cypher。", "confidence": 0.9},
        ],
    })
    extraction = LLMExtractor(fake).extract(note, chunks[0])
    assert {e.name for e in extraction.entities} == {"Neo4j", "Cypher"}
    assert extraction.entities[0].type == "TECHNOLOGY"
    assert extraction.relations[0].type == "USES"
    assert extraction.relations[0].confidence == 0.9


def test_llm_extractor_rejects_non_json():
    note, chunks = _chunks("Neo4j 使用 Cypher。")
    fake = FakeLLM()
    fake.chat = lambda messages, **kwargs: "not json"  # type: ignore[assignment]
    try:
        LLMExtractor(fake).extract(note, chunks[0])
    except LLMError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected LLMError")


def test_pipeline_auto_falls_back_to_heuristic(settings: Settings):
    settings.extractor = "auto"
    note, chunks = _chunks("Neo4j 是一个图数据库。")
    pipeline = ExtractorPipeline(settings, FakeLLM(fail=True))
    extraction = pipeline.extract(note, chunks[0])
    assert pipeline.name == "llm+heuristic-fallback"
    assert ("Neo4j", "IS_A", "图数据库") in {
        (r.source, r.type, r.target) for r in extraction.relations
    }


def test_pipeline_strict_llm_propagates_errors(settings: Settings):
    settings.extractor = "llm"
    note, chunks = _chunks("Neo4j 是一个图数据库。")
    pipeline = ExtractorPipeline(settings, FakeLLM(fail=True))
    try:
        pipeline.extract(note, chunks[0])
    except LLMError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected LLMError")
