from __future__ import annotations

import json

from kgnotes.agent import KnowledgeAgent


NOTES = {
    "kgnotes-project.md": (
        "# 知识图谱助手项目\n\n"
        "知识图谱助手项目使用 Neo4j。知识图谱助手项目使用 FastAPI。\n"
        "知识图谱助手项目的作者是王小明。王小明就职于示例科技公司。\n"
        "知识图谱助手项目包含向量检索模块。"
    ),
    "neo4j.md": "# Neo4j\n\nNeo4j 是一个图数据库。Neo4j 使用 Cypher。",
}


def _seed(agent: KnowledgeAgent) -> None:
    for filename, content in NOTES.items():
        agent.ingest_text(content, path=filename)


def test_ingest_builds_graph_and_vectors(agent: KnowledgeAgent):
    summary = agent.ingest_text(NOTES["kgnotes-project.md"], path="kgnotes-project.md")
    assert summary.chunks >= 1
    assert summary.entities >= 3
    assert summary.relations >= 2
    assert summary.new_note is True

    stats = agent.stats()
    assert stats["graph"]["notes"] == 1
    assert stats["graph"]["entities"] >= 3
    assert stats["graph"]["relation_types"].get("USES", 0) >= 1
    assert stats["vectors"]["count"] == summary.chunks

    updated = agent.ingest_text(NOTES["kgnotes-project.md"], path="kgnotes-project.md")
    assert updated.new_note is False


def test_reingest_replaces_stale_chunks_and_vectors(agent: KnowledgeAgent):
    long_text = "。".join(f"第{i}段内容介绍 Neo4j 图数据库" for i in range(40)) + "。"
    first = agent.ingest_text(long_text, title="会缩短的笔记", path="resize.md")
    assert first.chunks > 2
    second = agent.ingest_text("Neo4j 是一个图数据库。", title="会缩短的笔记", path="resize.md")
    assert second.new_note is False
    stats = agent.stats()
    assert stats["graph"]["chunks"] == second.chunks
    assert stats["vectors"]["count"] == second.chunks


def test_search_returns_hybrid_citations(agent: KnowledgeAgent):
    _seed(agent)
    result = agent.search("王小明在哪里工作？")
    assert result.citations
    assert any("王小明" in entity.name for entity in result.entities)
    assert any(rel.type in {"WORKS_AT", "AUTHORED"} for rel in result.relations)
    assert result.stats["keyword_hits"] > 0
    assert result.stats["indexed_vectors"] >= 1


def test_ask_extractive_and_followup(agent: KnowledgeAgent):
    _seed(agent)
    first = agent.ask("王小明在哪里工作？")
    assert first.citations
    assert "示例科技公司" in first.answer
    assert first.mode == "extractive"
    assert first.session_id

    second = agent.ask("那他是谁？", session_id=first.session_id)
    assert second.session_id == first.session_id
    assert "王小明" in second.rewritten_question
    assert second.rewritten_question != second.question

    messages = agent.sessions.messages(first.session_id)
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]


def test_no_context_is_explicit(agent: KnowledgeAgent):
    answer = agent.ask("火星上的量子咖啡机保修几年？")
    assert answer.mode == "no-context"
    assert ("没有找到" in answer.answer) or ("couldn't find" in answer.answer.lower())


def test_delete_note_cleans_vectors(agent: KnowledgeAgent):
    _seed(agent)
    total_chunks = agent.stats()["graph"]["chunks"]
    note = agent.store.list_notes()[0]
    assert agent.delete_note(note.id) is True
    stats = agent.stats()
    assert stats["graph"]["notes"] == 1
    assert stats["vectors"]["count"] < total_chunks


def test_reindex_and_export(agent: KnowledgeAgent, tmp_path):
    _seed(agent)
    result = agent.reindex()
    assert result["chunks"] == agent.stats()["graph"]["chunks"]
    out = tmp_path / "graph.json"
    data = agent.export_graph(out)
    assert data["entities"] and data["relations"]
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["entities"]) == len(data["entities"])


def test_reset_clears_everything(agent: KnowledgeAgent):
    _seed(agent)
    agent.ask("Neo4j 是什么？")
    agent.reset()
    stats = agent.stats()
    assert stats["graph"]["notes"] == 0
    assert stats["vectors"]["count"] == 0
    assert stats["sessions"] == 0


def test_graph_neighbourhood(agent: KnowledgeAgent):
    _seed(agent)
    data = agent.graph("知识图谱助手项目", depth=1)
    assert data["entity"]["name"] == "知识图谱助手项目"
    assert data["entities"]
    assert any(rel["type"] == "USES" for rel in data["relations"])
    assert agent.graph("不存在的实体")["entity"] is None
