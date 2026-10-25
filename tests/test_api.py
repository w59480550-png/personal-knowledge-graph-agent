from __future__ import annotations

from fastapi.testclient import TestClient

from kgnotes.agent import KnowledgeAgent
from kgnotes.api import create_app
from kgnotes.config import Settings


def _client(settings: Settings, agent: KnowledgeAgent) -> TestClient:
    return TestClient(create_app(settings, agent))


def test_note_crud_search_ask_and_graph(settings: Settings, agent: KnowledgeAgent):
    client = _client(settings, agent)

    created = client.post("/notes", json={
        "title": "API 测试笔记",
        "content": "# API 测试笔记\n\n知识图谱助手项目使用 Neo4j。王小明就职于示例科技公司。",
        "tags": ["api", "kg"],
    })
    assert created.status_code == 201
    note_id = created.json()["note_id"]
    assert created.json()["relations"] >= 1

    listed = client.get("/notes")
    assert listed.status_code == 200
    assert listed.json()["count"] == 1

    detail = client.get(f"/notes/{note_id}")
    assert detail.status_code == 200
    assert detail.json()["note"]["title"] == "API 测试笔记"
    assert detail.json()["chunks"]

    entities = client.get("/entities", params={"query": "Neo4j"})
    assert entities.status_code == 200
    assert entities.json()["count"] >= 1

    graph = client.get("/graph", params={"entity": "Neo4j", "depth": 1})
    assert graph.status_code == 200
    assert graph.json()["entity"]["name"] == "Neo4j"
    assert client.get("/graph", params={"entity": "does-not-exist"}).status_code == 404

    search = client.post("/search", json={"query": "王小明", "top_k": 3})
    assert search.status_code == 200
    assert search.json()["citations"]

    first = client.post("/ask", json={"question": "王小明在哪里工作？"})
    assert first.status_code == 200
    payload = first.json()
    assert "示例科技公司" in payload["answer"]
    assert payload["citations"]
    session_id = payload["session_id"]

    followup = client.post("/ask", json={"question": "他是谁？", "session_id": session_id})
    assert followup.status_code == 200
    assert followup.json()["session_id"] == session_id

    messages = client.get(f"/sessions/{session_id}")
    assert messages.status_code == 200
    assert len(messages.json()["messages"]) == 4

    stats = client.get("/stats")
    assert stats.status_code == 200
    assert stats.json()["graph"]["notes"] == 1

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["config"]["llm_api_key"] == ""

    assert client.delete(f"/notes/{note_id}").status_code == 200
    assert client.get(f"/notes/{note_id}").status_code == 404


def test_api_validation_errors(settings: Settings, agent: KnowledgeAgent):
    client = _client(settings, agent)
    assert client.post("/notes", json={"content": ""}).status_code == 422
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.get("/sessions/not-found").status_code == 404
    assert client.delete("/sessions/not-found").status_code == 404


def test_api_reset_and_reindex(settings: Settings, agent: KnowledgeAgent):
    client = _client(settings, agent)
    client.post("/notes", json={"content": "Neo4j 是一个图数据库。"})
    assert client.post("/reindex").status_code == 200
    reset = client.post("/admin/reset")
    assert reset.status_code == 200
    assert client.get("/stats").json()["graph"]["notes"] == 0


def test_api_reset_can_be_disabled(settings: Settings, agent: KnowledgeAgent):
    settings.allow_reset = False
    client = _client(settings, agent)
    assert client.post("/admin/reset").status_code == 403
