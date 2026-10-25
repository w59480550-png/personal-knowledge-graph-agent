"""Optional Neo4j integration test.

Skipped unless ``KGNOTES_TEST_NEO4J_URI`` is set:

    KGNOTES_TEST_NEO4J_URI=bolt://localhost:7687 \
    NEO4J_PASSWORD=kgnotes-pass \
    python -m pytest tests/test_neo4j_optional.py -q
"""

from __future__ import annotations

import os

import pytest

from kgnotes.agent import KnowledgeAgent
from kgnotes.config import Settings

URI = os.getenv("KGNOTES_TEST_NEO4J_URI")

pytestmark = pytest.mark.skipif(not URI, reason="KGNOTES_TEST_NEO4J_URI not set")


def test_neo4j_ingest_search_and_reset(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        graph_store="neo4j",
        neo4j_uri=URI,
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "kgnotes-pass"),
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
        extractor="heuristic",
        embedding_provider="hash",
        embedding_dim=128,
        llm_api_key="",
    )
    with KnowledgeAgent(settings) as agent:
        agent.reset(include_sessions=False)
        summary = agent.ingest_text(
            "知识图谱助手项目使用 Neo4j。王小明就职于示例科技公司。",
            title="Neo4j 集成测试",
            path="neo4j-integration.md",
        )
        assert summary.relations >= 2
        stats = agent.stats()
        assert stats["graph"]["backend"] == "neo4j"
        assert stats["graph"]["notes"] == 1
        assert stats["vectors"]["count"] == summary.chunks

        result = agent.search("王小明在哪里工作")
        assert result.citations

        answer = agent.ask("王小明在哪里工作？")
        assert "示例科技公司" in answer.answer

        graph = agent.graph("Neo4j", depth=1)
        assert graph["entity"]["name"] == "Neo4j"
        assert graph["relations"]

        agent.reset(include_sessions=False)
        assert agent.stats()["graph"]["notes"] == 0
