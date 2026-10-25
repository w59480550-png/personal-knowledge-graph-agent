from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from kgnotes.agent import KnowledgeAgent  # noqa: E402
from kgnotes.config import Settings  # noqa: E402


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        graph_store="sqlite",
        extractor="heuristic",
        embedding_provider="hash",
        embedding_dim=256,
        llm_api_key="",
        top_k=5,
        max_chunk_chars=240,
        chunk_overlap=40,
    )


@pytest.fixture
def agent(settings: Settings):
    instance = KnowledgeAgent(settings)
    yield instance
    instance.close()
