from __future__ import annotations

import json

import httpx
import pytest

from kgnotes.llm import LLMClient, LLMError, extract_json_object
from kgnotes.config import Settings


def _client(settings: Settings, handler) -> LLMClient:
    settings.llm_api_key = "test-key"
    settings.llm_base_url = "https://example.test/v1"
    return LLMClient(settings, transport=httpx.MockTransport(handler))


def test_chat_and_json_mode(settings: Settings):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["payload"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "{\"ok\": true}"}}],
        })

    client = _client(settings, handler)
    assert client.chat([{"role": "user", "content": "hi"}], json_mode=True) == '{"ok": true}'
    assert seen["path"].endswith("/chat/completions")
    assert seen["payload"]["response_format"] == {"type": "json_object"}
    assert seen["auth"] == "Bearer test-key"


def test_embeddings_batching(settings: Settings):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload["input"])
        return httpx.Response(200, json={
            "data": [
                {"index": i, "embedding": [float(i), 1.0]}
                for i, _ in enumerate(payload["input"])
            ]
        })

    settings.embedding_batch_size = 2
    client = _client(settings, handler)
    vectors = client.embed(["a", "b", "c"])
    assert len(vectors) == 3
    assert vectors[0] == [0.0, 1.0]
    assert [len(call) for call in calls] == [2, 1]


def test_errors_and_retries(settings: Settings, monkeypatch):
    attempts = {"n": 0}
    monkeypatch.setattr("kgnotes.llm.time.sleep", lambda *_: None)

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(500, text="server error")

    settings.llm_timeout = 0.1
    client = _client(settings, handler)
    with pytest.raises(LLMError):
        client.chat([{"role": "user", "content": "hi"}])
    assert attempts["n"] == 3


def test_extract_json_object_handles_fences_and_noise():
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_object('Here you go: {"a": {"b": 2}} thanks') == {"a": {"b": 2}}
    assert extract_json_object("no json here") == {}
