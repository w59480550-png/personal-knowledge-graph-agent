"""OpenAI-compatible chat + embedding client (works with OpenAI, DeepSeek,
Qwen/DashScope compatible mode, Ollama, vLLM, LM Studio, ...)."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Sequence

import httpx

from .config import Settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Thin, synchronous HTTP client with graceful failure semantics."""

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self._transport = transport  # injectable for tests

    @property
    def enabled(self) -> bool:
        return bool(self.settings.llm_api_key)

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, path: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        url = f"{self.settings.llm_base_url}{path}"
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with httpx.Client(
                    timeout=timeout or self.settings.llm_timeout, transport=self._transport
                ) as client:
                    response = client.post(url, headers=self._headers, json=payload)
                if response.status_code >= 400:
                    raise LLMError(f"{response.status_code} from {path}: {response.text[:400]}")
                return response.json()
            except (httpx.HTTPError, LLMError, ValueError) as exc:  # noqa: PERF203
                last_error = exc
                if attempt < 2:
                    time.sleep(0.6 * (attempt + 1))
        raise LLMError(f"request to {url} failed: {last_error}")

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        if not self.enabled:
            raise LLMError("LLM is not configured (set LLM_API_KEY)")
        payload: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": list(messages),
            "temperature": self.settings.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = self._post("/chat/completions", payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected chat response: {json.dumps(data)[:300]}") from exc
        if isinstance(content, list):  # some providers return content parts
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return (content or "").strip()

    def embed(self, texts: Sequence[str], model: str | None = None) -> list[list[float]]:
        if not self.enabled:
            raise LLMError("LLM is not configured (set LLM_API_KEY)")
        clean = [t.replace("\n", " ").strip() or " " for t in texts]
        vectors: list[list[float]] = []
        batch = max(1, self.settings.embedding_batch_size)
        for start in range(0, len(clean), batch):
            payload = {"model": model or self.settings.embedding_model, "input": clean[start:start + batch]}
            data = self._post("/embeddings", payload)
            try:
                items = sorted(data["data"], key=lambda item: item.get("index", 0))
                vectors.extend([list(map(float, item["embedding"])) for item in items])
            except (KeyError, TypeError, ValueError) as exc:
                raise LLMError(f"unexpected embeddings response: {json.dumps(data)[:300]}") from exc
        return vectors


def extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction from an LLM reply (handles ``` fences)."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}
