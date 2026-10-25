---
title: Python asyncio notes
tags: [python, asyncio, backend]
---

# Python asyncio notes

asyncio is a Python library. asyncio uses an event loop. FastAPI is built with asyncio.

## Patterns

Use `asyncio.gather` for independent I/O tasks. Avoid blocking calls inside coroutines; offload
CPU-bound work to `run_in_executor`. The `httpx.AsyncClient` supports connection pooling.

## Pitfalls

Creating a task without keeping a reference can be garbage collected. Always handle cancellation
with `try/finally`. Structured concurrency via `TaskGroup` is available in Python 3.11.

Related: [[知识图谱助手项目]] #python #asyncio
