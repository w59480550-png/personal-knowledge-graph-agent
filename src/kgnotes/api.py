"""HTTP API (FastAPI): ingest notes, search, follow-up Q&A, graph inspection."""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import __version__
from .agent import KnowledgeAgent
from .config import Settings

log = logging.getLogger(__name__)


class NoteIn(BaseModel):
    content: str = Field(min_length=1, description="Markdown / plain-text note body")
    title: str | None = Field(default=None, max_length=200)
    path: str = Field(default="", max_length=500)
    tags: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=100)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)


def create_app(settings: Settings | None = None, agent: KnowledgeAgent | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        if app.state.owns_agent and app.state.agent is not None:
            app.state.agent.close()

    app = FastAPI(
        title="Personal Knowledge Graph Notes Agent",
        version=__version__,
        description="Notes -> entities/relations -> graph + vector index -> follow-up Q&A",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.agent = agent
    app.state.owns_agent = agent is None

    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_agent() -> KnowledgeAgent:
        if app.state.agent is None:
            with lock:
                if app.state.agent is None:
                    app.state.agent = KnowledgeAgent(settings)
                    app.state.owns_agent = True
        return app.state.agent

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return _json_error(400, str(exc))

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError):
        log.warning("runtime error on %s: %s", request.url.path, exc)
        return _json_error(503, str(exc))

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "name": "kgnotes",
            "version": __version__,
            "docs": "/docs",
            "endpoints": [
                "POST /notes", "GET /notes", "GET /notes/{id}", "DELETE /notes/{id}",
                "POST /search", "POST /ask", "GET /graph", "GET /entities",
                "GET /stats", "GET /sessions", "POST /reindex",
            ],
        }

    @app.get("/health")
    def health(a: KnowledgeAgent = Depends(get_agent)) -> dict[str, Any]:
        try:
            stats = a.stats()
            status = "ok"
        except Exception as exc:  # noqa: BLE001
            log.exception("health check failed")
            stats = {"error": str(exc)}
            status = "degraded"
        return {
            "status": status,
            "version": __version__,
            "config": settings.as_public_dict(),
            "stats": stats,
        }

    @app.post("/notes", status_code=201)
    def create_note(payload: NoteIn, a: KnowledgeAgent = Depends(get_agent)) -> dict[str, Any]:
        summary = a.ingest_text(
            payload.content,
            title=payload.title,
            path=payload.path,
            tags=payload.tags,
            aliases=payload.aliases,
            metadata=payload.metadata,
        )
        return summary.as_dict()

    @app.get("/notes")
    def list_notes(
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        a: KnowledgeAgent = Depends(get_agent),
    ) -> dict[str, Any]:
        notes = a.store.list_notes(limit=limit, offset=offset)
        return {
            "notes": [
                {
                    "id": note.id,
                    "title": note.title,
                    "path": note.path,
                    "tags": note.tags,
                    "created_at": note.created_at,
                    "preview": note.content[:160],
                }
                for note in notes
            ],
            "count": len(notes),
        }

    @app.get("/notes/{note_id}")
    def get_note(note_id: str, a: KnowledgeAgent = Depends(get_agent)) -> dict[str, Any]:
        note = a.store.get_note(note_id)
        if note is None:
            raise HTTPException(status_code=404, detail="note not found")
        chunks = [chunk.as_dict() for chunk, _ in a.store.iter_chunks() if chunk.note_id == note_id]
        return {"note": note.as_dict(), "chunks": chunks}

    @app.delete("/notes/{note_id}")
    def delete_note(note_id: str, a: KnowledgeAgent = Depends(get_agent)) -> dict[str, Any]:
        if not a.delete_note(note_id):
            raise HTTPException(status_code=404, detail="note not found")
        return {"deleted": note_id}

    @app.post("/search")
    def search(payload: SearchIn, a: KnowledgeAgent = Depends(get_agent)) -> dict[str, Any]:
        result = a.search(payload.query, top_k=payload.top_k)
        return {
            "query": payload.query,
            "citations": [c.as_dict() for c in result.citations],
            "entities": [e.as_dict() for e in result.entities],
            "relations": [r.as_dict() for r in result.relations],
            "stats": result.stats,
        }

    @app.post("/ask")
    def ask(payload: AskIn, a: KnowledgeAgent = Depends(get_agent)) -> dict[str, Any]:
        return a.ask(payload.question, session_id=payload.session_id, top_k=payload.top_k).as_dict()

    @app.get("/graph")
    def graph(
        entity: str = Query(..., min_length=1, max_length=200),
        depth: int = Query(1, ge=1, le=3),
        limit: int = Query(50, ge=1, le=200),
        a: KnowledgeAgent = Depends(get_agent),
    ) -> dict[str, Any]:
        data = a.graph(entity, depth=depth, limit=limit)
        if not data.get("entity"):
            raise HTTPException(status_code=404, detail=f"entity {entity!r} not found")
        return data

    @app.get("/entities")
    def entities(
        query: str | None = Query(None, max_length=200),
        type: str | None = Query(None, max_length=40),
        limit: int = Query(100, ge=1, le=500),
        a: KnowledgeAgent = Depends(get_agent),
    ) -> dict[str, Any]:
        rows = a.store.list_entities(query=query, entity_type=type, limit=limit)
        return {"entities": [e.as_dict() for e in rows], "count": len(rows)}

    @app.get("/stats")
    def stats(a: KnowledgeAgent = Depends(get_agent)) -> dict[str, Any]:
        return a.stats()

    @app.get("/sessions")
    def list_sessions(
        limit: int = Query(50, ge=1, le=200), a: KnowledgeAgent = Depends(get_agent)
    ) -> dict[str, Any]:
        return {"sessions": a.sessions.list_sessions(limit=limit)}

    @app.get("/sessions/{session_id}")
    def get_session(
        session_id: str,
        limit: int = Query(50, ge=1, le=200),
        a: KnowledgeAgent = Depends(get_agent),
    ) -> dict[str, Any]:
        if not a.sessions.exists(session_id):
            raise HTTPException(status_code=404, detail="session not found")
        return {"session_id": session_id, "messages": a.sessions.messages(session_id, limit=limit)}

    @app.delete("/sessions/{session_id}")
    def delete_session(session_id: str, a: KnowledgeAgent = Depends(get_agent)) -> dict[str, Any]:
        if not a.sessions.delete(session_id):
            raise HTTPException(status_code=404, detail="session not found")
        return {"deleted": session_id}

    @app.post("/reindex")
    def reindex(a: KnowledgeAgent = Depends(get_agent)) -> dict[str, Any]:
        return a.reindex()

    @app.post("/admin/reset")
    def reset(a: KnowledgeAgent = Depends(get_agent)) -> dict[str, Any]:
        if not settings.allow_reset:
            raise HTTPException(status_code=403, detail="reset disabled (KGNOTES_ALLOW_RESET=false)")
        a.reset(include_sessions=True)
        return {"reset": True}

    return app


def _json_error(status_code: int, message: str):
    from fastapi.responses import JSONResponse  # local import keeps module import light

    return JSONResponse(status_code=status_code, content={"detail": message})


app = create_app()
