"""Persistent multi-turn conversation sessions (SQLite)."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class SessionStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT DEFAULT '',
                    created_at TEXT,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    meta TEXT DEFAULT '{}',
                    created_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
                """
            )
            self._conn.commit()

    def create(self, session_id: str | None = None, title: str = "") -> str:
        session_id = session_id or uuid.uuid4().hex
        now = _now()
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO sessions(id,title,created_at,updated_at) VALUES (?,?,?,?)",
                (session_id, title[:120], now, now),
            )
            self._conn.commit()
        return session_id

    def exists(self, session_id: str) -> bool:
        with self._lock:
            return (
                self._conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone()
                is not None
            )

    def append(
        self,
        session_id: str,
        role: str,
        content: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.create(session_id)
        now = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages(session_id,role,content,meta,created_at) VALUES (?,?,?,?,?)",
                (session_id, role, content, json.dumps(meta or {}, ensure_ascii=False, default=str), now),
            )
            self._conn.execute(
                "UPDATE sessions SET updated_at=?, title=CASE WHEN title='' AND ?='user' THEN ? ELSE title END "
                "WHERE id=?",
                (now, role, content[:120], session_id),
            )
            self._conn.commit()

    def messages(self, session_id: str, limit: int = 20, before_id: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM messages WHERE session_id=?"
        params: list[Any] = [session_id]
        if before_id is not None:
            sql += " AND id < ?"
            params.append(before_id)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(max(1, limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        out = []
        for row in reversed(rows):
            out.append({
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "meta": json.loads(row["meta"] or "{}"),
                "created_at": row["created_at"],
            })
        return out

    def history(self, session_id: str, limit: int = 8) -> list[dict[str, str]]:
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self.messages(session_id, limit=limit)
            if m["role"] in {"user", "assistant"}
        ]

    def last_user_questions(self, session_id: str, n: int = 2) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT content FROM messages WHERE session_id=? AND role='user' ORDER BY id DESC LIMIT ?",
                (session_id, max(1, n)),
            ).fetchall()
        return [row["content"] for row in rows]

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id=s.id) AS message_count
                FROM sessions s ORDER BY s.updated_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "id": row["id"], "title": row["title"], "created_at": row["created_at"],
                "updated_at": row["updated_at"], "message_count": row["message_count"],
            }
            for row in rows
        ]

    def delete(self, session_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def reset(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM messages")
            self._conn.execute("DELETE FROM sessions")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
