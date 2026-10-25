"""Vector index backends.

* ``sqlite`` -> portable local index (default, no services required)
* ``neo4j``  -> native Neo4j vector index over :Chunk.embedding

Both expose the same tiny interface and store only ids + vectors, so payloads
always come from the graph store (single source of truth).
"""

from __future__ import annotations

import array
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Sequence

from .config import Settings
from .embeddings import Embedder
from .text import cosine

log = logging.getLogger(__name__)


class VectorIndexMismatch(RuntimeError):
    pass


class SqliteVectorIndex:
    backend = "sqlite"

    def __init__(self, path: str | Path, model: str, dim: int = 0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.dim = dim
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
                CREATE TABLE IF NOT EXISTS vectors (
                    id TEXT PRIMARY KEY,
                    vector BLOB NOT NULL,
                    dim INTEGER NOT NULL,
                    updated_at TEXT
                );
                """
            )
            self._conn.commit()
            stored_model = self._get_meta("model")
            stored_dim = self._get_meta("dim")
            count = self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
            if count and (stored_model != self.model or (stored_dim and dim and int(stored_dim) != dim)):
                log.warning(
                    "embedding backend changed (%s/%s -> %s/%s); clearing %d stale vectors. "
                    "Re-ingest your notes to rebuild the index.",
                    stored_model, stored_dim, self.model, dim or "?", count,
                )
                self._conn.execute("DELETE FROM vectors")
            self._set_meta("model", self.model)
            if dim:
                self._set_meta("dim", str(dim))
            self._conn.commit()

    def _get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def _ensure_dim(self, dim: int) -> None:
        if not dim:
            return
        if self.dim and self.dim != dim:
            raise VectorIndexMismatch(
                f"embedding dimension mismatch: index={self.dim}, embedder={dim}. "
                "Delete the vector index or re-ingest with a consistent embedding model."
            )
        self.dim = dim
        with self._lock:
            self._set_meta("dim", str(dim))
            self._conn.commit()

    def add(self, items: Sequence[tuple[str, Sequence[float]]]) -> int:
        if not items:
            return 0
        dim = len(items[0][1])
        self._ensure_dim(dim)
        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO vectors(id,vector,dim) VALUES (?,?,?)
                ON CONFLICT(id) DO UPDATE SET vector=excluded.vector, dim=excluded.dim
                """,
                [(item_id, array.array("f", vec).tobytes(), dim) for item_id, vec in items],
            )
            self._conn.commit()
        return len(items)

    @staticmethod
    def _decode(blob: bytes) -> list[float]:
        arr = array.array("f")
        arr.frombytes(blob)
        return list(arr)

    def get(self, item_id: str) -> list[float] | None:
        with self._lock:
            row = self._conn.execute("SELECT vector FROM vectors WHERE id=?", (item_id,)).fetchone()
        return self._decode(row["vector"]) if row else None

    def search(
        self, vector: Sequence[float], limit: int = 10, allowed_ids: set[str] | None = None
    ) -> list[tuple[str, float]]:
        with self._lock:
            rows = self._conn.execute("SELECT id, vector FROM vectors").fetchall()
        scored: list[tuple[str, float]] = []
        for row in rows:
            item_id = row["id"]
            if allowed_ids is not None and item_id not in allowed_ids:
                continue
            score = cosine(vector, self._decode(row["vector"]))
            if score > 0:
                scored.append((item_id, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[: max(1, limit)]

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0])

    def delete(self, item_ids: Sequence[str]) -> int:
        if not item_ids:
            return 0
        with self._lock:
            cur = self._conn.executemany("DELETE FROM vectors WHERE id=?", [(i,) for i in item_ids])
            self._conn.commit()
            return cur.rowcount

    def reset(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM vectors")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class Neo4jVectorIndex:
    backend = "neo4j"
    index_name = "chunk_embeddings"

    def __init__(self, settings: Settings, model: str, dim: int = 0) -> None:
        from neo4j import GraphDatabase  # noqa: PLC0415

        self.settings = settings
        self.model = model
        self.dim = dim
        self._driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
        self._ensure_index()

    def _run(self, query: str, **params):
        with self._driver.session(database=self.settings.neo4j_database) as session:
            result = session.run(query, **params)
            rows = [record.data() for record in result]
            result.consume()
            return rows

    def _ensure_index(self) -> None:
        if not self.dim:
            return
        try:
            rows = self._run(
                "SHOW VECTOR INDEXES YIELD name, options WHERE name = $name RETURN options",
                name=self.index_name,
            )
        except Exception:  # noqa: BLE001 - older servers without SHOW support
            rows = []
        if rows:
            options = rows[0].get("options") or {}
            config = options.get("indexConfig") or {}
            existing_dim = config.get("vector.dimensions")
            if existing_dim and int(existing_dim) != self.dim:
                log.warning("vector index dimension %s -> %s; recreating", existing_dim, self.dim)
                self._run(f"DROP INDEX {self.index_name} IF EXISTS")
        self._run(
            f"""
            CREATE VECTOR INDEX {self.index_name} IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {{indexConfig: {{`vector.dimensions`: {int(self.dim)},
                                     `vector.similarity_function`: 'cosine'}}}}
            """
        )

    def add(self, items: Sequence[tuple[str, Sequence[float]]]) -> int:
        if not items:
            return 0
        dim = len(items[0][1])
        if not self.dim:
            self.dim = dim
            self._ensure_index()
        self._run(
            """
            UNWIND $rows AS row
            MATCH (c:Chunk {id:row.id})
            SET c.embedding = row.vector
            """,
            rows=[{"id": item_id, "vector": [float(v) for v in vec]} for item_id, vec in items],
        )
        return len(items)

    def get(self, item_id: str) -> list[float] | None:
        rows = self._run("MATCH (c:Chunk {id:$id}) RETURN c.embedding AS embedding", id=item_id)
        embedding = rows[0]["embedding"] if rows else None
        return list(embedding) if embedding else None

    def search(
        self, vector: Sequence[float], limit: int = 10, allowed_ids: set[str] | None = None
    ) -> list[tuple[str, float]]:
        if not self.dim:
            return []
        candidates = max(limit * 4, 20) if allowed_ids else limit
        rows = self._run(
            f"""
            CALL db.index.vector.queryNodes('{self.index_name}', $k, $vector)
            YIELD node, score
            RETURN node.id AS id, score
            """,
            k=max(1, candidates), vector=[float(v) for v in vector],
        )
        out = [
            (row["id"], float(row["score"])) for row in rows
            if allowed_ids is None or row["id"] in allowed_ids
        ]
        return out[: max(1, limit)]

    def count(self) -> int:
        rows = self._run("MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c) AS n")
        return int(rows[0]["n"]) if rows else 0

    def delete(self, item_ids: Sequence[str]) -> int:
        if not item_ids:
            return 0
        self._run(
            "MATCH (c:Chunk) WHERE c.id IN $ids REMOVE c.embedding",
            ids=list(item_ids),
        )
        return len(item_ids)

    def reset(self) -> None:
        self._run("MATCH (c:Chunk) REMOVE c.embedding")

    def close(self) -> None:
        self._driver.close()


def build_vector_index(settings: Settings, embedder: Embedder):
    dim = embedder.dim or (settings.embedding_dim if not settings.use_openai_embeddings else 0)
    if settings.graph_store == "neo4j":
        return Neo4jVectorIndex(settings, embedder.name, dim)
    return SqliteVectorIndex(settings.vector_path, embedder.name, dim)

