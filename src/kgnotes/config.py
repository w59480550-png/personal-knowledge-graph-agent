"""Runtime configuration for kgnotes.

Environment driven (optionally via ``.env``) so the same code base runs in
three modes:

* fully offline -> SQLite graph store + deterministic local embeddings, no LLM
* Neo4j         -> Neo4j graph store + native vector indexes
* hosted LLM    -> any OpenAI-compatible chat / embeddings endpoint
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

DEFAULT_SYSTEM_PROMPT = (
    "你是一个严谨的个人知识库助手。只依据提供的【知识库上下文】回答，"
    "不要编造。引用信息时在句末标注来源编号，如 [1]。"
    "如果上下文不足以回答，请直接说明“知识库中没有找到相关信息”，并说明还需要什么信息。"
    "回答使用与用户提问相同的语言，先给结论，再给必要的要点。"
)


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader (KEY=VALUE, # comments, optional quotes)."""
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip())
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip())
    except (TypeError, ValueError):
        return default



@dataclass
class Settings:
    # --- storage -----------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("KGNOTES_DATA_DIR", "data")))
    graph_store: str = field(default_factory=lambda: os.getenv("KGNOTES_GRAPH_STORE", "sqlite").lower())
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", ""))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "kgnotes-pass"))
    neo4j_database: str = field(default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j"))

    # --- extraction --------------------------------------------------------
    extractor: str = field(default_factory=lambda: os.getenv("KGNOTES_EXTRACTOR", "auto").lower())
    max_chunk_chars: int = field(default_factory=lambda: _env_int("KGNOTES_MAX_CHUNK_CHARS", 700))
    chunk_overlap: int = field(default_factory=lambda: _env_int("KGNOTES_CHUNK_OVERLAP", 80))
    max_triples_per_chunk: int = field(default_factory=lambda: _env_int("KGNOTES_MAX_TRIPLES_PER_CHUNK", 20))

    # --- embeddings --------------------------------------------------------
    embedding_provider: str = field(
        default_factory=lambda: os.getenv("KGNOTES_EMBEDDING_PROVIDER", "auto").lower()
    )
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    embedding_dim: int = field(default_factory=lambda: _env_int("EMBEDDING_DIM", 384))
    embedding_batch_size: int = field(default_factory=lambda: _env_int("KGNOTES_EMBEDDING_BATCH", 32))

    # --- LLM ---------------------------------------------------------------
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    )
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.1))
    llm_timeout: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT", 60.0))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 1200))
    system_prompt: str = field(default_factory=lambda: os.getenv("KGNOTES_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT))

    # --- retrieval ---------------------------------------------------------
    top_k: int = field(default_factory=lambda: _env_int("KGNOTES_TOP_K", 6))
    graph_depth: int = field(default_factory=lambda: _env_int("KGNOTES_GRAPH_DEPTH", 1))
    graph_neighbor_limit: int = field(default_factory=lambda: _env_int("KGNOTES_GRAPH_NEIGHBORS", 12))
    max_context_chars: int = field(default_factory=lambda: _env_int("KGNOTES_MAX_CONTEXT_CHARS", 7000))
    use_llm_rewrite: bool = field(default_factory=lambda: _env_bool("KGNOTES_LLM_REWRITE", True))

    # --- API ---------------------------------------------------------------
    allow_reset: bool = field(default_factory=lambda: _env_bool("KGNOTES_ALLOW_RESET", True))
    cors_origins: str = field(default_factory=lambda: os.getenv("KGNOTES_CORS_ORIGINS", "*"))

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir).expanduser().resolve()
        if self.graph_store not in {"sqlite", "neo4j"}:
            raise ValueError(f"KGNOTES_GRAPH_STORE must be 'sqlite' or 'neo4j', got {self.graph_store!r}")
        if self.extractor not in {"auto", "heuristic", "llm"}:
            raise ValueError(f"KGNOTES_EXTRACTOR must be auto|heuristic|llm, got {self.extractor!r}")
        if self.embedding_provider not in {"auto", "hash", "openai"}:
            raise ValueError(
                f"KGNOTES_EMBEDDING_PROVIDER must be auto|hash|openai, got {self.embedding_provider!r}"
            )
        if self.graph_store == "neo4j" and not self.neo4j_uri:
            raise ValueError("KGNOTES_GRAPH_STORE=neo4j requires NEO4J_URI")
        if self.chunk_overlap >= self.max_chunk_chars:
            raise ValueError("KGNOTES_CHUNK_OVERLAP must be smaller than KGNOTES_MAX_CHUNK_CHARS")

    # -- helpers ------------------------------------------------------------
    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "kgnotes.sqlite"

    @property
    def vector_path(self) -> Path:
        return self.data_dir / "vectors.sqlite"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def use_openai_embeddings(self) -> bool:
        if self.embedding_provider == "openai":
            return True
        if self.embedding_provider == "hash":
            return False
        return self.llm_enabled  # auto

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "exports").mkdir(parents=True, exist_ok=True)

    def as_public_dict(self) -> dict[str, Any]:
        """Config snapshot for /health - never leaks secrets."""
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["data_dir"] = str(self.data_dir)
        data["llm_api_key"] = "***" if self.llm_api_key else ""
        data["neo4j_password"] = "***" if self.neo4j_password else ""
        data["llm_enabled"] = self.llm_enabled
        data["embedding_backend"] = "openai" if self.use_openai_embeddings else "hash"
        data["system_prompt"] = "…" if self.system_prompt else ""
        return data

    @classmethod
    def from_env(cls, dotenv: str | Path | None = ".env") -> "Settings":
        if dotenv:
            load_dotenv(dotenv)
        return cls()

