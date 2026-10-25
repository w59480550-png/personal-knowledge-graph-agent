"""Command line interface: ``python -m kgnotes ...`` / ``kgnotes ...``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .agent import KnowledgeAgent
from .config import Settings


def _apply_overrides(args: argparse.Namespace) -> None:
    mapping = {
        "data_dir": "KGNOTES_DATA_DIR",
        "graph_store": "KGNOTES_GRAPH_STORE",
        "extractor": "KGNOTES_EXTRACTOR",
        "neo4j_uri": "NEO4J_URI",
        "embedding_provider": "KGNOTES_EMBEDDING_PROVIDER",
        "llm_model": "LLM_MODEL",
    }
    for attr, env in mapping.items():
        value = getattr(args, attr, None)
        if value:
            os.environ[env] = str(value)


def _settings(args: argparse.Namespace) -> Settings:
    _apply_overrides(args)
    return Settings.from_env()


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def _echo(message: str = "") -> None:
    print(message)


# -- commands ---------------------------------------------------------------
def cmd_init(args: argparse.Namespace) -> int:
    settings = _settings(args)
    settings.ensure_dirs()
    with KnowledgeAgent(settings) as agent:
        _echo(f"✓ 数据目录: {settings.data_dir}")
        _echo(f"✓ 图存储: {settings.graph_store}")
        _echo(f"✓ 向量后端: {getattr(agent.vector_index, 'backend', '?')} / {agent.embedder.name}")
        _echo(f"✓ 抽取器: {agent.extractors.name}")
        _echo(f"{'✓' if agent.llm.enabled else '·'} LLM: "
              f"{settings.llm_model if agent.llm.enabled else '未配置（抽取式回答模式）'}")
        _echo(f"✓ 统计: {json.dumps(agent.stats()['graph'], ensure_ascii=False)}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = _settings(args)
    paths = [Path(p) for p in args.paths]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        _echo(f"✗ 路径不存在: {', '.join(missing)}")
        return 2
    if args.reset:
        _echo("· 正在清空现有图与向量索引…")
    with KnowledgeAgent(settings) as agent:
        if args.reset:
            agent.reset(include_sessions=False)
        summaries = []
        for path in paths:
            if path.is_dir():
                summaries.extend(agent.ingest_dir(path))
            else:
                summaries.append(agent.ingest_file(path, root=path.parent))
    if args.json:
        _print_json([s.as_dict() for s in summaries])
        return 0
    total_entities = sum(s.entities for s in summaries)
    total_relations = sum(s.relations for s in summaries)
    total_chunks = sum(s.chunks for s in summaries)
    for summary in summaries:
        flag = "新增" if summary.new_note else "更新"
        _echo(
            f"✓ [{flag}] {summary.title}  "
            f"块={summary.chunks} 实体={summary.entities} 关系={summary.relations} "
            f"({summary.elapsed_ms}ms)"
        )
        for warning in summary.warnings:
            _echo(f"  ! {warning}")
    _echo(
        f"\n完成：{len(summaries)} 篇笔记，{total_chunks} 个片段，"
        f"{total_entities} 个实体提及，{total_relations} 条关系。"
    )
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    settings = _settings(args)
    with KnowledgeAgent(settings) as agent:
        answer = agent.ask(args.question, session_id=args.session, top_k=args.top_k)
    if args.json:
        _print_json(answer.as_dict())
        return 0
    _echo(answer.answer)
    if args.show_context and answer.citations:
        _echo("\n--- 引用片段 ---")
        for citation in answer.citations:
            _echo(f"[{citation.index}] 《{citation.note_title}》 第{citation.chunk_index + 1}段 "
                  f"({citation.reason})")
            _echo(citation.text)
            _echo()
    if answer.citations:
        _echo("\n来源：")
        for citation in answer.citations:
            _echo(f"  [{citation.index}] 《{citation.note_title}》 · 第{citation.chunk_index + 1}段")
    _echo(f"\n(session: {answer.session_id} · mode: {answer.mode} · {answer.latency_ms}ms)")
    if answer.rewritten_question != answer.question:
        _echo(f"(检索改写: {answer.rewritten_question})")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    settings = _settings(args)
    _echo("知识图谱问答（输入 exit / quit 退出，/new 开始新会话）")
    with KnowledgeAgent(settings) as agent:
        session_id = args.session
        while True:
            try:
                question = input("\n你> ").strip()
            except (EOFError, KeyboardInterrupt):
                _echo()
                break
            if not question:
                continue
            if question.lower() in {"exit", "quit", "q"}:
                break
            if question == "/new":
                session_id = None
                _echo("· 已开始新会话")
                continue
            try:
                answer = agent.ask(question, session_id=session_id)
            except Exception as exc:  # noqa: BLE001
                _echo(f"✗ {exc}")
                continue
            session_id = answer.session_id
            _echo(f"\n助手> {answer.answer}")
            if answer.citations:
                refs = "；".join(
                    f"[{c.index}]《{c.note_title}》" for c in answer.citations[:4]
                )
                _echo(f"\n来源：{refs}")
            if answer.rewritten_question != answer.question:
                _echo(f"(检索改写: {answer.rewritten_question})")

def cmd_search(args: argparse.Namespace) -> int:
    settings = _settings(args)
    with KnowledgeAgent(settings) as agent:
        result = agent.search(args.query, top_k=args.top_k)
    if args.json:
        _print_json({
            "query": args.query,
            "citations": [c.as_dict() for c in result.citations],
            "entities": [e.as_dict() for e in result.entities],
            "relations": [r.as_dict() for r in result.relations],
            "stats": result.stats,
        })
        return 0
    if not result.citations:
        _echo("没有找到相关内容。")
    for citation in result.citations:
        _echo(f"[{citation.index}] 《{citation.note_title}》 第{citation.chunk_index + 1}段 "
              f"score={citation.score:.4f} ({citation.reason})")
        _echo(f"    {citation.text[:300]}")
    if result.entities:
        _echo("\n相关实体：" + "、".join(f"{e.name}({e.type})" for e in result.entities))
    if result.relations:
        _echo("\n图谱关系：")
        for rel in result.relations[:10]:
            _echo(f"  {rel.source_name} --{rel.type}--> {rel.target_name}")
    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    settings = _settings(args)
    with KnowledgeAgent(settings) as agent:
        data = agent.graph(args.entity, depth=args.depth, limit=args.limit)
    if not data.get("entity"):
        _echo(f"未找到实体：{args.entity}")
        return 1
    if args.json:
        _print_json(data)
        return 0
    entity = data["entity"]
    _echo(f"实体：{entity['name']} ({entity['type']}) id={entity['id']}")
    if entity.get("description"):
        _echo(f"说明：{entity['description']}")
    if data["relations"]:
        _echo(f"\n关系（{len(data['relations'])}）：")
        for rel in data["relations"]:
            _echo(f"  {rel['source']} --{rel['type']}--> {rel['target']} "
                  f"(置信度 {rel['confidence']})")
    if data["entities"]:
        _echo(f"\n邻居实体（{len(data['entities'])}）：")
        for item in data["entities"]:
            _echo(f"  - {item['name']} ({item['type']})")
    return 0


def cmd_entities(args: argparse.Namespace) -> int:
    settings = _settings(args)
    with KnowledgeAgent(settings) as agent:
        rows = agent.store.list_entities(query=args.query, entity_type=args.type, limit=args.limit)
    if args.json:
        _print_json([e.as_dict() for e in rows])
        return 0
    for entity in rows:
        _echo(f"- {entity.name} ({entity.type}) 笔记数={len(entity.note_ids)}"
              + (f"  {entity.description[:60]}" if entity.description else ""))
    _echo(f"\n共 {len(rows)} 个实体")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    settings = _settings(args)
    with KnowledgeAgent(settings) as agent:
        data = agent.stats()
    _print_json(data)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    settings = _settings(args)
    out = Path(args.out) if args.out else settings.data_dir / "exports" / "graph.json"
    with KnowledgeAgent(settings) as agent:
        data = agent.export_graph(out)
    _echo(f"✓ 已导出 {len(data['entities'])} 个实体、{len(data['relations'])} 条关系 -> {out}")
    return 0


def cmd_reindex(args: argparse.Namespace) -> int:
    settings = _settings(args)
    with KnowledgeAgent(settings) as agent:
        result = agent.reindex()
    _echo(f"✓ 已重建向量索引：{result['chunks']} 个片段（{result['model']}）")
    return 0


def cmd_delete_note(args: argparse.Namespace) -> int:
    settings = _settings(args)
    with KnowledgeAgent(settings) as agent:
        deleted = agent.delete_note(args.note_id)
    _echo("✓ 已删除" if deleted else "✗ 未找到该笔记")
    return 0 if deleted else 1


def cmd_reset(args: argparse.Namespace) -> int:
    if not args.yes:
        _echo("此操作会清空所有笔记、图谱、向量和会话。确认请加 --yes。")
        return 2
    settings = _settings(args)
    with KnowledgeAgent(settings) as agent:
        agent.reset(include_sessions=True)
    _echo("✓ 已重置")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    _settings(args)  # apply overrides before uvicorn imports kgnotes.api
    try:
        import uvicorn
    except ImportError:
        _echo("✗ 需要安装 uvicorn：pip install uvicorn")
        return 2
    _echo(f"→ http://{args.host}:{args.port}  (OpenAPI: /docs)")
    uvicorn.run("kgnotes.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0



# -- parser -----------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kgnotes",
        description="个人知识图谱笔记 Agent：实体关系抽取 + 图存储 + 向量检索 + 追问问答",
    )
    parser.add_argument("--version", action="version", version=f"kgnotes {__version__}")
    parser.add_argument("--data-dir", help="数据目录（默认 data/）")
    parser.add_argument("--graph-store", choices=["sqlite", "neo4j"], help="图存储后端")
    parser.add_argument("--extractor", choices=["auto", "heuristic", "llm"], help="抽取器")
    parser.add_argument("--neo4j-uri", help="Neo4j URI，如 bolt://localhost:7687")
    parser.add_argument("--embedding-provider", choices=["auto", "hash", "openai"], help="向量后端")
    parser.add_argument("--llm-model", help="LLM 模型名")

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="初始化数据目录并显示配置")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("ingest", help="导入 Markdown 文件或目录")
    p.add_argument("paths", nargs="+", help="笔记文件或目录")
    p.add_argument("--reset", action="store_true", help="导入前清空现有数据")
    p.add_argument("--json", action="store_true", help="输出 JSON")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="提问（支持多轮，用 --session 续接）")
    p.add_argument("question")
    p.add_argument("--session", help="会话 id（用于追问）")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--show-context", action="store_true", help="显示引用原文")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="交互式多轮问答")
    p.add_argument("--session", help="续接已有会话 id")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("search", help="只检索，不生成回答")
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("graph", help="查看某个实体的邻域子图")
    p.add_argument("entity", help="实体名称或 id")
    p.add_argument("--depth", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_graph)

    p = sub.add_parser("entities", help="列出/搜索实体")
    p.add_argument("--query")
    p.add_argument("--type")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_entities)

    p = sub.add_parser("stats", help="查看统计信息")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("export", help="导出图谱 JSON")
    p.add_argument("--out")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("reindex", help="重建向量索引")
    p.set_defaults(func=cmd_reindex)

    p = sub.add_parser("delete-note", help="按 id 删除笔记")
    p.add_argument("note_id")
    p.set_defaults(func=cmd_delete_note)

    p = sub.add_parser("reset", help="清空全部数据")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_reset)

    p = sub.add_parser("serve", help="启动 HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        _echo(f"✗ {exc}")
        return 1
    except KeyboardInterrupt:
        _echo("\n已取消")
        return 130


if __name__ == "__main__":
    sys.exit(main())
