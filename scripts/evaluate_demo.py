"""Reproducible demo evaluation for the report.

Runs fully offline (heuristic extractor + hash embeddings) against the sample
notes, then prints / writes:

* corpus statistics
* relation extraction precision & recall on a curated set and a held-out
  paraphrase set
* entity type accuracy for non-note entities
* keyword-graded QA accuracy (13 questions + 1 follow-up)
* end-to-end ask / retrieval latency (avg, p50, p95)

Usage:
    python scripts/evaluate_demo.py
    python scripts/evaluate_demo.py --markdown docs/evaluation.md --json docs/evaluation.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from kgnotes.agent import KnowledgeAgent
from kgnotes.config import Settings
from kgnotes.models import Relation
from kgnotes.text import normalize_name

# --- QA set: each group requires at least one keyword from the list ----------
QA_CASES: list[tuple[str, list[list[str]]]] = [
    ("知识图谱助手项目使用了哪些技术？", [["Neo4j"], ["FastAPI"]]),
    ("知识图谱助手项目的作者是谁？", [["王小明"]]),
    ("王小明在哪里工作？", [["示例科技公司"]]),
    ("Neo4j 是什么？", [["图数据库"]]),
    ("Neo4j 使用什么查询语言？", [["Cypher"]]),
    ("GraphRAG 是什么？", [["检索增强生成"]]),
    ("团队周会中谁负责向量检索模块？", [["李雷"]]),
    ("谁负责前端可视化？", [["韩梅梅"]]),
    ("下次周会定在什么时候？", [["周五"]]),
    ("知识图谱助手项目包含哪些模块？", [["向量检索模块"], ["图构建模块"]]),
    ("Neo4j 支持向量索引吗？", [["向量索引"]]),
    ("Python asyncio 有哪些注意事项？", [["TaskGroup", "取消", "阻塞", "引用"]]),
    (
        "知识图谱助手项目和 GraphRAG 的关注点有什么不同？",
        [["个人笔记", "增量", "引用溯源", "多轮追问"]],
    ),
]

# --- Curated gold relations (all explicitly stated in notes/) ----------------
CURATED_GOLD: list[tuple[str, str, str]] = [
    ("GraphRAG", "IS_A", "检索增强生成方法"),
    ("GraphRAG", "USES", "知识图谱"),
    ("GraphRAG", "USES", "Neo4j"),
    ("王小明", "AUTHORED", "知识图谱助手项目"),
    ("王小明", "LEADS", "知识图谱助手项目"),
    ("图谱构建模块", "USES", "Neo4j 与向量索引"),
    ("Neo4j", "IS_A", "图数据库"),
    ("Neo4j", "USES", "Cypher"),
    ("知识图谱助手项目", "IS_A", "个人笔记问答系统"),
    ("知识图谱助手项目", "USES", "Neo4j"),
    ("知识图谱助手项目", "USES", "FastAPI"),
    ("FastAPI", "USES", "asyncio"),
    ("王小明", "WORKS_AT", "示例科技公司"),
    ("asyncio", "IS_A", "Python library"),
    ("asyncio", "USES", "event loop"),
    ("图构建模块", "PART_OF", "知识图谱助手项目"),
    ("向量检索模块", "PART_OF", "知识图谱助手项目"),
    ("王小明", "OWNS", "整理评测集"),
    ("韩梅梅", "OWNS", "前端可视化"),
    ("向量检索模块", "SUPPORTS", "混合检索"),
    ("Neo4j", "SUPPORTS", "向量索引"),
    ("李雷", "OWNS", "向量检索模块"),
]

GOLD_ENTITY_TYPES = {
    "Neo4j": "TECHNOLOGY",
    "王小明": "PERSON",
    "示例科技公司": "ORGANIZATION",
    "GraphRAG": "CONCEPT",
    "图数据库": "TECHNOLOGY",
    "李雷": "PERSON",
    "韩梅梅": "PERSON",
    "Cypher": "TECHNOLOGY",
    "向量检索模块": "CONCEPT",
}

# --- Held-out paraphrase note: deliberately avoids the rule patterns --------
HELDOUT_TEXT = """# 读书会随记

我在上周的读书会上认识了陈晨，她现在在星河数据做知识图谱工程师。
星河数据总部位于杭州。
星河数据正在做一个叫「星图」的内部知识库产品。
星图的技术栈是 Neo4j 和 FastAPI。
"""

HELDOUT_GOLD: list[tuple[str, str, str]] = [
    ("陈晨", "WORKS_AT", "星河数据"),
    ("陈晨", "WORKS_AS", "知识图谱工程师"),
    ("星河数据", "LOCATED_IN", "杭州"),
    ("星图", "IS_A", "知识库产品"),
    ("星图", "USES", "Neo4j"),
    ("星图", "USES", "FastAPI"),
]

_EXPLICIT_EXCLUDE = {"RELATED_TO", "TAGGED_WITH", "LINKS_TO", "MENTIONS"}


def _triples(relations: list[Relation]) -> set[tuple[str, str, str]]:
    return {
        (normalize_name(rel.source_name), rel.type, normalize_name(rel.target_name))
        for rel in relations
    }


def _gold_triples(gold: list[tuple[str, str, str]]) -> set[tuple[str, str, str]]:
    return {(normalize_name(s), t, normalize_name(o)) for s, t, o in gold}


def _prf(predicted: set, gold: set) -> dict[str, float | int]:
    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "predicted": len(predicted),
        "gold": len(gold),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(len(ordered) * pct)) - 1))
    return ordered[index]


def evaluate(notes_dir: Path, repeat: int = 3) -> dict:
    data_dir = Path(tempfile.mkdtemp(prefix="kgnotes-eval-"))
    settings = Settings(
        data_dir=data_dir,
        graph_store="sqlite",
        extractor="heuristic",
        embedding_provider="hash",
        embedding_dim=384,
        llm_api_key="",
        top_k=6,
    )
    report: dict = {"data_dir": str(data_dir), "mode": "offline/heuristic+hash"}

    with KnowledgeAgent(settings) as agent:
        ingest_started = time.perf_counter()
        summaries = agent.ingest_dir(notes_dir)
        ingest_ms = (time.perf_counter() - ingest_started) * 1000
        stats = agent.stats()
        report["corpus"] = {
            "notes": stats["graph"]["notes"],
            "chunks": stats["graph"]["chunks"],
            "entities": stats["graph"]["entities"],
            "relations": stats["graph"]["relations"],
            "vectors": stats["vectors"]["count"],
            "note_files": [s.title for s in summaries],
            "ingest_ms": round(ingest_ms, 1),
        }

        # ---- relation extraction (curated corpus) -------------------------
        all_entities = agent.store.list_entities(limit=5000)
        relations = agent.store.relations_for([e.id for e in all_entities], limit=5000)
        explicit = [r for r in relations if r.type not in _EXPLICIT_EXCLUDE]
        curated_pred = _triples(explicit)
        curated_gold = _gold_triples(CURATED_GOLD)
        report["extraction_curated"] = _prf(curated_pred, curated_gold)
        report["extraction_curated"]["unmatched_predicted"] = sorted(
            f"{s} | {t} | {o}" for s, t, o in curated_pred - curated_gold
        )
        report["extraction_curated"]["missed_gold"] = sorted(
            f"{s} | {t} | {o}" for s, t, o in curated_gold - curated_pred
        )

        # ---- entity type accuracy (note-title entities excluded) ----------
        entity_rows = []
        correct = 0
        for name, expected in GOLD_ENTITY_TYPES.items():
            entity = agent.store.get_entity(name=name)
            actual = entity.type if entity else None
            ok = actual == expected
            correct += ok
            entity_rows.append({"name": name, "expected": expected, "actual": actual, "ok": ok})
        report["entity_types"] = {
            "correct": correct,
            "total": len(GOLD_ENTITY_TYPES),
            "accuracy": round(correct / len(GOLD_ENTITY_TYPES), 4),
            "details": entity_rows,
        }

        # ---- QA + latency --------------------------------------------------
        agent.ask("预热：Neo4j 是什么？")
        qa_rows = []
        ask_latencies: list[float] = []
        for question, groups in QA_CASES:
            runs: list[float] = []
            answer = None
            for run in range(max(1, repeat)):
                started = time.perf_counter()
                answer = agent.ask(question)
                runs.append((time.perf_counter() - started) * 1000)
            ask_latencies.extend(runs)
            text = answer.answer.lower()
            missing = [g for g in groups if not any(k.lower() in text for k in g)]
            qa_rows.append({
                "question": question,
                "passed": not missing,
                "missing_keywords": missing,
                "mode": answer.mode,
                "citations": len(answer.citations),
                "latency_ms_avg": round(statistics.mean(runs), 1),
                "latency_ms_min": round(min(runs), 1),
                "latency_ms_max": round(max(runs), 1),
            })

        followup_first = agent.ask("王小明是谁？")
        followup_started = time.perf_counter()
        followup = agent.ask("他在哪工作？", session_id=followup_first.session_id)
        followup_ms = (time.perf_counter() - followup_started) * 1000
        followup_ok = (
            "王小明" in followup.rewritten_question
            and "示例科技公司" in followup.answer
        )
        report["followup"] = {
            "first_question": followup_first.question,
            "followup_question": followup.question,
            "rewritten_question": followup.rewritten_question,
            "answer_excerpt": followup.answer[:200],
            "passed": followup_ok,
            "latency_ms": round(followup_ms, 1),
        }

        # ---- held-out paraphrase extraction --------------------------------
        heldout_summary = agent.ingest_text(
            HELDOUT_TEXT, title="读书会随记（held-out）", path="heldout-paraphrase.md"
        )
        heldout_relations = [
            r for r in agent.store.relations_for(
                [e.id for e in agent.store.list_entities(limit=5000)], limit=5000
            )
            if r.note_id == heldout_summary.note_id and r.type not in _EXPLICIT_EXCLUDE
        ]
        heldout_pred = _triples(heldout_relations)
        heldout_gold = _gold_triples(HELDOUT_GOLD)
        report["extraction_heldout"] = _prf(heldout_pred, heldout_gold)
        report["extraction_heldout"]["unmatched_predicted"] = sorted(
            f"{s} | {t} | {o}" for s, t, o in heldout_pred - heldout_gold
        )
        report["extraction_heldout"]["missed_gold"] = sorted(
            f"{s} | {t} | {o}" for s, t, o in heldout_gold - heldout_pred
        )
        report["heldout_note"] = HELDOUT_TEXT.strip()

        # ---- retrieval-only latency ---------------------------------------
        search_latencies: list[float] = []
        for question, _ in QA_CASES[:5]:
            for _ in range(max(1, repeat)):
                started = time.perf_counter()
                agent.search(question)
                search_latencies.append((time.perf_counter() - started) * 1000)

    passed = sum(1 for row in qa_rows if row["passed"])
    report["qa"] = {
        "passed": passed,
        "total": len(qa_rows),
        "accuracy": round(passed / len(qa_rows), 4),
        "rows": qa_rows,
    }
    report["latency"] = {
        "ask_runs": len(ask_latencies),
        "ask_avg_ms": round(statistics.mean(ask_latencies), 1),
        "ask_p50_ms": round(statistics.median(ask_latencies), 1),
        "ask_p95_ms": round(_percentile(ask_latencies, 0.95), 1),
        "ask_max_ms": round(max(ask_latencies), 1),
        "search_runs": len(search_latencies),
        "search_avg_ms": round(statistics.mean(search_latencies), 1),
        "search_p95_ms": round(_percentile(search_latencies, 0.95), 1),
    }
    return report



def to_markdown(report: dict) -> str:
    corpus = report["corpus"]
    curated = report["extraction_curated"]
    heldout = report["extraction_heldout"]
    entities = report["entity_types"]
    qa = report["qa"]
    latency = report["latency"]
    followup = report["followup"]

    lines = [
        "# Demo 评测结果（自动生成）",
        "",
        "> 运行方式：`python scripts/evaluate_demo.py`  ",
        f"> 模式：{report['mode']}（无 Neo4j、无 LLM、无 API Key）  ",
        "> 评测使用临时数据目录，不会污染项目数据。",
        "",
        "## 1. 语料规模",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 笔记数 | {corpus['notes']} |",
        f"| 片段数（chunks） | {corpus['chunks']} |",
        f"| 实体数 | {corpus['entities']} |",
        f"| 关系数 | {corpus['relations']} |",
        f"| 向量数 | {corpus['vectors']} |",
        f"| 导入耗时 | {corpus['ingest_ms']} ms |",
        "",
        f"笔记文件：{', '.join(corpus['note_files'])}",
        "",
        "## 2. 实体 / 关系抽取",
        "",
        "| 数据集 | 预测数 | 标注数 | TP | Precision | Recall | F1 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        f"| 示例集（规则友好） | {curated['predicted']} | {curated['gold']} | {curated['tp']} | "
        f"{curated['precision']:.1%} | {curated['recall']:.1%} | {curated['f1']:.1%} |",
        f"| 留出集（改写文本） | {heldout['predicted']} | {heldout['gold']} | {heldout['tp']} | "
        f"{heldout['precision']:.1%} | {heldout['recall']:.1%} | {heldout['f1']:.1%} |",
        "",
        f"关键实体类型准确率：**{entities['correct']}/{entities['total']} = {entities['accuracy']:.1%}**"
        "（不含笔记标题实体；标题实体统一为 NOTE）",
        "",
        "留出集说明：文本为规则未覆盖的同义改写（“在…做…”、“技术栈是…”、“叫「…」的…”），"
        "用于展示纯规则抽取的泛化上限，也是接入 LLM 抽取的依据。",
        "",
        "## 3. 问答",
        "",
        f"测试问题：**{qa['total']} 个独立问题 + 1 个多轮追问（共 {qa['total'] + 1} 个）**，"
        f"关键词命中：**{qa['passed']}/{qa['total']} = {qa['accuracy']:.1%}**",
        "",
        "| # | 问题 | 结果 | 模式 | 引用数 | 平均耗时 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for index, row in enumerate(qa["rows"], start=1):
        result = "PASS" if row["passed"] else "FAIL"
        lines.append(
            f"| {index} | {row['question']} | {result} | {row['mode']} | "
            f"{row['citations']} | {row['latency_ms_avg']} ms |"
        )
    lines += [
        "",
        f"多轮追问：{'PASS' if followup['passed'] else 'FAIL'}  ",
        f"- 首轮：{followup['first_question']}",
        f"- 追问：{followup['followup_question']}",
        f"- 改写后：{followup['rewritten_question']}",
        f"- 回答摘录：{followup['answer_excerpt'].replace(chr(10), ' ')}",
        "",
        "## 4. 响应时间（离线模式）",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 问答样本数 | {latency['ask_runs']} |",
        f"| 问答平均 | {latency['ask_avg_ms']} ms |",
        f"| 问答 P50 | {latency['ask_p50_ms']} ms |",
        f"| 问答 P95 | {latency['ask_p95_ms']} ms |",
        f"| 问答最大 | {latency['ask_max_ms']} ms |",
        f"| 纯检索平均（{latency['search_runs']} 次） | {latency['search_avg_ms']} ms |",
        f"| 纯检索 P95 | {latency['search_p95_ms']} ms |",
        "",
        "## 5. 口径与限制",
        "",
        "- 问答判分为**关键词命中**（标准答案中的关键事实是否出现在回答里），不是人工语义评分。",
        "- 示例集由规则友好的短句构成，100% 仅代表当前 demo 数据，不代表真实笔记上的准确率。",
        "- 延迟为单机、SQLite、离线哈希向量的端到端耗时；接入 LLM 后主要受模型与网络延迟影响。",
        "- Neo4j 后端与真实 LLM 调用路径已有实现，但本次评测未启用（环境无 Neo4j / API Key）。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="kgnotes offline demo evaluation")
    parser.add_argument("--notes", default="notes", help="示例笔记目录")
    parser.add_argument("--repeat", type=int, default=3, help="每个问题的计时重复次数")
    parser.add_argument("--markdown", help="写出 Markdown 报告路径")
    parser.add_argument("--json", help="写出 JSON 结果路径")
    args = parser.parse_args()

    report = evaluate(Path(args.notes), repeat=max(1, args.repeat))
    markdown = to_markdown(report)

    if args.markdown:
        target = Path(args.markdown)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(markdown, encoding="utf-8")
    if args.json:
        target = Path(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

