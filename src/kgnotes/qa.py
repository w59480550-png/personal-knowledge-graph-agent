"""Follow-up aware question answering over the hybrid retriever."""

from __future__ import annotations

import logging
import re
import time

from .config import Settings
from .llm import LLMClient, LLMError
from .models import Answer, Citation, Entity, Relation
from .retrieval import HybridRetriever, RetrievalResult
from .sessions import SessionStore
from .text import truncate

log = logging.getLogger(__name__)

_CUES = (
    "它", "他", "她", "这个", "那个", "这些", "那些", "该", "其", "上述", "前面", "刚才",
    "还有", "那么", "那", "为什么", "怎么", "如何做到", "继续", "接着",
    "this", "that", "it", "they", "those", "these", "why", "how about", "what about", "more",
)

_REWRITE_PROMPT = """根据对话历史，把用户的最新问题改写成一个独立、完整、适合知识库检索的问题。
要求：
- 补全代词和省略信息（例如“它”指代谁），保留专有名词。
- 不要回答问题，不要添加历史中没有的信息。
- 只输出改写后的问题，不要引号或解释。

对话历史：
{history}

最新问题：{question}
改写后的问题："""


def _is_english(text: str) -> bool:
    letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return letters > cjk * 2


class QAService:
    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        sessions: SessionStore,
        client: LLMClient | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.sessions = sessions
        self.client = client

    # -- public -------------------------------------------------------------
    def ask(
        self, question: str, session_id: str | None = None, top_k: int | None = None
    ) -> Answer:
        started = time.perf_counter()
        question = (question or "").strip()
        if not question:
            raise ValueError("question must not be empty")
        session_id = self.sessions.create(session_id)
        history = self.sessions.history(session_id, limit=6)
        rewritten = self.rewrite_question(question, session_id, history)
        result = self.retriever.retrieve(rewritten, top_k=top_k)

        if not result.citations:
            answer_text = self._no_context_answer(question, result)
            mode = "no-context"
        elif self.client is not None and self.client.enabled:
            try:
                answer_text = self._llm_answer(question, rewritten, result, history)
                mode = "llm"
            except LLMError as exc:
                log.warning("LLM answer failed: %s", exc)
                answer_text = self._extractive_answer(question, result)
                mode = "extractive-fallback"
        else:
            answer_text = self._extractive_answer(question, result)
            mode = "extractive"

        answer = Answer(
            answer=answer_text,
            session_id=session_id,
            question=question,
            rewritten_question=rewritten,
            citations=result.citations,
            entities=[e.as_dict() for e in result.entities[:8]],
            relations=[r.as_dict() for r in result.relations[:12]],
            mode=mode,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        self.sessions.append(session_id, "user", question, {"rewritten": rewritten})
        self.sessions.append(
            session_id, "assistant", answer_text,
            {"mode": mode, "citation_ids": [c.chunk_id for c in result.citations]},
        )
        return answer

    def rewrite_question(
        self, question: str, session_id: str, history: list[dict[str, str]] | None = None
    ) -> str:
        previous = self.sessions.last_user_questions(session_id, 2)
        if not previous:
            return question
        needs_context = len(question) < 12 or any(cue in question.lower() for cue in _CUES)
        if not needs_context:
            return question
        history = history if history is not None else self.sessions.history(session_id, limit=6)
        if self.client is not None and self.client.enabled and self.settings.use_llm_rewrite:
            history_text = "\n".join(
                f"{m['role']}: {truncate(m['content'], 300)}" for m in history[-6:]
            )
            try:
                rewritten = self.client.chat(
                    [
                        {"role": "system", "content": "你是对话查询改写器，只输出改写后的问题。"},
                        {"role": "user", "content": _REWRITE_PROMPT.format(
                            history=history_text or "-", question=question
                        )},
                    ],
                    temperature=0.0,
                    max_tokens=200,
                ).strip().strip('"').strip("“”")
                if 2 <= len(rewritten) <= 300:
                    return rewritten
            except LLMError as exc:
                log.warning("query rewrite failed: %s", exc)
        # Heuristic fallback: prepend the most recent standalone question.
        return f"{previous[0]} {question}".strip()

    # -- context & answering ------------------------------------------------
    def _build_context(self, result: RetrievalResult) -> str:
        parts: list[str] = ["【知识库上下文】"]
        used = 0
        for citation in result.citations:
            block = [
                f"[{citation.index}] 《{citation.note_title}》 第{citation.chunk_index + 1}段"
                f"（匹配：{citation.reason}）",
                citation.text,
            ]
            if citation.relations:
                block.append("图谱关系：" + "；".join(citation.relations))
            text = "\n".join(block)
            if used + len(text) > self.settings.max_context_chars and len(parts) > 1:
                break
            parts.append(text)
            used += len(text)

        if result.entities:
            parts.append("\n【识别到的实体】")
            for entity in result.entities[:8]:
                desc = f"：{truncate(entity.description, 120)}" if entity.description else ""
                parts.append(f"- {entity.name}（{entity.type}）{desc}")

        if result.relations:
            parts.append("\n【图谱关系（含证据）】")
            for rel in result.relations[:12]:
                evidence = f"；证据：{truncate(rel.evidence, 160)}" if rel.evidence else ""
                parts.append(f"- {rel.source_name} --{rel.type}--> {rel.target_name}{evidence}")
        return "\n".join(parts)

    def _llm_answer(
        self,
        question: str,
        rewritten: str,
        result: RetrievalResult,
        history: list[dict[str, str]],
    ) -> str:
        context = self._build_context(result)
        messages: list[dict[str, str]] = [{"role": "system", "content": self.settings.system_prompt}]
        for message in history[-4:]:
            messages.append({"role": message["role"], "content": truncate(message["content"], 800)})
        user_content = (
            f"{context}\n\n【用户问题】\n{question}\n\n"
            f"（检索用问题：{rewritten}）\n"
            "请只依据以上上下文回答，并在关键结论后标注 [编号]。"
        )
        messages.append({"role": "user", "content": user_content})
        answer = self.client.chat(messages).strip()
        if not answer:
            raise LLMError("empty completion")
        return self._sanitize_citations(answer, result.citations)

    @staticmethod
    def _sanitize_citations(answer: str, citations: list[Citation]) -> str:
        valid = {str(c.index) for c in citations}
        answer = re.sub(
            r"\[(\d+)\]",
            lambda m: m.group(0) if m.group(1) in valid else "",
            answer,
        )
        if citations and not any(f"[{c.index}]" in answer for c in citations):
            refs = "；".join(f"[{c.index}]《{c.note_title}》" for c in citations[:4])
            answer = f"{answer}\n\n参考来源：{refs}"
        return answer

    def _extractive_answer(self, question: str, result: RetrievalResult) -> str:
        english = _is_english(question)
        lines: list[str] = []
        lead_entity = result.entities[0].name if result.entities else ""
        if english:
            lines.append(
                f"Based on your notes, here is what relates to **{lead_entity}**:"
                if lead_entity else "Based on your notes, here are the most relevant passages:"
            )
        else:
            lines.append(
                f"根据你的笔记，与「{lead_entity}」相关的内容如下："
                if lead_entity else "根据你的笔记，找到以下相关信息："
            )

        strong = [r for r in result.relations if r.confidence >= 0.5][:5]
        if strong:
            lines.append("\n**图谱关系**" if not english else "\n**Graph relations**")
            for rel in strong:
                evidence = f"（{truncate(rel.evidence, 120)}）" if rel.evidence else ""
                lines.append(f"- {rel.source_name} —{rel.type}→ {rel.target_name}{evidence}")

        lines.append("\n**相关原文**" if not english else "\n**Relevant passages**")
        for citation in result.citations[:4]:
            excerpt = truncate(re.sub(r"\s+", " ", citation.text), 260)
            lines.append(f"- {excerpt} [{citation.index}]（《{citation.note_title}》）")

        if english:
            lines.append(
                f"\n{len(result.citations)} passage(s) retrieved. "
                "Connect an LLM for a synthesized answer; citations above are the evidence."
            )
        else:
            lines.append(
                f"\n以上内容来自 {len(result.citations)} 条笔记片段，点击引用可查看原文。"
                "（当前为抽取式回答，配置 LLM_API_KEY 后可获得综合生成式回答。）"
            )
        return "\n".join(lines)

    def _no_context_answer(self, question: str, result: RetrievalResult) -> str:
        english = _is_english(question)
        if english:
            text = "I couldn't find anything directly relevant in your knowledge base."
        else:
            text = "我在知识库中没有找到与这个问题直接相关的内容。"
        if result.entities:
            names = "、".join(e.name for e in result.entities[:5])
            text += (
                f" I did recognise these entities: {names}. Try asking about one of them."
                if english else f"不过我识别到了这些相关实体：{names}。可以尝试围绕它们追问，或补充相关笔记。"
            )
        else:
            text += (
                " Try adding a note about this topic, then ask again."
                if english else "可以先把相关笔记导入知识库，再重新提问。"
            )
        return text
