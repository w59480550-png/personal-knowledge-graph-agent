from __future__ import annotations

from kgnotes.agent import KnowledgeAgent
from kgnotes.config import Settings
from kgnotes.llm import LLMError


class FakeChat:
    enabled = True

    def __init__(self, fail_on_answer: bool = False) -> None:
        self.fail_on_answer = fail_on_answer
        self.rewrite_calls = 0
        self.answer_calls = 0

    def chat(self, messages, temperature=None, max_tokens=None, json_mode=False) -> str:
        prompt = messages[-1]["content"]
        if "改写后的问题" in prompt:
            self.rewrite_calls += 1
            return "王小明在哪里工作？"
        self.answer_calls += 1
        if self.fail_on_answer:
            raise LLMError("provider down")
        return "王小明就职于示例科技公司 [1]。无关引用 [9] 应被移除。"


def _agent(settings: Settings, fake: FakeChat) -> KnowledgeAgent:
    settings.llm_api_key = "test-key"
    settings.extractor = "heuristic"
    settings.embedding_provider = "hash"
    agent = KnowledgeAgent(settings, llm=fake)
    agent.ingest_text(
        "知识图谱助手项目的作者是王小明。王小明就职于示例科技公司。",
        title="知识图谱助手项目",
        path="project.md",
    )
    return agent


def test_llm_answer_with_citation_sanitising(settings: Settings):
    fake = FakeChat()
    with _agent(settings, fake) as agent:
        answer = agent.ask("王小明在哪里工作？")
        assert answer.mode == "llm"
        assert "示例科技公司" in answer.answer
        assert "[1]" in answer.answer
        assert "[9]" not in answer.answer
        assert fake.answer_calls == 1


def test_llm_rewrite_for_followup(settings: Settings):
    fake = FakeChat()
    with _agent(settings, fake) as agent:
        first = agent.ask("王小明是谁？")
        second = agent.ask("他在哪里工作？", session_id=first.session_id)
        assert fake.rewrite_calls == 1
        assert second.rewritten_question == "王小明在哪里工作？"


def test_llm_failure_falls_back_to_extractive(settings: Settings):
    fake = FakeChat(fail_on_answer=True)
    with _agent(settings, fake) as agent:
        answer = agent.ask("王小明在哪里工作？")
        assert answer.mode == "extractive-fallback"
        assert "示例科技公司" in answer.answer
