from __future__ import annotations

from kgnotes.notes import NoteLoader, parse_frontmatter


def test_parse_frontmatter_roundtrip():
    raw = "---\ntitle: 测试\ntags: [a, b]\naliases: 别名一\n---\n# 正文\n\n内容"
    meta, body = parse_frontmatter(raw)
    assert meta["title"] == "测试"
    assert meta["tags"] == ["a", "b"]
    assert body.startswith("# 正文")


def test_from_text_extracts_title_tags_links():
    loader = NoteLoader()
    note = loader.from_text(
        "---\ntags: [知识图谱, neo4j]\n---\n# 我的笔记\n\n参考 [[Neo4j|图数据库]] 和 #检索。",
        path="notes/demo.md",
    )
    assert note.title == "我的笔记"
    assert note.tags == ["知识图谱", "neo4j"]
    assert ("Neo4j", "图数据库") in note.wikilinks()
    assert "检索" in note.all_tags()
    assert note.id.startswith("note_")


def test_chunking_respects_size_and_keeps_headings():
    loader = NoteLoader(max_chunk_chars=120, chunk_overlap=20)
    body = "## 第一节\n\n" + "。".join(f"这是第{i}句测试内容" for i in range(30)) + "。"
    note = loader.from_text(body, title="长笔记")
    chunks = loader.chunk(note)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 160 for chunk in chunks)
    assert all(chunk.note_id == note.id for chunk in chunks)
    assert chunks[0].seq == 0 and chunks[-1].seq == len(chunks) - 1
    assert "第一节" in chunks[0].text


def test_chunking_empty_note_returns_one_chunk():
    loader = NoteLoader()
    note = loader.from_text("", title="空笔记")
    chunks = loader.chunk(note)
    assert len(chunks) == 1
    assert chunks[0].text == "空笔记"
