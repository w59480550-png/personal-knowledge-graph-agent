---
title: GraphRAG 读书笔记
tags: [rag, graphrag, retrieval]
---

# GraphRAG 读书笔记

GraphRAG 是一种检索增强生成方法。GraphRAG 使用知识图谱。GraphRAG 基于 Neo4j。

## 核心观点

传统 RAG 只做向量召回，遇到“跨文档、多跳”的问题容易丢上下文。
GraphRAG 先从问题中识别实体，再沿图谱扩展邻居，最后把子图与原文一起交给大模型。
在个人知识库场景中，图谱规模不大，一跳邻居通常就足够。

## 和本项目的区别

GraphRAG 更强调社区摘要与全局问题；本项目更关注个人笔记的增量导入、引用溯源和多轮追问。
本项目采用 [[Neo4j 与向量索引]] 中的混合检索方案，并用 RRF 融合向量与 BM25 排名。

#RAG #知识图谱 #检索增强
