---
title: Neo4j 与向量索引
tags: [neo4j, graph-database, vector-search]
aliases: [Neo4j 图数据库]
---

# Neo4j 与向量索引

Neo4j 是一个图数据库。Neo4j 使用 Cypher。Neo4j 支持向量索引。

## 为什么用图数据库

知识图谱中的实体和关系天然适合用属性图表示。Neo4j 的 Cypher 查询语言适合做多跳邻居遍历，
例如从“知识图谱助手项目”出发找到相关的技术栈和负责人。

## 向量检索

Neo4j 5 提供原生的向量索引，可以在 `Chunk` 节点上建立 embedding 索引，并结合全文索引做混合检索。
实践上，图检索负责“关系”，向量检索负责“语义相似”，两者用 RRF 融合效果更稳。

相关笔记：[[知识图谱助手项目]]、[[GraphRAG 读书笔记]] #知识图谱 #检索增强
