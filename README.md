# Personal Knowledge Graph Notes Agent

把零散的 Markdown 笔记自动整理成**实体关系图 + 向量索引**，并支持带引用的**多轮追问**。

- 输入：Markdown / 纯文本笔记（支持 YAML front-matter、`[[双链]]`、`#标签`）
- 抽取：LLM（任何 OpenAI 兼容接口）或**零依赖规则抽取**，LLM 失败自动回退
- 存储：**Neo4j**（可选）或内置 **SQLite** 图存储，二者行为一致
- 索引：Neo4j 原生向量索引 / 本地 SQLite 向量索引；Embedding 可走 API，也可离线哈希向量
- 检索：向量召回 + BM25 + 图谱邻居（RRF 融合）+ 实体链接
- 问答：多轮会话、指代改写、引用溯源；无 LLM 时使用抽取式回答
- 接口：CLI、FastAPI HTTP API、可导出的图谱 JSON

> 默认配置**完全离线可跑**：不需要 Neo4j、不需要 API Key。配置 `LLM_API_KEY`
> 与 `NEO4J_URI` 后自动升级为「LLM 抽取 + Neo4j 图 + 向量索引」的完整形态。

---

## 架构

```text
 Markdown 笔记
      │
      ▼
 NoteLoader ──► 按标题分节 / 段落切块（带 overlap）
      │
      ▼
 ExtractorPipeline ──► LLM 抽取（JSON schema）──失败──► 规则抽取
      │                        │
      │                        ├─ 实体归一化 / 别名合并 / 类型升级
      │                        └─ 关系去重（source+type+target+note）
      ▼
 ┌───────────────┐        ┌─────────────────────┐
 │ Graph Store   │        │ Embedding + Vector  │
 │ SQLite/Neo4j  │        │ SQLite/Neo4j index  │
 └───────┬───────┘        └──────────┬──────────┘
         │                           │
         └──────────┬────────────────┘
                    ▼
              HybridRetriever
   实体链接 + 向量 TopK + BM25 + 图谱 1~2 跳邻居
                    │  RRF 融合
                    ▼
               QAService
   历史感知的查询改写 → 上下文拼装 → LLM / 抽取式回答 + 引用
```

关键设计：

1. **图与向量分离但可互换**：向量索引只存 `chunk_id -> vector`，正文永远来自图存储，
   避免两套数据不一致。
2. **关系建模采用“再化”（reified）节点**：`(:Relation {type})` 连接两个实体，并保留
   笔记、片段、证据、置信度，天然支持多来源同一关系。
3. **混合检索**：向量负责语义相似，BM25 负责精确术语，图谱负责多跳关系；RRF 对三路
   排名做无参数融合，再用图谱证据与实体命中做小幅加权。
4. **可解释**：每条引用带 `reason`（vector / keyword / graph / entity）、
   笔记标题、段落号与命中分数。

---

## 快速开始（离线，1 分钟）

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # 或 pip install -e ".[dev]" 安装测试依赖

python -m kgnotes ingest notes/          # 导入仓库自带示例笔记
python -m kgnotes ask "知识图谱助手项目使用了哪些技术？" --show-context
python -m kgnotes ask "王小明是谁？"      # 记下返回的 session
python -m kgnotes ask "他在哪工作？" --session <上一步的 session>
python -m kgnotes chat                   # 交互式多轮
python -m kgnotes graph "Neo4j" --depth 2
python -m kgnotes serve                  # http://127.0.0.1:8000/docs
```

默认数据落在 `data/`：

```text
data/kgnotes.sqlite     # 笔记、片段、实体、关系、提及
data/vectors.sqlite     # 向量索引
data/sessions.sqlite    # 多轮会话
data/exports/graph.json # `kgnotes export` 的输出
```

离线模式说明：

| 能力 | 离线模式 |
| --- | --- |
| 实体/关系抽取 | 规则 + Markdown 结构（双链、标签、中英文句式） |
| Embedding | 字符 n-gram 特征哈希（384 维，确定性、无需下载模型） |
| 回答 | 抽取式：图谱关系 + 原文摘录 + 引用编号 |
| 追问 | 基于历史问题的启发式改写 + 实体链接 |

---

## 接入 Neo4j

```bash
pip install -e ".[neo4j]"   # 首次使用 Neo4j 需要驱动
docker compose up -d neo4j
export KGNOTES_GRAPH_STORE=neo4j
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=kgnotes-pass

python -m kgnotes ingest notes/ --reset
```

Neo4j Browser：<http://localhost:7474>（用户名 `neo4j`，密码 `kgnotes-pass`）。

图模型：

```cypher
(:Note {id, title, path, content, tags, aliases, content_hash})
  -[:HAS_CHUNK]-> (:Chunk {id, seq, heading, text, embedding})

(:Entity {id, name, name_key, type, description, aliases, note_ids})
  -[:MENTIONED_IN {chunk_id}]-> (:Note)

(:Relation {id, type, note_id, chunk_id, confidence, evidence, description})
  -[:FROM]-> (:Entity)
  -[:TO]->   (:Entity)
  -[:EVIDENCE_OF]-> (:Chunk)
```

示例查询：

```cypher
MATCH (s:Entity)-[:FROM]-(r:Relation)-[:TO]->(t:Entity)
WHERE r.type = 'USES'
RETURN s.name, r.type, t.name, r.confidence
ORDER BY r.confidence DESC LIMIT 20;
```

> 切换后端后需要重新导入（`--reset` 只清理当前后端的图与向量）。
> 向量索引名固定为 `chunk_embeddings`，维度随 embedding 模型自动创建/重建。


## 接入 LLM（可选）

任何 OpenAI 兼容的 `/chat/completions` 与 `/embeddings` 接口均可，例如 OpenAI、DeepSeek、
通义千问兼容模式、Ollama、vLLM、LM Studio：

```bash
cp .env.example .env
# 编辑 .env：
# LLM_BASE_URL=https://api.openai.com/v1
# LLM_API_KEY=sk-...
# LLM_MODEL=gpt-4o-mini
# EMBEDDING_MODEL=text-embedding-3-small

python -m kgnotes ingest notes/ --reset
python -m kgnotes ask "王小明负责哪些工作？"
```

只想用 LLM 做抽取/回答、但保留离线向量：

```bash
KGNOTES_EMBEDDING_PROVIDER=hash
```

只允许 LLM 抽取（失败不降级，便于发现配置错误）：

```bash
KGNOTES_EXTRACTOR=llm
```

Ollama 示例（无需 API Key，但需本地服务）：

```bash
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen2.5:7b
KGNOTES_EMBEDDING_PROVIDER=hash
```

---

## CLI

| 命令 | 说明 |
| --- | --- |
| `kgnotes init` | 初始化数据目录，显示后端/抽取器/LLM 状态 |
| `kgnotes ingest PATH... [--reset] [--json]` | 导入文件或目录；重复导入按路径覆盖 |
| `kgnotes ask QUESTION [--session ID] [--show-context] [--json]` | 提问；不传 session 会自动新建 |
| `kgnotes chat [--session ID]` | 交互式多轮（`/new` 开新会话，`exit` 退出） |
| `kgnotes search QUERY [--top-k N] [--json]` | 只检索，不生成回答 |
| `kgnotes graph ENTITY [--depth 1-3] [--json]` | 查看实体邻域子图 |
| `kgnotes entities [--query Q] [--type T]` | 列出/搜索实体 |
| `kgnotes stats` | 图、向量、会话统计 |
| `kgnotes export [--out FILE]` | 导出实体与关系 JSON |
| `kgnotes reindex` | 重建向量索引（切换 embedding 模型后使用） |
| `kgnotes delete-note NOTE_ID` | 删除笔记及其片段/关系/向量 |
| `kgnotes reset --yes` | 清空所有数据 |
| `kgnotes serve [--host H] [--port P] [--reload]` | 启动 HTTP API |

全局参数（写在子命令前）：`--data-dir`、`--graph-store`、`--extractor`、`--neo4j-uri`、
`--embedding-provider`、`--llm-model`，它们等价于对应的环境变量。

---

## HTTP API

```bash
python -m kgnotes serve
# OpenAPI: http://127.0.0.1:8000/docs
```

```bash
# 导入笔记
curl -s localhost:8000/notes -H 'content-type: application/json' -d '{
  "title": "周会",
  "content": "王小明就职于示例科技公司。示例科技公司位于北京。",
  "tags": ["meeting"]
}'

# 提问 + 多轮追问
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"王小明在哪工作？"}'
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"那家公司在哪里？","session_id":"<上一步返回的 session_id>"}'

# 检索 / 图谱 / 统计
curl -s localhost:8000/search -H 'content-type: application/json' \
  -d '{"query":"向量检索","top_k":5}'
curl -s 'localhost:8000/graph?entity=王小明&depth=2'
curl -s localhost:8000/stats
```

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 配置（脱敏）+ 统计；向量/图故障时返回 `degraded` |
| POST | `/notes` | 导入一篇笔记，返回 chunks/entities/relations |
| GET | `/notes` | 笔记列表（分页） |
| GET | `/notes/{id}` | 笔记详情 + 片段 |
| DELETE | `/notes/{id}` | 删除笔记 |
| POST | `/search` | 混合检索，返回引用/实体/关系/诊断指标 |
| POST | `/ask` | 问答；传 `session_id` 续接多轮 |
| GET | `/graph` | 实体邻域（`entity` 支持名称或 id） |
| GET | `/entities` | 实体搜索（`query`、`type`、`limit`） |
| GET | `/stats` | 图 + 向量 + 会话统计 |
| GET/DELETE | `/sessions...` | 会话列表、消息、删除 |
| POST | `/reindex` | 重建向量索引 |
| POST | `/admin/reset` | 清空全部数据（`KGNOTES_ALLOW_RESET=false` 可禁用） |

错误约定：参数校验 `422`；资源不存在 `404`；业务校验失败 `400`；未配置/服务不可用 `503`。


## 抽取与检索细节

### 规则抽取（离线）

- 结构信号：front-matter 标题/标签/别名、`[[双链]]`、`#标签`、`**加粗**`、`` `代码` ``
- 中英文句式：`X 是/属于 Y`、`X 使用/依赖 Y`、`X 位于 Y`、`X 由 Y 创建`、
  `X is a Y`、`X uses Y`、`X was founded by Y` 等
- 实体类型：`PERSON / ORGANIZATION / PROJECT / TECHNOLOGY / LOCATION / EVENT /
  CONCEPT / TAG / NOTE`，按名称词缀与上下文推断
- 同一片段内共现实体生成低置信度 `RELATED_TO`（最多 4 条），可按需过滤
- 别名写法：`Neo4j（简称 N4J）` / `X (aka Y)`

### LLM 抽取

- 结构化 JSON 输出（`response_format=json_object` + 本地 JSON 容错解析）
- 约束实体类型与关系类型；要求 `evidence` 为原文原句；`confidence` 0~1
- `auto` 模式下单块失败自动回退规则抽取，不影响整篇导入

### 实体归一化

- 名称 key：NFKC + 小写 + 去标点/空白（中英文通用）
- 别名索引命中即复用已有实体；同 key 不同类型时按信息量升级
  （`TAG → CONCEPT → TECHNOLOGY → ORGANIZATION/PERSON`），不会把 `NOTE` 降级
- 展示名优先保留大小写更规范的写法（`GraphRAG` 优于 `graphrag`）

### 混合检索打分

1. 向量 TopK（归一化向量点积）与 BM25 TopK 分别计算 RRF：`1 / (60 + rank)`
2. 图谱证据片段加 `0.02 + 0.01 × confidence`
3. 含问题中实体的片段加 `0.015`
4. 融合后取 TopK 作为引用；`reason` 记录命中来源

### 多轮追问

- 会话持久化在 `data/sessions.sqlite`
- 触发指代改写：问题较短或包含「它/他/这个/上述/那/this/it...」
- 优先用 LLM 改写为独立问题；无 LLM 时将上一轮问题拼到当前问题前
- 回答只使用检索到的上下文，并要求 `[n]` 引用；不存在的引用编号会被清理

---

## 配置

所有配置都可用环境变量或 `.env`（见 `.env.example`）：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `KGNOTES_DATA_DIR` | `data` | 数据目录 |
| `KGNOTES_GRAPH_STORE` | `sqlite` | `sqlite` / `neo4j` |
| `NEO4J_URI/USER/PASSWORD/DATABASE` | - | Neo4j 连接 |
| `KGNOTES_EXTRACTOR` | `auto` | `auto` / `heuristic` / `llm` |
| `KGNOTES_MAX_CHUNK_CHARS` | `700` | 片段最大字符数 |
| `KGNOTES_CHUNK_OVERLAP` | `80` | 片段重叠字符数 |
| `KGNOTES_EMBEDDING_PROVIDER` | `auto` | `auto` / `hash` / `openai` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | API embedding 模型 |
| `EMBEDDING_DIM` | `384` | 离线哈希向量维度 |
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | - | OpenAI 兼容接口 |
| `KGNOTES_LLM_REWRITE` | `true` | 是否用 LLM 改写追问 |
| `KGNOTES_TOP_K` | `6` | 默认引用数 |
| `KGNOTES_GRAPH_DEPTH` | `1` | 图谱扩展跳数（1~3） |
| `KGNOTES_MAX_CONTEXT_CHARS` | `7000` | 提示词上下文上限 |
| `KGNOTES_ALLOW_RESET` | `true` | 是否允许 `/admin/reset` |
| `KGNOTES_CORS_ORIGINS` | `*` | 逗号分隔的 CORS 白名单 |

---

## 测试

```bash
pip install -e ".[dev]"
python -m pytest -q
```

覆盖：front-matter/分块、规则与 LLM 抽取、回退策略、实体合并、关系幂等、
邻域遍历、向量索引、混合检索、多轮改写、引用清理、SQLite 图存储、FastAPI 全部端点、
OpenAI 兼容客户端（MockTransport，含重试）。

Neo4j 集成测试默认跳过，设置以下变量后可运行：

```bash
KGNOTES_TEST_NEO4J_URI=bolt://localhost:7687 \
NEO4J_PASSWORD=kgnotes-pass \
python -m pytest tests/test_neo4j_optional.py -q
```

---

## 目录结构

```text
src/kgnotes/
  agent.py            # 编排：导入、检索、问答、维护
  api.py              # FastAPI 应用
  cli.py              # 命令行
  config.py           # 环境变量配置
  models.py           # Entity/Relation/Citation/Answer
  notes.py            # front-matter、双链、标签、分块
  text.py             # 分词、BM25、归一化、余弦
  embeddings.py       # HashEmbedder / OpenAIEmbedder
  llm.py              # OpenAI 兼容 chat/embeddings 客户端
  extraction/         # heuristic / llm / pipeline
  graph/              # sqlite_store / neo4j_store / base
  vector.py           # SQLite / Neo4j 向量索引
  retrieval.py        # 混合检索与 RRF
  sessions.py         # 多轮会话存储
  qa.py               # 追问改写、上下文、回答、引用
tests/                # pytest（44 个用例；Neo4j 集成用例默认跳过）
notes/                # 示例笔记
docker-compose.yml    # Neo4j 5
```

---

## 已知限制

- 规则抽取面向常见句式，复杂长句/隐含关系需要 LLM 抽取；`RELATED_TO` 共现关系会有噪声
- 离线哈希向量是词法级相似度，不理解同义词；需要语义召回时接入 embedding API
- 只处理文本；PDF/图片/网页抓取需在上游转换
- 会话按单进程 SQLite 共享，多副本部署需改为共享数据库
- API 未内置鉴权，公网部署请置于反向代理/网关之后，并关闭 `KGNOTES_ALLOW_RESET`

## License

MIT

