PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: help install test ingest ask serve neo4j-up neo4j-down reset lint

help:
	@echo "make install    创建虚拟环境并安装依赖"
	@echo "make test       运行测试"
	@echo "make ingest     导入 notes/ 示例笔记"
	@echo "make ask Q=...  提问"
	@echo "make serve      启动 HTTP API (127.0.0.1:8000)"
	@echo "make neo4j-up   启动 Neo4j 容器"
	@echo "make reset      清空本地数据"

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

test:
	$(BIN)/python -m pytest

ingest:
	$(BIN)/python -m kgnotes ingest notes/

ask:
	$(BIN)/python -m kgnotes ask "$(Q)"

serve:
	$(BIN)/python -m kgnotes serve --host 127.0.0.1 --port 8000

neo4j-up:
	docker compose up -d neo4j

neo4j-down:
	docker compose down

reset:
	$(BIN)/python -m kgnotes reset --yes
