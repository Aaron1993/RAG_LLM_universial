# RAG-Service · 生产级 RAG+LLM 问答框架

面向后端系统调用的通用 RAG(检索增强生成)问答服务。换项目时,**大部分改动集中在 `.env` 与 `app/prompts/templates.py`**,即可快速交付。

## 技术栈
- **框架**:Python 3.11+ / FastAPI(全异步)
- **大模型**:阿里百炼(DashScope)Qwen,OpenAI 兼容协议接入;Provider 抽象层,可切换 DeepSeek / GLM / OpenAI
- **向量库**:Qdrant(抽象层,可替换)
- **能力**:SSE 流式 · 多轮对话记忆 · 引用溯源 · API Key 鉴权 · 文档入库流水线 · Rerank(预留,默认关闭)

## 快速开始

```bash
# 1. 配置
cp .env.example .env
# 编辑 .env,至少填入 DASHSCOPE_API_KEY,生产环境再配置 API_KEYS

# 2. 一键启动(含 Qdrant)
docker compose up -d --build

# 3. 验证
curl http://localhost:8000/healthz
```

本地开发(不走 Docker):

```bash
pip install -r requirements.txt
# 需自行启动 Qdrant: docker run -p 6333:6333 qdrant/qdrant
uvicorn app.main:app --reload
```

## 三步用起来

```bash
# 入库一段文本
curl -X POST http://localhost:8000/v1/documents/text \
  -H "Content-Type: application/json" -H "X-API-Key: $KEY" \
  -d '{"text":"公司年假为每年 10 天。","source":"hr.md"}'

# 提问
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" -H "X-API-Key: $KEY" \
  -d '{"query":"年假有几天?"}'
```

接口文档(Swagger)在线访问:`http://localhost:8000/docs`

详尽说明见 [docs/USAGE.md](docs/USAGE.md)。

## 测试

```bash
pip install -e ".[dev]"
pytest
```
