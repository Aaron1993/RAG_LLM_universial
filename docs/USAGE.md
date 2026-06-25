# 使用说明 · RAG-Service

本文档面向**接入方后端开发**与**框架交付/二次开发者**。

---

## 一、整体架构

```
请求 → 鉴权中间件 → API 路由 → RAGService(编排)
                                  ├── Retriever  → Embedding → VectorStore(Qdrant) →(可选)Reranker
                                  ├── ConversationStore(会话记忆)
                                  └── LLMProvider(Qwen / OpenAI 兼容)
```

所有外部依赖均面向接口编程,实现可插拔:

| 抽象 | 默认实现 | 位置 |
|---|---|---|
| `LLMProvider` | OpenAI 兼容(Qwen) | `app/core/llm/` |
| `EmbeddingProvider` | OpenAI 兼容(text-embedding-v3) | `app/core/embeddings/` |
| `VectorStore` | Qdrant | `app/core/vectorstore/` |
| `Reranker` | NoOp(默认关闭)/ DashScope | `app/core/reranker/` |
| `ConversationStore` | SQLite(可切 memory / redis) | `app/memory/` |
| `DocumentLoader` | pdf / docx / txt / md / html | `app/ingestion/loaders/` |

---

## 二、配置(.env)

复制 `.env.example` 为 `.env`。关键项:

| 变量 | 说明 |
|---|---|
| `DASHSCOPE_API_KEY` | 阿里百炼密钥(LLM 与 Embedding 默认共用) |
| `LLM_MODEL` | 对话模型,默认 `qwen-plus`(可选 `qwen-max`/`qwen-turbo`) |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | 向量模型与维度,**维度必须与已建集合一致** |
| `QDRANT_URL` / `QDRANT_COLLECTION` | 向量库地址与集合名 |
| `API_KEYS` | 逗号分隔的服务访问密钥;**留空则不鉴权(仅限本地开发)** |
| `RETRIEVAL_TOP_K` / `SCORE_THRESHOLD` | 检索条数与分数阈值 |
| `RERANKER_ENABLED` / `RERANKER_PROVIDER` | 重排开关与实现 |
| `MEMORY_BACKEND` | 会话记忆后端:`sqlite`(默认)/`memory`/`redis` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 文档分块大小与重叠 |

> ⚠️ 修改 `EMBEDDING_DIM` 后需重建集合(删除旧集合或换 `QDRANT_COLLECTION`),否则维度不匹配会写入失败。

---

## 三、API 一览

所有 `/v1/*` 业务接口需带请求头 `X-API-Key`(当配置了 `API_KEYS` 时)。错误统一格式:

```json
{ "error": { "code": "unauthorized", "message": "...", "request_id": "..." } }
```

### 1. 文档入库

**上传文件**(multipart):
```bash
curl -X POST http://localhost:8000/v1/documents \
  -H "X-API-Key: $KEY" \
  -F "file=@./合同.pdf" -F "tenant=acme"
# → {"document_id":"...", "chunks": 42}
```

**直接传文本**(JSON):
```bash
curl -X POST http://localhost:8000/v1/documents/text \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"text":"正文...","source":"hr.md","metadata":{"dept":"hr"},"tenant":"acme"}'
```
支持类型:`.pdf .docx .txt .md .html .csv .json` 等(未知类型按纯文本处理)。

### 2. 删除文档
```bash
curl -X DELETE http://localhost:8000/v1/documents \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"document_ids":["<document_id>"]}'
# 或按元数据过滤: {"filters":{"tenant":"acme"}}
```

### 3. 问答(非流式)
```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"query":"付款期限是多久?","conversation_id":null,"filters":{"tenant":"acme"},"top_k":5}'
```
响应:
```json
{
  "answer": "根据合同 [1],付款期限为 30 日。",
  "citations": [
    {"index":1,"document_id":"...","source":"合同.pdf","page":2,"score":0.91,"snippet":"..."}
  ],
  "conversation_id": "..."
}
```
- 传相同 `conversation_id` 即可实现**多轮对话**;不传则自动新建并在响应中返回。
- `filters` 用于多租户 / 分库隔离(按入库时写入的元数据过滤)。

### 4. 问答(SSE 流式)
```bash
curl -N -X POST http://localhost:8000/v1/chat/stream \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"query":"付款期限是多久?"}'
```
事件流(`text/event-stream`),每行 `data: {json}`:
```
data: {"type":"token","data":"根据"}
data: {"type":"token","data":"合同"}
data: {"type":"citations","data":[{"index":1,"source":"合同.pdf",...}]}
data: {"type":"done","data":{"conversation_id":"..."}}
```
出错时产出 `{"type":"error","data":{"code":"...","message":"..."}}`。

### 5. 会话管理
```bash
curl http://localhost:8000/v1/conversations/<cid> -H "X-API-Key: $KEY"        # 查历史
curl -X DELETE http://localhost:8000/v1/conversations/<cid> -H "X-API-Key: $KEY"  # 清空
```

### 6. 健康检查(无需鉴权)
- `GET /healthz`:进程存活
- `GET /readyz`:依赖(Qdrant)就绪,未就绪返回 503

---

## 四、引用溯源原理

国产 OpenAI 兼容模型无原生 citations,框架在应用层实现:
1. 检索片段拼装为带编号 `[1][2]...` 的参考资料块(`app/prompts/templates.py`);
2. 系统提示要求模型用 `[编号]` 标注引用;
3. 响应的 `citations[]` 把编号映射回来源(文档、页码、片段、分数)。

接入方可直接展示 `answer` 中的 `[n]` 并与 `citations` 联动高亮。

---

## 五、交付新项目:你通常只改这些

1. **`.env`**:模型 endpoint/key、集合名、`API_KEYS`、分块参数。
2. **`app/prompts/templates.py`**:`DEFAULT_SYSTEM_PROMPT` 与拼装话术(领域定制核心)。
3. **元数据 / 过滤字段**:入库时通过 `metadata` / `tenant` 写入,问答时通过 `filters` 过滤。
4. **(可选)新文件类型**:在 `app/ingestion/loaders/base.py` 注册一个 `DocumentLoader`。
5. **(可选)开启重排**:`.env` 设 `RERANKER_ENABLED=true`、`RERANKER_PROVIDER=dashscope`。

---

## 六、扩展指南(替换实现)

以替换向量库为例:
1. 在 `app/core/vectorstore/` 新增 `xxx_store.py`,实现 `VectorStore` 抽象;
2. 在 `app/core/vectorstore/factory.py` 的 `build_vectorstore` 中按 `settings.vector_store` 分支返回;
3. 在 `config.py` 的 `vector_store` 字面量类型中加入新值。

LLM / Embedding / Reranker / Memory 的替换方式完全一致(实现接口 → 注册工厂 → 配置项)。

---

## 七、生产部署建议

- **务必配置 `API_KEYS`**,并按需收紧 `CORS_ORIGINS`。
- 多实例部署时把 `MEMORY_BACKEND` 改为 `redis`(并安装 `pip install redis`、启用 compose 中的 redis 服务)。
- 反向代理(Nginx)上对 `/v1/chat/stream` **关闭响应缓冲**(已设置 `X-Accel-Buffering: no`)。
- 日志系统:`LOG_FORMAT=json`(生产,便于 ELK/Loki 采集)或 `console`(本地,便于阅读);
  每条日志自动携带 `request_id`(可由调用方通过 `X-Request-ID` 透传以做链路追踪);
  uvicorn 日志已统一接管为同一格式。二次开发时可用 `log_event(logger, level, "event", **字段)`
  记录结构化事件,用 `with log_context(**字段):` 为代码块内所有日志绑定上下文(如 conversation_id)。
- 通过 `/healthz`(存活)与 `/readyz`(就绪)接入 K8s 探针。
- 上游(LLM/Embedding/向量库)已内置超时与指数退避重试(`LLM_MAX_RETRIES` 等可调)。

---

## 八、测试

```bash
pip install -e ".[dev]"
pytest
```
测试使用 fakes 替换外部依赖,**不触网、不需要 Qdrant**,可在 CI 中直接运行。
