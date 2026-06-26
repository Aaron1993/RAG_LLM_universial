"""应用配置:基于 pydantic-settings,从环境变量 / .env 读取。

设计要点:
- LLM 与 Embedding 默认复用阿里百炼(DASHSCOPE_*)的 base_url / api_key,留空则回退。
- 列表类字段(API_KEYS / CORS_ORIGINS)以逗号分隔的字符串配置,用 NoDecode 关闭 JSON 解析。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- 应用 ----------
    app_name: str = "rag-service"
    environment: Literal["dev", "staging", "prod"] = "dev"
    debug: bool = False
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"  # json=生产采集 / console=本地阅读
    host: str = "0.0.0.0"
    port: int = 8000

    # ---------- 安全 ----------
    api_keys: Annotated[list[str], NoDecode] = []
    cors_origins: Annotated[list[str], NoDecode] = ["*"]
    max_upload_mb: int = 25

    # ---------- 阿里百炼 / DashScope ----------
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # ---------- 大模型 ----------
    llm_provider: Literal["openai_compat"] = "openai_compat"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "qwen-plus"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048
    llm_timeout: float = 60.0
    llm_max_retries: int = 2

    # ---------- 向量化 ----------
    # openai_compat: 百炼/DashScope 等 OpenAI 兼容 API
    # tei: 本地 HuggingFace text-embeddings-inference(如 BAAI/bge-large-zh-v1.5)
    embedding_provider: Literal["openai_compat", "tei"] = "openai_compat"
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = "text-embedding-v3"
    embedding_dim: int = 1024
    embedding_batch_size: int = 16
    embedding_timeout: float = 30.0
    embedding_max_retries: int = 2

    # 本地 TEI 服务(embedding_provider=tei 时生效)
    tei_url: str = "http://localhost:8080"
    tei_api_key: str = ""
    tei_normalize: bool = True
    tei_truncate: bool = True

    # ---------- 向量数据库 ----------
    vector_store: Literal["qdrant"] = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "rag_documents"
    retrieval_top_k: int = 5
    score_threshold: float = 0.0

    # ---------- 重排序(默认关闭) ----------
    reranker_enabled: bool = False
    reranker_provider: Literal["noop", "dashscope"] = "noop"
    reranker_model: str = "gte-rerank-v2"
    rerank_top_n: int = 5

    # ---------- 会话记忆 ----------
    memory_backend: Literal["sqlite", "memory", "redis"] = "sqlite"
    sqlite_path: str = "./data/conversations.db"
    redis_url: str = "redis://localhost:6379/0"
    max_history_turns: int = 10

    # ---------- 文档分块 ----------
    chunk_size: int = 800
    chunk_overlap: int = 120

    @field_validator("api_keys", "cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """允许用逗号分隔的字符串配置列表字段。"""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def _apply_fallbacks(self) -> "Settings":
        # LLM / Embedding 的 base_url 与 key 默认回退到 DashScope 配置
        self.llm_base_url = self.llm_base_url or self.dashscope_base_url
        self.embedding_base_url = self.embedding_base_url or self.dashscope_base_url
        self.llm_api_key = self.llm_api_key or self.dashscope_api_key
        self.embedding_api_key = self.embedding_api_key or self.dashscope_api_key
        if not self.cors_origins:
            self.cors_origins = ["*"]
        return self


@lru_cache
def get_settings() -> Settings:
    """全局单例配置。"""
    return Settings()
