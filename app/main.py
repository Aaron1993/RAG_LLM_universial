"""FastAPI 应用入口与组件装配。"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import chat, conversations, documents, health
from .config import Settings, get_settings
from .core.embeddings.factory import build_embeddings
from .core.llm.factory import build_llm
from .core.reranker.factory import build_reranker
from .core.retrieval.retriever import Retriever
from .core.vectorstore.factory import build_vectorstore
from .exceptions import register_exception_handlers
from .ingestion.service import IngestionService
from .ingestion.splitter import RecursiveTextSplitter
from .memory.factory import build_memory
from .observability.logging import configure_logging, get_logger
from .observability.middleware import RequestContextMiddleware
from .services.rag_service import RAGService

logger = get_logger("app.main")


async def _init_components(app: FastAPI, settings: Settings) -> None:
    """构建并注入运行时组件。"""
    llm = build_llm(settings)
    embeddings = build_embeddings(settings)
    vectorstore = build_vectorstore(settings)
    await vectorstore.ensure_collection(settings.embedding_dim)
    reranker = build_reranker(settings)
    memory = build_memory(settings)

    retriever = Retriever(embeddings, vectorstore, reranker, settings)
    splitter = RecursiveTextSplitter(settings.chunk_size, settings.chunk_overlap)

    app.state.llm = llm
    app.state.embeddings = embeddings
    app.state.vectorstore = vectorstore
    app.state.memory = memory
    app.state.rag_service = RAGService(llm, retriever, memory, settings)
    app.state.ingestion_service = IngestionService(embeddings, vectorstore, splitter)

    if not settings.api_keys:
        logger.warning("api_keys_not_configured", extra={"extra_fields": {"hint": "生产环境请配置 API_KEYS"}})


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 若已预置组件(如测试),跳过真实初始化
    if getattr(app.state, "rag_service", None) is None:
        await _init_components(app, app.state.settings)
    try:
        yield
    finally:
        store = getattr(app.state, "vectorstore", None)
        memory = getattr(app.state, "memory", None)
        embeddings = getattr(app.state, "embeddings", None)
        if store is not None:
            await store.close()
        if memory is not None:
            await memory.close()
        if embeddings is not None and hasattr(embeddings, "close"):
            await embeddings.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="生产级 RAG+LLM 问答服务",
        lifespan=lifespan,
    )
    app.state.settings = settings
    # 占位,lifespan 据此判断是否需要真实初始化
    if not hasattr(app.state, "rag_service"):
        app.state.rag_service = None

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(documents.router)
    app.include_router(conversations.router)
    return app


app = create_app()
