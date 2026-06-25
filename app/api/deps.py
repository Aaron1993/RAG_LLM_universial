"""API 依赖:鉴权与组件注入。

组件实例在应用启动(lifespan)时构建并挂到 app.state,
此处通过 Request 取出,便于测试时以 dependency_overrides 替换。
"""

from __future__ import annotations

import hmac

from fastapi import Depends, Header, Request

from ..config import Settings, get_settings
from ..core.vectorstore.base import VectorStore
from ..exceptions import AuthError
from ..memory.base import ConversationStore
from ..services.rag_service import RAGService
from ..ingestion.service import IngestionService


def settings_dep() -> Settings:
    return get_settings()


async def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(settings_dep),
) -> None:
    keys = settings.api_keys
    if not keys:
        # 未配置 API_KEYS:仅限本地开发,放行
        return
    if not x_api_key or not any(hmac.compare_digest(x_api_key, k) for k in keys):
        raise AuthError("无效或缺失的 API Key")


def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service


def get_ingestion_service(request: Request) -> IngestionService:
    return request.app.state.ingestion_service


def get_memory(request: Request) -> ConversationStore:
    return request.app.state.memory


def get_vectorstore(request: Request) -> VectorStore:
    return request.app.state.vectorstore
