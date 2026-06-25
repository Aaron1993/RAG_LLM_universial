"""健康检查:存活与就绪(探测向量库连通)。"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict:
    """存活探针:进程在线即返回 200。"""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    """就绪探针:依赖(向量库)可用才返回 200。"""
    store = getattr(request.app.state, "vectorstore", None)
    ready = bool(store) and await store.health()
    status_code = 200 if ready else 503
    return JSONResponse(status_code=status_code, content={"status": "ok" if ready else "unavailable"})
