"""问答接口:非流式 + SSE 流式。"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ...exceptions import AppException
from ...observability.logging import get_logger
from ...services.rag_service import RAGService
from ..deps import get_rag_service, require_api_key
from ..schemas import ChatRequest, ChatResponse, CitationModel

logger = get_logger("app.chat")

router = APIRouter(prefix="/v1", tags=["chat"], dependencies=[Depends(require_api_key)])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, svc: RAGService = Depends(get_rag_service)) -> ChatResponse:
    result = await svc.answer(
        req.query,
        conversation_id=req.conversation_id,
        filters=req.filters,
        top_k=req.top_k,
    )
    return ChatResponse(
        answer=result.answer,
        citations=[CitationModel(**c.__dict__) for c in result.citations],
        conversation_id=result.conversation_id,
    )


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, svc: RAGService = Depends(get_rag_service)) -> StreamingResponse:
    async def event_generator():
        try:
            async for event in svc.answer_stream(
                req.query,
                conversation_id=req.conversation_id,
                filters=req.filters,
                top_k=req.top_k,
            ):
                yield _sse(event)
        except AppException as exc:
            yield _sse({"type": "error", "data": {"code": exc.code, "message": exc.message}})
        except Exception:  # 流已开始,无法改 HTTP 状态码,只能在流内告知
            logger.exception("stream_failed")
            yield _sse({"type": "error", "data": {"code": "internal_error", "message": "服务内部错误"}})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
