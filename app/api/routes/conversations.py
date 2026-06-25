"""会话查询 / 清除接口。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ...memory.base import ConversationStore
from ..deps import get_memory, require_api_key
from ..schemas import ConversationResponse, MessageModel, StatusResponse

router = APIRouter(prefix="/v1/conversations", tags=["conversations"], dependencies=[Depends(require_api_key)])


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    memory: ConversationStore = Depends(get_memory),
) -> ConversationResponse:
    messages = await memory.load(conversation_id)
    return ConversationResponse(
        conversation_id=conversation_id,
        messages=[MessageModel(role=m.role, content=m.content) for m in messages],
    )


@router.delete("/{conversation_id}", response_model=StatusResponse)
async def clear_conversation(
    conversation_id: str,
    memory: ConversationStore = Depends(get_memory),
) -> StatusResponse:
    await memory.clear(conversation_id)
    return StatusResponse(status="cleared")
