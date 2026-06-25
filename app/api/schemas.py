"""API 请求 / 响应模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, description="用户问题")
    conversation_id: str | None = Field(default=None, description="会话 ID,不传则新建")
    filters: dict[str, str] | None = Field(default=None, description="按元数据过滤,如 {'tenant': 'a'}")
    top_k: int | None = Field(default=None, ge=1, le=50, description="检索条数")


class CitationModel(BaseModel):
    index: int
    document_id: str | None = None
    source: str | None = None
    page: int | None = None
    score: float
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationModel]
    conversation_id: str


class IngestTextRequest(BaseModel):
    text: str = Field(min_length=1)
    source: str = "inline"
    metadata: dict[str, str] | None = None
    tenant: str | None = None


class IngestResponse(BaseModel):
    document_id: str
    chunks: int


class DeleteDocumentsRequest(BaseModel):
    document_ids: list[str] | None = None
    filters: dict[str, str] | None = None


class MessageModel(BaseModel):
    role: str
    content: str


class ConversationResponse(BaseModel):
    conversation_id: str
    messages: list[MessageModel]


class StatusResponse(BaseModel):
    status: str
