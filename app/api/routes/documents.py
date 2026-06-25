"""文档入库 / 删除接口。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ...config import Settings
from ...core.vectorstore.base import VectorStore
from ...exceptions import ValidationAppError
from ...ingestion.service import IngestionService
from ..deps import get_ingestion_service, get_vectorstore, require_api_key, settings_dep
from ..schemas import DeleteDocumentsRequest, IngestResponse, IngestTextRequest, StatusResponse

router = APIRouter(prefix="/v1", tags=["documents"], dependencies=[Depends(require_api_key)])


@router.post("/documents", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(..., description="待入库文件(pdf/docx/txt/md/html 等)"),
    tenant: str | None = Form(default=None),
    svc: IngestionService = Depends(get_ingestion_service),
    settings: Settings = Depends(settings_dep),
) -> IngestResponse:
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise ValidationAppError(f"文件超过大小上限 {settings.max_upload_mb}MB")
    if not data:
        raise ValidationAppError("文件为空")
    result = await svc.ingest_bytes(data, file.filename or "upload", tenant=tenant)
    return IngestResponse(document_id=result.document_id, chunks=result.chunks)


@router.post("/documents/text", response_model=IngestResponse)
async def ingest_text(
    req: IngestTextRequest,
    svc: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    result = await svc.ingest_text(
        req.text,
        source=req.source,
        metadata=req.metadata,
        tenant=req.tenant,
    )
    return IngestResponse(document_id=result.document_id, chunks=result.chunks)


@router.delete("/documents", response_model=StatusResponse)
async def delete_documents(
    req: DeleteDocumentsRequest,
    store: VectorStore = Depends(get_vectorstore),
) -> StatusResponse:
    if not req.document_ids and not req.filters:
        raise ValidationAppError("需提供 document_ids 或 filters 之一")
    if req.document_ids:
        for document_id in req.document_ids:
            await store.delete(filters={"document_id": document_id})
    if req.filters:
        await store.delete(filters=req.filters)
    return StatusResponse(status="deleted")
