"""统一异常体系与错误响应封装。

所有对外错误统一为:
    {"error": {"code": "...", "message": "...", "request_id": "..."}}
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .observability.logging import get_logger

logger = get_logger("app.error")


class AppException(Exception):
    """领域异常基类。"""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(
        self,
        message: str = "Internal server error",
        *,
        code: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code


class AuthError(AppException):
    status_code = 401
    code = "unauthorized"


class NotFoundError(AppException):
    status_code = 404
    code = "not_found"


class ValidationAppError(AppException):
    status_code = 422
    code = "invalid_request"


class RateLimitedError(AppException):
    status_code = 429
    code = "rate_limited"


class UpstreamError(AppException):
    """上游服务(LLM/Embedding/向量库/Rerank)错误。"""

    status_code = 502
    code = "upstream_error"


def _error_body(code: str, message: str, request: Request) -> dict:
    request_id = getattr(request.state, "request_id", None)
    return {"error": {"code": code, "message": message, "request_id": request_id}}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppException)
    async def _handle_app_exception(request: Request, exc: AppException) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error("app_exception", extra={"extra_fields": {"code": exc.code, "message": exc.message}})
        return JSONResponse(status_code=exc.status_code, content=_error_body(exc.code, exc.message, request))

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        body = _error_body("invalid_request", "Request validation failed", request)
        body["error"]["detail"] = jsonable_encoder(exc.errors())
        return JSONResponse(status_code=422, content=body)

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_exception")
        return JSONResponse(
            status_code=500,
            content=_error_body("internal_error", "Internal server error", request),
        )
