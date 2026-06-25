"""请求上下文中间件:为每个请求分配 request_id,并记录访问日志与耗时。"""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .logging import get_logger, request_id_var

logger = get_logger("app.http")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        # 1. 取用调用方传入的 request_id(便于链路追踪),否则新生成一个
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        # 2. 写入 ContextVar,使本请求内的所有日志自动携带 request_id
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            # 3. 回写响应头,方便前端/调用方记录
            response.headers["X-Request-ID"] = request_id
            # 4. 记录访问日志;5xx 用 warning 级别突出
            log = logger.warning if response.status_code >= 500 else logger.info
            log(
                "request",
                extra={
                    "extra_fields": {
                        "method": request.method,
                        "path": request.url.path,
                        "status": response.status_code,
                        "duration_ms": duration_ms,
                        "client": request.client.host if request.client else None,
                    }
                },
            )
            return response
        except Exception:
            # 未被路由层捕获的异常:记录后交给异常处理器(其依赖 request.state.request_id)
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.exception(
                "request_failed",
                extra={"extra_fields": {"method": request.method, "path": request.url.path, "duration_ms": duration_ms}},
            )
            raise
        finally:
            # 6. 还原 ContextVar,避免污染后续请求
            request_id_var.reset(token)
