"""结构化日志系统。

设计目标:
- **双格式**:`json`(生产,便于 ELK/Loki 采集)与 `console`(本地,便于阅读),由 LOG_FORMAT 控制。
- **上下文透传**:通过 ContextVar 注入 `request_id` 与请求级自定义字段,
  所有日志记录(含三方库)自动携带,无需逐处传参。
- **统一日志出口**:接管 uvicorn / gunicorn 日志,避免格式不一致与重复输出。
- **便捷 API**:`log_event()` 记录结构化事件,`log_context()` 在代码块内绑定字段。

用法:
    from app.observability.logging import get_logger, log_event, log_context
    logger = get_logger("app.xxx")
    log_event(logger, logging.INFO, "did_something", count=3, cost_ms=12)
    with log_context(conversation_id=cid):
        ...  # 块内所有日志都会带上 conversation_id
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

# 当前请求的 request_id(由 RequestContextMiddleware 设置)
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
# 当前请求 / 任务绑定的额外字段(由 log_context 设置,会合并进每条日志)
_context_fields: ContextVar[dict[str, Any]] = ContextVar("log_context_fields", default={})


def _collect_fields(record: logging.LogRecord) -> dict[str, Any]:
    """汇总一条日志要附带的结构化字段:request_id + 上下文字段 + 调用处 extra_fields。"""
    fields: dict[str, Any] = {}
    request_id = request_id_var.get()
    if request_id:
        fields["request_id"] = request_id
    context = _context_fields.get()
    if context:
        fields.update(context)
    extra = getattr(record, "extra_fields", None)
    if isinstance(extra, dict):
        fields.update(extra)
    return fields


class JsonFormatter(logging.Formatter):
    """单行 JSON 输出,适合日志采集系统。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(_collect_fields(record))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    """人类可读的彩色无依赖输出,适合本地开发。"""

    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"{self.formatTime(record, '%H:%M:%S')} "
            f"{record.levelname:<7} {record.name}: {record.getMessage()}"
        )
        fields = _collect_fields(record)
        if fields:
            base += " | " + " ".join(f"{k}={v}" for k, v in fields.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """初始化全局日志。应在应用启动最早期调用一次。"""
    formatter: logging.Formatter = ConsoleFormatter() if fmt == "console" else JsonFormatter()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # 统一接管 uvicorn / gunicorn 日志:清空其私有 handler,改为向 root 传播,
    # 这样它们也走我们的 JSON/控制台格式,且自动带上 request_id。
    for name in (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "gunicorn.error",
        "gunicorn.access",
    ):
        managed = logging.getLogger(name)
        managed.handlers.clear()
        managed.propagate = True

    # access 日志较吵,且我们的中间件已逐请求记录,降噪到 WARNING
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """以结构化字段记录一条事件日志(等价于 logger.log(level, msg, extra={...}))。"""
    logger.log(level, message, extra={"extra_fields": fields})


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """在代码块内为所有日志绑定额外字段(如 conversation_id);退出后自动还原。"""
    merged = dict(_context_fields.get())
    merged.update(fields)
    token = _context_fields.set(merged)
    try:
        yield
    finally:
        _context_fields.reset(token)
