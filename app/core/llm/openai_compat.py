"""基于 OpenAI 兼容协议的 LLM 实现(适配阿里百炼/Qwen、DeepSeek、GLM、OpenAI 等)。"""

from __future__ import annotations

from typing import AsyncIterator, Sequence

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError as OpenAIRateLimitError,
)
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ...exceptions import RateLimitedError, UpstreamError
from .base import LLMProvider, Message

_RETRYABLE = (APIConnectionError, APITimeoutError)


class OpenAICompatLLM(LLMProvider):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        # max_retries=0:由 tenacity 统一控制重试,避免双重退避
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "EMPTY",
            timeout=timeout,
            max_retries=0,
        )
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries

    async def _create(self, messages: Sequence[Message], *, stream: bool, temperature, max_tokens):
        """统一的 chat.completions 调用入口。

        负责:组装参数、对可重试错误(连接/超时)做指数退避、把厂商异常
        归一为框架异常(限流→RateLimitedError,其余→UpstreamError)。
        """
        kwargs = dict(
            model=self._model,
            messages=list(messages),
            temperature=self._temperature if temperature is None else temperature,
            max_tokens=max_tokens or self._max_tokens,
            stream=stream,
        )
        try:
            async for attempt in AsyncRetrying(
                reraise=True,
                stop=stop_after_attempt(self._max_retries + 1),
                wait=wait_exponential(multiplier=0.5, max=8),
                retry=retry_if_exception_type(_RETRYABLE),
            ):
                with attempt:
                    return await self._client.chat.completions.create(**kwargs)
        except OpenAIRateLimitError as exc:
            raise RateLimitedError("LLM 服务限流,请稍后重试") from exc
        except APIError as exc:
            raise UpstreamError(f"LLM 上游错误: {exc}") from exc
        except _RETRYABLE as exc:  # 重试耗尽
            raise UpstreamError(f"LLM 连接失败: {exc}") from exc

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        resp = await self._create(messages, stream=False, temperature=temperature, max_tokens=max_tokens)
        if not resp.choices:
            return ""
        return resp.choices[0].message.content or ""

    async def chat_stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        stream = await self._create(messages, stream=True, temperature=temperature, max_tokens=max_tokens)
        # 流已建立后再出错无法重试(可能已产出部分内容),只能向上抛出归一异常
        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                # 逐块取增量文本;部分块可能只含角色/结束标记,delta 为空则跳过
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except OpenAIRateLimitError as exc:
            raise RateLimitedError("LLM 服务限流,请稍后重试") from exc
        except APIError as exc:
            raise UpstreamError(f"LLM 流式输出错误: {exc}") from exc
