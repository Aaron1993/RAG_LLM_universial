"""提示词与 RAG 上下文拼装。

交付新项目时,通常只需改本文件中的 DEFAULT_SYSTEM_PROMPT 与拼装话术。
引用溯源策略:给每个上下文片段编号 [1][2]...,提示模型用编号标注引用,
回包再把编号映射回来源(见 services/rag_service.py)。
"""

from __future__ import annotations

from typing import Sequence

from ..core.llm.base import Message
from ..core.vectorstore.base import RetrievedChunk
from ..memory.base import StoredMessage

DEFAULT_SYSTEM_PROMPT = """你是一个严谨的企业知识库问答助手。请遵守以下规则:
1. 仅依据「参考资料」作答,不要编造资料之外的信息。
2. 当资料不足以回答问题时,明确告知「未在知识库中找到相关信息」。
3. 在回答中使用 [编号] 标注引用的资料来源,例如:根据合同条款 [1]……
4. 回答使用简体中文,简洁、准确、结构清晰。"""


def build_context_block(chunks: Sequence[RetrievedChunk]) -> str:
    """把检索片段拼成带编号的参考资料块。"""
    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "未知来源")
        page = chunk.metadata.get("page")
        header = f"[{index}] 来源: {source}"
        if page:
            header += f"(第 {page} 页)"
        lines.append(f"{header}\n{chunk.text}")
    return "\n\n".join(lines)


def build_messages(
    query: str,
    chunks: Sequence[RetrievedChunk],
    history: Sequence[StoredMessage],
    *,
    system_prompt: str | None = None,
) -> list[Message]:
    """组装发送给 LLM 的消息列表:system + 历史 + 当前(带上下文)。"""
    messages: list[Message] = [
        {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT}
    ]
    for item in history:
        messages.append({"role": item.role, "content": item.content})

    if chunks:
        context = build_context_block(chunks)
        user_content = (
            f"参考资料:\n\n{context}\n\n"
            f"用户问题: {query}\n\n"
            "请基于以上参考资料作答,并用 [编号] 标注引用来源。"
        )
    else:
        user_content = f"(知识库中未检索到相关资料)\n\n用户问题: {query}"

    messages.append({"role": "user", "content": user_content})
    return messages
