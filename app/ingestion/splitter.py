"""递归文本分块器:按句子/段落边界切分,合并到目标大小并保留重叠。"""

from __future__ import annotations

import re

# 句末或换行边界(中英文)
_BOUNDARY = re.compile(r"(\n\n+|\n|(?<=[。！？!?.;；])\s*)")


class RecursiveTextSplitter:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def _segment(self, text: str) -> list[str]:
        """切成不超过 chunk_size 的原子片段(保留分隔符)。"""
        parts = _BOUNDARY.split(text)
        segments: list[str] = []
        buffer = ""
        for part in parts:
            if not part:
                continue
            buffer += part
            if part.endswith("\n") or re.search(r"[。！？!?.;；]\s*$", buffer):
                segments.append(buffer)
                buffer = ""
        if buffer:
            segments.append(buffer)

        # 超长片段硬切
        atoms: list[str] = []
        for seg in segments:
            if len(seg) <= self._chunk_size:
                atoms.append(seg)
            else:
                for i in range(0, len(seg), self._chunk_size):
                    atoms.append(seg[i : i + self._chunk_size])
        return [a for a in atoms if a]

    def split_text(self, text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []

        atoms = self._segment(text)
        chunks: list[str] = []
        current = ""
        for atom in atoms:
            if not current:
                current = atom
            elif len(current) + len(atom) <= self._chunk_size:
                current += atom
            else:
                chunks.append(current.strip())
                overlap = current[-self._chunk_overlap :] if self._chunk_overlap else ""
                current = overlap + atom
        if current.strip():
            chunks.append(current.strip())
        return [c for c in chunks if c]
