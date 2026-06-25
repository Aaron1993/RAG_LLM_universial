"""文档加载器:按扩展名分发,把原始字节抽取为带元数据的文本段。"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import PurePosixPath


@dataclass
class LoadedSection:
    """文档中的一段文本,metadata 可携带 page 等定位信息。"""

    text: str
    metadata: dict = field(default_factory=dict)


class DocumentLoader(ABC):
    @abstractmethod
    def load(self, data: bytes, filename: str) -> list[LoadedSection]:
        raise NotImplementedError


class TextLoader(DocumentLoader):
    """纯文本 / Markdown。"""

    def load(self, data: bytes, filename: str) -> list[LoadedSection]:
        text = data.decode("utf-8", errors="ignore")
        return [LoadedSection(text=text)] if text.strip() else []


class PdfLoader(DocumentLoader):
    def load(self, data: bytes, filename: str) -> list[LoadedSection]:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        sections: list[LoadedSection] = []
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                sections.append(LoadedSection(text=text, metadata={"page": index}))
        return sections


class DocxLoader(DocumentLoader):
    def load(self, data: bytes, filename: str) -> list[LoadedSection]:
        import docx  # python-docx

        document = docx.Document(io.BytesIO(data))
        text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
        return [LoadedSection(text=text)] if text.strip() else []


class HtmlLoader(DocumentLoader):
    def load(self, data: bytes, filename: str) -> list[LoadedSection]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(data.decode("utf-8", errors="ignore"), "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n").strip()
        return [LoadedSection(text=text)] if text.strip() else []


# 扩展名 -> 加载器(单例即可,加载器无状态)
_TEXT = TextLoader()
_LOADERS: dict[str, DocumentLoader] = {
    ".txt": _TEXT,
    ".md": _TEXT,
    ".markdown": _TEXT,
    ".csv": _TEXT,
    ".json": _TEXT,
    ".log": _TEXT,
    ".pdf": PdfLoader(),
    ".docx": DocxLoader(),
    ".html": HtmlLoader(),
    ".htm": HtmlLoader(),
}


def get_loader(filename: str) -> DocumentLoader:
    """根据扩展名选择加载器,未知类型按纯文本处理。"""
    suffix = PurePosixPath(filename).suffix.lower()
    return _LOADERS.get(suffix, _TEXT)
