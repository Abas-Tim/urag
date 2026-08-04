"""Extractor registry: language -> extractor."""

from __future__ import annotations

from .base import Extractor
from .python_ext import PythonExtractor
from .ts_ext import TsExtractor
from .markdown_ext import MarkdownExtractor
from .native_ext import GoExtractor, RustExtractor, JavaExtractor, CExtractor, CSharpExtractor

_REGISTRY: dict[str, Extractor] = {
    "python": PythonExtractor(),
    "typescript": TsExtractor("typescript"),
    "tsx": TsExtractor("tsx"),
    "javascript": TsExtractor("javascript"),
    "markdown": MarkdownExtractor(),
    "go": GoExtractor(),
    "rust": RustExtractor(),
    "java": JavaExtractor(),
    "c": CExtractor("c"),
    "cpp": CExtractor("cpp"),
    "csharp": CSharpExtractor(),
}


def get_extractor(language: str) -> Extractor | None:
    return _REGISTRY.get(language)
