"""Extractor registry: language -> extractor."""

from __future__ import annotations

from .base import Extractor
from .config_ext import ConfigExtractor
from .markdown_ext import MarkdownExtractor
from .native_ext import (
    CExtractor,
    CSharpExtractor,
    GoExtractor,
    JavaExtractor,
    RustExtractor,
)
from .python_ext import PythonExtractor
from .ts_ext import TsExtractor
from .xml_ext import XmlExtractor

_REGISTRY: dict[str, Extractor] = {
    "python": PythonExtractor(),
    "typescript": TsExtractor("typescript"),
    "tsx": TsExtractor("tsx"),
    "javascript": TsExtractor("javascript"),
    "markdown": MarkdownExtractor(),
    "json": ConfigExtractor("json"),
    "yaml": ConfigExtractor("yaml"),
    "toml": ConfigExtractor("toml"),
    "ini": ConfigExtractor("ini"),
    "env": ConfigExtractor("env"),
    "go": GoExtractor(),
    "rust": RustExtractor(),
    "java": JavaExtractor(),
    "c": CExtractor("c"),
    "cpp": CExtractor("cpp"),
    "csharp": CSharpExtractor(),
    "xml": XmlExtractor(),
}


def get_extractor(language: str) -> Extractor | None:
    return _REGISTRY.get(language)
