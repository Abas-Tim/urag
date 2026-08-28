"""Compatibility shim: native-language extractors (Go, Rust, Java, C/C++, C#).

Extractors live in per-language modules (go_ext, rust_ext, java_ext, c_ext,
csharp_ext); this module re-exports them so existing imports keep working.
"""

from .c_ext import CExtractor
from .csharp_ext import CSharpExtractor
from .go_ext import GoExtractor
from .java_ext import JavaExtractor
from .rust_ext import RustExtractor

__all__ = ["CExtractor", "CSharpExtractor", "GoExtractor", "JavaExtractor", "RustExtractor"]
