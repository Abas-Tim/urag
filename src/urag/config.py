"""Project configuration: .urag/urag.toml written by `urag init`."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .models import CHUNK_TYPES, SYMBOL_TYPES

UURAG_DIR = ".urag"
CONFIG_NAME = "urag.toml"
DB_NAME = "index.db"

DEFAULT_EXCLUDES = [
    ".urag",
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "dist",
    "build",
    "target",
    ".venv",
    "venv",
    "__pycache__",
    ".next",
    ".nuxt",
    ".cache",
    "coverage",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
    "vendor",
    "bin",
    "obj",
    "out",
]

SUPPORTED_LANGUAGES = {
    "python": {"ext": (".py", ".pyi"), "kind": "source"},
    "typescript": {"ext": (".ts", ".mts", ".cts"), "kind": "source"},
    "tsx": {"ext": (".tsx",), "kind": "source"},
    "javascript": {"ext": (".js", ".jsx", ".mjs", ".cjs"), "kind": "source"},
    "go": {"ext": (".go",), "kind": "source"},
    "rust": {"ext": (".rs",), "kind": "source"},
    "java": {"ext": (".java",), "kind": "source"},
    "c": {"ext": (".c", ".h"), "kind": "source"},
    "cpp": {"ext": (".cpp", ".cc", ".cxx", ".hpp", ".hh"), "kind": "source"},
    "csharp": {"ext": (".cs",), "kind": "source"},
    "markdown": {"ext": (".md", ".markdown", ".mdx"), "kind": "doc"},
}


def language_for_path(path: Path) -> tuple[str, str] | None:
    """Return (language, kind) for a file, or None if unsupported."""
    ext = path.suffix.lower()
    for lang, spec in SUPPORTED_LANGUAGES.items():
        if ext in spec["ext"]:
            return lang, spec["kind"]
    return None


@dataclass
class EmbeddingConfig:
    provider: str = "local"  # local | http | none
    model: str = "BAAI/bge-small-en-v1.5"
    dimension: int = 384
    http_url: str = ""
    http_api_key: str = ""
    http_model: str = ""
    http_timeout: float = 30.0

    def fingerprint(self) -> str:
        return "|".join(
            [self.provider, self.model, str(self.dimension), self.http_url, self.http_model]
        )


@dataclass
class IndexConfig:
    languages: list[str] = field(default_factory=lambda: list(SUPPORTED_LANGUAGES))
    exclude: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDES))
    include: list[str] = field(default_factory=list)
    ignore_gitignore: bool = False
    max_file_bytes: int = 1_000_000


@dataclass
class RetrievalConfig:
    default_top_k: int = 10
    rrf_k: int = 60
    max_evidence_tokens: int = 1500
    dense_candidates: int = 30
    lexical_candidates: int = 30
    max_results_per_file: int = 3
    lexical_weight: float = 1.0
    dense_weight: float = 1.0


@dataclass
class Config:
    project_root: Path
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    @property
    def urag_dir(self) -> Path:
        return self.project_root / UURAG_DIR

    @property
    def db_path(self) -> Path:
        return self.urag_dir / DB_NAME

    @property
    def config_path(self) -> Path:
        return self.urag_dir / CONFIG_NAME

    def save(self) -> None:
        self.urag_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            "# urag configuration",
            "[embedding]",
            f"provider = {self.embedding.provider!r}",
            f"model = {self.embedding.model!r}",
            f"dimension = {self.embedding.dimension}",
            f"http_url = {self.embedding.http_url!r}",
            f"http_api_key = {self.embedding.http_api_key!r}",
            f"http_model = {self.embedding.http_model!r}",
            "",
            "[index]",
            f"languages = {self.index.languages!r}",
            f"exclude = {self.index.exclude!r}",
            f"include = {self.index.include!r}",
            f"ignore_gitignore = {str(self.index.ignore_gitignore).lower()}",
            f"max_file_bytes = {self.index.max_file_bytes}",
            "",
            "[retrieval]",
            f"default_top_k = {self.retrieval.default_top_k}",
            f"rrf_k = {self.retrieval.rrf_k}",
            f"max_evidence_tokens = {self.retrieval.max_evidence_tokens}",
            f"dense_candidates = {self.retrieval.dense_candidates}",
            f"lexical_candidates = {self.retrieval.lexical_candidates}",
            f"max_results_per_file = {self.retrieval.max_results_per_file}",
            f"lexical_weight = {self.retrieval.lexical_weight}",
            f"dense_weight = {self.retrieval.dense_weight}",
            "",
        ]
        self.config_path.write_text("\n".join(lines), encoding="utf-8")


def default_config(project_root: Path) -> Config:
    return Config(project_root=project_root)


def _parse_toml(path: Path, cfg: Config) -> None:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    emb = data.get("embedding", {})
    idx = data.get("index", {})
    ret = data.get("retrieval", {})
    if emb:
        for k in ("provider", "model", "dimension", "http_url", "http_api_key", "http_model", "http_timeout"):
            if k in emb:
                setattr(cfg.embedding, k, emb[k])
    if idx:
        for k in ("languages", "exclude", "include", "ignore_gitignore", "max_file_bytes"):
            if k in idx:
                setattr(cfg.index, k, idx[k])
    if ret:
        for k in (
            "default_top_k", "rrf_k", "max_evidence_tokens", "dense_candidates",
            "lexical_candidates", "max_results_per_file", "lexical_weight", "dense_weight",
        ):
            if k in ret:
                setattr(cfg.retrieval, k, ret[k])


def load_config(project_root: Path) -> Config:
    cfg = default_config(project_root)
    cfg_path = cfg.config_path
    if cfg_path.exists():
        _parse_toml(cfg_path, cfg)
    else:
        cfg.save()
    return cfg


def discover_project_root(start: Path | None = None) -> Path:
    """Walk up from start looking for a .urag dir, git root, or pyproject."""
    cur = (start or Path.cwd()).resolve()
    for d in (cur, *cur.parents):
        if (d / UURAG_DIR).exists():
            return d
        if (d / ".git").exists() or (d / ".git").is_file():
            return d
        if (d / "pyproject.toml").exists() or (d / "package.json").exists():
            return d
    return cur


def default_model_cache_dir() -> Path:
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / "urag"
    return Path.home() / ".cache" / "urag"
