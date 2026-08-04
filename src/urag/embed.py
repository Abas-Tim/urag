"""Pluggable embedding providers: local ONNX (fastembed) and OpenAI-compatible HTTP.

The embedder interface has a query/passage distinction so providers that
support retrieval prefixes (bge) can use them; others fall back to plain embed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import httpx

from .config import EmbeddingConfig, default_model_cache_dir


class Embedder(ABC):
    """Model-agnostic embedding provider."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @abstractmethod
    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents/summaries for indexing."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""


class LocalEmbedder(Embedder):
    """fastembed-based local ONNX models. No API key, works offline."""

    def __init__(self, cfg: EmbeddingConfig, cache_dir: Path | None = None):
        from fastembed import TextEmbedding

        self.cfg = cfg
        self.cache_dir = Path(cache_dir or default_model_cache_dir())
        self.model = TextEmbedding(
            model_name=cfg.model,
            cache_dir=str(self.cache_dir),
        )
        self._dim = cfg.dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(x) for x in v] for v in self.model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return [float(x) for x in next(self.model.query_embed(text))]


class HttpEmbedder(Embedder):
    """OpenAI-compatible /api/embeddings endpoint (OpenAI, Ollama, LiteLLM...)."""

    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg

    @property
    def dimension(self) -> int:
        return self.cfg.dimension

    def _call(self, texts: Sequence[str]) -> list[list[float]]:
        if not self.cfg.http_url:
            raise RuntimeError("embedding.http_url is not configured")
        headers = {"Content-Type": "application/json"}
        if self.cfg.http_api_key:
            headers["Authorization"] = f"Bearer {self.cfg.http_api_key}"
        payload = {"input": list(texts)}
        if self.cfg.http_model:
            payload["model"] = self.cfg.http_model
        with httpx.Client(timeout=self.cfg.http_timeout) as client:
            resp = client.post(self.cfg.http_url.rstrip("/") + "/embeddings", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        items = sorted(data["data"], key=lambda d: d.get("index", 0))
        return [list(i["embedding"]) for i in items]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return self._call(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._call([text])[0]


class NoopEmbedder(Embedder):
    """Lexical-only mode; dense search is disabled."""

    @property
    def dimension(self) -> int:
        return 0

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("embedding provider is 'none'")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("embedding provider is 'none'")


def create_embedder(cfg: EmbeddingConfig, cache_dir: Path | None = None) -> Embedder:
    if cfg.provider == "http":
        return HttpEmbedder(cfg)
    if cfg.provider == "none":
        return NoopEmbedder()
    return LocalEmbedder(cfg, cache_dir=cache_dir)
