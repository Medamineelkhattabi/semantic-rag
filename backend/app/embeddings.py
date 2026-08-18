"""Shared embedding client.

Both pipelines embed through the same object. Basic RAG uses it for chunk
vectors; Semantic RAG uses it for question-to-entity linking. Using one model
for both keeps the comparison honest.

Two backends are supported, chosen with ``EMBEDDING_PROVIDER``:

* ``api``   — any OpenAI-compatible ``/v1/embeddings`` endpoint (default).
* ``local`` — fastembed, running in-process with no API key at all.

The split matters in practice: most free chat APIs (Groq, OpenRouter, Cerebras)
serve **no** embedding model, so the embedding backend is configured
independently of the chat one and can stay local while chat is remote.

Vectors are cached on disk keyed by (model, text) so repeated benchmark runs do
not re-pay the cost.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import httpx
import numpy as np

from .config import settings
from .llm import RETRYABLE_STATUS


class EmbeddingError(RuntimeError):
    pass


# ----------------------------------------------------------------------
class BaseEmbeddingClient:
    """Disk-cached, L2-normalising embedding client.

    Subclasses implement ``_fetch`` for one batch of uncached texts.
    """

    model: str = "unset"

    def __init__(self, cache_path: Optional[Path] = None) -> None:
        self._cache_path = cache_path or (settings.cache_dir / "embeddings.jsonl")
        self._cache: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        self._load_cache()

    # ------------------------------------------------------------------
    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}::{text}".encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        if not self._cache_path.exists():
            return
        try:
            with self._cache_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        self._cache[record["k"]] = record["v"]
                    except (json.JSONDecodeError, KeyError):
                        continue
        except OSError:
            pass

    def _append_cache(self, key: str, vector: List[float]) -> None:
        try:
            with self._cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"k": key, "v": vector}) + "\n")
        except OSError:  # pragma: no cover - cache is best effort
            pass

    # ------------------------------------------------------------------
    def _fetch(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed(self, texts: Sequence[str], *, batch_size: int = 16) -> np.ndarray:
        """Embed texts, returning an L2-normalised ``(n, dim)`` float32 matrix."""
        if not texts:
            return np.zeros((0, 0), dtype="float32")

        texts = list(texts)
        pending: List[int] = []
        with self._lock:
            for i, text in enumerate(texts):
                if self._key(text) not in self._cache:
                    pending.append(i)

        for start in range(0, len(pending), batch_size):
            batch = [texts[i] for i in pending[start : start + batch_size]]
            vectors = self._fetch(batch)
            with self._lock:
                for text, vector in zip(batch, vectors):
                    key = self._key(text)
                    self._cache[key] = vector
                    self._append_cache(key, vector)

        with self._lock:
            matrix = np.array([self._cache[self._key(t)] for t in texts], dtype="float32")

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


# ----------------------------------------------------------------------
class ApiEmbeddingClient(BaseEmbeddingClient):
    """OpenAI-compatible ``/v1/embeddings`` backend."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        cache_path: Optional[Path] = None,
    ) -> None:
        self.base_url = (base_url or settings.effective_embedding_base_url).rstrip("/")
        self.api_key = api_key or settings.effective_embedding_api_key
        self.model = model or settings.embedding_model
        self._client = httpx.Client(timeout=settings.llm_timeout)
        super().__init__(cache_path)

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        headers.update(settings.extra_headers)
        return headers

    def _fetch(self, texts: List[str]) -> List[List[float]]:
        payload = {"model": self.model, "input": texts}
        data: Optional[Dict] = None
        last_error: Optional[Exception] = None

        # Same transient-saturation handling as the chat client.
        for attempt in range(settings.llm_max_retries + 1):
            try:
                response = self._client.post(
                    f"{self.base_url}/embeddings", headers=self._headers(), json=payload
                )
                response.raise_for_status()
                data = response.json()
                break
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status not in RETRYABLE_STATUS or attempt == settings.llm_max_retries:
                    raise EmbeddingError(
                        f"Embedding endpoint returned {status}: {exc.response.text[:300]}"
                    ) from exc
                last_error = exc
            except httpx.HTTPError as exc:
                if attempt == settings.llm_max_retries:
                    raise EmbeddingError(f"Embedding endpoint unreachable: {exc}") from exc
                last_error = exc

            time.sleep(settings.llm_retry_base_delay * (2**attempt))

        if data is None:  # pragma: no cover - defensive
            raise EmbeddingError(f"Embedding endpoint failed after retries: {last_error}")

        rows = sorted(data.get("data", []), key=lambda r: r.get("index", 0))
        if len(rows) != len(texts):
            raise EmbeddingError(
                f"Embedding count mismatch: sent {len(texts)}, received {len(rows)}"
            )
        return [row["embedding"] for row in rows]


# ----------------------------------------------------------------------
class LocalEmbeddingClient(BaseEmbeddingClient):
    """In-process fastembed backend — no API key, no network at query time.

    The ONNX weights are downloaded once on first use and cached by fastembed.
    """

    def __init__(
        self, model: Optional[str] = None, cache_path: Optional[Path] = None
    ) -> None:
        self.model = f"local:{model or settings.local_embedding_model}"
        self._model_name = model or settings.local_embedding_model
        self._embedder = None
        super().__init__(cache_path)

    def _ensure_model(self):
        if self._embedder is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:  # pragma: no cover
                raise EmbeddingError(
                    "EMBEDDING_PROVIDER=local needs fastembed. "
                    "It ships with semantica; reinstall requirements.txt."
                ) from exc
            try:
                self._embedder = TextEmbedding(self._model_name)
            except Exception as exc:
                raise EmbeddingError(
                    f"Could not load local embedding model {self._model_name!r}: {exc}"
                ) from exc
        return self._embedder

    def _fetch(self, texts: List[str]) -> List[List[float]]:
        embedder = self._ensure_model()
        try:
            return [vector.tolist() for vector in embedder.embed(texts)]
        except Exception as exc:  # pragma: no cover - runtime dependent
            raise EmbeddingError(f"Local embedding failed: {exc}") from exc


# ----------------------------------------------------------------------
_embedding_client: Optional[BaseEmbeddingClient] = None


def build_embedder() -> BaseEmbeddingClient:
    if settings.embedding_provider.strip().lower() == "local":
        return LocalEmbeddingClient()
    return ApiEmbeddingClient()


def get_embedder() -> BaseEmbeddingClient:
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = build_embedder()
    return _embedding_client


def reset_embedder() -> None:
    """Drop the cached singleton (used by tests)."""
    global _embedding_client
    _embedding_client = None


# Backwards-compatible alias: the API backend was the only one originally.
EmbeddingClient = ApiEmbeddingClient
