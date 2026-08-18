"""Embedding backends: caching, normalisation, provider selection."""

from __future__ import annotations

import httpx
import numpy as np
import pytest

from app.config import settings
from app.embeddings import (
    ApiEmbeddingClient,
    BaseEmbeddingClient,
    EmbeddingError,
    LocalEmbeddingClient,
    build_embedder,
)


class FakeBackend(BaseEmbeddingClient):
    """Counts fetches so cache behaviour is observable."""

    model = "fake-model"

    def __init__(self, cache_path):
        self.fetches: list[list[str]] = []
        super().__init__(cache_path)

    def _fetch(self, texts):
        self.fetches.append(list(texts))
        return [[float(len(t)), 1.0, 0.0] for t in texts]


@pytest.fixture
def cache_path(tmp_path):
    return tmp_path / "emb.jsonl"


# ----------------------------------------------------------------------
def test_returns_normalised_matrix(cache_path):
    client = FakeBackend(cache_path)
    matrix = client.embed(["abc", "de"])

    assert matrix.shape == (2, 3)
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0)


def test_empty_input_short_circuits(cache_path):
    client = FakeBackend(cache_path)
    assert client.embed([]).shape == (0, 0)
    assert client.fetches == []


def test_repeat_calls_hit_the_cache(cache_path):
    client = FakeBackend(cache_path)
    client.embed(["alpha", "beta"])
    client.embed(["alpha", "beta"])

    assert len(client.fetches) == 1  # second call served entirely from cache


def test_only_uncached_texts_are_fetched(cache_path):
    client = FakeBackend(cache_path)
    client.embed(["alpha"])
    client.embed(["alpha", "beta"])

    assert client.fetches == [["alpha"], ["beta"]]


def test_cache_persists_across_instances(cache_path):
    first = FakeBackend(cache_path)
    first.embed(["alpha"])

    second = FakeBackend(cache_path)
    second.embed(["alpha"])
    assert second.fetches == []          # loaded from disk


def test_cache_key_is_model_scoped(cache_path):
    """Switching embedding model must not reuse the previous model's vectors."""
    first = FakeBackend(cache_path)
    first.embed(["alpha"])

    second = FakeBackend(cache_path)
    second.model = "different-model"
    second.embed(["alpha"])
    assert second.fetches == [["alpha"]]


def test_embed_one_returns_a_vector(cache_path):
    assert FakeBackend(cache_path).embed_one("alpha").shape == (3,)


def test_zero_vector_does_not_divide_by_zero(cache_path):
    class ZeroBackend(FakeBackend):
        def _fetch(self, texts):
            return [[0.0, 0.0, 0.0] for _ in texts]

    matrix = ZeroBackend(cache_path).embed(["x"])
    assert not np.isnan(matrix).any()


# ----------------------------------------------------------------------
# Provider selection
# ----------------------------------------------------------------------
def test_build_embedder_defaults_to_api(monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "api")
    assert isinstance(build_embedder(), ApiEmbeddingClient)


def test_build_embedder_selects_local(monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "local")
    client = build_embedder()
    assert isinstance(client, LocalEmbeddingClient)
    # Model name is namespaced so local and API vectors never share a cache key.
    assert client.model.startswith("local:")


def test_provider_selection_is_case_insensitive(monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "LOCAL")
    assert isinstance(build_embedder(), LocalEmbeddingClient)


def test_active_embedding_model_reflects_provider(monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "local")
    monkeypatch.setattr(settings, "local_embedding_model", "some/model")
    assert settings.active_embedding_model == "local:some/model"

    monkeypatch.setattr(settings, "embedding_provider", "api")
    monkeypatch.setattr(settings, "embedding_model", "bge-m3:latest")
    assert settings.active_embedding_model == "bge-m3:latest"


def test_embedding_endpoint_falls_back_to_chat_endpoint(monkeypatch):
    monkeypatch.setattr(settings, "embedding_base_url", "")
    monkeypatch.setattr(settings, "llm_base_url", "https://chat.example/v1")
    assert settings.effective_embedding_base_url == "https://chat.example/v1"


def test_embedding_endpoint_can_differ_from_chat(monkeypatch):
    """Chat on Groq, embeddings elsewhere — the whole point of the split."""
    monkeypatch.setattr(settings, "llm_base_url", "https://api.groq.com/openai/v1")
    monkeypatch.setattr(settings, "embedding_base_url", "https://emb.example/v1")
    monkeypatch.setattr(settings, "embedding_api_key", "emb-key")

    assert settings.effective_embedding_base_url == "https://emb.example/v1"
    assert settings.effective_embedding_api_key == "emb-key"


# ----------------------------------------------------------------------
# API backend transport
# ----------------------------------------------------------------------
def _api_client(handler, cache_path):
    client = ApiEmbeddingClient(cache_path=cache_path)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def _payload(n: int):
    return {"data": [{"index": i, "embedding": [1.0, 0.0]} for i in range(n)]}


def test_api_backend_orders_by_index(cache_path):
    def handler(_request):
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 2.0]},
                    {"index": 0, "embedding": [3.0, 0.0]},
                ]
            },
        )

    matrix = _api_client(handler, cache_path).embed(["first", "second"])
    assert matrix[0].tolist() == [1.0, 0.0]   # index 0 vector, normalised
    assert matrix[1].tolist() == [0.0, 1.0]


def test_api_backend_detects_count_mismatch(cache_path):
    with pytest.raises(EmbeddingError, match="mismatch"):
        _api_client(lambda _r: httpx.Response(200, json=_payload(1)), cache_path).embed(
            ["a", "b"]
        )


def test_api_backend_retries_then_succeeds(cache_path, monkeypatch):
    from app import embeddings as module

    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def handler(_request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="server busy")
        return httpx.Response(200, json=_payload(1))

    _api_client(handler, cache_path).embed(["a"])
    assert calls["n"] == 3


def test_api_backend_does_not_retry_auth_failure(cache_path, monkeypatch):
    from app import embeddings as module

    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def handler(_request):
        calls["n"] += 1
        return httpx.Response(401, text="bad key")

    with pytest.raises(EmbeddingError, match="401"):
        _api_client(handler, cache_path).embed(["a"])
    assert calls["n"] == 1
