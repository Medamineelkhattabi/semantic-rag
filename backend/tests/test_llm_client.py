"""LLM client: JSON salvage, response quirks and transient-failure retries."""

from __future__ import annotations

import httpx
import pytest

from app.llm import RETRYABLE_STATUS, LLMClient, LLMError, extract_json


# ----------------------------------------------------------------------
# JSON salvage
# ----------------------------------------------------------------------
def test_extract_plain_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_fenced_json():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('```\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_with_surrounding_prose():
    assert extract_json('Sure! Here you go:\n{"a": 1}\nHope that helps.') == {"a": 1}


def test_extract_json_array():
    assert extract_json('[{"a": 1}]') == [{"a": 1}]


def test_extract_json_tolerates_trailing_commas():
    assert extract_json('prefix {"a": 1, "b": [2, 3,],} suffix') == {"a": 1, "b": [2, 3]}


def test_extract_json_rejects_empty():
    with pytest.raises(LLMError):
        extract_json("")


def test_extract_json_rejects_unparseable():
    with pytest.raises(LLMError):
        extract_json("no json anywhere here")


# ----------------------------------------------------------------------
# Transport behaviour
# ----------------------------------------------------------------------
def _client(handler: httpx.MockTransport) -> LLMClient:
    client = LLMClient()
    client._client = httpx.Client(transport=handler)
    return client


def _completion(content: str = "hello", reasoning: str | None = None):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning"] = reasoning
    return {
        "model": "test-model",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


def test_chat_returns_text_and_meta():
    transport = httpx.MockTransport(lambda _r: httpx.Response(200, json=_completion("hi")))
    text, meta = _client(transport).chat([{"role": "user", "content": "x"}])

    assert text == "hi"
    assert meta["model"] == "test-model"
    assert meta["prompt_tokens"] == 5
    assert meta["latency_ms"] >= 0


def test_chat_falls_back_to_reasoning_field():
    """Some reasoning models leave `content` empty and answer in `reasoning`."""
    transport = httpx.MockTransport(
        lambda _r: httpx.Response(200, json=_completion("", reasoning="the answer"))
    )
    text, _meta = _client(transport).chat([{"role": "user", "content": "x"}])
    assert text == "the answer"


def test_chat_retries_transient_failure_then_succeeds(monkeypatch):
    from app import llm as llm_module

    monkeypatch.setattr(llm_module.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="server busy, maximum pending requests exceeded")
        return httpx.Response(200, json=_completion("recovered"))

    text, _meta = _client(httpx.MockTransport(handler)).chat([{"role": "user", "content": "x"}])

    assert text == "recovered"
    assert calls["n"] == 3


def test_chat_gives_up_after_max_retries(monkeypatch):
    from app import llm as llm_module

    monkeypatch.setattr(llm_module.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="always busy")

    with pytest.raises(LLMError, match="503"):
        _client(httpx.MockTransport(handler)).chat([{"role": "user", "content": "x"}])

    assert calls["n"] == llm_module.settings.llm_max_retries + 1


def test_chat_does_not_retry_client_error(monkeypatch):
    """A 400 is our bug, not the gateway's — fail fast."""
    from app import llm as llm_module

    monkeypatch.setattr(llm_module.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    with pytest.raises(LLMError, match="400"):
        _client(httpx.MockTransport(handler)).chat([{"role": "user", "content": "x"}])

    assert calls["n"] == 1


def test_retryable_statuses_cover_saturation_codes():
    assert {429, 503, 502, 504}.issubset(RETRYABLE_STATUS)
    assert 400 not in RETRYABLE_STATUS
    assert 403 not in RETRYABLE_STATUS


def test_request_carries_gateway_headers():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, json=_completion())

    _client(httpx.MockTransport(handler)).chat([{"role": "user", "content": "x"}])

    # The gateway's WAF rejects the OpenAI SDK's default agent, so ours must be sent.
    assert seen.get("user-agent") == "rag-intelligence-lab/1.0"
    assert "authorization" in seen
