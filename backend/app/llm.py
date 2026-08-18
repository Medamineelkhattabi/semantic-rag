"""Shared LLM client.

Both pipelines answer through this exact object, so any observed quality
difference is attributable to the retrieved context and not to the generator.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .config import settings


class LLMError(RuntimeError):
    pass


#: Statuses worth retrying: rate limiting and transient gateway saturation.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class LLMClient:
    """Thin OpenAI-compatible chat client with usage accounting."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.api_key = api_key or settings.llm_api_key
        self.model = model or settings.llm_model
        self._client = httpx.Client(timeout=settings.llm_timeout)

    # ------------------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        headers.update(settings.extra_headers)
        return headers

    def _post_with_retry(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST with exponential backoff on transient gateway failures."""
        last_error: Optional[Exception] = None

        for attempt in range(settings.llm_max_retries + 1):
            try:
                response = self._client.post(
                    f"{self.base_url}{path}", headers=self._headers(), json=payload
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status not in RETRYABLE_STATUS or attempt == settings.llm_max_retries:
                    raise LLMError(
                        f"LLM gateway returned {status}: {exc.response.text[:400]}"
                    ) from exc
                last_error = exc
            except httpx.HTTPError as exc:
                if attempt == settings.llm_max_retries:
                    raise LLMError(f"LLM gateway unreachable: {exc}") from exc
                last_error = exc

            time.sleep(settings.llm_retry_base_delay * (2**attempt))

        raise LLMError(f"LLM gateway failed after retries: {last_error}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Run a chat completion.

        Returns ``(text, meta)`` where meta carries latency and token usage.
        """
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": settings.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or settings.llm_max_tokens,
        }

        started = time.perf_counter()
        data = self._post_with_retry("/chat/completions", payload)
        elapsed_ms = (time.perf_counter() - started) * 1000

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = (message.get("content") or "").strip()

        # Some reasoning-tuned models on OpenAI-compatible gateways return the
        # visible answer in a non-standard `reasoning` field and leave `content`
        # empty. Fall back to it rather than surfacing a blank answer.
        if not text:
            text = (message.get("reasoning") or "").strip()

        usage = data.get("usage") or {}
        meta = {
            "latency_ms": round(elapsed_ms, 2),
            "model": data.get("model", payload["model"]),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "finish_reason": choice.get("finish_reason"),
        }
        return text, meta

    # ------------------------------------------------------------------
    def complete_json(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        max_tokens: int = 3000,
    ) -> Any:
        """Chat completion whose response is parsed as JSON."""
        text, _meta = self.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a precise information extraction engine. "
                        "You reply with raw JSON only. No prose, no markdown fences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=model,
            max_tokens=max_tokens,
        )
        return extract_json(text)

    def health(self) -> Dict[str, Any]:
        try:
            response = self._client.get(
                f"{self.base_url}/models", headers=self._headers(), timeout=20.0
            )
            response.raise_for_status()
            models = [m.get("id") for m in response.json().get("data", [])]
            return {"reachable": True, "models": models}
        except Exception as exc:  # pragma: no cover - network dependent
            return {"reachable": False, "error": str(exc)}


# ----------------------------------------------------------------------
# JSON salvage helpers
# ----------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Parse JSON out of an LLM response, tolerating fences and stray prose."""
    if not text or not text.strip():
        raise LLMError("Empty response from LLM")

    candidate = text.strip()

    fenced = _FENCE_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost balanced object/array in the text.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        end = candidate.rfind(closer)
        if start != -1 and end > start:
            blob = candidate[start : end + 1]
            blob = re.sub(r",\s*([}\]])", r"\1", blob)  # trailing commas
            try:
                return json.loads(blob)
            except json.JSONDecodeError:
                continue

    raise LLMError(f"Could not parse JSON from LLM response: {text[:300]}")


_llm_client: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
