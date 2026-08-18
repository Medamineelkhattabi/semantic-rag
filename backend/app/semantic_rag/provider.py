"""Semantica provider adapter for the Helios LLM gateway.

Semantica 0.6.5 ships ``semantic_extract.OpenAIProvider``, which already accepts
a custom ``base_url`` for OpenAI-compatible endpoints. What it does not do is
forward custom auth headers: ``_init_client()`` passes only ``api_key`` and
``base_url`` to the OpenAI SDK.

Our gateway sits behind Cloudflare Access, so it needs
``CF-Access-Client-Id`` / ``CF-Access-Client-Secret`` on every request. We
therefore subclass the real provider, override only client construction to add
``default_headers``, and register the result through Semantica's own
``ProviderRegistry`` so that ``create_provider()`` resolves it like any
first-class provider.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from semantica.semantic_extract import (  # type: ignore
    OpenAIProvider,
    ProviderRegistry,
    create_provider,
)

from ..config import settings

PROVIDER_NAME = "helios_gateway"


class HeliosGatewayProvider(OpenAIProvider):
    """OpenAI-compatible provider that carries Cloudflare Access headers."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        self._default_headers = default_headers or settings.extra_headers
        super().__init__(
            api_key=api_key or settings.llm_api_key,
            model=model or settings.llm_extraction_model,
            base_url=base_url or settings.llm_base_url,
            **kwargs,
        )

    def _init_client(self) -> None:
        """Build the OpenAI client with gateway headers attached.

        Mirrors ``OpenAIProvider._init_client`` including its http/https scheme
        guard, adding only ``default_headers`` and ``timeout``.
        """
        if self.base_url:
            from urllib.parse import urlparse

            scheme = urlparse(self.base_url).scheme
            if scheme not in ("http", "https"):
                raise ValueError(
                    f"base_url must use http or https, got scheme {scheme!r}."
                )

        try:
            from openai import OpenAI

            init_kwargs: Dict[str, Any] = {
                "api_key": self.api_key or "unused",
                "timeout": settings.llm_timeout,
                # Extraction issues one call per document back to back; shared
                # gateways answer bursts with 429/503. The SDK retries those
                # with backoff natively.
                "max_retries": settings.llm_max_retries,
            }
            if self.base_url:
                init_kwargs["base_url"] = self.base_url
            if self._default_headers:
                init_kwargs["default_headers"] = dict(self._default_headers)

            self.client = OpenAI(**init_kwargs)
        except (ImportError, OSError) as exc:  # pragma: no cover
            self.client = None
            self.logger.warning("Could not initialise OpenAI client: %s", exc)


def register_provider() -> str:
    """Register the gateway provider with Semantica. Idempotent."""
    if ProviderRegistry.get(PROVIDER_NAME) is None:
        ProviderRegistry.register(PROVIDER_NAME, HeliosGatewayProvider)
    return PROVIDER_NAME


def get_semantica_provider(model: Optional[str] = None):
    """Return a Semantica provider instance bound to the gateway."""
    register_provider()
    return create_provider(
        PROVIDER_NAME,
        use_pool=False,
        api_key=settings.llm_api_key,
        model=model or settings.llm_extraction_model,
        base_url=settings.llm_base_url,
        default_headers=settings.extra_headers,
    )
