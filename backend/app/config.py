"""Central configuration.

Every knob that affects a retrieval pipeline lives here so that the Basic and
Semantic pipelines can be shown to run under provably identical settings
(same LLM, same embedding model, same corpus).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # LLM provider (OpenAI-compatible gateway)
    # ------------------------------------------------------------------
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "not-used-by-this-gateway"

    #: Model used to synthesise the final answer. Shared by BOTH pipelines.
    llm_model: str = "qwen3-coder:30b"
    #: Model used for entity/relationship extraction (Semantic RAG ingest only).
    llm_extraction_model: str = "qwen3-coder:30b"
    #: Embedding model. Shared by BOTH pipelines.
    embedding_model: str = "bge-m3:latest"

    # ------------------------------------------------------------------
    # Embedding provider — configured independently of chat on purpose.
    #
    # Most free chat APIs (Groq, OpenRouter, Cerebras) serve no embedding model
    # at all, so pinning embeddings to the chat endpoint would make them
    # unusable. Set EMBEDDING_PROVIDER=local to run fastembed in-process with
    # no API key, or point EMBEDDING_BASE_URL at a different provider.
    # ------------------------------------------------------------------
    #: "api" (OpenAI-compatible endpoint) or "local" (fastembed, no key).
    embedding_provider: str = "api"
    #: Defaults to llm_base_url when blank.
    embedding_base_url: str = ""
    #: Defaults to llm_api_key when blank.
    embedding_api_key: str = ""
    #: Model used when embedding_provider="local".
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"

    llm_temperature: float = 0.0
    #: Enumeration answers ('which risks reach Titan?') run long once the
    #: context carries several chains. At 900 they were truncated mid-table,
    #: which scored as a retrieval failure when retrieval had actually worked.
    llm_max_tokens: int = 2000
    llm_timeout: float = 180.0

    #: Shared gateways reject bursts with 429/503 ("maximum pending requests
    #: exceeded"). A benchmark run issues dozens of calls back to back, so
    #: transient rejections are retried rather than failing the question.
    llm_max_retries: int = 5
    llm_retry_base_delay: float = 2.0

    # Cloudflare Access service-token headers, if the gateway sits behind one.
    cf_access_client_id: str = ""
    cf_access_client_secret: str = ""

    #: Sent as User-Agent on every gateway request.
    #: Some WAFs (Cloudflare Access rules, for one) reject the OpenAI SDK's
    #: default "OpenAI/Python x.y.z" agent outright with
    #: a 403, which surfaces as PermissionDeniedError and silently degrades
    #: extraction to the regex fallback. Overriding it keeps the SDK usable.
    llm_user_agent: str = "rag-intelligence-lab/1.0"

    # ------------------------------------------------------------------
    # Retrieval settings
    # ------------------------------------------------------------------
    chunk_size: int = 700          # characters
    chunk_overlap: int = 120       # characters
    top_k: int = 5                 # chunks retrieved by Basic RAG

    #: Max hops the graph walker expands from a seed entity.
    #: The headline supply-chain question is a 4-hop chain
    #: (Project -> Component -> Supplier -> Contract -> Risk), and asking who
    #: owns that risk is a 5th hop — so the default must be at least 5.
    graph_max_hops: int = 5
    #: Max seed entities linked from a single question.
    graph_max_seeds: int = 6
    #: Entity-linking similarity floor (cosine, 0..1).
    entity_link_threshold: float = 0.45

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    documents_dir: Path = BACKEND_ROOT / "data" / "documents"
    cache_dir: Path = BACKEND_ROOT / "cache"

    #: When true, entity/relationship extraction falls back to a deterministic
    #: rule-based extractor if the LLM is unreachable, so the demo still boots.
    allow_extraction_fallback: bool = True

    cors_origins: str = "*"

    @property
    def effective_embedding_base_url(self) -> str:
        return self.embedding_base_url or self.llm_base_url

    @property
    def effective_embedding_api_key(self) -> str:
        return self.embedding_api_key or self.llm_api_key

    @property
    def active_embedding_model(self) -> str:
        """The embedding model actually in use, for status reporting."""
        if self.embedding_provider.strip().lower() == "local":
            return f"local:{self.local_embedding_model}"
        return self.embedding_model

    @property
    def extra_headers(self) -> Dict[str, str]:
        """Headers appended to every gateway request."""
        headers: Dict[str, str] = {}
        if self.llm_user_agent:
            headers["User-Agent"] = self.llm_user_agent
        if self.cf_access_client_id:
            headers["CF-Access-Client-Id"] = self.cf_access_client_id
        if self.cf_access_client_secret:
            headers["CF-Access-Client-Secret"] = self.cf_access_client_secret
        return headers


settings = Settings()
settings.cache_dir.mkdir(parents=True, exist_ok=True)
