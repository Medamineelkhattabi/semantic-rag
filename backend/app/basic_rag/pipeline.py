"""Basic RAG: chunk -> embed -> vector search -> top-K -> LLM.

This is a deliberately *good* baseline, not a straw man:

* paragraph-aware chunking that avoids splitting mid-sentence,
* the same bge-m3 embedding model the semantic pipeline uses,
* exact inner-product search over L2-normalised vectors (true cosine ranking),
* the same LLM, temperature and answer prompt shape as the semantic pipeline.

The only thing it cannot do is follow a relationship it never saw in one chunk.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import settings
from ..dataset.corpus import Document, load_corpus
from ..embeddings import EmbeddingClient, get_embedder
from ..llm import LLMClient, get_llm

try:  # FAISS is the default index; numpy is a functional fallback.
    import faiss  # type: ignore

    _HAS_FAISS = True
except ImportError:  # pragma: no cover - exercised only without faiss
    faiss = None  # type: ignore
    _HAS_FAISS = False


# ----------------------------------------------------------------------
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    start: int
    end: int


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    score: float
    rank: int


@dataclass
class BasicRagResult:
    answer: str
    chunks: List[RetrievedChunk]
    context: str
    sources: List[str]
    latency: Dict[str, float]
    stats: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def chunk_document(doc: Document, size: int, overlap: int) -> List[Chunk]:
    """Split a document into overlapping, paragraph-aligned windows."""
    paragraphs = [p for p in _PARAGRAPH_SPLIT.split(doc.text) if p.strip()]

    chunks: List[Chunk] = []
    buffer: List[str] = []
    buffer_len = 0
    cursor = 0

    def flush() -> None:
        nonlocal buffer, buffer_len, cursor
        if not buffer:
            return
        text = "\n\n".join(buffer).strip()
        if text:
            start = doc.text.find(buffer[0], cursor)
            start = start if start != -1 else cursor
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::c{len(chunks):03d}",
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    text=text,
                    start=start,
                    end=start + len(text),
                )
            )
            cursor = start
        buffer = []
        buffer_len = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        # A single oversized paragraph is split on sentence boundaries.
        if len(paragraph) > size:
            flush()
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            current: List[str] = []
            current_len = 0
            for sentence in sentences:
                if current_len + len(sentence) > size and current:
                    buffer = ["  ".join(current)]
                    buffer_len = current_len
                    flush()
                    # Carry the tail of the window forward as overlap.
                    carry = " ".join(current)[-overlap:] if overlap else ""
                    current = [carry, sentence] if carry else [sentence]
                    current_len = len(carry) + len(sentence)
                else:
                    current.append(sentence)
                    current_len += len(sentence) + 1
            if current:
                buffer = [" ".join(current)]
                buffer_len = current_len
                flush()
            continue

        if buffer_len + len(paragraph) > size and buffer:
            tail = buffer[-1]
            flush()
            if overlap and len(tail) <= overlap * 2:
                buffer = [tail]
                buffer_len = len(tail)

        buffer.append(paragraph)
        buffer_len += len(paragraph) + 2

    flush()
    return chunks


# ----------------------------------------------------------------------
class VectorIndex:
    """Exact cosine search over normalised vectors (FAISS, numpy fallback)."""

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = vectors
        self.dim = int(vectors.shape[1]) if vectors.size else 0
        self.backend = "faiss" if _HAS_FAISS and self.dim else "numpy"

        if self.backend == "faiss":
            self._index = faiss.IndexFlatIP(self.dim)
            self._index.add(vectors)
        else:
            self._index = None

    def search(self, query: np.ndarray, top_k: int):
        query = query.reshape(1, -1).astype("float32")
        if self.backend == "faiss":
            scores, indices = self._index.search(query, top_k)
            return scores[0], indices[0]
        scores = (self.vectors @ query.T).ravel()
        indices = np.argsort(-scores)[:top_k]
        return scores[indices], indices


# ----------------------------------------------------------------------
ANSWER_SYSTEM_PROMPT = (
    "You are an enterprise knowledge assistant for Helios Aerospace. "
    "Answer strictly from the supplied context. "
    "If the context does not contain enough information to answer fully, say "
    "exactly which part is missing. Never invent identifiers, names or figures. "
    "Cite the source document ids you relied on in square brackets."
)


class BasicRagPipeline:
    """Classic dense-retrieval RAG."""

    name = "basic"

    def __init__(
        self,
        embedder: Optional[EmbeddingClient] = None,
        llm: Optional[LLMClient] = None,
    ) -> None:
        self.embedder = embedder or get_embedder()
        self.llm = llm or get_llm()
        self.chunks: List[Chunk] = []
        self.index: Optional[VectorIndex] = None
        self.build_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def build(self, documents: Optional[List[Document]] = None) -> Dict[str, Any]:
        documents = documents or load_corpus()
        started = time.perf_counter()

        self.chunks = []
        for doc in documents:
            self.chunks.extend(
                chunk_document(doc, settings.chunk_size, settings.chunk_overlap)
            )

        vectors = self.embedder.embed([c.text for c in self.chunks])
        self.index = VectorIndex(vectors)

        self.build_stats = {
            "documents": len(documents),
            "chunks": len(self.chunks),
            "avg_chunk_chars": round(
                sum(len(c.text) for c in self.chunks) / max(len(self.chunks), 1), 1
            ),
            "embedding_dim": self.index.dim,
            "index_backend": self.index.backend,
            "embedding_model": self.embedder.model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "build_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        return self.build_stats

    # ------------------------------------------------------------------
    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        if self.index is None:
            raise RuntimeError("BasicRagPipeline.build() must be called first")

        top_k = top_k or settings.top_k
        query_vector = self.embedder.embed_one(question)
        scores, indices = self.index.search(query_vector, min(top_k, len(self.chunks)))

        retrieved: List[RetrievedChunk] = []
        for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    doc_title=chunk.doc_title,
                    text=chunk.text,
                    score=float(score),
                    rank=rank,
                )
            )
        return retrieved

    # ------------------------------------------------------------------
    @staticmethod
    def build_context(chunks: List[RetrievedChunk]) -> str:
        blocks = []
        for chunk in chunks:
            blocks.append(
                f"[{chunk.doc_id}] (chunk {chunk.chunk_id}, similarity {chunk.score:.4f})\n"
                f"{chunk.text}"
            )
        return "\n\n---\n\n".join(blocks)

    # ------------------------------------------------------------------
    def answer(self, question: str, top_k: Optional[int] = None) -> BasicRagResult:
        total_started = time.perf_counter()

        retrieval_started = time.perf_counter()
        chunks = self.retrieve(question, top_k=top_k)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        context = self.build_context(chunks)

        prompt = (
            f"Context:\n\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the context above."
        )
        answer_text, meta = self.llm.chat(
            [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )

        total_ms = (time.perf_counter() - total_started) * 1000
        return BasicRagResult(
            answer=answer_text,
            chunks=chunks,
            context=context,
            sources=sorted({c.doc_id for c in chunks}),
            latency={
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": meta["latency_ms"],
                "total_ms": round(total_ms, 2),
            },
            stats={
                "chunks_retrieved": len(chunks),
                "context_chars": len(context),
                "top_k": top_k or settings.top_k,
                "index_backend": self.index.backend if self.index else None,
                "llm_model": meta.get("model"),
                "prompt_tokens": meta.get("prompt_tokens"),
                "completion_tokens": meta.get("completion_tokens"),
            },
        )
