"""Basic RAG chunking, indexing and retrieval."""

from __future__ import annotations

import numpy as np
import pytest

from app.basic_rag.pipeline import (
    BasicRagPipeline,
    VectorIndex,
    chunk_document,
)
from app.dataset.corpus import Document, load_corpus


def test_chunking_respects_size_budget():
    doc = load_corpus()[0]
    chunks = chunk_document(doc, size=700, overlap=120)
    assert chunks
    # Paragraph-aligned chunking may overshoot slightly on a long paragraph,
    # but never unboundedly.
    assert all(len(c.text) <= 700 * 2 for c in chunks)


def test_chunking_preserves_all_documents():
    for doc in load_corpus():
        chunks = chunk_document(doc, size=700, overlap=120)
        assert chunks, f"{doc.doc_id} produced no chunks"
        assert all(c.doc_id == doc.doc_id for c in chunks)
        assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunking_retains_key_facts():
    """Chunking must not destroy the facts the benchmark depends on."""
    index = {d.doc_id: d for d in load_corpus()}
    doc = index["02-component-register"]
    joined = " ".join(c.text for c in chunk_document(doc, size=700, overlap=120))
    assert "C-17" in joined
    assert "Alpha Precision Systems" in joined


def test_chunking_handles_oversized_paragraph():
    doc = Document(
        doc_id="big",
        title="Big",
        path="big.md",
        text="# Big\n\n" + ("This is a sentence that repeats. " * 200),
    )
    chunks = chunk_document(doc, size=300, overlap=50)
    assert len(chunks) > 1
    assert all(c.text.strip() for c in chunks)


def test_chunking_empty_document():
    doc = Document(doc_id="empty", title="Empty", path="e.md", text="")
    assert chunk_document(doc, size=500, overlap=50) == []


# ----------------------------------------------------------------------
def test_vector_index_ranks_by_cosine():
    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype="float32"
    )
    index = VectorIndex(vectors)
    scores, indices = index.search(np.array([1.0, 0.0], dtype="float32"), top_k=3)

    assert indices[0] == 0
    assert scores[0] == pytest.approx(1.0, abs=1e-4)
    # Scores must be non-increasing.
    assert list(scores) == sorted(scores, reverse=True)


def test_vector_index_reports_backend():
    index = VectorIndex(np.eye(4, dtype="float32"))
    assert index.backend in {"faiss", "numpy"}
    assert index.dim == 4


# ----------------------------------------------------------------------
def test_pipeline_build_and_retrieve(fake_embedder, fake_llm, sample_documents):
    pipeline = BasicRagPipeline(embedder=fake_embedder, llm=fake_llm)
    stats = pipeline.build(sample_documents)

    assert stats["documents"] == len(sample_documents)
    assert stats["chunks"] > 0
    assert stats["embedding_model"] == "fake-embedder"

    retrieved = pipeline.retrieve("Which supplier provides component C-17?", top_k=3)
    assert len(retrieved) == 3
    assert [c.rank for c in retrieved] == [1, 2, 3]
    assert all(-1.001 <= c.score <= 1.001 for c in retrieved)


def test_pipeline_answer_shape(fake_embedder, fake_llm, sample_documents):
    pipeline = BasicRagPipeline(embedder=fake_embedder, llm=fake_llm)
    pipeline.build(sample_documents)

    result = pipeline.answer("Which supplier provides component C-17?", top_k=2)

    assert result.answer
    assert len(result.chunks) == 2
    assert result.sources
    assert set(result.sources).issubset({d.doc_id for d in sample_documents})
    assert result.context
    assert result.latency["total_ms"] >= 0
    assert "retrieval_ms" in result.latency
    assert result.stats["chunks_retrieved"] == 2


def test_context_carries_provenance(fake_embedder, fake_llm, sample_documents):
    pipeline = BasicRagPipeline(embedder=fake_embedder, llm=fake_llm)
    pipeline.build(sample_documents)
    result = pipeline.answer("component C-17", top_k=2)

    for chunk in result.chunks:
        assert f"[{chunk.doc_id}]" in result.context


def test_retrieve_before_build_raises(fake_embedder, fake_llm):
    pipeline = BasicRagPipeline(embedder=fake_embedder, llm=fake_llm)
    with pytest.raises(RuntimeError):
        pipeline.retrieve("anything")
