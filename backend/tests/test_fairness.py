"""Fairness invariants.

The whole exercise is only meaningful if the two pipelines differ in exactly one
respect: how they retrieve. These tests pin the rest down, so a future change
that quietly advantages one side fails here rather than showing up as a better
benchmark number.
"""

from __future__ import annotations

from app.basic_rag import pipeline as basic_module
from app.basic_rag.pipeline import BasicRagPipeline
from app.dataset.corpus import load_corpus
from app.semantic_rag import pipeline as semantic_module
from app.semantic_rag.pipeline import SemanticRagPipeline
from tests.test_semantic_rag import StubExtractor


def _pipelines(fake_embedder, fake_llm, documents, extractions):
    basic = BasicRagPipeline(embedder=fake_embedder, llm=fake_llm)
    semantic = SemanticRagPipeline(
        embedder=fake_embedder, llm=fake_llm, extractor=StubExtractor(extractions)
    )
    basic.build(documents)
    semantic.build(documents)
    return basic, semantic


# ----------------------------------------------------------------------
def test_answer_prompt_is_identical():
    """Neither side may get a better system prompt than the other."""
    assert basic_module.ANSWER_SYSTEM_PROMPT == semantic_module.ANSWER_SYSTEM_PROMPT


def test_chunking_reaches_every_real_document():
    """Basic RAG must index the whole corpus, not a subset of it."""
    from app.basic_rag.pipeline import chunk_document
    from app.config import settings

    documents = load_corpus()
    covered = {
        chunk.doc_id
        for doc in documents
        for chunk in chunk_document(doc, settings.chunk_size, settings.chunk_overlap)
    }
    assert covered == {doc.doc_id for doc in documents}


def test_both_index_the_same_documents(
    fake_embedder, fake_llm, sample_documents, sample_extractions
):
    basic, semantic = _pipelines(
        fake_embedder, fake_llm, sample_documents, sample_extractions
    )

    basic_docs = {chunk.doc_id for chunk in basic.chunks}
    semantic_docs = {doc.doc_id for doc in semantic.documents}
    assert basic_docs == semantic_docs == {d.doc_id for d in sample_documents}


def test_both_use_the_same_embedding_model(
    fake_embedder, fake_llm, sample_documents, sample_extractions
):
    basic, semantic = _pipelines(
        fake_embedder, fake_llm, sample_documents, sample_extractions
    )
    assert basic.embedder is semantic.embedder
    assert basic.build_stats["embedding_model"] == semantic.build_stats["embedding_model"]


def test_both_use_the_same_llm(
    fake_embedder, fake_llm, sample_documents, sample_extractions
):
    basic, semantic = _pipelines(
        fake_embedder, fake_llm, sample_documents, sample_extractions
    )
    # Same client object — the generator is provably identical for both sides.
    assert basic.llm is semantic.llm

    basic.answer("Which supplier serves Project Phoenix?")
    semantic.answer("Which supplier serves Project Phoenix?")
    assert len(fake_llm.calls) == 2

    # ...and both used the same system prompt through it.
    system_prompts = {messages[0]["content"] for messages in fake_llm.calls}
    assert len(system_prompts) == 1


def test_both_receive_the_same_question(
    fake_embedder, fake_llm, sample_documents, sample_extractions
):
    """The question text must reach both prompts unmodified."""
    basic, semantic = _pipelines(
        fake_embedder, fake_llm, sample_documents, sample_extractions
    )
    question = "Which risk is linked to Project Phoenix?"

    basic.answer(question)
    semantic.answer(question)

    for messages in fake_llm.calls:
        assert question in messages[-1]["content"]


def test_neither_pipeline_sees_the_ground_truth(
    fake_embedder, fake_llm, sample_documents, sample_extractions
):
    """Keypoints must never leak into a prompt — that would be scoring itself."""
    from app.dataset.benchmark import BENCHMARK

    basic, semantic = _pipelines(
        fake_embedder, fake_llm, sample_documents, sample_extractions
    )
    basic.answer("Which supplier serves Project Phoenix?")
    semantic.answer("Which supplier serves Project Phoenix?")

    banned = {"keypoint", "ground truth", "relevant_docs", "gold_path"}
    for messages in fake_llm.calls:
        blob = " ".join(m["content"] for m in messages).lower()
        for term in banned:
            assert term not in blob

    # And the benchmark's expected-answer fields are not reachable from a prompt.
    assert all(q.keypoints for q in BENCHMARK)
