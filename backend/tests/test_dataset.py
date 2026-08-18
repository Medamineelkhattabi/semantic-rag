"""Corpus and ground-truth integrity.

The most valuable test here is ``test_keypoints_are_supported_by_relevant_docs``:
it proves the benchmark answers are actually derivable from the documents the
ground truth names, so a pipeline scoring badly is a retrieval failure and not a
broken fixture.
"""

from __future__ import annotations

import re

import pytest

from app.dataset.benchmark import BENCHMARK, BENCHMARK_BY_ID
from app.dataset.corpus import corpus_index, load_corpus


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def test_corpus_loads():
    documents = load_corpus()
    assert len(documents) >= 10
    assert all(doc.text.strip() for doc in documents)
    assert len({doc.doc_id for doc in documents}) == len(documents)


def test_documents_have_titles_and_metadata():
    for doc in load_corpus():
        assert doc.title
        assert doc.metadata.get("document_id"), f"{doc.doc_id} is missing a Document ID"


def test_benchmark_shape():
    assert 10 <= len(BENCHMARK) <= 20
    assert len(BENCHMARK_BY_ID) == len(BENCHMARK)
    for question in BENCHMARK:
        assert question.question.strip().endswith("?")
        assert question.keypoints, f"{question.id} has no keypoints"
        assert question.relevant_docs, f"{question.id} has no relevant docs"
        assert question.difficulty in {"easy", "medium", "hard"}
        assert question.hops >= 1


def test_benchmark_covers_easy_and_multihop():
    hops = [q.hops for q in BENCHMARK]
    assert any(h == 1 for h in hops), "need single-hop questions Basic RAG can win"
    assert any(h >= 4 for h in hops), "need deep multi-hop questions"


def test_relevant_docs_exist():
    known = set(corpus_index())
    for question in BENCHMARK:
        unknown = set(question.relevant_docs) - known
        assert not unknown, f"{question.id} references unknown documents: {unknown}"


@pytest.mark.parametrize("question", BENCHMARK, ids=lambda q: q.id)
def test_keypoints_are_supported_by_relevant_docs(question):
    """Every keypoint must appear in at least one of its relevant documents."""
    index = corpus_index()
    haystack = _normalise(
        " ".join(index[doc_id].text for doc_id in question.relevant_docs)
    )
    for forms in question.keypoints:
        assert any(_normalise(form) in haystack for form in forms), (
            f"{question.id}: none of {forms} appears in {question.relevant_docs}"
        )


@pytest.mark.parametrize(
    "question",
    [q for q in BENCHMARK if q.hops >= 3],
    ids=lambda q: q.id,
)
def test_multihop_questions_span_multiple_documents(question):
    """A multi-hop question must genuinely need more than one document."""
    assert len(question.relevant_docs) >= 3, (
        f"{question.id} claims {question.hops} hops but names "
        f"{len(question.relevant_docs)} document(s)"
    )


def test_chain_is_split_across_documents():
    """No single document may contain two consecutive links of the core chain.

    This is what forces multi-hop retrieval: if one chunk held
    'Phoenix uses C-17' *and* 'C-17 supplied by Alpha', plain vector search
    would answer the demo question without any graph.
    """
    index = corpus_index()

    def mentions(doc_id: str, *needles: str) -> bool:
        text = _normalise(index[doc_id].text)
        return all(_normalise(n) in text for n in needles)

    # The portfolio knows Phoenix->C-17 but must not name the supplier.
    assert mentions("01-project-portfolio", "project phoenix", "C-17")
    assert not mentions("01-project-portfolio", "alpha precision systems")

    # The component register knows C-17->Alpha but must not name the project.
    assert mentions("02-component-register", "C-17", "alpha precision systems")
    assert not mentions("02-component-register", "project phoenix")

    # The supplier directory knows Alpha->C-2048 but must not name the risk.
    assert mentions("03-supplier-directory", "alpha precision systems", "C-2048")
    assert not mentions("03-supplier-directory", "R-17")

    # The contract register knows C-2048->R-17 but must not name the supplier.
    assert mentions("04-contract-register", "C-2048", "R-17")
    assert not mentions("04-contract-register", "alpha precision systems")
