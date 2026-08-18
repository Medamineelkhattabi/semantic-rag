"""Semantic RAG retrieval: entity linking, target typing, traversal, expansion."""

from __future__ import annotations

import pytest

from app.semantic_rag.pipeline import SemanticRagPipeline, split_sections


class StubExtractor:
    """Returns pre-built extractions so no LLM call is made."""

    model = "stub-extractor"

    def __init__(self, extractions):
        self._by_doc = {e.doc_id: e for e in extractions}

    def extract(self, doc, **_):
        return self._by_doc[doc.doc_id]


@pytest.fixture
def pipeline(fake_embedder, fake_llm, sample_documents, sample_extractions):
    pipe = SemanticRagPipeline(
        embedder=fake_embedder,
        llm=fake_llm,
        extractor=StubExtractor(sample_extractions),
    )
    pipe.build(sample_documents)
    return pipe


# ----------------------------------------------------------------------
def test_build_stats(pipeline):
    stats = pipeline.build_stats
    assert stats["entities"] > 0
    assert stats["relationships"] > 0
    assert "Project" in stats["entity_types"]
    assert "supplied_by" in stats["relation_types"]


# ----------------------------------------------------------------------
def test_split_sections(sample_documents):
    sections = split_sections(sample_documents[0])
    headings = [h for h, _ in sections]
    assert "PRJ-PHOENIX" in headings
    assert all(body.strip() for _, body in sections)


# ----------------------------------------------------------------------
# Entity linking
# ----------------------------------------------------------------------
def test_links_explicit_identifier(pipeline):
    seeds = pipeline.link_entities("What is risk R-17?")
    top = {s.id for s in seeds if s.method == "identifier"}
    assert "R-17" in top


def test_links_identifier_case_insensitively(pipeline):
    seeds = pipeline.link_entities("tell me about component c-17")
    assert any(s.id == "C-17" and s.method == "identifier" for s in seeds)


def test_links_entity_by_name(pipeline):
    seeds = pipeline.link_entities("Who supplies Project Phoenix?")
    assert any(s.id == "Project Phoenix" for s in seeds)


def test_links_project_by_bare_name(pipeline):
    seeds = pipeline.link_entities("What does Phoenix depend on?")
    assert any(s.id == "Project Phoenix" for s in seeds)


def test_linking_respects_seed_limit(pipeline):
    seeds = pipeline.link_entities("supplier contract risk component project", limit=2)
    assert len(seeds) <= 2


# ----------------------------------------------------------------------
# Target typing
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "question,expected",
    [
        ("Which supplier makes this?", "Supplier"),
        ("What contract governs it?", "Contract"),
        ("What risk is associated?", "Risk"),
        ("Which component is used?", "Component"),
        ("Which programme is affected?", "Project"),
        ("Which incident was raised?", "Incident"),
        ("Who owns this risk?", "Employee"),
    ],
)
def test_detects_target_type(question, expected):
    assert expected in SemanticRagPipeline.detect_target_types(question)


def test_detects_multiple_target_types():
    detected = SemanticRagPipeline.detect_target_types(
        "Which supplier, under what contract, carries which risk?"
    )
    assert {"Supplier", "Contract", "Risk"}.issubset(set(detected))


def test_no_false_target_type():
    assert SemanticRagPipeline.detect_target_types("What is the weather?") == []


# ----------------------------------------------------------------------
# Traversal
# ----------------------------------------------------------------------
def test_traversal_reaches_the_risk_from_the_project(pipeline):
    seeds = pipeline.link_entities("Which risk relates to Project Phoenix?")
    paths, reachable, edges = pipeline.traverse(seeds, ["Risk"], hops=4)

    assert paths, "expected at least one path"
    assert "R-17" in reachable
    assert edges

    chains = [[p[0].source] + [s.target for s in p] for p in paths]
    assert any(
        chain[:5]
        == ["Project Phoenix", "C-17", "Alpha Precision Systems", "C-2048", "R-17"]
        for chain in chains
    )


def test_traversal_without_seeds_is_empty(pipeline):
    paths, reachable, edges = pipeline.traverse([], ["Risk"], hops=3)
    assert paths == []
    assert reachable == set()
    assert edges == []


# ----------------------------------------------------------------------
# Context expansion
# ----------------------------------------------------------------------
def test_collect_passages_matches_entities(pipeline):
    passages = pipeline.collect_passages(
        {"R-17"}, {"doc-risks"}
    )
    assert passages
    assert any("Critical" in p.text for p in passages)
    assert all(p.matched_entities for p in passages)


def test_context_includes_chain_facts_and_evidence(pipeline):
    result = pipeline.answer(
        "Which supplier serves Project Phoenix, under what contract, and what risk?"
    )
    context = result.context

    assert "REASONING CHAINS" in context
    assert "VERIFIED RELATIONSHIPS" in context
    assert "SOURCE EVIDENCE" in context
    for token in ["C-17", "Alpha Precision Systems", "C-2048", "R-17"]:
        assert token in context, f"{token} missing from assembled context"


# ----------------------------------------------------------------------
# End-to-end result shape
# ----------------------------------------------------------------------
def test_answer_result_shape(pipeline):
    result = pipeline.answer("Which risk is linked to Project Phoenix?")

    assert result.answer
    assert result.seeds
    assert result.facts
    assert result.paths
    assert result.sources
    assert result.subgraph_nodes
    assert result.latency["total_ms"] >= 0
    for key in ("entity_linking_ms", "traversal_ms", "context_expansion_ms", "generation_ms"):
        assert key in result.latency
    assert result.stats["paths_found"] == len(result.paths)


def test_facts_on_path_are_flagged(pipeline):
    result = pipeline.answer(
        "Which supplier serves Project Phoenix, under what contract, and what risk?"
    )
    on_path = [f for f in result.facts if f.on_path]
    assert on_path
    assert result.stats["facts_on_path"] == len(on_path)
    # Path facts must sort ahead of the rest so they enter the prompt first.
    assert result.facts[0].on_path


def test_sources_are_real_documents(pipeline, sample_documents):
    result = pipeline.answer("Which risk is linked to Project Phoenix?")
    known = {d.doc_id for d in sample_documents}
    assert set(result.sources).issubset(known)
