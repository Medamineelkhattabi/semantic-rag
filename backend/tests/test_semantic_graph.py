"""Knowledge graph construction, canonicalisation and multi-hop traversal."""

from __future__ import annotations

import pytest

from app.semantic_rag.extraction import (
    DocumentExtraction,
    ExtractedEntity,
    ExtractedRelationship,
    canonical_id,
    fallback_extract,
)
from app.semantic_rag.graph import SemanticGraph
from app.semantic_rag.ontology import ONTOLOGY, RELATION_NAMES


# ----------------------------------------------------------------------
# Canonicalisation — this is what makes cross-document merging work.
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,entity_type,expected",
    [
        ("c-17", "Component", "C-17"),
        ("C-17", "Component", "C-17"),
        ("c-2048", "Contract", "C-2048"),
        ("r-17", "Risk", "R-17"),
        ("inc-1042", "Incident", "INC-1042"),
        ("PRJ-PHOENIX", "Project", "Project Phoenix"),
        ("Project Phoenix", "Project", "Project Phoenix"),
        ("phoenix", "Project", "Project Phoenix"),
        ("Alpha Precision Systems S.A.S.", "Supplier", "Alpha Precision Systems"),
        ("Borealis Components GmbH", "Supplier", "Borealis Components"),
        ("Cygnus Aerospace Limited", "Supplier", "Cygnus Aerospace"),
        ("Delta Microwave Inc", "Supplier", "Delta Microwave"),
        ("  Elena   Vasquez ", "Employee", "Elena Vasquez"),
        ("", "Component", ""),
    ],
)
def test_canonical_id(raw, entity_type, expected):
    assert canonical_id(raw, entity_type) == expected


def test_canonicalisation_merges_supplier_variants():
    """The same supplier written three ways must collapse to one node."""
    variants = [
        "Alpha Precision Systems",
        "Alpha Precision Systems S.A.S.",
        "alpha precision systems",
    ]
    canonical = {canonical_id(v, "Supplier").lower() for v in variants}
    assert len(canonical) == 1


# ----------------------------------------------------------------------
def test_ontology_is_self_consistent():
    for relation, source, target, _ in ONTOLOGY.relation_types:
        assert source in ONTOLOGY.entity_types, f"{relation} has unknown source {source}"
        assert target in ONTOLOGY.entity_types, f"{relation} has unknown target {target}"
    assert RELATION_NAMES == {r[0] for r in ONTOLOGY.relation_types}


def test_ontology_prompt_block_lists_everything():
    block = ONTOLOGY.prompt_block()
    for entity_type in ONTOLOGY.entity_types:
        assert entity_type in block
    for relation, *_ in ONTOLOGY.relation_types:
        assert relation in block


# ----------------------------------------------------------------------
def test_graph_build(sample_extractions):
    graph = SemanticGraph()
    stats = graph.build(sample_extractions)

    assert stats["entities"] > 0
    assert stats["relationships"] > 0
    assert graph.kg is not None
    # The framework dataclass must be populated, not just our own structures.
    assert len(graph.kg.entities) == len(graph.nodes)
    assert len(graph.kg.relationships) == len(graph.edges)


def test_graph_merges_entities_across_documents(sample_extractions):
    graph = SemanticGraph()
    graph.build(sample_extractions)

    # C-17 appears in two documents; it must be one node carrying both.
    node = graph.nodes["C-17"]
    assert set(node.doc_ids) == {"doc-projects", "doc-components"}


def test_graph_keeps_edge_provenance(sample_extractions):
    graph = SemanticGraph()
    graph.build(sample_extractions)

    edge = graph.edges[("C-17", "supplied_by", "Alpha Precision Systems")]
    assert edge.doc_ids == ["doc-components"]
    assert edge.evidence


def test_graph_drops_dangling_edges():
    extraction = DocumentExtraction(
        doc_id="d1",
        entities=[ExtractedEntity(id="A", name="A", type="Project", doc_id="d1")],
        relationships=[
            ExtractedRelationship(source="A", target="GHOST", type="uses", doc_id="d1")
        ],
        method="fixture",
        latency_ms=0.0,
    )
    graph = SemanticGraph()
    graph.build([extraction])
    assert graph.edges == {}


# ----------------------------------------------------------------------
# Multi-hop path finding — the core capability being demonstrated.
# ----------------------------------------------------------------------
def test_finds_the_four_hop_chain(sample_extractions):
    graph = SemanticGraph()
    graph.build(sample_extractions)

    path = graph.find_path("Project Phoenix", "R-17")
    assert path is not None

    nodes = [path[0].source] + [step.target for step in path]
    assert nodes == [
        "Project Phoenix",
        "C-17",
        "Alpha Precision Systems",
        "C-2048",
        "R-17",
    ]
    assert len(path) == 4


def test_path_steps_carry_relations_and_provenance(sample_extractions):
    graph = SemanticGraph()
    graph.build(sample_extractions)
    path = graph.find_path("Project Phoenix", "R-17")

    relations = [step.relation for step in path]
    assert relations == ["uses", "supplied_by", "governed_by", "has_risk"]
    assert all(step.doc_ids for step in path)
    # The chain must span more than one source document.
    assert len({doc for step in path for doc in step.doc_ids}) >= 4


def test_path_traverses_edges_backwards_when_needed(sample_extractions):
    """R-17 -> Phoenix runs against edge direction; it must still resolve."""
    graph = SemanticGraph()
    graph.build(sample_extractions)

    path = graph.find_path("R-17", "Project Phoenix")
    assert path is not None
    assert all(step.direction == "reverse" for step in path)
    assert path[0].phrase().startswith("R-17")


def test_no_path_between_disconnected_nodes(sample_extractions):
    graph = SemanticGraph()
    graph.build(sample_extractions)
    assert graph.find_path("Project Titan", "R-17") is None


def test_missing_node_returns_none(sample_extractions):
    graph = SemanticGraph()
    graph.build(sample_extractions)
    assert graph.find_path("Nope", "R-17") is None
    assert graph.find_path("R-17", "R-17") == []


# ----------------------------------------------------------------------
def test_neighbourhood_expansion_is_bounded(sample_extractions):
    graph = SemanticGraph()
    graph.build(sample_extractions)

    one_hop, _ = graph.neighbourhood(["Project Phoenix"], hops=1)
    assert one_hop == {"Project Phoenix", "C-17"}

    two_hop, _ = graph.neighbourhood(["Project Phoenix"], hops=2)
    assert "Alpha Precision Systems" in two_hop
    assert "C-2048" not in two_hop

    four_hop, edges = graph.neighbourhood(["Project Phoenix"], hops=4)
    assert {"C-2048", "R-17"}.issubset(four_hop)
    assert edges


def test_neighbourhood_ignores_unknown_seeds(sample_extractions):
    graph = SemanticGraph()
    graph.build(sample_extractions)
    nodes, edges = graph.neighbourhood(["does-not-exist"], hops=3)
    assert nodes == set()
    assert edges == []


# ----------------------------------------------------------------------
def test_payload_is_serialisable(sample_extractions):
    graph = SemanticGraph()
    graph.build(sample_extractions)
    payload = graph.to_payload()

    assert payload["nodes"] and payload["edges"]
    node_ids = {n["id"] for n in payload["nodes"]}
    for edge in payload["edges"]:
        assert edge["source"] in node_ids
        assert edge["target"] in node_ids
        assert edge["docIds"]


# ----------------------------------------------------------------------
def test_fallback_extractor_finds_the_chain():
    """The offline fallback must still recover the core supply-chain links."""
    from app.dataset.corpus import corpus_index

    index = corpus_index()
    relations = set()
    for doc_id in [
        "01-project-portfolio",
        "02-component-register",
        "03-supplier-directory",
        "04-contract-register",
    ]:
        _entities, rels = fallback_extract(index[doc_id])
        relations.update((r.source, r.type, r.target) for r in rels)

    assert ("Project Phoenix", "uses", "C-17") in relations
    assert ("C-17", "supplied_by", "Alpha Precision Systems") in relations
    assert ("Alpha Precision Systems", "governed_by", "C-2048") in relations
    assert ("C-2048", "has_risk", "R-17") in relations
