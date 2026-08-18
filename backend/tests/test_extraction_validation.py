"""Ontology enforcement during extraction.

The single most damaging extraction error for this demo is a *shortcut edge*:
a model that reads "Phoenix is exposed to R-17" and emits
``Project --has_risk--> Risk`` collapses a genuine four-hop supply chain into
one hop, and the comparison silently stops demonstrating anything.

These tests pin that behaviour down.
"""

from __future__ import annotations

import pytest

from app.dataset.corpus import Document
from app.semantic_rag.extraction import SemanticExtractor
from app.semantic_rag.ontology import RELATION_SIGNATURES, RELATION_TYPES


@pytest.fixture
def doc() -> Document:
    return Document(doc_id="d1", title="T", path="d1.md", text="body")


@pytest.fixture
def extractor() -> SemanticExtractor:
    # No provider is constructed: _normalise is exercised directly.
    return SemanticExtractor(provider=object())


def _payload(entities, relationships):
    return {"entities": entities, "relationships": relationships}


PHOENIX = {"id": "Project Phoenix", "name": "Project Phoenix", "type": "Project"}
C17 = {"id": "C-17", "name": "C-17", "type": "Component"}
ALPHA = {"id": "Alpha Precision Systems", "name": "Alpha Precision Systems", "type": "Supplier"}
C2048 = {"id": "C-2048", "name": "C-2048", "type": "Contract"}
R17 = {"id": "R-17", "name": "R-17", "type": "Risk"}


# ----------------------------------------------------------------------
def test_signatures_cover_every_relation():
    assert set(RELATION_SIGNATURES) == {r[0] for r in RELATION_TYPES}


def test_valid_relationship_is_kept(extractor, doc):
    entities, relationships = extractor._normalise(
        _payload([PHOENIX, C17], [{"source": "Project Phoenix", "target": "C-17", "type": "uses"}]),
        doc,
    )
    assert len(entities) == 2
    assert len(relationships) == 1
    assert relationships[0].type == "uses"
    assert extractor._rejected == []


def test_shortcut_edge_is_rejected(extractor, doc):
    """Project --has_risk--> Risk is not in the ontology and must be dropped."""
    _entities, relationships = extractor._normalise(
        _payload(
            [PHOENIX, R17],
            [{"source": "Project Phoenix", "target": "R-17", "type": "has_risk"}],
        ),
        doc,
    )
    assert relationships == []
    assert len(extractor._rejected) == 1
    assert "has_risk" in extractor._rejected[0]


def test_transposed_endpoints_are_repaired_not_dropped(extractor, doc):
    """A merely reversed edge carries a real fact; fix direction, keep it."""
    _entities, relationships = extractor._normalise(
        _payload(
            [C17, ALPHA],
            # Declared signature is Component -> Supplier; this is backwards.
            [{"source": "Alpha Precision Systems", "target": "C-17", "type": "supplied_by"}],
        ),
        doc,
    )
    assert len(relationships) == 1
    assert relationships[0].source == "C-17"
    assert relationships[0].target == "Alpha Precision Systems"
    assert extractor._rejected == []


def test_unknown_relation_name_is_dropped(extractor, doc):
    _entities, relationships = extractor._normalise(
        _payload([PHOENIX, C17], [{"source": "Project Phoenix", "target": "C-17", "type": "blah"}]),
        doc,
    )
    assert relationships == []


def test_unknown_entity_type_is_dropped(extractor, doc):
    entities, _rel = extractor._normalise(
        _payload([{"id": "X", "name": "X", "type": "Alien"}], []), doc
    )
    assert entities == []


def test_edge_with_unextracted_endpoint_is_dropped(extractor, doc):
    _entities, relationships = extractor._normalise(
        _payload([PHOENIX], [{"source": "Project Phoenix", "target": "C-99", "type": "uses"}]),
        doc,
    )
    assert relationships == []


def test_empty_extraction_is_valid_not_an_error(extractor, doc):
    """A generic policy document legitimately contains no ontology entities."""
    entities, relationships = extractor._normalise(_payload([], []), doc)
    assert entities == []
    assert relationships == []


def test_ids_are_canonicalised_and_aliased(extractor, doc):
    """Legal suffixes are stripped and the edge still resolves."""
    entities, relationships = extractor._normalise(
        _payload(
            [
                C17,
                {
                    "id": "Alpha Precision Systems S.A.S.",
                    "name": "Alpha Precision Systems S.A.S.",
                    "type": "Supplier",
                },
            ],
            [
                {
                    "source": "C-17",
                    "target": "Alpha Precision Systems S.A.S.",
                    "type": "supplied_by",
                }
            ],
        ),
        doc,
    )
    ids = {e.id for e in entities}
    assert "Alpha Precision Systems" in ids
    assert len(relationships) == 1
    assert relationships[0].target == "Alpha Precision Systems"


def test_duplicate_relationships_are_collapsed(extractor, doc):
    _entities, relationships = extractor._normalise(
        _payload(
            [PHOENIX, C17],
            [
                {"source": "Project Phoenix", "target": "C-17", "type": "uses"},
                {"source": "Project Phoenix", "target": "C-17", "type": "uses"},
            ],
        ),
        doc,
    )
    assert len(relationships) == 1


def test_self_loop_is_dropped(extractor, doc):
    _entities, relationships = extractor._normalise(
        _payload([C2048], [{"source": "C-2048", "target": "C-2048", "type": "has_risk"}]),
        doc,
    )
    assert relationships == []


def test_full_chain_survives_normalisation(extractor, doc):
    """The whole demo chain must pass validation unchanged."""
    _entities, relationships = extractor._normalise(
        _payload(
            [PHOENIX, C17, ALPHA, C2048, R17],
            [
                {"source": "Project Phoenix", "target": "C-17", "type": "uses"},
                {"source": "C-17", "target": "Alpha Precision Systems", "type": "supplied_by"},
                {
                    "source": "Alpha Precision Systems",
                    "target": "C-2048",
                    "type": "governed_by",
                },
                {"source": "C-2048", "target": "R-17", "type": "has_risk"},
            ],
        ),
        doc,
    )
    assert len(relationships) == 4
    assert extractor._rejected == []
