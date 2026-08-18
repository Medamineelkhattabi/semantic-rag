"""Test fixtures.

Everything here runs offline: the LLM and embedding clients are replaced with
deterministic fakes so the suite never touches the network.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pytest

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.dataset.corpus import Document  # noqa: E402
from app.semantic_rag.extraction import (  # noqa: E402
    DocumentExtraction,
    ExtractedEntity,
    ExtractedRelationship,
)


class FakeEmbedder:
    """Deterministic hash embedder with lexical overlap signal.

    Vectors are built from character trigrams so texts sharing vocabulary land
    near each other — enough structure for retrieval tests without a network.
    """

    model = "fake-embedder"
    dim = 64

    def embed(self, texts: Sequence[str], **_: Any) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        matrix = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            lowered = (text or "").lower()
            for i in range(max(len(lowered) - 2, 0)):
                trigram = lowered[i : i + 3]
                bucket = int(hashlib.md5(trigram.encode()).hexdigest()[:8], 16) % self.dim
                matrix[row, bucket] += 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class FakeLLM:
    """Echoes the retrieved context back, so answer quality tracks retrieval."""

    def __init__(self, response: str = "") -> None:
        self.response = response
        self.calls: List[List[Dict[str, str]]] = []

    def chat(self, messages, **_: Any) -> Tuple[str, Dict[str, Any]]:
        self.calls.append(messages)
        if self.response:
            text = self.response
        else:
            # Echo the context so keypoint coverage reflects what was retrieved.
            text = messages[-1]["content"][:4000]
        return text, {
            "latency_ms": 1.0,
            "model": "fake-llm",
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "total_tokens": 20,
        }


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def sample_documents() -> List[Document]:
    return [
        Document(
            doc_id="doc-projects",
            title="Portfolio",
            path="doc-projects.md",
            text=(
                "# Portfolio\n\n"
                "## PRJ-PHOENIX\n\n"
                "Project Phoenix uses component C-17 as its primary navigation element.\n\n"
                "## PRJ-TITAN\n\n"
                "Project Titan uses component C-31 in its cryogenic system.\n"
            ),
        ),
        Document(
            doc_id="doc-components",
            title="Components",
            path="doc-components.md",
            text=(
                "# Components\n\n"
                "## C-17\n\n"
                "Component C-17 is supplied by Alpha Precision Systems.\n\n"
                "## C-31\n\n"
                "Component C-31 is supplied by Cygnus Aerospace.\n"
            ),
        ),
        Document(
            doc_id="doc-suppliers",
            title="Suppliers",
            path="doc-suppliers.md",
            text=(
                "# Suppliers\n\n"
                "## Alpha Precision Systems\n\n"
                "The relationship with Alpha Precision Systems is governed by contract C-2048.\n"
            ),
        ),
        Document(
            doc_id="doc-contracts",
            title="Contracts",
            path="doc-contracts.md",
            text=(
                "# Contracts\n\n"
                "## C-2048\n\n"
                "Contract C-2048 has risk R-17 recorded against it. Value EUR 31.4 million.\n"
            ),
        ),
        Document(
            doc_id="doc-risks",
            title="Risks",
            path="doc-risks.md",
            text=(
                "# Risks\n\n"
                "## R-17\n\n"
                "Risk R-17 is a single-source dependency. Severity: Critical. "
                "Risk owner: Yuki Tanaka.\n"
            ),
        ),
    ]


@pytest.fixture
def sample_extractions() -> List[DocumentExtraction]:
    """A hand-built chain: Phoenix -> C-17 -> Alpha -> C-2048 -> R-17."""

    def entity(eid: str, etype: str, doc: str) -> ExtractedEntity:
        return ExtractedEntity(id=eid, name=eid, type=etype, doc_id=doc, evidence=f"{eid} evidence")

    def relation(src: str, rel: str, dst: str, doc: str) -> ExtractedRelationship:
        return ExtractedRelationship(
            source=src, target=dst, type=rel, doc_id=doc, evidence=f"{src} {rel} {dst}"
        )

    return [
        DocumentExtraction(
            doc_id="doc-projects",
            entities=[
                entity("Project Phoenix", "Project", "doc-projects"),
                entity("Project Titan", "Project", "doc-projects"),
                entity("C-17", "Component", "doc-projects"),
                entity("C-31", "Component", "doc-projects"),
            ],
            relationships=[
                relation("Project Phoenix", "uses", "C-17", "doc-projects"),
                relation("Project Titan", "uses", "C-31", "doc-projects"),
            ],
            method="fixture",
            latency_ms=0.0,
        ),
        DocumentExtraction(
            doc_id="doc-components",
            entities=[
                entity("C-17", "Component", "doc-components"),
                entity("C-31", "Component", "doc-components"),
                entity("Alpha Precision Systems", "Supplier", "doc-components"),
                entity("Cygnus Aerospace", "Supplier", "doc-components"),
            ],
            relationships=[
                relation("C-17", "supplied_by", "Alpha Precision Systems", "doc-components"),
                relation("C-31", "supplied_by", "Cygnus Aerospace", "doc-components"),
            ],
            method="fixture",
            latency_ms=0.0,
        ),
        DocumentExtraction(
            doc_id="doc-suppliers",
            entities=[
                entity("Alpha Precision Systems", "Supplier", "doc-suppliers"),
                entity("C-2048", "Contract", "doc-suppliers"),
            ],
            relationships=[
                relation("Alpha Precision Systems", "governed_by", "C-2048", "doc-suppliers"),
            ],
            method="fixture",
            latency_ms=0.0,
        ),
        DocumentExtraction(
            doc_id="doc-contracts",
            entities=[
                entity("C-2048", "Contract", "doc-contracts"),
                entity("R-17", "Risk", "doc-contracts"),
            ],
            relationships=[
                relation("C-2048", "has_risk", "R-17", "doc-contracts"),
            ],
            method="fixture",
            latency_ms=0.0,
        ),
        DocumentExtraction(
            doc_id="doc-risks",
            entities=[
                entity("R-17", "Risk", "doc-risks"),
                entity("Yuki Tanaka", "Employee", "doc-risks"),
            ],
            relationships=[
                relation("Yuki Tanaka", "owns_risk", "R-17", "doc-risks"),
            ],
            method="fixture",
            latency_ms=0.0,
        ),
    ]
