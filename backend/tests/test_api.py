"""API surface tests.

The engine is replaced with pipelines wired to offline fakes, so these exercise
routing, serialisation and error handling without any network access.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import service
from app.basic_rag.pipeline import BasicRagPipeline
from app.semantic_rag.pipeline import SemanticRagPipeline
from tests.test_semantic_rag import StubExtractor


@pytest.fixture
def client(fake_embedder, fake_llm, sample_documents, sample_extractions, monkeypatch):
    engine = service.Engine()
    engine.basic = BasicRagPipeline(embedder=fake_embedder, llm=fake_llm)
    engine.semantic = SemanticRagPipeline(
        embedder=fake_embedder,
        llm=fake_llm,
        extractor=StubExtractor(sample_extractions),
    )
    engine.basic.build(sample_documents)
    engine.semantic.build(sample_documents)
    engine.state = "ready"

    monkeypatch.setattr(service, "engine", engine)

    from app import main

    monkeypatch.setattr(main, "engine", engine)
    # Skip the background warm-up; the engine above is already built.
    monkeypatch.setattr(engine, "start_build", lambda **_: None)

    with TestClient(main.app) as test_client:
        yield test_client


# ----------------------------------------------------------------------
def test_status(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    assert response.json()["state"] == "ready"


def test_config_reports_shared_settings(client):
    payload = client.get("/api/config").json()
    for key in ("llmModel", "embeddingModel", "topK", "graphMaxHops", "chunkSize"):
        assert key in payload


def test_dataset_listing(client):
    payload = client.get("/api/dataset").json()
    assert payload["count"] >= 10
    assert all("docId" in d and "title" in d for d in payload["documents"])


def test_single_document(client):
    doc_id = client.get("/api/dataset").json()["documents"][0]["docId"]
    payload = client.get(f"/api/dataset/{doc_id}").json()
    assert payload["docId"] == doc_id
    assert payload["text"]


def test_missing_document_404(client):
    assert client.get("/api/dataset/nope").status_code == 404


def test_benchmark_questions(client):
    payload = client.get("/api/benchmark/questions").json()
    assert payload["count"] >= 10
    first = payload["questions"][0]
    for key in ("id", "question", "hops", "difficulty", "relevantDocs"):
        assert key in first


# ----------------------------------------------------------------------
def test_graph_endpoint(client):
    payload = client.get("/api/graph").json()
    assert payload["nodes"]
    assert payload["edges"]
    node_ids = {n["id"] for n in payload["nodes"]}
    assert all(e["source"] in node_ids and e["target"] in node_ids for e in payload["edges"])


def test_compare_returns_both_pipelines(client):
    response = client.post(
        "/api/compare",
        json={"question": "Which supplier serves Project Phoenix and what risk applies?"},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["basic"]["answer"]
    assert payload["semantic"]["answer"]
    assert payload["basic"]["chunks"]
    assert payload["semantic"]["relationships"]
    assert payload["semantic"]["paths"]
    assert payload["basic"]["context"]
    assert payload["semantic"]["context"]


def test_compare_reports_shared_configuration(client):
    payload = client.post(
        "/api/compare", json={"question": "Which risk affects Project Phoenix?"}
    ).json()
    shared = payload["shared"]
    assert shared["llmModel"]
    assert shared["embeddingModel"]
    assert shared["documents"] > 0


def test_compare_primary_path_is_a_chain(client):
    payload = client.post(
        "/api/compare",
        json={
            "question": "Which supplier serves Project Phoenix, under what contract, "
            "and what risk is attached?"
        },
    ).json()

    primary = payload["semantic"]["primaryPath"]
    assert primary is not None
    assert primary["hops"] >= 1
    assert "->" in primary["chain"]
    assert len(primary["nodes"]) == primary["hops"] + 1


def test_compare_rejects_empty_question(client):
    assert client.post("/api/compare", json={"question": "  "}).status_code == 422


# ----------------------------------------------------------------------
def test_benchmark_run_subset(client):
    response = client.post("/api/benchmark/run", json={"questionIds": ["Q01", "Q10"]})
    assert response.status_code == 200
    payload = response.json()

    assert len(payload["questions"]) == 2
    assert set(payload["summary"]) == {"basic", "semantic"}
    for side in ("basic", "semantic"):
        summary = payload["summary"][side]
        for metric in ("precision", "recall", "mrr", "context_coverage", "latency_ms"):
            assert metric in summary
    assert payload["meta"]["questionCount"] == 2
