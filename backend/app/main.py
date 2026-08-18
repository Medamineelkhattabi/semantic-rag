"""FastAPI application for the RAG Intelligence Lab."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import settings
from .dataset.benchmark import BENCHMARK
from .dataset.corpus import load_corpus
from .llm import get_llm
from .service import engine


class CompareRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


class BenchmarkRequest(BaseModel):
    question_ids: Optional[List[str]] = Field(default=None, alias="questionIds")

    model_config = {"populate_by_name": True}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the pipelines in the background so the first request is fast.
    engine.start_build()
    yield


app = FastAPI(
    title="RAG Intelligence Lab",
    description="Side-by-side comparison of Basic RAG and Semantic (graph) RAG.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")] if settings.cors_origins != "*" else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "engine": engine.state,
        "llm": get_llm().health(),
    }


@app.get("/api/status")
def status() -> Dict[str, Any]:
    return engine.status()


@app.post("/api/build")
def build() -> Dict[str, Any]:
    engine.start_build()
    return engine.status()


@app.get("/api/config")
def config() -> Dict[str, Any]:
    """The settings both pipelines share — proof the comparison is like-for-like."""
    return {
        "llmModel": settings.llm_model,
        "extractionModel": settings.llm_extraction_model,
        "embeddingModel": settings.active_embedding_model,
        "embeddingProvider": settings.embedding_provider,
        "temperature": settings.llm_temperature,
        "chunkSize": settings.chunk_size,
        "chunkOverlap": settings.chunk_overlap,
        "topK": settings.top_k,
        "graphMaxHops": settings.graph_max_hops,
        "baseUrl": settings.llm_base_url,
    }


@app.get("/api/dataset")
def dataset() -> Dict[str, Any]:
    documents = load_corpus()
    return {
        "count": len(documents),
        "documents": [
            {
                "docId": doc.doc_id,
                "title": doc.title,
                "path": doc.path,
                "chars": doc.char_count,
                "metadata": doc.metadata,
            }
            for doc in documents
        ],
    }


@app.get("/api/dataset/{doc_id}")
def document(doc_id: str) -> Dict[str, Any]:
    for doc in load_corpus():
        if doc.doc_id == doc_id:
            return {
                "docId": doc.doc_id,
                "title": doc.title,
                "path": doc.path,
                "text": doc.text,
                "metadata": doc.metadata,
            }
    raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")


@app.get("/api/benchmark/questions")
def benchmark_questions() -> Dict[str, Any]:
    return {
        "count": len(BENCHMARK),
        "questions": [
            {
                "id": q.id,
                "question": q.question,
                "hops": q.hops,
                "difficulty": q.difficulty,
                "relevantDocs": q.relevant_docs,
                "goldPath": q.gold_path,
                "note": q.note,
            }
            for q in BENCHMARK
        ],
    }


@app.post("/api/compare")
def compare(request: CompareRequest) -> Dict[str, Any]:
    try:
        return engine.compare(request.question.strip())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.get("/api/graph")
def graph() -> Dict[str, Any]:
    try:
        engine.ensure_ready()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return engine.semantic.graph.to_payload()


@app.post("/api/benchmark/run")
def run_benchmark(request: Optional[BenchmarkRequest] = None) -> Dict[str, Any]:
    try:
        return engine.run_benchmark(request.question_ids if request else None)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
