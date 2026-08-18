"""Engine singleton and response serialisation.

Holds one built instance of each pipeline. Building the semantic pipeline
requires LLM extraction over the corpus, so the build runs once in a background
thread and its progress is exposed through ``/api/status``.
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Any, Dict, List, Optional

from .basic_rag.pipeline import BasicRagPipeline, BasicRagResult
from .config import settings
from .dataset.benchmark import BENCHMARK
from .dataset.corpus import load_corpus
from .evaluation.runner import BenchmarkRunner, basic_ranked_docs, semantic_ranked_docs
from .semantic_rag.pipeline import SemanticRagPipeline, SemanticRagResult


class Engine:
    """Lazily-built holder for both pipelines."""

    def __init__(self) -> None:
        self.basic = BasicRagPipeline()
        self.semantic = SemanticRagPipeline()
        self.state: str = "idle"          # idle | building | ready | error
        self.stage: str = ""
        self.error: str = ""
        self.built_at: Optional[float] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def status(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "stage": self.stage,
            "error": self.error,
            "builtAt": self.built_at,
            "basic": self.basic.build_stats,
            "semantic": self.semantic.build_stats,
        }

    # ------------------------------------------------------------------
    def _build(self) -> None:
        try:
            self.state = "building"
            self.error = ""

            self.stage = "loading corpus"
            documents = load_corpus()

            self.stage = "basic rag: chunking + embedding"
            self.basic.build(documents)

            self.stage = "semantic rag: extraction + knowledge graph"
            self.semantic.build(documents)

            self.stage = ""
            self.state = "ready"
            self.built_at = time.time()
        except Exception as exc:  # pragma: no cover - startup path
            self.state = "error"
            self.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

    def start_build(self, *, block: bool = False) -> None:
        with self._lock:
            if self.state in ("building", "ready"):
                if block and self._thread is not None:
                    self._thread.join()
                return
            self._thread = threading.Thread(target=self._build, daemon=True)
            self._thread.start()
        if block and self._thread is not None:
            self._thread.join()

    def ensure_ready(self, timeout: float = 900.0) -> None:
        """Block until built, raising if the build failed."""
        if self.state == "ready":
            return
        self.start_build()
        deadline = time.time() + timeout
        while self.state == "building" and time.time() < deadline:
            time.sleep(0.25)
        if self.state != "ready":
            raise RuntimeError(self.error or f"Engine not ready (state={self.state})")

    # ------------------------------------------------------------------
    def compare(self, question: str) -> Dict[str, Any]:
        self.ensure_ready()
        basic_result = self.basic.answer(question)
        semantic_result = self.semantic.answer(question)
        return {
            "question": question,
            "basic": serialise_basic(basic_result),
            "semantic": serialise_semantic(semantic_result, self.semantic),
            "shared": {
                "llmModel": settings.llm_model,
                "embeddingModel": settings.active_embedding_model,
                "documents": len(self.semantic.documents),
                "temperature": settings.llm_temperature,
            },
        }

    def run_benchmark(self, question_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        self.ensure_ready()
        questions = BENCHMARK
        if question_ids:
            wanted = set(question_ids)
            questions = [q for q in BENCHMARK if q.id in wanted]
        return BenchmarkRunner(self.basic, self.semantic).run(questions)


# ----------------------------------------------------------------------
# Serialisers
# ----------------------------------------------------------------------
def serialise_basic(result: BasicRagResult) -> Dict[str, Any]:
    return {
        "answer": result.answer,
        "chunks": [
            {
                "chunkId": chunk.chunk_id,
                "docId": chunk.doc_id,
                "docTitle": chunk.doc_title,
                "text": chunk.text,
                "score": round(chunk.score, 4),
                "rank": chunk.rank,
            }
            for chunk in result.chunks
        ],
        "sources": result.sources,
        "rankedDocs": basic_ranked_docs(result),
        "context": result.context,
        "latency": result.latency,
        "stats": result.stats,
    }


def serialise_semantic(
    result: SemanticRagResult, pipeline: SemanticRagPipeline
) -> Dict[str, Any]:
    def node_label(node_id: str) -> str:
        node = pipeline.graph.nodes.get(node_id)
        return node.name if node else node_id

    def node_type(node_id: str) -> str:
        node = pipeline.graph.nodes.get(node_id)
        return node.type if node else "Unknown"

    paths = []
    for path in result.paths:
        if not path:
            continue
        node_ids = [path[0].source] + [step.target for step in path]
        paths.append(
            {
                "nodes": [
                    {"id": nid, "label": node_label(nid), "type": node_type(nid)}
                    for nid in node_ids
                ],
                "steps": [
                    {
                        "source": step.source,
                        "relation": step.relation,
                        "target": step.target,
                        "direction": step.direction,
                        "docIds": step.doc_ids,
                        "evidence": step.evidence,
                        "phrase": step.phrase(),
                    }
                    for step in path
                ],
                "hops": len(path),
                "chain": "  ->  ".join(node_ids),
            }
        )

    return {
        "answer": result.answer,
        "entities": [
            {
                "id": seed.id,
                "label": seed.name,
                "type": seed.type,
                "score": round(seed.score, 4),
                "method": seed.method,
            }
            for seed in result.seeds
        ],
        "targetTypes": result.target_types,
        "relationships": [
            {
                "source": fact.source,
                "relation": fact.relation,
                "target": fact.target,
                "docIds": fact.doc_ids,
                "evidence": fact.evidence,
                "onPath": fact.on_path,
                "sentence": fact.to_sentence(),
            }
            for fact in result.facts
        ],
        "paths": paths,
        "primaryPath": paths[0] if paths else None,
        "passages": [
            {
                "docId": passage.doc_id,
                "docTitle": passage.doc_title,
                "heading": passage.heading,
                "text": passage.text,
                "matchedEntities": passage.matched_entities,
            }
            for passage in result.passages
        ],
        "subgraphNodes": result.subgraph_nodes,
        "sources": result.sources,
        "rankedDocs": semantic_ranked_docs(result),
        "context": result.context,
        "latency": result.latency,
        "stats": result.stats,
    }


engine = Engine()
