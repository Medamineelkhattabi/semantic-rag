"""Benchmark runner.

Runs the identical question set through both pipelines and scores them with the
identical rubric, then aggregates overall and by difficulty / hop count.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from ..basic_rag.pipeline import BasicRagPipeline, BasicRagResult
from ..dataset.benchmark import BENCHMARK, BenchmarkQuestion
from ..semantic_rag.pipeline import SemanticRagPipeline, SemanticRagResult
from .metrics import QuestionScore, aggregate, aggregate_by, score_question


def basic_ranked_docs(result: BasicRagResult) -> List[str]:
    """Documents in the order Basic RAG surfaced them (best chunk first)."""
    ordered: List[str] = []
    for chunk in result.chunks:
        if chunk.doc_id not in ordered:
            ordered.append(chunk.doc_id)
    return ordered


def semantic_ranked_docs(result: SemanticRagResult) -> List[str]:
    """Documents in the order Semantic RAG surfaced them.

    Path-supporting provenance ranks first, then the expanded passages —
    mirroring the order in which the evidence enters the prompt.
    """
    ordered: List[str] = []
    for fact in result.facts:
        if not fact.on_path:
            continue
        for doc_id in fact.doc_ids:
            if doc_id not in ordered:
                ordered.append(doc_id)
    for passage in result.passages:
        if passage.doc_id not in ordered:
            ordered.append(passage.doc_id)
    for fact in result.facts:
        for doc_id in fact.doc_ids:
            if doc_id not in ordered:
                ordered.append(doc_id)
    return ordered


class BenchmarkRunner:
    def __init__(
        self, basic: BasicRagPipeline, semantic: SemanticRagPipeline
    ) -> None:
        self.basic = basic
        self.semantic = semantic

    # ------------------------------------------------------------------
    def _run_one(
        self, question: BenchmarkQuestion, pipeline_name: str
    ) -> QuestionScore:
        common = dict(
            question_id=question.id,
            question=question.question,
            pipeline=pipeline_name,
            hops=question.hops,
            difficulty=question.difficulty,
            relevant_docs=question.relevant_docs,
            keypoints=question.keypoints,
        )

        try:
            if pipeline_name == "basic":
                result = self.basic.answer(question.question)
                ranked = basic_ranked_docs(result)
            else:
                result = self.semantic.answer(question.question)
                ranked = semantic_ranked_docs(result)

            return score_question(
                ranked_docs=ranked,
                context=result.context,
                answer=result.answer,
                latency=result.latency,
                **common,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            score = score_question(
                ranked_docs=[],
                context="",
                answer="",
                latency={},
                **common,
            )
            score.error = f"{type(exc).__name__}: {exc}"
            return score

    # ------------------------------------------------------------------
    def run(
        self, questions: Optional[Sequence[BenchmarkQuestion]] = None
    ) -> Dict[str, Any]:
        questions = list(questions or BENCHMARK)
        started = time.perf_counter()

        basic_scores: List[QuestionScore] = []
        semantic_scores: List[QuestionScore] = []

        for question in questions:
            basic_scores.append(self._run_one(question, "basic"))
            semantic_scores.append(self._run_one(question, "semantic"))

        return {
            "questions": [
                {
                    "id": q.id,
                    "question": q.question,
                    "hops": q.hops,
                    "difficulty": q.difficulty,
                    "relevantDocs": q.relevant_docs,
                    "basic": basic.to_dict(),
                    "semantic": semantic.to_dict(),
                }
                for q, basic, semantic in zip(questions, basic_scores, semantic_scores)
            ],
            "summary": {
                "basic": aggregate(basic_scores),
                "semantic": aggregate(semantic_scores),
            },
            "byDifficulty": {
                "basic": aggregate_by(basic_scores, "difficulty"),
                "semantic": aggregate_by(semantic_scores, "difficulty"),
            },
            "byHops": {
                "basic": aggregate_by(basic_scores, "hops"),
                "semantic": aggregate_by(semantic_scores, "hops"),
            },
            "meta": {
                "questionCount": len(questions),
                "wallClockMs": round((time.perf_counter() - started) * 1000, 2),
                "errors": [
                    s.error
                    for s in (*basic_scores, *semantic_scores)
                    if s.error
                ],
            },
        }
