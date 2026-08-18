"""Evaluation metrics.

Every number the UI displays is computed here from ground truth. Nothing is
hardcoded, and both pipelines are scored by the identical functions.

Retrieval is scored at *document* level, which is the only unit the two
pipelines have in common: Basic RAG returns chunks, Semantic RAG returns facts
and passages, but both ultimately claim a set of source documents.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Sequence


#: Typographic characters that LLMs emit inside identifiers, folded to ASCII
#: before matching. Without this, a model that renders "C-2048" with a
#: non-breaking hyphen (U+2011) scores zero on a keypoint it answered
#: perfectly — the metric would be measuring typography, not correctness.
_UNICODE_FOLD = str.maketrans(
    {
        "‐": "-",  # hyphen
        "‑": "-",  # non-breaking hyphen
        "‒": "-",  # figure dash
        "–": "-",  # en dash
        "—": "-",  # em dash
        "―": "-",  # horizontal bar
        "−": "-",  # minus sign
        "－": "-",  # fullwidth hyphen-minus
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
    }
)


def _normalise(text: str) -> str:
    """Lowercase, fold typographic characters, and collapse whitespace.

    ``\\s`` already covers Unicode spaces (NBSP, narrow NBSP) for str patterns.
    """
    return re.sub(r"\s+", " ", (text or "").translate(_UNICODE_FOLD)).strip().lower()


# ----------------------------------------------------------------------
# Retrieval metrics
# ----------------------------------------------------------------------
def precision(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    retrieved_set = set(retrieved)
    if not retrieved_set:
        return 0.0
    relevant_set = set(relevant)
    return len(retrieved_set & relevant_set) / len(retrieved_set)


def recall(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    return len(set(retrieved) & relevant_set) / len(relevant_set)


def f1_score(prec: float, rec: float) -> float:
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    """1 / rank of the first relevant document (0 if none retrieved)."""
    relevant_set = set(relevant)
    for position, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant_set:
            return 1.0 / position
    return 0.0


# ----------------------------------------------------------------------
# Content metrics
# ----------------------------------------------------------------------
def keypoint_hits(text: str, keypoints: Sequence[Sequence[str]]) -> List[bool]:
    """For each keypoint, whether any of its accepted surface forms appears."""
    haystack = _normalise(text)
    return [
        any(_normalise(form) in haystack for form in forms if form)
        for forms in keypoints
    ]


def coverage(text: str, keypoints: Sequence[Sequence[str]]) -> float:
    if not keypoints:
        return 1.0
    hits = keypoint_hits(text, keypoints)
    return sum(hits) / len(hits)


# ----------------------------------------------------------------------
@dataclass
class QuestionScore:
    question_id: str
    question: str
    pipeline: str
    hops: int
    difficulty: str

    precision: float
    recall: float
    f1: float
    mrr: float
    context_coverage: float
    answer_correctness: float

    latency_ms: float
    retrieval_ms: float
    generation_ms: float

    retrieved_docs: List[str] = field(default_factory=list)
    relevant_docs: List[str] = field(default_factory=list)
    missed_keypoints: List[str] = field(default_factory=list)
    answer: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def score_question(
    *,
    question_id: str,
    question: str,
    pipeline: str,
    hops: int,
    difficulty: str,
    ranked_docs: Sequence[str],
    relevant_docs: Sequence[str],
    keypoints: Sequence[Sequence[str]],
    context: str,
    answer: str,
    latency: Dict[str, float],
) -> QuestionScore:
    prec = precision(ranked_docs, relevant_docs)
    rec = recall(ranked_docs, relevant_docs)
    hits = keypoint_hits(answer, keypoints)
    missed = [forms[0] for forms, hit in zip(keypoints, hits) if not hit]

    return QuestionScore(
        question_id=question_id,
        question=question,
        pipeline=pipeline,
        hops=hops,
        difficulty=difficulty,
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1_score(prec, rec), 4),
        mrr=round(reciprocal_rank(ranked_docs, relevant_docs), 4),
        context_coverage=round(coverage(context, keypoints), 4),
        answer_correctness=round(sum(hits) / len(hits) if hits else 1.0, 4),
        latency_ms=round(latency.get("total_ms", 0.0), 2),
        retrieval_ms=round(latency.get("retrieval_ms", 0.0), 2),
        generation_ms=round(latency.get("generation_ms", 0.0), 2),
        retrieved_docs=list(ranked_docs),
        relevant_docs=list(relevant_docs),
        missed_keypoints=missed,
        answer=answer,
    )


# ----------------------------------------------------------------------
_AGGREGATED_FIELDS = (
    "precision",
    "recall",
    "f1",
    "mrr",
    "context_coverage",
    "answer_correctness",
    "latency_ms",
    "retrieval_ms",
    "generation_ms",
)


def aggregate(scores: Sequence[QuestionScore]) -> Dict[str, float]:
    """Mean of each metric across the supplied scores.

    Questions that errored (e.g. the LLM gateway was unreachable) are excluded
    rather than counted as zero. Averaging in a transport failure would report
    an infrastructure outage as a retrieval failure, which is the opposite of
    what the benchmark is measuring. The count is surfaced as ``errored`` so a
    degraded run is never mistaken for a complete one.
    """
    scored = [s for s in scores if not s.error]
    errored = len(scores) - len(scored)

    if not scored:
        return (
            {field_name: 0.0 for field_name in _AGGREGATED_FIELDS}
            | {"questions": 0, "errored": errored}
        )

    summary: Dict[str, float] = {}
    for field_name in _AGGREGATED_FIELDS:
        values = [getattr(score, field_name) for score in scored]
        summary[field_name] = round(sum(values) / len(values), 4)
    summary["questions"] = len(scored)
    summary["errored"] = errored
    return summary


def aggregate_by(
    scores: Sequence[QuestionScore], attribute: str
) -> Dict[str, Dict[str, float]]:
    """Aggregate grouped by an attribute such as ``difficulty`` or ``hops``."""
    buckets: Dict[str, List[QuestionScore]] = {}
    for score in scores:
        key = str(getattr(score, attribute))
        buckets.setdefault(key, []).append(score)
    return {key: aggregate(group) for key, group in sorted(buckets.items())}
