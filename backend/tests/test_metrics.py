"""Retrieval and content metric maths."""

from __future__ import annotations

import pytest

from app.evaluation.metrics import (
    aggregate,
    aggregate_by,
    coverage,
    f1_score,
    keypoint_hits,
    precision,
    recall,
    reciprocal_rank,
    score_question,
)


# ----------------------------------------------------------------------
def test_precision():
    assert precision(["a", "b", "c", "d"], ["a", "b"]) == 0.5
    assert precision(["a"], ["a"]) == 1.0
    assert precision([], ["a"]) == 0.0
    # Duplicates must not inflate the denominator.
    assert precision(["a", "a"], ["a"]) == 1.0


def test_recall():
    assert recall(["a", "b"], ["a", "b", "c", "d"]) == 0.5
    assert recall(["a", "b", "c"], ["a"]) == 1.0
    assert recall(["x"], []) == 0.0


def test_f1():
    assert f1_score(1.0, 1.0) == 1.0
    assert f1_score(0.0, 0.0) == 0.0
    assert f1_score(0.5, 1.0) == pytest.approx(0.6666, abs=1e-3)


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0
    assert reciprocal_rank(["a", "b", "c"], ["b"]) == 0.5
    assert reciprocal_rank(["a", "b", "c"], ["c"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a", "b"], ["z"]) == 0.0
    assert reciprocal_rank([], ["a"]) == 0.0


def test_reciprocal_rank_uses_first_relevant():
    assert reciprocal_rank(["x", "b", "a"], ["a", "b"]) == 0.5


# ----------------------------------------------------------------------
def test_keypoint_hits_accepts_any_surface_form():
    keypoints = [["Alpha Precision Systems", "Alpha Precision"], ["C-2048"]]
    hits = keypoint_hits("The supplier is Alpha Precision under C-2048.", keypoints)
    assert hits == [True, True]


def test_keypoint_hits_is_case_and_whitespace_insensitive():
    assert keypoint_hits("alpha   precision   systems", [["Alpha Precision Systems"]]) == [True]


def test_keypoint_hits_detects_absence():
    hits = keypoint_hits("The supplier is Borealis.", [["Alpha Precision"], ["C-2048"]])
    assert hits == [False, False]


def test_coverage():
    keypoints = [["C-17"], ["C-2048"], ["R-17"], ["Yuki Tanaka"]]
    assert coverage("C-17 is covered by C-2048.", keypoints) == 0.5
    assert coverage("C-17 C-2048 R-17 Yuki Tanaka", keypoints) == 1.0
    assert coverage("nothing relevant here", keypoints) == 0.0
    assert coverage("anything", []) == 1.0


# ----------------------------------------------------------------------
def _score(**overrides):
    payload = dict(
        question_id="Q01",
        question="Which supplier?",
        pipeline="basic",
        hops=2,
        difficulty="medium",
        ranked_docs=["d1", "d2"],
        relevant_docs=["d1", "d3"],
        keypoints=[["alpha"], ["c-2048"]],
        context="alpha appears here",
        answer="the supplier is alpha",
        latency={"total_ms": 100.0, "retrieval_ms": 20.0, "generation_ms": 80.0},
    )
    payload.update(overrides)
    return score_question(**payload)


def test_score_question_computes_all_metrics():
    score = _score()
    assert score.precision == 0.5
    assert score.recall == 0.5
    assert score.mrr == 1.0
    assert score.context_coverage == 0.5          # only "alpha" in context
    assert score.answer_correctness == 0.5        # only "alpha" in answer
    assert score.missed_keypoints == ["c-2048"]
    assert score.latency_ms == 100.0


def test_score_question_perfect_run():
    score = _score(
        ranked_docs=["d1", "d3"],
        context="alpha and c-2048",
        answer="alpha via c-2048",
    )
    assert score.precision == 1.0
    assert score.recall == 1.0
    assert score.f1 == 1.0
    assert score.answer_correctness == 1.0
    assert score.missed_keypoints == []


def test_aggregate_means():
    scores = [_score(), _score(ranked_docs=["d1", "d3"], answer="alpha c-2048")]
    summary = aggregate(scores)
    assert summary["questions"] == 2
    assert summary["precision"] == pytest.approx(0.75)
    assert summary["answer_correctness"] == pytest.approx(0.75)


def test_aggregate_empty():
    summary = aggregate([])
    assert summary["questions"] == 0
    assert summary["precision"] == 0.0


def test_aggregate_by_difficulty():
    scores = [_score(difficulty="easy"), _score(difficulty="hard")]
    grouped = aggregate_by(scores, "difficulty")
    assert set(grouped) == {"easy", "hard"}
    assert grouped["easy"]["questions"] == 1


# ----------------------------------------------------------------------
# Errored questions must not be averaged in as zeros.
# ----------------------------------------------------------------------
def _errored(**overrides):
    score = _score(ranked_docs=[], context="", answer="", latency={}, **overrides)
    score.error = "LLMError: gateway returned 503"
    return score


def test_aggregate_excludes_errored_questions():
    """A gateway outage must not be reported as a retrieval failure."""
    good = _score(ranked_docs=["d1", "d3"], context="alpha c-2048", answer="alpha c-2048")
    summary = aggregate([good, _errored()])

    assert summary["questions"] == 1          # only the scored one counts
    assert summary["errored"] == 1
    assert summary["precision"] == 1.0        # not halved by a transport failure
    assert summary["answer_correctness"] == 1.0


def test_aggregate_all_errored_reports_zero_questions():
    summary = aggregate([_errored(), _errored()])
    assert summary["questions"] == 0
    assert summary["errored"] == 2
    assert summary["precision"] == 0.0


def test_aggregate_clean_run_reports_no_errors():
    summary = aggregate([_score(), _score()])
    assert summary["errored"] == 0
    assert summary["questions"] == 2


def test_aggregate_by_difficulty_also_excludes_errors():
    grouped = aggregate_by([_score(difficulty="easy"), _errored(difficulty="easy")], "difficulty")
    assert grouped["easy"]["questions"] == 1
    assert grouped["easy"]["errored"] == 1


# ----------------------------------------------------------------------
# Typographic folding — the metric must measure correctness, not punctuation.
# ----------------------------------------------------------------------
def test_keypoints_match_across_unicode_hyphens():
    """Models render identifiers with typographic hyphens; that is not a miss."""
    keypoints = [["C-17"], ["C-2048"], ["R-17"]]
    # U+2011 non-breaking hyphen, U+2013 en dash, U+2010 hyphen
    answer = "Component C‑17 under contract C–2048 carries risk R‐17."
    assert keypoint_hits(answer, keypoints) == [True, True, True]


def test_keypoints_match_across_unicode_spaces():
    answer = "The supplier is Alpha Precision Systems."
    assert keypoint_hits(answer, [["Alpha Precision Systems"]]) == [True]


def test_keypoints_match_across_smart_quotes():
    assert keypoint_hits("the “R-17” entry", [["R-17"]]) == [True]


def test_folding_does_not_create_false_positives():
    """Folding must not make a wrong identifier match a right one."""
    assert keypoint_hits("risk R‑42 applies", [["R-17"]]) == [False]
    assert keypoint_hits("supplier Borealis", [["Alpha Precision"]]) == [False]


def test_coverage_uses_the_same_folding():
    context = "chain: C‑2048 -> R‑17"
    assert coverage(context, [["C-2048"], ["R-17"]]) == 1.0
