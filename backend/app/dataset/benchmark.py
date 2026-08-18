"""Benchmark questions with ground truth.

Ground truth is authored against the corpus, never against a pipeline's output.
Both pipelines are scored with the identical rubric in ``app.evaluation``.

Each keypoint is a list of acceptable surface forms; the keypoint counts as
covered if *any* form appears. This tolerates phrasing differences without
tolerating a wrong answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class BenchmarkQuestion:
    id: str
    question: str
    #: 1 = answerable from one document; >1 requires joining that many facts.
    hops: int
    difficulty: str  # "easy" | "medium" | "hard"
    #: Minimal set of documents required to answer correctly.
    relevant_docs: List[str]
    #: Each entry is a set of acceptable surface forms for one required fact.
    keypoints: List[List[str]] = field(default_factory=list)
    #: Expected entity chain, used to validate the graph path visualisation.
    gold_path: List[str] = field(default_factory=list)
    note: str = ""


BENCHMARK: List[BenchmarkQuestion] = [
    # ------------------------------------------------------------------
    # Single-hop — a well-tuned Basic RAG should do well on these.
    # ------------------------------------------------------------------
    BenchmarkQuestion(
        id="Q01",
        question="What is the severity and current status of risk R-17?",
        hops=1,
        difficulty="easy",
        relevant_docs=["05-risk-register"],
        keypoints=[["Critical"], ["mitigation overdue", "overdue"]],
        note="Single lookup in the risk register.",
    ),
    BenchmarkQuestion(
        id="Q02",
        question="Who is the programme director of Project Phoenix?",
        hops=1,
        difficulty="easy",
        relevant_docs=["01-project-portfolio"],
        keypoints=[["Elena Vasquez"]],
    ),
    BenchmarkQuestion(
        id="Q03",
        question="What is the total contract value of contract C-2048 and when does it expire?",
        hops=1,
        difficulty="easy",
        relevant_docs=["04-contract-register"],
        keypoints=[["31.4"], ["2028-12-31", "December 2028"]],
    ),
    BenchmarkQuestion(
        id="Q04",
        question="What is the unit cost and lead time of component C-31?",
        hops=1,
        difficulty="easy",
        relevant_docs=["02-component-register"],
        keypoints=[["88,700", "88700"], ["31 weeks"]],
    ),
    BenchmarkQuestion(
        id="Q05",
        question="What is the on-time delivery performance and supplier rating of Cygnus Aerospace Ltd?",
        hops=1,
        difficulty="easy",
        relevant_docs=["03-supplier-directory"],
        keypoints=[["78.2"], ["Marginal"]],
    ),
    # ------------------------------------------------------------------
    # Two hops.
    # ------------------------------------------------------------------
    BenchmarkQuestion(
        id="Q06",
        question=(
            "Which supplier is responsible for the inertial navigation component "
            "used by Project Phoenix?"
        ),
        hops=2,
        difficulty="medium",
        relevant_docs=["01-project-portfolio", "02-component-register"],
        keypoints=[["C-17"], ["Alpha Precision Systems", "Alpha Precision"]],
        gold_path=["Project Phoenix", "C-17", "Alpha Precision Systems"],
        note="Portfolio gives Phoenix->C-17; register gives C-17->Alpha. Never co-located.",
    ),
    BenchmarkQuestion(
        id="Q07",
        question="Which contract governs the supplier of component C-17?",
        hops=2,
        difficulty="medium",
        relevant_docs=["02-component-register", "03-supplier-directory"],
        keypoints=[["Alpha Precision Systems", "Alpha Precision"], ["C-2048"]],
        gold_path=["C-17", "Alpha Precision Systems", "C-2048"],
    ),
    BenchmarkQuestion(
        id="Q08",
        question="Which contract does Samuel Okonkwo manage, and which supplier does it govern?",
        hops=2,
        difficulty="medium",
        relevant_docs=["07-employee-directory", "03-supplier-directory"],
        keypoints=[["C-2048"], ["Alpha Precision Systems", "Alpha Precision"]],
        gold_path=["Samuel Okonkwo", "C-2048", "Alpha Precision Systems"],
    ),
    BenchmarkQuestion(
        id="Q09",
        question=(
            "Which component is shared between Project Phoenix and Project Titan, "
            "and which supplier and contract cover it?"
        ),
        hops=3,
        difficulty="hard",
        relevant_docs=[
            "01-project-portfolio",
            "02-component-register",
            "03-supplier-directory",
        ],
        keypoints=[
            ["C-22"],
            ["Borealis Components", "Borealis"],
            ["C-3120"],
        ],
        gold_path=["Project Phoenix", "C-22", "Borealis Components GmbH", "C-3120"],
    ),
    # ------------------------------------------------------------------
    # Deep multi-hop — the reason graph retrieval exists.
    # ------------------------------------------------------------------
    BenchmarkQuestion(
        id="Q10",
        # The spec's headline wording was "the component used by Project Phoenix".
        # Phoenix uses three (C-17, C-22, C-63), each leading to a *different*
        # supplier, contract and risk — so that phrasing has three equally
        # correct answers and cannot be scored. Naming the element preserves the
        # same four-hop chain while giving the question a unique answer.
        question=(
            "Which supplier is responsible for the inertial navigation component "
            "used by Project Phoenix, what contract governs that relationship, and "
            "what risk is associated with that supplier?"
        ),
        hops=4,
        difficulty="hard",
        relevant_docs=[
            "01-project-portfolio",
            "02-component-register",
            "03-supplier-directory",
            "04-contract-register",
            "05-risk-register",
        ],
        keypoints=[
            ["C-17"],
            ["Alpha Precision Systems", "Alpha Precision"],
            ["C-2048"],
            ["R-17"],
        ],
        gold_path=[
            "Project Phoenix",
            "C-17",
            "Alpha Precision Systems",
            "C-2048",
            "R-17",
        ],
        note="The headline demo question: a five-node chain across five documents.",
    ),
    BenchmarkQuestion(
        id="Q11",
        question=(
            "Who owns the risk attached to the contract that governs the supplier of "
            "Project Phoenix's inertial navigation module, and what is that risk's "
            "exposure value?"
        ),
        hops=5,
        difficulty="hard",
        relevant_docs=[
            "01-project-portfolio",
            "02-component-register",
            "03-supplier-directory",
            "04-contract-register",
            "05-risk-register",
        ],
        keypoints=[["R-17"], ["Yuki Tanaka"], ["14.8"]],
        gold_path=[
            "Project Phoenix",
            "C-17",
            "Alpha Precision Systems",
            "C-2048",
            "R-17",
            "Yuki Tanaka",
        ],
    ),
    BenchmarkQuestion(
        id="Q12",
        question="Which risks reach Project Titan through its supply chain?",
        hops=4,
        difficulty="hard",
        relevant_docs=[
            "01-project-portfolio",
            "02-component-register",
            "03-supplier-directory",
            "04-contract-register",
        ],
        keypoints=[["R-31"], ["R-08"], ["R-24"]],
        gold_path=["Project Titan", "C-31", "Cygnus Aerospace Ltd", "C-4501", "R-31"],
        note="Titan touches Cygnus (C-4501: R-31, R-08) and Borealis (C-3120: R-24).",
    ),
    BenchmarkQuestion(
        id="Q13",
        question=(
            "Which projects are exposed to the supplier involved in incident INC-1042?"
        ),
        hops=4,
        difficulty="hard",
        relevant_docs=[
            "06-incident-log",
            "02-component-register",
            "01-project-portfolio",
        ],
        keypoints=[
            ["Alpha Precision Systems", "Alpha Precision"],
            ["Project Phoenix", "Phoenix"],
            ["Project Vega", "Vega"],
        ],
        gold_path=["INC-1042", "C-17", "Alpha Precision Systems", "C-63", "Project Vega"],
        note="Alpha supplies C-17 (Phoenix) and C-63 (Phoenix + Vega).",
    ),
    BenchmarkQuestion(
        id="Q14",
        question=(
            "Which export-control risk exists in the supply chain, which supplier does "
            "it stem from, and which project does it ultimately affect?"
        ),
        hops=4,
        difficulty="hard",
        relevant_docs=[
            "05-risk-register",
            "04-contract-register",
            "03-supplier-directory",
            "02-component-register",
            "01-project-portfolio",
        ],
        keypoints=[
            ["R-42"],
            ["Delta Microwave", "Delta"],
            ["Project Aurora", "Aurora"],
        ],
        gold_path=[
            "R-42",
            "C-5077",
            "Delta Microwave Inc",
            "C-45",
            "Project Aurora",
        ],
    ),
    BenchmarkQuestion(
        id="Q15",
        question=(
            "Which supplier has more than one risk recorded against its governing "
            "contract, and what are those risks?"
        ),
        hops=3,
        difficulty="hard",
        relevant_docs=[
            "03-supplier-directory",
            "04-contract-register",
            "05-risk-register",
        ],
        keypoints=[
            ["Cygnus Aerospace", "Cygnus"],
            ["C-4501"],
            ["R-31"],
            ["R-08"],
        ],
        gold_path=["Cygnus Aerospace Ltd", "C-4501", "R-31"],
    ),
]


BENCHMARK_BY_ID = {q.id: q for q in BENCHMARK}


def get_question(question_id: str) -> BenchmarkQuestion:
    if question_id not in BENCHMARK_BY_ID:
        raise KeyError(f"Unknown benchmark question: {question_id}")
    return BENCHMARK_BY_ID[question_id]
