"""Path ranking.

Many paths connect any two seeds in a dense graph. The one shown as *the*
answer chain has to be the explanatory one, not merely the longest — an early
version sorted by length and surfaced an eight-hop ramble through unrelated
incidents instead of the four-hop supply chain.
"""

from __future__ import annotations

import pytest

from app.semantic_rag.pipeline import SemanticRagPipeline
from tests.test_semantic_rag import StubExtractor


@pytest.fixture
def pipeline(fake_embedder, fake_llm, sample_documents, sample_extractions):
    pipe = SemanticRagPipeline(
        embedder=fake_embedder,
        llm=fake_llm,
        extractor=StubExtractor(sample_extractions),
    )
    pipe.build(sample_documents)
    return pipe


def chain_of(path):
    return [path[0].source] + [step.target for step in path]


# ----------------------------------------------------------------------
def test_supply_chain_outranks_hub_wander(pipeline):
    """A Project->…->Risk chain must beat a Risk->Employee->Risk shortcut.

    Both connect risks, but only one is a supply-chain explanation.
    """
    seeds = pipeline.link_entities("Which risks reach Project Phoenix?")
    paths, _reachable, _edges = pipeline.traverse(seeds, ["Risk", "Project"], hops=5)

    assert paths
    best = chain_of(paths[0])
    assert best[0] == "Project Phoenix"
    assert "R-17" in best
    # The winning path must not be a two-hop hop through the risk owner.
    assert best != ["R-17", "Yuki Tanaka", "R-17"]


def test_ranking_prefers_type_diversity_over_length(pipeline):
    seeds = pipeline.link_entities("Which supplier and contract and risk for Project Phoenix?")
    paths, _r, _e = pipeline.traverse(seeds, ["Supplier", "Contract", "Risk"], hops=5)

    assert paths
    types = [
        pipeline.graph.nodes[n].type for n in chain_of(paths[0]) if n in pipeline.graph.nodes
    ]
    # The best chain should visit distinct types, not revisit one repeatedly.
    assert len(set(types)) == len(types)


def test_headline_chain_is_ranked_first(pipeline):
    question = (
        "Which supplier is responsible for the component used by Project Phoenix, "
        "what contract governs the relationship, and what risk is associated with "
        "that supplier?"
    )
    seeds = pipeline.link_entities(question)
    targets = pipeline.detect_target_types(question)
    paths, _r, _e = pipeline.traverse(seeds, targets, hops=5)

    assert paths
    assert chain_of(paths[0]) == [
        "Project Phoenix",
        "C-17",
        "Alpha Precision Systems",
        "C-2048",
        "R-17",
    ]


def test_rank_is_stable_and_total(pipeline):
    """Ranking must be deterministic — the hero visual cannot flicker."""
    question = "Which supplier and risk relate to Project Phoenix?"
    seeds = pipeline.link_entities(question)
    targets = pipeline.detect_target_types(question)

    first, _r, _e = pipeline.traverse(seeds, targets, hops=5)
    second, _r2, _e2 = pipeline.traverse(seeds, targets, hops=5)

    assert [chain_of(p) for p in first] == [chain_of(p) for p in second]


def test_empty_path_ranks_without_error(pipeline):
    assert pipeline._path_rank([], {}, set(), set()) == (0.0, 0)


# ----------------------------------------------------------------------
# Role relations: fine at the end of a chain, wrong in the middle.
# ----------------------------------------------------------------------
def test_interior_role_hop_is_penalised(pipeline):
    """Two contracts sharing a manager are not connected in the business.

    'C-2048 -> Yuki Tanaka -> R-17' routes through a person to reach a risk
    that is already reachable structurally, and must not outrank the real link.
    """
    seeds = pipeline.link_entities("Which risks relate to Project Phoenix?")
    paths, _r, _e = pipeline.traverse(seeds, ["Risk"], hops=5)
    assert paths

    best = paths[0]
    interior = [s.relation for s in best[:-1]]
    assert "owns_risk" not in interior
    assert "manages_contract" not in interior


def test_role_hop_allowed_as_final_step(pipeline):
    """'Who owns the risk?' legitimately ends on a person."""
    question = "Who owns the risk attached to Project Phoenix's component supplier?"
    seeds = pipeline.link_entities(question)
    targets = pipeline.detect_target_types(question)
    assert "Employee" in targets

    paths, _r, _e = pipeline.traverse(seeds, targets, hops=6)
    ending_on_person = [
        p for p in paths if pipeline.graph.nodes.get(p[-1].target, None)
        and pipeline.graph.nodes[p[-1].target].type == "Employee"
    ]
    assert ending_on_person, "expected at least one chain ending on the risk owner"


def test_path_does_not_wander_past_the_answer(pipeline):
    """A chain reaching R-17 must not be beaten by one continuing past it."""
    seeds = pipeline.link_entities("Which risk relates to Project Phoenix?")
    paths, _r, _e = pipeline.traverse(seeds, ["Risk"], hops=6)
    assert paths

    chain = chain_of(paths[0])
    assert chain[-1] == "R-17"


# ----------------------------------------------------------------------
# The highlighted path must illustrate the answer that was actually given.
# ----------------------------------------------------------------------
def test_answer_alignment_promotes_the_cited_chain(pipeline):
    a = pipeline.link_entities("Which risk relates to Project Phoenix?")
    paths, _r, _e = pipeline.traverse(a, ["Risk"], hops=5)
    assert len(paths) >= 1

    answer = "The supplier is Alpha Precision Systems under C-2048, carrying R-17."
    reordered = pipeline._prefer_path_matching_answer(paths, answer)
    assert "Alpha Precision Systems" in chain_of(reordered[0])


def test_answer_alignment_is_a_noop_without_an_answer(pipeline):
    seeds = pipeline.link_entities("Which risk relates to Project Phoenix?")
    paths, _r, _e = pipeline.traverse(seeds, ["Risk"], hops=5)
    assert pipeline._prefer_path_matching_answer(paths, "") is paths
    assert pipeline._prefer_path_matching_answer([], "text") == []


def test_answer_alignment_preserves_order_among_equals(pipeline):
    seeds = pipeline.link_entities("Which risk relates to Project Phoenix?")
    paths, _r, _e = pipeline.traverse(seeds, ["Risk"], hops=5)
    # An answer citing nothing leaves the structural ranking untouched.
    same = pipeline._prefer_path_matching_answer(paths, "no entities named here")
    assert [chain_of(p) for p in same] == [chain_of(p) for p in paths]


# ----------------------------------------------------------------------
# Entity linking: embeddings are a fallback, not a co-equal signal.
# ----------------------------------------------------------------------
def test_named_entity_suppresses_embedding_noise(pipeline):
    seeds = pipeline.link_entities("Which supplier serves Project Phoenix?")
    assert seeds
    assert all(s.method in ("identifier", "alias") for s in seeds), (
        "a named entity must not be diluted by topical embedding matches"
    )


def test_embedding_linking_still_used_when_nothing_is_named(pipeline):
    seeds = pipeline.link_entities("Which organisation carries the most exposure?")
    assert all(s.method == "embedding" for s in seeds) or not seeds
