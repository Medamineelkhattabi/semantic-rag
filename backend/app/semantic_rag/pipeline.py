"""Semantic RAG: extract -> graph -> entity link -> traverse -> expand -> LLM.

Retrieval works in four stages:

1. **Entity linking** — resolve the question to seed nodes using exact
   identifier matches, alias matches and embedding similarity over entity
   labels (same embedding model Basic RAG uses).
2. **Target typing** — read which *entity types* the question asks about
   ("which supplier", "what risk") even when no such entity is named.
3. **Graph traversal** — find paths from each seed to each requested type via
   ``semantica.kg.PathFinder``, plus a k-hop neighbourhood for local context.
4. **Context expansion** — pull the source passages that provenance-link to the
   retrieved facts, so the LLM gets grounded prose as well as triples.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..config import settings
from ..dataset.corpus import Document, load_corpus
from ..embeddings import EmbeddingClient, get_embedder
from ..llm import LLMClient, get_llm
from .extraction import SemanticExtractor
from .graph import GraphEdge, PathStep, SemanticGraph
from .ontology import ROLE_RELATIONS

ANSWER_SYSTEM_PROMPT = (
    "You are an enterprise knowledge assistant for Helios Aerospace. "
    "Answer strictly from the supplied context. "
    "If the context does not contain enough information to answer fully, say "
    "exactly which part is missing. Never invent identifiers, names or figures. "
    "Cite the source document ids you relied on in square brackets."
)

#: Words that signal which entity type the question is asking *for*.
TYPE_HINTS: Dict[str, Tuple[str, ...]] = {
    "Supplier": ("supplier", "suppliers", "vendor", "vendors", "manufacturer", "manufacturers"),
    "Contract": ("contract", "contracts", "agreement", "agreements", "instrument", "instruments", "ltsa"),
    "Risk": ("risk", "risks", "exposure", "exposures"),
    "Component": ("component", "components", "part", "parts", "module", "hardware"),
    "Project": ("project", "projects", "programme", "programmes", "program", "programs"),
    "Incident": ("incident", "incidents", "non-conformance", "defect", "quality issue"),
    # "responsible"/"accountable" are deliberately absent: they modify
    # organisations at least as often as people ("which supplier is responsible"),
    # and treating them as a person-signal drags irrelevant staff into the path.
    "Employee": ("who", "whom", "owner", "owns", "manager", "manages", "director", "engineer", "person", "staff"),
}

_ID_TOKEN_RE = re.compile(r"\b(?:C-\d{2,4}|R-\d{2}|INC-\d{3,4}|PRJ-[A-Z]+)\b", re.IGNORECASE)


# ----------------------------------------------------------------------
@dataclass
class SeedEntity:
    id: str
    name: str
    type: str
    score: float
    method: str  # "identifier" | "alias" | "embedding"


@dataclass
class RetrievedFact:
    source: str
    relation: str
    target: str
    doc_ids: List[str]
    evidence: str = ""
    on_path: bool = False

    def to_sentence(self) -> str:
        return f"{self.source} --[{self.relation}]--> {self.target}"


@dataclass
class SourcePassage:
    doc_id: str
    doc_title: str
    heading: str
    text: str
    matched_entities: List[str]


@dataclass
class SemanticRagResult:
    answer: str
    seeds: List[SeedEntity]
    target_types: List[str]
    facts: List[RetrievedFact]
    paths: List[List[PathStep]]
    passages: List[SourcePassage]
    subgraph_nodes: List[str]
    context: str
    sources: List[str]
    latency: Dict[str, float]
    stats: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
_HEADING_RE = re.compile(r"^(#{2,3})\s+(.*)$", re.MULTILINE)


def split_sections(doc: Document) -> List[Tuple[str, str]]:
    """Split a markdown document into ``(heading, body)`` sections."""
    matches = list(_HEADING_RE.finditer(doc.text))
    if not matches:
        return [(doc.title, doc.text)]

    sections: List[Tuple[str, str]] = []
    preamble = doc.text[: matches[0].start()].strip()
    if preamble:
        sections.append((doc.title, preamble))

    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc.text)
        body = doc.text[start:end].strip()
        if body:
            sections.append((match.group(2).strip(), body))
    return sections


# ----------------------------------------------------------------------
class SemanticRagPipeline:
    """Graph-aware retrieval over a Semantica knowledge graph."""

    name = "semantic"

    def __init__(
        self,
        embedder: Optional[EmbeddingClient] = None,
        llm: Optional[LLMClient] = None,
        extractor: Optional[SemanticExtractor] = None,
    ) -> None:
        self.embedder = embedder or get_embedder()
        self.llm = llm or get_llm()
        self.extractor = extractor or SemanticExtractor()
        self.graph = SemanticGraph()
        self.documents: List[Document] = []
        self._doc_index: Dict[str, Document] = {}
        self._sections: Dict[str, List[Tuple[str, str]]] = {}
        self._entity_ids: List[str] = []
        self._entity_vectors: Optional[np.ndarray] = None
        self.build_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def build(self, documents: Optional[List[Document]] = None) -> Dict[str, Any]:
        started = time.perf_counter()
        self.documents = documents or load_corpus()
        self._doc_index = {d.doc_id: d for d in self.documents}
        self._sections = {d.doc_id: split_sections(d) for d in self.documents}

        extract_started = time.perf_counter()
        extractions = [self.extractor.extract(doc) for doc in self.documents]
        extract_ms = (time.perf_counter() - extract_started) * 1000

        graph_stats = self.graph.build(extractions)

        # Embed entity labels once so question->entity linking is cheap.
        self._entity_ids = list(self.graph.nodes.keys())
        labels = [
            f"{node.type}: {node.name}" if node.name != node.id
            else f"{node.type}: {node.id}"
            for node in self.graph.nodes.values()
        ]
        self._entity_vectors = self.embedder.embed(labels) if labels else None

        self.build_stats = {
            **graph_stats,
            "extraction_ms": round(extract_ms, 2),
            "total_build_ms": round((time.perf_counter() - started) * 1000, 2),
            "extraction_model": self.extractor.model,
            "embedding_model": self.embedder.model,
        }
        return self.build_stats

    # ------------------------------------------------------------------
    # Stage 1 — entity linking
    # ------------------------------------------------------------------
    def link_entities(self, question: str, limit: Optional[int] = None) -> List[SeedEntity]:
        limit = limit or settings.graph_max_seeds
        lowered = question.lower()
        found: Dict[str, SeedEntity] = {}

        # (a) Explicit identifiers, e.g. "C-17", "R-42", "INC-1042".
        for token in _ID_TOKEN_RE.findall(question):
            key = token.upper()
            node = self.graph.nodes.get(key)
            if node:
                found[key] = SeedEntity(
                    id=node.id, name=node.name, type=node.type,
                    score=1.0, method="identifier",
                )

        # (b) Name / alias mentions.
        for node in self.graph.nodes.values():
            if node.id in found:
                continue
            candidates = {node.id.lower(), node.name.lower()}
            # "Project Phoenix" should also match a bare "Phoenix".
            if node.type == "Project":
                candidates.add(node.name.lower().replace("project ", ""))
            for candidate in candidates:
                if len(candidate) >= 4 and candidate in lowered:
                    found[node.id] = SeedEntity(
                        id=node.id, name=node.name, type=node.type,
                        score=0.95, method="alias",
                    )
                    break

        # (c) Embedding similarity — a FALLBACK for questions that name nothing.
        #
        # When the question does name an entity, that match is authoritative and
        # embedding neighbours are pure noise: "…used by Project Phoenix" pulls
        # in every supplier and risk in the corpus by topical similarity, and
        # those spurious seeds spawn well-formed paths that outrank the real
        # chain. Target-typing already steers traversal toward the entity types
        # being asked about, so nothing is lost by skipping them here.
        if found:
            return sorted(found.values(), key=lambda s: -s.score)[:limit]

        if self._entity_vectors is not None and len(self._entity_vectors):
            query = self.embedder.embed_one(question)
            similarities = self._entity_vectors @ query
            order = np.argsort(-similarities)
            for idx in order[: limit * 3]:
                score = float(similarities[int(idx)])
                if score < settings.entity_link_threshold:
                    break
                entity_id = self._entity_ids[int(idx)]
                if entity_id in found:
                    continue
                node = self.graph.nodes[entity_id]
                found[entity_id] = SeedEntity(
                    id=node.id, name=node.name, type=node.type,
                    score=score, method="embedding",
                )

        seeds = sorted(found.values(), key=lambda s: -s.score)[:limit]
        return seeds

    # ------------------------------------------------------------------
    # Stage 2 — target typing
    # ------------------------------------------------------------------
    @staticmethod
    def detect_target_types(question: str) -> List[str]:
        lowered = f" {question.lower()} "
        detected: List[str] = []
        for entity_type, hints in TYPE_HINTS.items():
            if any(re.search(rf"\b{re.escape(h)}\b", lowered) for h in hints):
                detected.append(entity_type)
        return detected

    # ------------------------------------------------------------------
    # Stage 3 — traversal
    # ------------------------------------------------------------------
    def traverse(
        self, seeds: Sequence[SeedEntity], target_types: Sequence[str], hops: Optional[int] = None
    ) -> Tuple[List[List[PathStep]], Set[str], List[GraphEdge]]:
        hops = hops or settings.graph_max_hops
        seed_ids = [s.id for s in seeds]

        reachable, edges = self.graph.neighbourhood(seed_ids, hops)

        paths: List[List[PathStep]] = []
        seen_signatures: Set[Tuple] = set()

        def record(path: Optional[List[PathStep]]) -> None:
            if not path:
                return
            signature = tuple((s.source, s.relation, s.target) for s in path)
            if signature in seen_signatures:
                return
            seen_signatures.add(signature)
            paths.append(path)

        # (a) Paths connecting the named seeds to each other.
        for i, source in enumerate(seed_ids):
            for target in seed_ids[i + 1 :]:
                record(self.graph.find_path(source, target))

        # (b) Paths from each seed to every node of a requested target type.
        seed_types = {s.type for s in seeds}
        for entity_type in target_types:
            if entity_type in seed_types and len(seed_ids) > 1:
                continue
            candidates = [
                node_id
                for node_id in reachable
                if self.graph.nodes.get(node_id) and self.graph.nodes[node_id].type == entity_type
            ]
            for seed in seed_ids:
                if self.graph.nodes.get(seed) and self.graph.nodes[seed].type == entity_type:
                    continue
                for candidate in candidates:
                    record(self.graph.find_path(seed, candidate))

        # Rank by explanatory value, not raw length. The longest path is usually
        # a ramble through unrelated entities; the best one covers the entity
        # types actually asked about, starts at a named seed, ends on a
        # requested type, and takes no detours to get there.
        seed_scores = {s.id: s.score for s in seeds}
        paths.sort(
            key=lambda p: self._path_rank(p, seed_scores, seed_types, set(target_types))
        )
        return paths, reachable, edges

    def _path_rank(
        self,
        path: List[PathStep],
        seed_scores: Dict[str, float],
        seed_types: Set[str],
        target_types: Set[str],
    ) -> Tuple[float, int]:
        """Sort key for candidate paths — lower is better."""
        if not path:
            return (0.0, 0)

        node_ids = [path[0].source] + [step.target for step in path]
        types = [
            self.graph.nodes[n].type for n in node_ids if n in self.graph.nodes
        ]
        if not types:
            return (0.0, 0)

        wanted = target_types or seed_types
        distinct = set(types)

        # Covering the types actually asked about is worth the most.
        covered = len(wanted & distinct)
        # Crossing many *different* types is the signature of a real chain
        # (Project -> Component -> Supplier -> Contract -> Risk). Intermediate
        # types are not penalised even when unnamed in the question: traversing
        # them is the entire point of graph retrieval.
        # Revisiting a type instead signals wandering through a hub node
        # (e.g. Risk -> Employee -> Risk, which is not a supply-chain fact).
        repeats = len(types) - len(distinct)

        score = 2.0 * covered + 1.0 * len(distinct) - 1.5 * repeats

        # Nodes trailing after the last one that first satisfied a requested
        # type add nothing to the explanation. A chain that reaches R-17 and
        # then wanders on through its owner to a second risk answers the
        # question no better than one that stops at R-17, and reads worse.
        satisfied: Set[str] = set()
        useful_end = 0
        for index, entity_type in enumerate(types):
            if entity_type in wanted and entity_type not in satisfied:
                satisfied.add(entity_type)
                useful_end = index
        score -= 1.5 * max(0, (len(types) - 1) - useful_end)

        # Passing *through* a person fabricates a link: two contracts sharing a
        # manager are not connected in the business. Ending on one is legitimate
        # ("who owns this risk?"), so only interior role hops are penalised.
        interior_role_hops = sum(
            1 for step in path[:-1] if step.relation in ROLE_RELATIONS
        )
        score -= 2.5 * interior_role_hops

        # Several chains can share an identical type signature (Phoenix reaches
        # both C-17/Alpha/C-2048/R-17 and C-22/Borealis/C-3120/R-24). Break that
        # tie by grounding: prefer the chain whose nodes the question actually
        # linked to, weighted by how confident each link was.
        score += 1.2 * sum(seed_scores.get(node_id, 0.0) for node_id in node_ids)

        if node_ids[0] in seed_scores:
            score += 1.0
        if types[-1] in target_types:
            score += 1.0

        # Negate for ascending sort; break ties toward the shorter explanation.
        return (-score, len(path))

    # ------------------------------------------------------------------
    # Stage 4 — context expansion
    # ------------------------------------------------------------------
    def collect_passages(
        self, entity_ids: Set[str], doc_ids: Set[str], max_passages: int = 10
    ) -> List[SourcePassage]:
        """Pull source sections that mention the retrieved entities."""
        names: Dict[str, str] = {}
        for entity_id in entity_ids:
            node = self.graph.nodes.get(entity_id)
            if node:
                names[entity_id] = node.name

        scored: List[Tuple[int, SourcePassage]] = []
        for doc_id in sorted(doc_ids):
            doc = self._doc_index.get(doc_id)
            if not doc:
                continue
            for heading, body in self._sections.get(doc_id, []):
                haystack = f"{heading}\n{body}".lower()
                matched = [
                    entity_id
                    for entity_id, name in names.items()
                    if entity_id.lower() in haystack or name.lower() in haystack
                ]
                if not matched:
                    continue
                scored.append(
                    (
                        len(matched),
                        SourcePassage(
                            doc_id=doc_id,
                            doc_title=doc.title,
                            heading=heading,
                            text=body,
                            matched_entities=sorted(matched),
                        ),
                    )
                )

        scored.sort(key=lambda pair: -pair[0])
        return [passage for _, passage in scored[:max_passages]]

    # ------------------------------------------------------------------
    def build_context(
        self,
        facts: Sequence[RetrievedFact],
        paths: Sequence[Sequence[PathStep]],
        passages: Sequence[SourcePassage],
        entity_ids: Sequence[str],
    ) -> str:
        blocks: List[str] = []

        if paths:
            chain_lines = []
            for path in list(paths)[:4]:
                if not path:
                    continue
                nodes = [path[0].source] + [step.target for step in path]
                chain_lines.append("  " + "  ->  ".join(nodes))
            if chain_lines:
                blocks.append("REASONING CHAINS FOUND IN THE KNOWLEDGE GRAPH:\n" + "\n".join(chain_lines))

        if entity_ids:
            entity_lines = []
            for entity_id in list(entity_ids)[:40]:
                node = self.graph.nodes.get(entity_id)
                if not node:
                    continue
                label = f"  - {node.id} ({node.type})"
                if node.name and node.name != node.id:
                    label += f" — {node.name}"
                entity_lines.append(label)
            if entity_lines:
                blocks.append("RELEVANT ENTITIES:\n" + "\n".join(entity_lines))

        if facts:
            fact_lines = []
            for fact in facts:
                marker = "*" if fact.on_path else " "
                provenance = ", ".join(fact.doc_ids)
                fact_lines.append(f" {marker} {fact.to_sentence()}   [{provenance}]")
            blocks.append(
                "VERIFIED RELATIONSHIPS (* = on the reasoning chain):\n" + "\n".join(fact_lines)
            )

        if passages:
            passage_blocks = []
            for passage in passages:
                passage_blocks.append(
                    f"[{passage.doc_id}] {passage.heading}\n{passage.text}"
                )
            blocks.append("SOURCE EVIDENCE:\n\n" + "\n\n---\n\n".join(passage_blocks))

        return "\n\n========\n\n".join(blocks)

    # ------------------------------------------------------------------
    @staticmethod
    def _prefer_path_matching_answer(
        paths: List[List[PathStep]], answer: str
    ) -> List[List[PathStep]]:
        """Stable-sort candidate paths by how many of their nodes the answer names.

        Ranking is preserved among equally-cited paths, so this only breaks ties
        that the structural score could not.
        """
        if not paths or not answer:
            return paths

        lowered = answer.lower()

        def cited(path: List[PathStep]) -> int:
            nodes = {path[0].source} | {step.target for step in path}
            return sum(1 for node in nodes if node.lower() in lowered)

        return sorted(paths, key=cited, reverse=True)

    # ------------------------------------------------------------------
    def answer(self, question: str, hops: Optional[int] = None) -> SemanticRagResult:
        total_started = time.perf_counter()

        link_started = time.perf_counter()
        seeds = self.link_entities(question)
        link_ms = (time.perf_counter() - link_started) * 1000

        target_types = self.detect_target_types(question)

        traverse_started = time.perf_counter()
        paths, reachable, edges = self.traverse(seeds, target_types, hops=hops)
        traverse_ms = (time.perf_counter() - traverse_started) * 1000

        # Facts on a path are flagged so the prompt (and the UI) can prioritise them.
        path_edge_keys: Set[Tuple[str, str, str]] = set()
        for path in paths:
            for step in path:
                if step.direction == "forward":
                    path_edge_keys.add((step.source, step.relation, step.target))
                else:
                    path_edge_keys.add((step.target, step.relation, step.source))

        facts: List[RetrievedFact] = []
        for edge in edges:
            facts.append(
                RetrievedFact(
                    source=edge.source,
                    relation=edge.type,
                    target=edge.target,
                    doc_ids=list(edge.doc_ids),
                    evidence=edge.evidence[0] if edge.evidence else "",
                    on_path=edge.key in path_edge_keys,
                )
            )
        facts.sort(key=lambda f: (not f.on_path, f.source))

        expand_started = time.perf_counter()
        path_nodes: Set[str] = set()
        for path in paths:
            for step in path:
                path_nodes.add(step.source)
                path_nodes.add(step.target)
        path_nodes.update(s.id for s in seeds)

        context_entities = path_nodes or set(reachable)
        doc_ids: Set[str] = set()
        for fact in facts:
            if fact.on_path or not paths:
                doc_ids.update(fact.doc_ids)
        for entity_id in context_entities:
            node = self.graph.nodes.get(entity_id)
            if node:
                doc_ids.update(node.doc_ids)

        passages = self.collect_passages(context_entities or set(reachable), doc_ids)
        context = self.build_context(
            facts, paths, passages, sorted(context_entities or reachable)
        )
        expand_ms = (time.perf_counter() - expand_started) * 1000

        prompt = (
            f"Context:\n\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the context above. When the answer follows a chain "
            "of relationships, state the chain explicitly."
        )
        answer_text, meta = self.llm.chat(
            [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )

        # Several chains can be equally well-formed (Phoenix reaches a risk via
        # both C-17/Alpha and C-22/Borealis), so structural ranking alone leaves
        # a genuine tie. Promote the chain the answer actually cites, otherwise
        # the highlighted path contradicts the prose beside it.
        paths = self._prefer_path_matching_answer(paths, answer_text)

        total_ms = (time.perf_counter() - total_started) * 1000
        sources = sorted({p.doc_id for p in passages} | doc_ids)

        return SemanticRagResult(
            answer=answer_text,
            seeds=seeds,
            target_types=target_types,
            facts=facts,
            paths=paths,
            passages=passages,
            subgraph_nodes=sorted(reachable),
            context=context,
            sources=sources,
            latency={
                "entity_linking_ms": round(link_ms, 2),
                "traversal_ms": round(traverse_ms, 2),
                "context_expansion_ms": round(expand_ms, 2),
                "retrieval_ms": round(link_ms + traverse_ms + expand_ms, 2),
                "generation_ms": meta["latency_ms"],
                "total_ms": round(total_ms, 2),
            },
            stats={
                "seeds": len(seeds),
                "target_types": target_types,
                "paths_found": len(paths),
                "longest_path_hops": max((len(p) for p in paths), default=0),
                "facts_retrieved": len(facts),
                "facts_on_path": sum(1 for f in facts if f.on_path),
                "subgraph_nodes": len(reachable),
                "passages": len(passages),
                "context_chars": len(context),
                "max_hops": hops or settings.graph_max_hops,
                "llm_model": meta.get("model"),
                "prompt_tokens": meta.get("prompt_tokens"),
                "completion_tokens": meta.get("completion_tokens"),
            },
        )
