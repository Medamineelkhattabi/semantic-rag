"""Knowledge graph construction and traversal.

Built with Semantica's own KG layer:

* ``semantica.kg.GraphBuilder``  — merges per-document extractions into one graph
  (entity resolution + conflict detection),
* ``semantica.kg.KnowledgeGraph`` — the framework's canonical graph dataclass,
* ``semantica.kg.PathFinder``     — multi-hop path search used to justify answers.

A NetworkX view is maintained alongside because ``PathFinder`` operates on
NetworkX-like graphs.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
from semantica.kg import GraphBuilder, KnowledgeGraph, PathFinder  # type: ignore

from .extraction import DocumentExtraction, ExtractedEntity, ExtractedRelationship
from .ontology import INVERSE_GLOSS


@dataclass
class GraphEdge:
    source: str
    target: str
    type: str
    doc_ids: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.source, self.type, self.target)


@dataclass
class GraphNode:
    id: str
    name: str
    type: str
    doc_ids: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)


@dataclass
class PathStep:
    source: str
    relation: str
    target: str
    direction: str          # "forward" | "reverse"
    doc_ids: List[str]
    evidence: str = ""

    def phrase(self) -> str:
        if self.direction == "forward":
            return f"{self.source} --[{self.relation}]--> {self.target}"
        return f"{self.source} --[{INVERSE_GLOSS.get(self.relation, self.relation)}]--> {self.target}"


class SemanticGraph:
    """Typed, provenance-carrying knowledge graph over the corpus."""

    def __init__(self) -> None:
        self.kg: Optional[KnowledgeGraph] = None
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[Tuple[str, str, str], GraphEdge] = {}
        self.nx_graph = nx.DiGraph()
        #: Undirected view, rebuilt with the graph. Path search treats edges as
        #: undirected (a risk is reachable *from* a project and vice versa), and
        #: traversal issues one query per seed/candidate pair — materialising
        #: this per call would deep-copy the whole graph hundreds of times.
        self._undirected: Optional[nx.Graph] = None
        self.path_finder = PathFinder()
        self.build_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def build(self, extractions: List[DocumentExtraction]) -> Dict[str, Any]:
        started = time.perf_counter()

        merged_entities: Dict[str, GraphNode] = {}
        merged_edges: Dict[Tuple[str, str, str], GraphEdge] = {}

        for extraction in extractions:
            for entity in extraction.entities:
                self._merge_entity(merged_entities, entity)
            for relationship in extraction.relationships:
                self._merge_edge(merged_edges, relationship)

        # Drop edges whose endpoints did not survive extraction.
        merged_edges = {
            key: edge
            for key, edge in merged_edges.items()
            if edge.source in merged_entities and edge.target in merged_entities
        }

        # Hand the merged payload to Semantica's GraphBuilder. It performs
        # entity resolution and conflict detection and returns the canonical
        # graph structure.
        builder = GraphBuilder(merge_entities=False, resolve_conflicts=False)
        built = builder.build(
            {
                "entities": [
                    {
                        "id": node.id,
                        "name": node.name,
                        "type": node.type,
                        "doc_ids": node.doc_ids,
                        "attributes": node.attributes,
                    }
                    for node in merged_entities.values()
                ],
                "relationships": [
                    {
                        "source": edge.source,
                        "target": edge.target,
                        "type": edge.type,
                        "doc_ids": edge.doc_ids,
                    }
                    for edge in merged_edges.values()
                ],
            },
            extract=False,
        )

        built_entities, built_relationships, built_metadata = _unpack(built)

        # Keep only entities the builder retained, preserving our provenance.
        retained_ids = {
            str(e.get("id")) for e in built_entities if isinstance(e, dict) and e.get("id")
        }
        if retained_ids:
            merged_entities = {
                k: v for k, v in merged_entities.items() if k in retained_ids
            }
            merged_edges = {
                key: edge
                for key, edge in merged_edges.items()
                if edge.source in merged_entities and edge.target in merged_entities
            }

        self.nodes = merged_entities
        self.edges = merged_edges

        self.kg = KnowledgeGraph(
            entities=[
                {
                    "id": node.id,
                    "name": node.name,
                    "type": node.type,
                    "doc_ids": node.doc_ids,
                    "attributes": node.attributes,
                }
                for node in self.nodes.values()
            ],
            relationships=[
                {
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.type,
                    "doc_ids": edge.doc_ids,
                }
                for edge in self.edges.values()
            ],
            metadata={
                "source": "semantica.kg.GraphBuilder",
                "builder_metadata": built_metadata,
                "documents": len(extractions),
            },
        )

        self._rebuild_nx()

        self.build_stats = {
            "entities": len(self.nodes),
            "relationships": len(self.edges),
            "entity_types": self.type_counts(),
            "relation_types": self.relation_counts(),
            "documents": len(extractions),
            "extraction_methods": sorted({e.method for e in extractions}),
            "rejected_facts": sum(len(e.rejected) for e in extractions),
            "rejected_examples": [r for e in extractions for r in e.rejected][:8],
            "build_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        return self.build_stats

    @staticmethod
    def _merge_entity(store: Dict[str, GraphNode], entity: ExtractedEntity) -> None:
        node = store.get(entity.id)
        if node is None:
            store[entity.id] = GraphNode(
                id=entity.id,
                name=entity.name or entity.id,
                type=entity.type,
                doc_ids=[entity.doc_id],
                attributes=dict(entity.attributes or {}),
                evidence=[entity.evidence] if entity.evidence else [],
            )
            return

        if entity.doc_id not in node.doc_ids:
            node.doc_ids.append(entity.doc_id)
        for key, value in (entity.attributes or {}).items():
            node.attributes.setdefault(key, value)
        if entity.evidence and entity.evidence not in node.evidence:
            node.evidence.append(entity.evidence)
        # Prefer the longer human-readable label.
        if len(entity.name or "") > len(node.name or ""):
            node.name = entity.name

    @staticmethod
    def _merge_edge(
        store: Dict[Tuple[str, str, str], GraphEdge], rel: ExtractedRelationship
    ) -> None:
        key = (rel.source, rel.type, rel.target)
        edge = store.get(key)
        if edge is None:
            store[key] = GraphEdge(
                source=rel.source,
                target=rel.target,
                type=rel.type,
                doc_ids=[rel.doc_id],
                evidence=[rel.evidence] if rel.evidence else [],
            )
            return
        if rel.doc_id not in edge.doc_ids:
            edge.doc_ids.append(rel.doc_id)
        if rel.evidence and rel.evidence not in edge.evidence:
            edge.evidence.append(rel.evidence)

    def _rebuild_nx(self) -> None:
        graph = nx.DiGraph()
        for node in self.nodes.values():
            graph.add_node(node.id, type=node.type, name=node.name, doc_ids=node.doc_ids)
        for edge in self.edges.values():
            graph.add_edge(
                edge.source,
                edge.target,
                type=edge.type,
                doc_ids=edge.doc_ids,
                weight=1.0,
            )
        self.nx_graph = graph
        self._undirected = graph.to_undirected(as_view=False)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def type_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for node in self.nodes.values():
            counts[node.type] = counts.get(node.type, 0) + 1
        return dict(sorted(counts.items()))

    def relation_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for edge in self.edges.values():
            counts[edge.type] = counts.get(edge.type, 0) + 1
        return dict(sorted(counts.items()))

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------
    def incident_edges(self, node_id: str) -> List[Tuple[GraphEdge, str]]:
        """All edges touching ``node_id`` with the direction of travel."""
        results: List[Tuple[GraphEdge, str]] = []
        for edge in self.edges.values():
            if edge.source == node_id:
                results.append((edge, "forward"))
            elif edge.target == node_id:
                results.append((edge, "reverse"))
        return results

    def neighbourhood(self, seeds: Iterable[str], hops: int) -> Tuple[Set[str], List[GraphEdge]]:
        """Undirected k-hop expansion from the seed set."""
        visited: Set[str] = set()
        collected: Dict[Tuple[str, str, str], GraphEdge] = {}

        queue: deque = deque((seed, 0) for seed in seeds if seed in self.nodes)
        visited.update(node for node, _ in queue)

        while queue:
            node_id, depth = queue.popleft()
            if depth >= hops:
                continue
            for edge, direction in self.incident_edges(node_id):
                collected[edge.key] = edge
                other = edge.target if direction == "forward" else edge.source
                if other not in visited:
                    visited.add(other)
                    queue.append((other, depth + 1))

        return visited, list(collected.values())

    def find_path(self, source: str, target: str) -> Optional[List[PathStep]]:
        """Shortest connecting path, treating edges as undirected.

        Uses ``semantica.kg.PathFinder`` over an undirected view of the graph.
        """
        if source not in self.nodes or target not in self.nodes:
            return None
        if source == target:
            return []

        undirected = self._undirected
        if undirected is None:
            undirected = self.nx_graph.to_undirected(as_view=False)
            self._undirected = undirected
        try:
            node_path = self.path_finder.find_shortest_path(undirected, source, target)
        except Exception:
            node_path = None

        if not node_path:
            try:
                node_path = nx.shortest_path(undirected, source, target)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None

        return self._to_steps(list(node_path))

    def _to_steps(self, node_path: List[str]) -> List[PathStep]:
        steps: List[PathStep] = []
        for left, right in zip(node_path, node_path[1:]):
            edge = self.edges.get(self._edge_key(left, right))
            direction = "forward"
            if edge is None:
                edge = self.edges.get(self._edge_key(right, left))
                direction = "reverse"
            if edge is None:
                continue
            steps.append(
                PathStep(
                    source=left,
                    relation=edge.type,
                    target=right,
                    direction=direction,
                    doc_ids=list(edge.doc_ids),
                    evidence=edge.evidence[0] if edge.evidence else "",
                )
            )
        return steps

    def _edge_key(self, source: str, target: str) -> Tuple[str, str, str]:
        for edge in self.edges.values():
            if edge.source == source and edge.target == target:
                return edge.key
        return (source, "", target)

    # ------------------------------------------------------------------
    def to_payload(self) -> Dict[str, Any]:
        """Serialise the whole graph for the frontend visualisation."""
        return {
            "nodes": [
                {
                    "id": node.id,
                    "label": node.name,
                    "type": node.type,
                    "docIds": node.doc_ids,
                    "attributes": node.attributes,
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "id": f"{edge.source}|{edge.type}|{edge.target}",
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.type,
                    "docIds": edge.doc_ids,
                }
                for edge in self.edges.values()
            ],
            "stats": self.build_stats,
        }


def _unpack(built: Any) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
    """Normalise GraphBuilder output (dict or KnowledgeGraph) to a tuple."""
    if isinstance(built, dict):
        return (
            built.get("entities") or [],
            built.get("relationships") or [],
            built.get("metadata") or {},
        )
    return (
        list(getattr(built, "entities", []) or []),
        list(getattr(built, "relationships", []) or []),
        dict(getattr(built, "metadata", {}) or {}),
    )
