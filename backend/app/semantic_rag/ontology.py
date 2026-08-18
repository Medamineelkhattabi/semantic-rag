"""Domain ontology for the enterprise corpus.

Constraining extraction to a typed schema is what makes graph retrieval
reliable: relationship names stay closed-vocabulary, so traversal can reason
about edge semantics instead of guessing at free-text predicates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

ENTITY_TYPES: Dict[str, str] = {
    "Project": "An engineering programme. Use the name, e.g. 'Project Phoenix'.",
    "Component": "A hardware component. Use its register id, e.g. 'C-17'.",
    "Supplier": "A manufacturing organisation. Use its trading name, e.g. 'Alpha Precision Systems'.",
    "Contract": "A commercial instrument. Use its contract number, e.g. 'C-2048'.",
    "Risk": "A register risk. Use its risk id, e.g. 'R-17'.",
    "Incident": "A quality incident. Use its incident id, e.g. 'INC-1042'.",
    "Employee": "A named person. Use their full name, e.g. 'Elena Vasquez'.",
}

#: (relation, source type, target type, natural-language gloss)
RELATION_TYPES: List[Tuple[str, str, str, str]] = [
    ("uses", "Project", "Component", "a project depends on a component"),
    ("supplied_by", "Component", "Supplier", "a component is manufactured by a supplier"),
    ("governed_by", "Supplier", "Contract", "a supplier relationship runs under a contract"),
    ("has_risk", "Contract", "Risk", "a risk is recorded against a contract"),
    ("affects", "Incident", "Component", "an incident was raised against a component"),
    ("directs", "Employee", "Project", "a person is programme director of a project"),
    ("manages_contract", "Employee", "Contract", "a person is accountable for a contract"),
    ("owns_risk", "Employee", "Risk", "a person owns a risk"),
    ("investigates", "Employee", "Incident", "a person investigates an incident"),
]

RELATION_NAMES = {r[0] for r in RELATION_TYPES}

#: Relations that carry the supply chain itself — the substance of a multi-hop
#: answer.
SPINE_RELATIONS = {"uses", "supplied_by", "governed_by", "has_risk", "affects"}

#: Relations that attach a *person* to a thing. These are administrative
#: annotations, not structural links: two unrelated contracts can share a
#: manager, so traversing *through* an employee invents a connection that does
#: not exist in the business ("Titan -> C-4501 -> Anna Kowalski -> R-42" implies
#: Titan is exposed to R-42, which belongs to a different supplier entirely).
#: Ending on one is fine — that answers "who owns this?".
ROLE_RELATIONS = {"directs", "manages_contract", "owns_risk", "investigates"}

#: relation -> (source entity type, target entity type).
#: Extraction is validated against this, so the LLM cannot invent a shortcut
#: edge such as ``Project --has_risk--> Risk`` that would collapse a genuine
#: multi-hop chain into a single hop.
RELATION_SIGNATURES: Dict[str, Tuple[str, str]] = {
    relation: (source, target) for relation, source, target, _ in RELATION_TYPES
}

#: Inverse labels used when a path traverses an edge backwards.
INVERSE_GLOSS: Dict[str, str] = {
    "uses": "is used by",
    "supplied_by": "supplies",
    "governed_by": "governs",
    "has_risk": "is recorded against",
    "affects": "is affected by",
    "directs": "is directed by",
    "manages_contract": "is managed by",
    "owns_risk": "is owned by",
    "investigates": "is investigated by",
}


@dataclass(frozen=True)
class OntologySpec:
    entity_types: Dict[str, str]
    relation_types: List[Tuple[str, str, str, str]]

    def prompt_block(self) -> str:
        lines = ["ENTITY TYPES:"]
        for name, description in self.entity_types.items():
            lines.append(f"  - {name}: {description}")
        lines.append("")
        lines.append("RELATIONSHIP TYPES (source_type -> target_type):")
        for relation, source, target, gloss in self.relation_types:
            lines.append(f"  - {relation}: {source} -> {target}  ({gloss})")
        return "\n".join(lines)


ONTOLOGY = OntologySpec(entity_types=ENTITY_TYPES, relation_types=RELATION_TYPES)
