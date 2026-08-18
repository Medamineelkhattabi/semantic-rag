"""Document -> entities + relationships, with provenance.

Extraction runs through Semantica's provider layer (``create_provider`` on a
registered custom provider), so the semantic pipeline uses the framework's own
LLM abstraction rather than a bespoke client.

Every extracted fact keeps the document it came from and the sentence that
supports it. That provenance is what lets the UI show *why* an answer is
believed, and what the evaluator scores for retrieval precision/recall.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..config import settings
from ..dataset.corpus import Document
from .ontology import ONTOLOGY, RELATION_NAMES, RELATION_SIGNATURES

PROMPT_VERSION = "v4"


# ----------------------------------------------------------------------
@dataclass
class ExtractedEntity:
    id: str
    name: str
    type: str
    doc_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    evidence: str = ""


@dataclass
class ExtractedRelationship:
    source: str
    target: str
    type: str
    doc_id: str
    evidence: str = ""
    confidence: float = 1.0


@dataclass
class DocumentExtraction:
    doc_id: str
    entities: List[ExtractedEntity]
    relationships: List[ExtractedRelationship]
    method: str
    latency_ms: float
    #: Facts discarded for violating the ontology's domain/range.
    rejected: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------
EXTRACTION_PROMPT = """You extract a typed knowledge graph from an enterprise document.

{ontology}

RULES
1. Extract ONLY entities and relationships that this document states explicitly.
   Do not infer, and do not import knowledge from other documents.
2. Entity `id` must be the identifier the document itself uses:
   components -> "C-17"; contracts -> "C-2048"; risks -> "R-17";
   incidents -> "INC-1042"; projects -> "Project Phoenix";
   suppliers -> trading name e.g. "Alpha Precision Systems";
   employees -> full name e.g. "Elena Vasquez".
   Use the SHORT canonical form. Never append legal suffixes such as
   "S.A.S.", "GmbH", "Limited", "Incorporated" unless the trading name in the
   document header itself contains it.
3. `type` must be exactly one of the entity types listed above.
4. Relationship `type` must be exactly one of the relationship types listed
   above, and must respect the declared source -> target direction.
5. `evidence` must be a short verbatim quote from the document (max 200 chars)
   that supports the fact.
6. Return an empty list rather than guessing.
7. Never create a shortcut edge. Only emit a relationship whose endpoint types
   match the declared source -> target signature exactly. If a document implies
   that a project is exposed to a risk, do NOT emit Project -> Risk: that link
   exists only through Component, Supplier and Contract, and inventing it
   destroys the real chain. Emit only the single link this document states.

Return raw JSON in exactly this shape and nothing else:
{{
  "entities": [
    {{"id": "...", "name": "...", "type": "...", "attributes": {{}}, "evidence": "..."}}
  ],
  "relationships": [
    {{"source": "...", "target": "...", "type": "...", "evidence": "..."}}
  ]
}}

DOCUMENT ID: {doc_id}
DOCUMENT TITLE: {title}

DOCUMENT TEXT:
\"\"\"
{text}
\"\"\"
"""


# ----------------------------------------------------------------------
# Canonicalisation
# ----------------------------------------------------------------------
_LEGAL_SUFFIXES = re.compile(
    r"\s*\b(S\.?A\.?S\.?|GmbH|Ltd\.?|Limited|Inc\.?|Incorporated|LLC|PLC|N\.?V\.?|S\.?A\.?)\b\.?$",
    re.IGNORECASE,
)
_ID_PATTERNS = [
    (re.compile(r"^(C-\d{2})$", re.IGNORECASE), lambda m: m.group(1).upper()),
    (re.compile(r"^(C-\d{4})$", re.IGNORECASE), lambda m: m.group(1).upper()),
    (re.compile(r"^(R-\d{2})$", re.IGNORECASE), lambda m: m.group(1).upper()),
    (re.compile(r"^(INC-\d{4})$", re.IGNORECASE), lambda m: m.group(1).upper()),
]


def canonical_id(raw: str, entity_type: str = "") -> str:
    """Normalise an entity id so the same real-world thing merges across docs."""
    value = (raw or "").strip().strip("\"'").strip()
    if not value:
        return ""

    for pattern, formatter in _ID_PATTERNS:
        match = pattern.match(value)
        if match:
            return formatter(match)

    # "PRJ-PHOENIX" / "phoenix" -> "Project Phoenix"
    if entity_type == "Project":
        bare = re.sub(r"^(PRJ-|Project\s+)", "", value, flags=re.IGNORECASE).strip()
        if bare:
            return f"Project {bare.title()}"

    if entity_type == "Supplier":
        value = _LEGAL_SUFFIXES.sub("", value).strip().rstrip(",")

    # Collapse internal whitespace.
    return re.sub(r"\s+", " ", value)


def _clean_evidence(text: str, limit: int = 240) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    return cleaned[:limit]


# ----------------------------------------------------------------------
# Deterministic fallback extractor
# ----------------------------------------------------------------------
_FALLBACK_PATTERNS: List[Tuple[re.Pattern, str, str, str]] = [
    (re.compile(r"(Project\s+\w+)\s+uses\s+component\s+(C-\d+)", re.I), "uses", "Project", "Component"),
    (re.compile(r"[Cc]omponent\s+(C-\d+)\s+is supplied by\s+([A-Z][\w\s\.]+?)(?:\.|,)", re.I), "supplied_by", "Component", "Supplier"),
    (re.compile(r"relationship with\s+([A-Z][\w\s\.]+?)\s+is governed by contract\s+(C-\d{4})", re.I), "governed_by", "Supplier", "Contract"),
    (re.compile(r"[Cc]ontract\s+(C-\d{4})\s+has risk\s+(R-\d+)", re.I), "has_risk", "Contract", "Risk"),
    (re.compile(r"[Ii]ncident\s+(INC-\d+)\s+affects component\s+(C-\d+)", re.I), "affects", "Incident", "Component"),
]


def fallback_extract(doc: Document) -> Tuple[List[ExtractedEntity], List[ExtractedRelationship]]:
    """Regex extractor used only when the LLM is unreachable, so the app boots."""
    entities: Dict[str, ExtractedEntity] = {}
    relationships: List[ExtractedRelationship] = []

    def ensure(entity_id: str, entity_type: str, evidence: str) -> str:
        key = canonical_id(entity_id, entity_type)
        if key and key not in entities:
            entities[key] = ExtractedEntity(
                id=key, name=key, type=entity_type, doc_id=doc.doc_id,
                evidence=_clean_evidence(evidence),
            )
        return key

    for pattern, relation, source_type, target_type in _FALLBACK_PATTERNS:
        for match in pattern.finditer(doc.text):
            sentence = doc.text[max(0, match.start() - 60) : match.end() + 60]
            source = ensure(match.group(1), source_type, sentence)
            target = ensure(match.group(2), target_type, sentence)
            if source and target:
                relationships.append(
                    ExtractedRelationship(
                        source=source, target=target, type=relation,
                        doc_id=doc.doc_id, evidence=_clean_evidence(sentence),
                        confidence=0.6,
                    )
                )

    return list(entities.values()), relationships


# ----------------------------------------------------------------------
class SemanticExtractor:
    """LLM extraction over Semantica's provider abstraction, with disk cache."""

    def __init__(self, provider: Optional[Any] = None, model: Optional[str] = None) -> None:
        self.model = model or settings.llm_extraction_model
        self._provider = provider
        self._rejected: List[str] = []
        self._cache_dir = settings.cache_dir / "extractions"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    @property
    def provider(self):
        if self._provider is None:
            from .provider import get_semantica_provider

            self._provider = get_semantica_provider(self.model)
        return self._provider

    def _cache_key(self, doc: Document) -> str:
        payload = f"{PROMPT_VERSION}|{self.model}|{doc.doc_id}|{doc.text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def _read_cache(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache(self, key: str, payload: Dict[str, Any]) -> None:
        try:
            (self._cache_dir / f"{key}.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        except OSError:  # pragma: no cover
            pass

    # ------------------------------------------------------------------
    def _normalise(
        self, raw: Dict[str, Any], doc: Document
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelationship]]:
        self._rejected = []
        entities: Dict[str, ExtractedEntity] = {}
        alias: Dict[str, str] = {}

        for item in raw.get("entities") or []:
            if not isinstance(item, dict):
                continue
            entity_type = str(item.get("type") or "").strip()
            if entity_type not in ONTOLOGY.entity_types:
                continue
            raw_id = str(item.get("id") or item.get("name") or "").strip()
            key = canonical_id(raw_id, entity_type)
            if not key:
                continue

            alias[raw_id.lower()] = key
            name = str(item.get("name") or key).strip()
            alias[name.lower()] = key

            attributes = item.get("attributes")
            if not isinstance(attributes, dict):
                attributes = {}

            if key in entities:
                entities[key].attributes.update(attributes)
                continue

            entities[key] = ExtractedEntity(
                id=key,
                name=name or key,
                type=entity_type,
                doc_id=doc.doc_id,
                attributes=attributes,
                evidence=_clean_evidence(str(item.get("evidence") or "")),
            )

        relationships: List[ExtractedRelationship] = []
        seen: set = set()
        for item in raw.get("relationships") or []:
            if not isinstance(item, dict):
                continue
            relation = str(item.get("type") or "").strip()
            if relation not in RELATION_NAMES:
                continue

            raw_source = str(item.get("source") or "").strip()
            raw_target = str(item.get("target") or "").strip()
            source = alias.get(raw_source.lower()) or canonical_id(raw_source)
            target = alias.get(raw_target.lower()) or canonical_id(raw_target)
            if not source or not target or source == target:
                continue

            # Drop edges whose endpoints were never extracted as entities.
            if source not in entities or target not in entities:
                continue

            # Enforce the ontology's domain/range. Models readily emit plausible
            # shortcuts (e.g. Project --has_risk--> Risk) that skip the real
            # chain; those must not enter the graph. If the endpoints are merely
            # transposed, repair the direction instead of discarding the fact.
            expected = RELATION_SIGNATURES[relation]
            actual = (entities[source].type, entities[target].type)
            if actual != expected:
                if (actual[1], actual[0]) == expected:
                    source, target = target, source
                else:
                    self._rejected.append(
                        f"{source}({actual[0]}) --{relation}--> {target}({actual[1]})"
                    )
                    continue

            signature = (source, relation, target)
            if signature in seen:
                continue
            seen.add(signature)

            relationships.append(
                ExtractedRelationship(
                    source=source,
                    target=target,
                    type=relation,
                    doc_id=doc.doc_id,
                    evidence=_clean_evidence(str(item.get("evidence") or "")),
                )
            )

        return list(entities.values()), relationships

    # ------------------------------------------------------------------
    def extract(self, doc: Document, *, use_cache: bool = True) -> DocumentExtraction:
        key = self._cache_key(doc)

        if use_cache:
            cached = self._read_cache(key)
            if cached:
                return DocumentExtraction(
                    doc_id=doc.doc_id,
                    entities=[ExtractedEntity(**e) for e in cached["entities"]],
                    relationships=[
                        ExtractedRelationship(**r) for r in cached["relationships"]
                    ],
                    method=cached.get("method", "llm") + "+cache",
                    latency_ms=0.0,
                    rejected=cached.get("rejected", []),
                )

        prompt = EXTRACTION_PROMPT.format(
            ontology=ONTOLOGY.prompt_block(),
            doc_id=doc.doc_id,
            title=doc.title,
            text=doc.text,
        )

        started = time.perf_counter()
        method = "llm"
        try:
            raw = self.provider.generate_structured(
                prompt, temperature=0.0, max_tokens=4000
            )
            if not isinstance(raw, dict):
                raise ValueError(f"Expected a JSON object, got {type(raw).__name__}")
            # An empty result is legitimate: some documents (e.g. a generic
            # policy) genuinely contain no ontology entities. Only a transport
            # or parse failure justifies the fallback extractor.
            entities, relationships = self._normalise(raw, doc)
        except Exception as exc:
            if not settings.allow_extraction_fallback:
                raise
            method = f"fallback ({type(exc).__name__})"
            entities, relationships = fallback_extract(doc)

        latency_ms = (time.perf_counter() - started) * 1000
        rejected = list(self._rejected)

        self._write_cache(
            key,
            {
                "doc_id": doc.doc_id,
                "method": method,
                "entities": [asdict(e) for e in entities],
                "relationships": [asdict(r) for r in relationships],
                "rejected": rejected,
            },
        )

        return DocumentExtraction(
            doc_id=doc.doc_id,
            entities=entities,
            relationships=relationships,
            method=method,
            latency_ms=round(latency_ms, 2),
            rejected=rejected,
        )
