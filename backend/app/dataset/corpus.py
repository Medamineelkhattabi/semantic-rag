"""Corpus loading.

Both pipelines ingest exactly this list of documents — the shared substrate of
the comparison.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from ..config import settings


@dataclass
class Document:
    doc_id: str
    title: str
    path: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)


_HEADER_RE = re.compile(r"^#\s+(.*)$", re.MULTILINE)
_META_RE = re.compile(r"^(Document ID|Classification|Owner):\s*(.+)$", re.MULTILINE)


def _parse(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")

    title_match = _HEADER_RE.search(text)
    title = title_match.group(1).strip() if title_match else path.stem

    metadata = {key.lower().replace(" ", "_"): value.strip() for key, value in _META_RE.findall(text)}

    return Document(
        doc_id=path.stem,
        title=title,
        path=path.name,
        text=text,
        metadata=metadata,
    )


@lru_cache(maxsize=1)
def load_corpus() -> List[Document]:
    """Load every markdown document, sorted by filename for determinism."""
    directory = settings.documents_dir
    if not directory.exists():
        raise FileNotFoundError(f"Document directory not found: {directory}")

    documents = [_parse(p) for p in sorted(directory.glob("*.md"))]
    if not documents:
        raise FileNotFoundError(f"No markdown documents found in {directory}")
    return documents


def corpus_index() -> Dict[str, Document]:
    return {doc.doc_id: doc for doc in load_corpus()}
