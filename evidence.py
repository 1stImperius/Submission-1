from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from .models import CandidateId, EvidenceItem


class EvidenceStore:
    """Candidate-isolated, immutable-after-ingestion evidence registry."""

    def __init__(self, candidate_id: CandidateId):
        self.candidate_id = candidate_id
        self._items: dict[str, EvidenceItem] = {}

    def add(self, item: EvidenceItem) -> None:
        if item.candidate_id != self.candidate_id:
            raise ValueError("Cross-candidate evidence insertion rejected")
        if item.evidence_id in self._items:
            raise ValueError(f"Duplicate evidence ID: {item.evidence_id}")
        self._items[item.evidence_id] = item

    def add_many(self, items: Iterable[EvidenceItem]) -> None:
        for item in items:
            self.add(item)

    def validate_ids(self, ids: Iterable[str]) -> bool:
        ids = list(ids)
        return bool(ids) and all(i in self._items for i in ids)

    def require_ids(self, ids: Iterable[str]) -> list[EvidenceItem]:
        ids = list(ids)
        if not ids:
            raise ValueError("At least one evidence ID is required")
        if not self.validate_ids(ids):
            raise ValueError(f"Unknown/cross-candidate evidence IDs: {ids}")
        return [self._items[i] for i in ids]

    def quote(self, evidence_id: str) -> str:
        return self.require_ids([evidence_id])[0].quote

    def all_items(self) -> list[EvidenceItem]:
        return list(self._items.values())


def build_evidence_store(candidate_id: CandidateId, resume: str, transcript: str) -> EvidenceStore:
    """Simple deterministic quote store. Production deployments may replace chunking with a PDF-aware extractor."""
    store = EvidenceStore(candidate_id)
    for source, text in (("resume", resume), ("transcript", transcript)):
        paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
        for idx, quote in enumerate(paragraphs, start=1):
            eid = f"E_{candidate_id}_{source.upper()}_{idx:03d}"
            store.add(EvidenceItem(evidence_id=eid, candidate_id=candidate_id, source=source, quote=quote))
    return store
