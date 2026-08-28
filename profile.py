from __future__ import annotations

import hashlib
import json
from .evidence import EvidenceStore
from .models import CandidateFact, CandidateInput, CandidateProfile, JobProfile, JDRequirement
from .llm import LLMClient


class ProfileBuilder:
    """Builds a structured candidate profile through a separate extraction task."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def build(self, candidate: CandidateInput, store: EvidenceStore, jd: JobProfile) -> CandidateProfile:
        evidence = [e.model_dump() for e in store.all_items()]
        payload = await self.llm.complete_json(
            system=(
                "Extract only explicit candidate facts from supplied evidence. "
                "Do not infer missing facts. Every fact must cite one or more evidence IDs. "
                "Return JSON with facts=[{fact_id,key,value,evidence_ids}]."
            ),
            user=json.dumps({"candidate": candidate.name, "jd": jd.model_dump(), "evidence": evidence}),
        )
        facts = [CandidateFact(candidate_id=candidate.candidate_id, **f) for f in payload.get("facts", [])]
        for fact in facts:
            if not store.validate_ids(fact.evidence_ids):
                raise ValueError(f"Profile fact cites invalid evidence: {fact.fact_id}")
        return CandidateProfile(
            candidate_id=candidate.candidate_id,
            facts=facts,
            evidence_ids=[e.evidence_id for e in store.all_items()],
        )


async def extract_job_profile(llm: LLMClient, jd_text: str) -> JobProfile:
    payload = await llm.complete_json(
        system=(
            "Extract the job requirements into machine-readable criteria. "
            "Mark mandatory requirements only when the JD clearly makes them required. "
            "Do not invent requirements. Return JSON: requirements=[{requirement_id,criterion,description,mandatory,evidence_expected}]."
        ),
        user=jd_text,
    )
    return JobProfile(requirements=[JDRequirement(**r) for r in payload.get("requirements", [])])
