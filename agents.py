from __future__ import annotations

import json
from typing import Any
from .llm import LLMClient
from .models import CandidateProfile, EvaluationOutput, RevisionOutput, CandidateId, DebateTurn
from .personas import PersonaConfig
from .evidence import EvidenceStore
from .validation import PythonEnforcer


class Agent:
    """One implementation, four independent epistemic configurations."""
    def __init__(self, config: PersonaConfig, llm: LLMClient):
        self.config = config
        self.llm = llm

    async def initial(self, profile: CandidateProfile, jd_text: str, store: EvidenceStore) -> EvaluationOutput:
        # Crucially: no other agent output is supplied here.
        payload = await self.llm.complete_json(
            self.config.system_prompt,
            json.dumps({"jd": jd_text, "profile": profile.model_dump()}),
        )
        payload["agent_role"] = self.config.role
        output = EvaluationOutput.model_validate(payload)
        PythonEnforcer.validate_evaluation(output, self.config, store)
        return output

    async def debate_challenge(self, defender: str, argument: str, evidence_ids: list[str], store: EvidenceStore) -> str:
        store.require_ids(evidence_ids)
        return await self.llm.complete_text(
            self.config.system_prompt,
            f"You are challenging {defender}. Directly address this argument:\n{argument}\n"
            f"Use only these evidence IDs: {evidence_ids}. Do not invent evidence.",
        )

    async def debate_response(self, challenger: str, challenge: str, evidence_ids: list[str], store: EvidenceStore) -> str:
        store.require_ids(evidence_ids)
        return await self.llm.complete_text(
            self.config.system_prompt,
            f"{challenger} challenged you with:\n{challenge}\nRespond directly. Concede or defend with evidence. "
            f"Use only these evidence IDs: {evidence_ids}.",
        )

    async def revise(self, initial: EvaluationOutput, debate: list[DebateTurn], store: EvidenceStore) -> RevisionOutput:
        payload = await self.llm.complete_json(
            self.config.system_prompt,
            json.dumps({
                "initial": initial.model_dump(),
                "debate": [d.model_dump() for d in debate],
                "instruction": "Return a complete final evaluation. Preserve initial_recommendation separately and compute a truthful delta.",
            }),
        )
        payload["agent_role"] = self.config.role
        output = RevisionOutput.model_validate(payload)
        PythonEnforcer.validate_revision(output, self.config, store)
        return output
