from __future__ import annotations

import asyncio
from .agents import Agent
from .debate import DebateEngine
from .engines import GateEngine, ScoreEngine, ReportEngine
from .evidence import build_evidence_store, EvidenceStore
from .llm import LLMClient
from .models import CandidateInput, FinalReport, ComparativeDecision, EvaluationOutput, RevisionOutput, CandidateId
from .personas import PERSONAS
from .profile import ProfileBuilder, extract_job_profile


class HiringOrchestrator:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def _agents(self) -> dict:
        return {role: Agent(config, self.llm) for role, config in PERSONAS.items()}

    async def _candidate(self, candidate: CandidateInput, jd_text: str, jd_profile):
        store = build_evidence_store(candidate.candidate_id, candidate.resume, candidate.transcript)
        profile = await ProfileBuilder(self.llm).build(candidate, store, jd_profile)
        agents = self._agents()

        # Four truly independent calls: each receives only shared source/profile information.
        initial = await asyncio.gather(*[
            agents[role].initial(profile, jd_text, store) for role in PERSONAS
        ])

        debate_engine = DebateEngine(agents)
        debate, disagreements = await debate_engine.run(candidate.candidate_id, initial, store)

        # Every revision passes through the same Python validation chain as initial output.
        revisions = await asyncio.gather(*[
            agents[e.agent_role].revise(e, debate, store) for e in initial
        ])

        eligible, failures = GateEngine.evaluate(revisions, jd_profile.requirements)
        score = ScoreEngine.calculate(revisions)
        report = ReportEngine.build_candidate_report(
            candidate.candidate_id, candidate.name, eligible, revisions, score,
            [d.model_copy(update={"description": d.description + (" | " + "; ".join(failures) if failures else "")}) for d in disagreements],
        )
        return {
            "candidate": candidate, "store": store, "initial": initial,
            "debate": debate, "revisions": revisions, "eligible": eligible,
            "failures": failures, "score": score, "report": report,
        }

    async def run(self, jd_text: str, candidate_a: CandidateInput, candidate_b: CandidateInput) -> FinalReport:
        jd_profile = await extract_job_profile(self.llm, jd_text)
        a, b = await asyncio.gather(
            self._candidate(candidate_a, jd_text, jd_profile),
            self._candidate(candidate_b, jd_text, jd_profile),
        )

        # Final reasoning is explicitly comparative and receives post-debate state, not initial-only data.
        comparison = await self.llm.complete_json(
            "You are the final decision agent. Compare both candidates using only the supplied post-debate reports. "
            "Do not select an ineligible candidate. Return selected_candidate_id, confidence, reasoning, evidence_ids.",
            str({"A": a["report"].model_dump(), "B": b["report"].model_dump()}),
        )
        decision = ComparativeDecision.model_validate(comparison)
        if decision.selected_candidate_id == "CAND_A" and not a["eligible"]:
            decision = decision.model_copy(update={"selected_candidate_id": "CAND_B" if b["eligible"] else "NEITHER"})
        if decision.selected_candidate_id == "CAND_B" and not b["eligible"]:
            decision = decision.model_copy(update={"selected_candidate_id": "CAND_A" if a["eligible"] else "NEITHER"})

        # Final evidence must belong to the selected candidate; NEITHER may have no deciding evidence.
        if decision.selected_candidate_id == "CAND_A":
            a["store"].require_ids(decision.evidence_ids)
        elif decision.selected_candidate_id == "CAND_B":
            b["store"].require_ids(decision.evidence_ids)
        elif decision.evidence_ids:
            raise ValueError("NEITHER decision cannot cite candidate-specific deciding evidence")

        return FinalReport(
            candidate_a=a["report"], candidate_b=b["report"], comparative_decision=decision
        )
