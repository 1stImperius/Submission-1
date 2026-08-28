from __future__ import annotations

from .agents import Agent
from .evidence import EvidenceStore
from .models import AgentRole, DebateTurn, Disagreement, EvaluationOutput

RING: tuple[tuple[AgentRole, AgentRole], ...] = (
    ("Technical", "Skeptic"),
    ("Skeptic", "HiringManager"),
    ("HiringManager", "HRCulture"),
    ("HRCulture", "Technical"),
)


def _all_evidence(e: EvaluationOutput) -> list[str]:
    ids = list(e.cited_evidence_ids)
    for a in e.criterion_assessments:
        ids.extend(a.evidence_ids)
    return sorted(set(ids))


def detect_adaptive_conflicts(evals: list[EvaluationOutput]) -> list[Disagreement]:
    by_criterion: dict[str, list[tuple[AgentRole, float, list[str]]]] = {}
    for e in evals:
        for a in e.criterion_assessments:
            if a.status == "ESTABLISHED" and a.score is not None:
                by_criterion.setdefault(a.criterion, []).append((e.agent_role, a.score, a.evidence_ids))
    out: list[Disagreement] = []
    n = 1
    for criterion, values in by_criterion.items():
        if len(values) >= 2:
            lo = min(values, key=lambda x: x[1])
            hi = max(values, key=lambda x: x[1])
            if hi[1] - lo[1] >= 3.0:
                out.append(Disagreement(
                    disagreement_id=f"D{n}",
                    candidate_id="CAND_A" if False else "CAND_A",  # overwritten by caller
                    criterion=criterion,
                    agents=[lo[0], hi[0]],
                    description=f"Score gap {lo[1]:.1f} vs {hi[1]:.1f} on {criterion}",
                    evidence_ids=sorted(set(lo[2] + hi[2])),
                ))
                n += 1
    return out


class DebateEngine:
    def __init__(self, agents: dict[AgentRole, Agent]):
        self.agents = agents

    async def run(self, candidate_id: str, initial: list[EvaluationOutput], store: EvidenceStore) -> tuple[list[DebateTurn], list[Disagreement]]:
        by_role = {e.agent_role: e for e in initial}
        turns: list[DebateTurn] = []

        # Fixed ring guarantees interaction on every run.
        for round_no, (challenger_role, defender_role) in enumerate(RING, start=1):
            challenger = by_role[challenger_role]
            evidence_ids = _all_evidence(challenger)
            challenge = await self.agents[challenger_role].debate_challenge(
                defender_role, challenger.rationale, evidence_ids, store
            )
            response = await self.agents[defender_role].debate_response(
                challenger_role, challenge, evidence_ids, store
            )
            turns.append(DebateTurn(
                turn_id=f"T{round_no}", candidate_id=candidate_id,
                challenger=challenger_role, defender=defender_role,
                round_no=round_no, challenge=challenge, response=response,
                evidence_ids=evidence_ids,
            ))

        # Adaptive extra edges ensure cross-cutting disagreements are not diluted by the ring.
        disagreements = detect_adaptive_conflicts(initial)
        disagreements = [d.model_copy(update={"candidate_id": candidate_id}) for d in disagreements]
        for idx, d in enumerate(disagreements, start=len(turns) + 1):
            a, b = d.agents
            challenge = await self.agents[a].debate_challenge(b, d.description, d.evidence_ids, store)
            response = await self.agents[b].debate_response(a, challenge, d.evidence_ids, store)
            turns.append(DebateTurn(
                turn_id=f"T{idx}", candidate_id=candidate_id,
                challenger=a, defender=b, round_no=idx,
                challenge=challenge, response=response,
                evidence_ids=d.evidence_ids,
            ))
        return turns, disagreements
