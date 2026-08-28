from __future__ import annotations

from .models import CandidateId, CandidateFinalReport, EvaluationOutput, JDRequirement, Recommendation, Disagreement, ReportPoint, RevisionOutput
from .evidence import EvidenceStore


class GateEngine:
    """Mandatory gates are requirement-driven, not hard-coded to arbitrary criteria."""
    @staticmethod
    def evaluate(evaluations: list[RevisionOutput], requirements: list[JDRequirement]) -> tuple[bool, list[str]]:
        mandatory = {r.criterion for r in requirements if r.mandatory}
        failures: list[str] = []
        for criterion in mandatory:
            assessments = [a for e in evaluations for a in e.criterion_assessments if a.criterion == criterion]
            if not assessments:
                failures.append(f"Mandatory criterion '{criterion}' was not assessed")
                continue
            # A mandatory criterion is satisfied only when at least one authoritative assessment establishes it.
            established = [a for a in assessments if a.status == "ESTABLISHED" and a.score is not None]
            if not established:
                failures.append(f"Mandatory criterion '{criterion}' is not established")
                continue
            if max(a.score for a in established) < 5.0:
                failures.append(f"Mandatory criterion '{criterion}' below threshold 5.0")
        return not failures, failures


class ScoreEngine:
    ROLE_WEIGHTS = {"Technical": .35, "HiringManager": .30, "Skeptic": .20, "HRCulture": .15}

    @classmethod
    def calculate(cls, evaluations: list[RevisionOutput]) -> float:
        weighted = 0.0
        total_weight = 0.0
        for e in evaluations:
            scores = [a.score for a in e.criterion_assessments if a.status == "ESTABLISHED" and a.score is not None]
            if not scores:
                continue
            avg = sum(scores) / len(scores)
            w = cls.ROLE_WEIGHTS[e.agent_role]
            weighted += avg * w
            total_weight += w
        return round(weighted / total_weight, 2) if total_weight else 0.0


class ReportEngine:
    @staticmethod
    def build_candidate_report(candidate_id: CandidateId, name: str, eligible: bool, evaluations: list[RevisionOutput], score: float, disagreements: list[Disagreement]) -> CandidateFinalReport:
        ordered = sorted(evaluations, key=lambda e: e.confidence, reverse=True)
        final = ordered[0]
        strengths: list[ReportPoint] = []
        concerns: list[ReportPoint] = []
        for e in evaluations:
            for a in e.criterion_assessments:
                if a.status != "ESTABLISHED" or a.score is None:
                    continue
                point = ReportPoint(text=f"{a.criterion}: {a.rationale}", evidence_ids=a.evidence_ids)
                (strengths if a.score >= 7 else concerns if a.score <= 4 else strengths).append(point)
        return CandidateFinalReport(
            candidate_id=candidate_id, name=name, eligible=eligible,
            final_recommendation=final.recommendation,
            confidence=round(sum(e.confidence for e in evaluations) / len(evaluations), 2),
            strengths=strengths[:8], concerns=concerns[:8], unresolved_disagreements=disagreements,
            final_score=score,
        )
