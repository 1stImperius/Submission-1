from __future__ import annotations

from enum import Enum
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

CandidateId = Literal["CAND_A", "CAND_B"]
AgentRole = Literal["Technical", "HRCulture", "HiringManager", "Skeptic"]
Recommendation = Literal["strong_hire", "hire", "lean_no_hire", "no_hire"]
AssessmentStatus = Literal["ESTABLISHED", "NOT_ESTABLISHED"]


class EvidenceItem(BaseModel):
    model_config = ConfigDict(frozen=True)
    evidence_id: str
    candidate_id: CandidateId
    source: Literal["resume", "transcript"]
    quote: str = Field(min_length=1)


class CandidateInput(BaseModel):
    candidate_id: CandidateId
    name: str
    resume: str
    transcript: str


class CandidateFact(BaseModel):
    fact_id: str
    candidate_id: CandidateId
    key: str
    value: str
    evidence_ids: list[str] = Field(min_length=1)


class CandidateProfile(BaseModel):
    candidate_id: CandidateId
    facts: list[CandidateFact]
    evidence_ids: list[str]


class JDRequirement(BaseModel):
    requirement_id: str
    criterion: str
    description: str
    mandatory: bool = False
    evidence_expected: bool = True


class JobProfile(BaseModel):
    requirements: list[JDRequirement]


class CriterionAssessment(BaseModel):
    criterion: str
    status: AssessmentStatus
    score: float | None = Field(default=None, ge=0.0, le=10.0)
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def enforce_state(self):
        if self.status == "ESTABLISHED" and self.score is None:
            raise ValueError("ESTABLISHED assessment requires a score")
        if self.status == "NOT_ESTABLISHED" and self.score is not None:
            raise ValueError("NOT_ESTABLISHED assessment cannot have a score")
        if self.status == "ESTABLISHED" and not self.evidence_ids:
            raise ValueError("ESTABLISHED assessment requires evidence_ids")
        return self


class EvaluationOutput(BaseModel):
    agent_role: AgentRole
    recommendation: Recommendation
    confidence: float = Field(ge=0.0, le=1.0)
    criterion_assessments: list[CriterionAssessment]
    rationale: str = Field(min_length=1)
    cited_evidence_ids: list[str] = Field(min_length=1)


class RevisionDelta(BaseModel):
    recommendation_changed: bool
    confidence_delta: float
    changed_criteria: list[str]
    score_deltas: dict[str, float]


class RevisionOutput(EvaluationOutput):
    initial_recommendation: Recommendation
    changed: bool
    revision_reason: str = Field(min_length=1)
    delta: RevisionDelta

    @model_validator(mode="after")
    def validate_change_flag(self):
        expected = self.initial_recommendation != self.recommendation
        if self.changed != expected:
            raise ValueError("changed must equal whether the recommendation changed")
        if self.delta.recommendation_changed != expected:
            raise ValueError("delta.recommendation_changed is inconsistent")
        return self


class DebateTurn(BaseModel):
    turn_id: str
    candidate_id: CandidateId
    challenger: AgentRole
    defender: AgentRole
    round_no: int = Field(ge=1)
    challenge: str = Field(min_length=1)
    response: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class Disagreement(BaseModel):
    disagreement_id: str
    candidate_id: CandidateId
    criterion: str
    agents: list[AgentRole]
    description: str
    evidence_ids: list[str]
    resolved: bool = False


class ReportPoint(BaseModel):
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class CandidateFinalReport(BaseModel):
    candidate_id: CandidateId
    name: str
    eligible: bool
    final_recommendation: Recommendation
    confidence: float = Field(ge=0.0, le=1.0)
    strengths: list[ReportPoint]
    concerns: list[ReportPoint]
    unresolved_disagreements: list[Disagreement]
    final_score: float


class ComparativeDecision(BaseModel):
    selected_candidate_id: Literal["CAND_A", "CAND_B", "NEITHER"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class FinalReport(BaseModel):
    candidate_a: CandidateFinalReport
    candidate_b: CandidateFinalReport
    comparative_decision: ComparativeDecision
