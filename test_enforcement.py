import pytest
from pydantic import ValidationError
from app.models import CriterionAssessment, EvaluationOutput
from app.personas import PERSONAS
from app.evidence import EvidenceStore
from app.models import EvidenceItem
from app.validation import PythonEnforcer


def store():
    s = EvidenceStore("CAND_A")
    s.add(EvidenceItem(evidence_id="E1", candidate_id="CAND_A", source="resume", quote="Python backend"))
    return s


def test_out_of_range_score_rejected():
    with pytest.raises(ValidationError):
        CriterionAssessment(criterion="python_backend", status="ESTABLISHED", score=11, evidence_ids=["E1"], rationale="x")


def test_insufficient_evidence_cannot_be_scored():
    with pytest.raises(ValidationError):
        CriterionAssessment(criterion="python_backend", status="NOT_ESTABLISHED", score=4, evidence_ids=[], rationale="unknown")


def test_persona_boundary_rejected():
    out = EvaluationOutput(
        agent_role="HRCulture", recommendation="hire", confidence=.8,
        criterion_assessments=[CriterionAssessment(criterion="python_backend", status="ESTABLISHED", score=8, evidence_ids=["E1"], rationale="x")],
        rationale="x", cited_evidence_ids=["E1"],
    )
    with pytest.raises(ValueError, match="Boundary violation"):
        PythonEnforcer.validate_evaluation(out, PERSONAS["HRCulture"], store())


def test_cross_candidate_evidence_rejected():
    s = store()
    out = EvaluationOutput(
        agent_role="Technical", recommendation="hire", confidence=.8,
        criterion_assessments=[CriterionAssessment(criterion="python_backend", status="ESTABLISHED", score=8, evidence_ids=["FAKE"], rationale="x")],
        rationale="x", cited_evidence_ids=["E1"],
    )
    with pytest.raises(ValueError):
        PythonEnforcer.validate_evaluation(out, PERSONAS["Technical"], s)
