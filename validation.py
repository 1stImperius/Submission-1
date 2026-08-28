from __future__ import annotations

from .evidence import EvidenceStore
from .models import EvaluationOutput, RevisionOutput
from .personas import PersonaConfig


class PythonEnforcer:
    @staticmethod
    def validate_evaluation(output: EvaluationOutput, config: PersonaConfig, store: EvidenceStore) -> None:
        if output.agent_role != config.role:
            raise ValueError(f"Agent identity mismatch: expected {config.role}, got {output.agent_role}")
        for assessment in output.criterion_assessments:
            if assessment.criterion not in config.dimensions:
                raise ValueError(
                    f"Boundary violation: {config.role} cannot score {assessment.criterion}"
                )
            if assessment.status == "ESTABLISHED":
                store.require_ids(assessment.evidence_ids)
            elif assessment.evidence_ids and not store.validate_ids(assessment.evidence_ids):
                raise ValueError("Invalid evidence IDs on NOT_ESTABLISHED assessment")
        if output.cited_evidence_ids and not store.validate_ids(output.cited_evidence_ids):
            raise ValueError("Invalid top-level evidence IDs")

    @staticmethod
    def validate_revision(output: RevisionOutput, config: PersonaConfig, store: EvidenceStore) -> None:
        PythonEnforcer.validate_evaluation(output, config, store)
        score_map = {a.criterion: a.score for a in output.criterion_assessments if a.score is not None}
        if set(output.delta.score_deltas) != set(output.delta.changed_criteria):
            raise ValueError("Revision delta criteria and score_deltas disagree")
        for criterion, delta in output.delta.score_deltas.items():
            if criterion not in score_map:
                raise ValueError(f"Revision delta references absent criterion: {criterion}")
