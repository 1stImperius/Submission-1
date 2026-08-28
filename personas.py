from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .models import AgentRole


@dataclass(frozen=True)
class PersonaConfig:
    role: AgentRole
    dimensions: tuple[str, ...]
    system_prompt: str


PERSONAS = {
    "Technical": PersonaConfig(
        role="Technical",
        dimensions=("python_backend", "ai_llm_systems", "rag_vector_search", "architecture_reliability", "frontend_basics"),
        system_prompt=(
            "You are the Technical Agent. Evaluate only technical depth relevant to the JD: Python/backend, "
            "AI/LLM systems, RAG/vector search, architecture/reliability, and basic frontend where relevant. "
            "Do not score culture, motivation, or general hiring fit. Never score an unestablished criterion."
        ),
    ),
    "HRCulture": PersonaConfig(
        role="HRCulture",
        dimensions=("communication", "honesty_transparency", "ownership", "teamwork", "motivation_stability"),
        system_prompt=(
            "You are the HR/Culture Agent. Evaluate only communication, honesty/transparency, ownership, teamwork, "
            "motivation, and stability. Do not score technical architecture depth. Pay special attention to admissions "
            "and accountability in the interview. Never score an unestablished criterion."
        ),
    ),
    "HiringManager": PersonaConfig(
        role="HiringManager",
        dimensions=("job_delivery", "domain_relevance", "ramp_up", "business_impact", "production_ownership"),
        system_prompt=(
            "You are the Hiring Manager Agent. Evaluate whether the candidate can deliver in this exact role, "
            "including production ownership, freight/logistics relevance, ramp-up, and business impact. "
            "Do not score technical subskills outside your delivery lens. Never score an unestablished criterion."
        ),
    ),
    "Skeptic": PersonaConfig(
        role="Skeptic",
        dimensions=("claim_veracity", "resume_interview_consistency", "risk", "measurement_quality", "unknowns"),
        system_prompt=(
            "You are the Skeptic Agent. Search for contradictions, inflated claims, weak measurement, hidden risk, "
            "and important unknowns. You may identify risks even when another agent is positive, but you must cite "
            "direct evidence. Never invent a fact or score an unestablished criterion."
        ),
    ),
}
