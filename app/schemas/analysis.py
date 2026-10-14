"""Schemas for newsletter analysis: executive summary, key insights,
actionable items, and extracted entities.

Every factual claim distinguishes `source_type`:
  - "explicit": directly stated in the document text.
  - "inferred": an AI-generated interpretation/recommendation that is NOT
    literally present in the text.
This distinction is enforced by the prompt (see `app/llm/prompts.py`) and
validated here. Actionable items additionally forbid inventing an owner or
deadline that is not literally present in the source chunks.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import Citation


class SourceType(str, Enum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"


class InsightCategory(str, Enum):
    FINDING = "finding"
    STATISTIC = "statistic"
    TREND = "trend"
    ANNOUNCEMENT = "announcement"
    OBSERVATION = "observation"


class KeyInsight(BaseModel):
    text: str
    category: InsightCategory
    source_type: SourceType
    citations: list[Citation] = Field(default_factory=list)


class ActionItem(BaseModel):
    action: str
    deadline: str | None = None
    owner: str | None = None
    # Actionable items must always be grounded in explicit document text;
    # the LLM is instructed never to fabricate an action that isn't stated.
    source_type: SourceType = SourceType.EXPLICIT
    citations: list[Citation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _forbid_placeholder_owner_or_deadline(self) -> "ActionItem":
        placeholders = {"unknown", "n/a", "not specified", "tbd", ""}
        if self.owner is not None and self.owner.strip().lower() in placeholders:
            self.owner = None
        if self.deadline is not None and self.deadline.strip().lower() in placeholders:
            self.deadline = None
        return self


class ExtractedEntities(BaseModel):
    dates: list[str] = Field(default_factory=list)
    organizations: list[str] = Field(default_factory=list)
    products: list[str] = Field(default_factory=list)
    people: list[str] = Field(default_factory=list)
    numerical_facts: list[str] = Field(default_factory=list)


class ExecutiveSummary(BaseModel):
    summary: str
    citations: list[Citation] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    document_id: str
    filename: str
    executive_summary: ExecutiveSummary
    key_insights: list[KeyInsight] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)


class AnalysisRequest(BaseModel):
    # Allows re-running analysis with a different top_k without changing
    # the stored document.
    top_k: int | None = Field(default=None, ge=1, le=50)
