"""
Schemas for JEV Router inputs, agent routing targets, and decisions.
"""

from typing import Optional, Dict
from pydantic import BaseModel, Field, model_validator, StrictBool, StrictInt, StrictStr

from backend.jev_router.config import ModelLevel, ROUTER_VERSION


# ============================================================================
# JEV Input Schemas
# ============================================================================

class DocumentNatureBlock(BaseModel):
    page_count: StrictInt = Field(..., ge=0, description="Total number of pages")
    word_count: StrictInt = Field(..., ge=0, description="Total word count")
    document_nature: StrictStr = Field(..., min_length=1, description="e.g. born-digital, scanned, mixed")


class StructureBlock(BaseModel):
    section_count: StrictInt = Field(..., ge=0, description="Number of detected sections")
    max_section_depth: StrictInt = Field(..., ge=0, description="Maximum section hierarchy depth")
    has_methodology: StrictBool = Field(..., description="Whether methodology section exists")
    has_experimental_results: StrictBool = Field(..., description="Whether experimental results exist")
    has_references: StrictBool = Field(..., description="Whether references exist")


class VisualContentBlock(BaseModel):
    figure_count: StrictInt = Field(..., ge=0, description="Confirmed figure count")
    table_count: StrictInt = Field(..., ge=0, description="Confirmed table count")
    has_visual_content: StrictBool = Field(..., description="Flag indicating presence of visual content")


class MathematicalContentBlock(BaseModel):
    equation_count: StrictInt = Field(..., ge=0, description="Confirmed equation count")
    has_mathematical_content: StrictBool = Field(..., description="Flag indicating mathematical content")


class ResearchCharacteristicsBlock(BaseModel):
    has_methodology: StrictBool = Field(..., description="Methodology characteristic")
    has_experimental_results: StrictBool = Field(..., description="Experimental results characteristic")
    has_tables: StrictBool = Field(..., description="Presence of tables characteristic")
    has_references: StrictBool = Field(..., description="Presence of references characteristic")


class JevInputState(BaseModel):
    """
    Compact, routing-relevant view extracted from Document Profile.
    """
    document: DocumentNatureBlock
    structure: StructureBlock
    visual_content: VisualContentBlock
    mathematical_content: MathematicalContentBlock
    research_characteristics: ResearchCharacteristicsBlock


# ============================================================================
# Agent Target Routing State Schemas
# ============================================================================

class AnalysisAgentState(BaseModel):
    """
    Analysis agent configuration.
    Analysis Agent is ALWAYS enabled. JEV does not decide whether it exists.
    """
    enabled: StrictBool = Field(True, description="Analysis agent must always be enabled")
    level: ModelLevel = Field(..., description="Capability tier: basic, medium, or advanced")
    probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence/probability of selected level")
    level_probabilities: Optional[Dict[str, float]] = Field(None, description="Probability distribution across levels")

    @model_validator(mode="after")
    def validate_analysis_always_enabled(self) -> "AnalysisAgentState":
        if not self.enabled:
            raise ValueError("analysis.enabled must always be True.")
        return self


class OptionalAgentState(BaseModel):
    """
    Configuration for conditionally enabled agents (RAG, Vision).
    If enabled is True, level must be a valid ModelLevel.
    If enabled is False, level may be None.
    """
    enabled: StrictBool = Field(..., description="Whether the agent is enabled")
    level: Optional[ModelLevel] = Field(None, description="Capability tier if enabled")
    probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Raw probability of enabled decision")
    level_probabilities: Optional[Dict[str, float]] = Field(None, description="Probability distribution across levels")

    @model_validator(mode="after")
    def validate_agent_level(self) -> "OptionalAgentState":
        if self.enabled and self.level is None:
            raise ValueError("Level must be specified (basic, medium, advanced) when agent is enabled.")
        return self


class AgentConfidence(BaseModel):
    """
    Optional confidence scores returned by JEV.
    Scores must be probabilities between 0.0 and 1.0.
    """
    rag: Optional[float] = Field(None, ge=0.0, le=1.0, description="RAG routing confidence")
    vision: Optional[float] = Field(None, ge=0.0, le=1.0, description="Vision routing confidence")
    analysis: Optional[float] = Field(None, ge=0.0, le=1.0, description="Analysis routing confidence")


class RoutingDecisionState(BaseModel):
    """
    Full representation of the routing decision contract.
    """
    router_version: StrictStr = Field(default=ROUTER_VERSION, description="Router contract version")
    analysis: AnalysisAgentState = Field(..., description="Analysis agent decision")
    rag: OptionalAgentState = Field(..., description="RAG agent decision")
    vision: OptionalAgentState = Field(..., description="Vision agent decision")
    confidence: Optional[AgentConfidence] = Field(None, description="Optional JEV confidence probabilities")

    @model_validator(mode="after")
    def validate_decision_rules(self) -> "RoutingDecisionState":
        if not self.router_version.strip():
            raise ValueError("router_version must not be empty.")
        if not self.analysis.enabled:
            raise ValueError("analysis.enabled must always be True.")
        if self.rag.enabled and self.rag.level is None:
            raise ValueError("rag.level must be one of [basic, medium, advanced] when rag is enabled.")
        if self.vision.enabled and self.vision.level is None:
            raise ValueError("vision.level must be one of [basic, medium, advanced] when vision is enabled.")
        return self

