"""Configuration management for crisis-lens pipelines.

Handles loading, validation, and merging of YAML-based pipeline configs.
Designed for T&S operational environments where misconfiguration can mean
missed escalations.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class Severity(str, Enum):
    """Incident severity levels following standard T&S escalation tiers."""

    P0 = "P0"  # Platform-wide, immediate executive escalation
    P1 = "P1"  # Regional or high-profile, senior on-call
    P2 = "P2"  # Significant but contained, standard escalation
    P3 = "P3"  # Low-volume emerging signal, monitoring
    P4 = "P4"  # Informational, trend tracking only


class IncidentType(str, Enum):
    POLITICAL_UNREST = "political_unrest"
    NATURAL_DISASTER = "natural_disaster"
    PLATFORM_MANIPULATION = "platform_manipulation"
    COORDINATED_HARASSMENT = "coordinated_harassment"
    CHILD_SAFETY = "child_safety"
    VIOLENT_EXTREMISM = "violent_extremism"
    SELF_HARM = "self_harm"
    MISINFORMATION = "misinformation"
    DATA_BREACH = "data_breach"
    REGULATORY = "regulatory"
    OTHER = "other"


class SeverityThresholds(BaseModel):
    """Score boundaries for severity assignment.

    Thresholds are inclusive lower bounds -- a score of 0.9 with p0_threshold=0.9
    maps to P0.
    """

    p0_threshold: float = 0.95
    p1_threshold: float = 0.85
    p2_threshold: float = 0.70
    p3_threshold: float = 0.50

    @field_validator("p0_threshold", "p1_threshold", "p2_threshold", "p3_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Threshold must be between 0 and 1")
        return v

    def assign_severity(self, score: float) -> Severity:
        if score >= self.p0_threshold:
            return Severity.P0
        if score >= self.p1_threshold:
            return Severity.P1
        if score >= self.p2_threshold:
            return Severity.P2
        if score >= self.p3_threshold:
            return Severity.P3
        return Severity.P4


class DetectionConfig(BaseModel):
    enabled: bool = True
    languages: list[str] = Field(default_factory=lambda: ["en", "es", "pt"])
    batch_size: int = 64
    dedup_window_seconds: int = 300
    min_confidence: float = 0.3
    thresholds: SeverityThresholds = Field(default_factory=SeverityThresholds)


class LLMProviderConfig(BaseModel):
    provider: str = "openai"  # "openai" | "huggingface"
    model: str = "gpt-4o-mini"
    api_base: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout_seconds: int = 30


class ClassificationConfig(BaseModel):
    llm: LLMProviderConfig = Field(default_factory=LLMProviderConfig)
    confidence_threshold: float = 0.6
    max_labels: int = 3
    boosted_prompt_iterations: int = 3
    golden_set_path: str | None = None


class ReportConfig(BaseModel):
    format: str = "markdown"
    include_raw_signals: bool = False
    max_signals_per_report: int = 50
    handover_lookback_hours: int = 8


class PipelineConfig(BaseModel):
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    reports: ReportConfig = Field(default_factory=ReportConfig)
    max_concurrent_tasks: int = 16
    pipeline_timeout_seconds: int = 300

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return cls(**raw)

    @classmethod
    def default(cls) -> PipelineConfig:
        return cls()

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
