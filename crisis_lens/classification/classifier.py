"""Crisis classification engine.

Wraps LLM providers with structured prompting, confidence calibration,
and the boosted-prompt refinement loop. The classifier is the analytical
core -- detection finds signals, classification determines what they mean
and how urgently they need a response.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from crisis_lens.classification.prompts import (
    SYSTEM_PROMPT,
    build_boosted_prompt,
    build_classification_prompt,
)
from crisis_lens.classification.providers import LLMProvider, LLMResponse, create_provider
from crisis_lens.config import (
    ClassificationConfig,
    IncidentType,
    Severity,
)
from crisis_lens.detection.rules import Signal


class LabelScore(BaseModel):
    incident_type: IncidentType
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class ClassificationResult(BaseModel):
    signal_id: str
    labels: list[LabelScore] = Field(default_factory=list)
    severity: Severity = Severity.P4
    requires_human_review: bool = False
    escalation_note: str | None = None
    raw_response: dict[str, Any] = Field(default_factory=dict)

    @property
    def primary_type(self) -> IncidentType | None:
        if not self.labels:
            return None
        return max(self.labels, key=lambda lb: lb.confidence).incident_type

    @property
    def max_confidence(self) -> float:
        if not self.labels:
            return 0.0
        return max(lb.confidence for lb in self.labels)


class CrisisClassifier:
    """LLM-backed multi-label crisis classifier with boosted prompt support."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        config: ClassificationConfig | None = None,
    ):
        self.config = config or ClassificationConfig()
        self.provider = provider or create_provider(self.config.llm)
        self._failure_modes: list[str] = []
        self._calibration_notes: list[str] = []
        self._system_prompt = SYSTEM_PROMPT

    @property
    def system_prompt(self) -> str:
        if self._failure_modes or self._calibration_notes:
            return build_boosted_prompt(self._failure_modes, self._calibration_notes)
        return self._system_prompt

    def add_failure_mode(self, description: str) -> None:
        """Register a known failure pattern from golden set evaluation."""
        self._failure_modes.append(description)

    def add_calibration_note(self, note: str) -> None:
        """Add a calibration adjustment learned from validation."""
        self._calibration_notes.append(note)

    def reset_boost(self) -> None:
        self._failure_modes.clear()
        self._calibration_notes.clear()

    async def classify(self, signal: Signal) -> ClassificationResult:
        """Classify a single signal using the LLM provider."""
        user_prompt = build_classification_prompt(
            text=signal.text,
            source=signal.source,
            language=signal.language,
        )

        try:
            response: LLMResponse = await self.provider.complete(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
            )
        except Exception as e:
            return ClassificationResult(
                signal_id=signal.signal_id,
                requires_human_review=True,
                escalation_note=f"Classification failed: {e}",
                raw_response={"error": str(e)},
            )

        return self._parse_response(signal.signal_id, response)

    async def classify_batch(self, signals: list[Signal]) -> list[ClassificationResult]:
        """Classify multiple signals. Sequential for now; async batching planned."""
        results = []
        for signal in signals:
            result = await self.classify(signal)
            results.append(result)
        return results

    def classify_from_dict(
        self, signal_id: str, response_data: dict[str, Any]
    ) -> ClassificationResult:
        """Parse a pre-existing response dict into a ClassificationResult.

        Useful for offline evaluation and golden set comparison where
        we already have the LLM output.
        """
        return self._parse_response(signal_id, LLMResponse(response_data))

    def _parse_response(
        self, signal_id: str, response: LLMResponse
    ) -> ClassificationResult:
        if response.get("_parse_error"):
            return ClassificationResult(
                signal_id=signal_id,
                requires_human_review=True,
                escalation_note="Failed to parse LLM response as JSON",
                raw_response=dict(response),
            )

        labels: list[LabelScore] = []
        for label_data in response.get("labels", []):
            try:
                incident_type = IncidentType(label_data["type"])
                confidence = float(label_data.get("confidence", 0.0))
                confidence = max(0.0, min(1.0, confidence))

                if confidence >= self.config.confidence_threshold:
                    labels.append(
                        LabelScore(
                            incident_type=incident_type,
                            confidence=confidence,
                            reasoning=label_data.get("reasoning", ""),
                        )
                    )
            except (ValueError, KeyError):
                continue

        # Enforce max labels (keep highest confidence)
        labels.sort(key=lambda lb: lb.confidence, reverse=True)
        labels = labels[: self.config.max_labels]

        severity_str = response.get("overall_severity", "P4")
        try:
            severity = Severity(severity_str)
        except ValueError:
            severity = Severity.P4

        return ClassificationResult(
            signal_id=signal_id,
            labels=labels,
            severity=severity,
            requires_human_review=response.get("requires_human_review", False),
            escalation_note=response.get("escalation_note"),
            raw_response=dict(response),
        )
