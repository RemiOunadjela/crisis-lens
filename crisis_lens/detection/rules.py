"""Detection rules for crisis signal identification.

Rules are the first line of defense in the detection pipeline. They operate
on raw text and produce scored signals that feed into classification.

The layered approach (keyword -> pattern -> anomaly) reflects operational
reality: keyword rules catch known threats fast, pattern rules handle
evolving tactics, and anomaly detection surfaces unknown-unknowns.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from hashlib import sha256
from typing import Any

from pydantic import BaseModel, Field

from crisis_lens.config import IncidentType, Severity, SeverityThresholds


class Signal(BaseModel):
    """A detected crisis signal before classification."""

    signal_id: str
    text: str
    source: str = "unknown"
    language: str = "en"
    score: float = Field(ge=0.0, le=1.0)
    severity: Severity = Severity.P4
    matched_rules: list[str] = Field(default_factory=list)
    suggested_types: list[IncidentType] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    dedup_key: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.dedup_key:
            normalized = re.sub(r"\s+", " ", self.text.lower().strip())
            self.dedup_key = sha256(normalized.encode()).hexdigest()[:16]


class DetectionRule(ABC):
    """Base class for all detection rules."""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    @abstractmethod
    def evaluate(self, text: str, language: str = "en") -> float:
        """Return a score between 0 and 1 indicating match strength."""
        ...


class KeywordRule(DetectionRule):
    """Term-frequency based detection for known crisis vocabulary."""

    CRISIS_TERMS: dict[str, dict[str, list[str]]] = {
        "violence": {
            "en": ["shooting", "shooter", "gunfire", "attack", "bombing", "stabbing", "massacre", "assault"],
        },
        "natural_disaster": {
            "en": ["earthquake", "tsunami", "hurricane", "flood", "wildfire", "tornado"],
        },
        "platform_abuse": {
            "en": ["raid", "brigading", "bot network", "coordinated", "mass report", "doxxing"],
        },
        "child_safety": {
            "en": ["csam", "grooming", "minor", "underage", "exploitation"],
        },
        "self_harm": {
            "en": ["suicide", "self-harm", "cutting", "overdose", "end my life"],
        },
    }

    def __init__(
        self,
        name: str,
        categories: list[str] | None = None,
        custom_terms: dict[str, list[str]] | None = None,
        weight: float = 1.0,
    ):
        super().__init__(name, weight)
        self.categories = categories or list(self.CRISIS_TERMS.keys())
        self.custom_terms = custom_terms or {}

    def _get_terms(self, language: str) -> list[str]:
        terms: list[str] = []
        for cat in self.categories:
            cat_terms = self.CRISIS_TERMS.get(cat, {})
            terms.extend(cat_terms.get(language, cat_terms.get("en", [])))

        for lang, lang_terms in self.custom_terms.items():
            if lang == language:
                terms.extend(lang_terms)

        return terms

    def evaluate(self, text: str, language: str = "en") -> float:
        text_lower = text.lower()
        terms = self._get_terms(language)
        if not terms:
            return 0.0

        # Use word boundary matching to avoid substring false positives
        import re

        matches = sum(
            1 for t in terms if re.search(r"\b" + re.escape(t) + r"\b", text_lower)
        )
        # Normalize: 1 match = 0.4, 2 = 0.65, 3+ = 0.8+
        if matches == 0:
            return 0.0
        raw = min(matches / len(terms) * 5, 1.0)
        return min(0.3 + raw * 0.7, 1.0)


class PatternRule(DetectionRule):
    """Regex-based pattern matching for structured threat indicators.

    Useful for detecting coordinated activity patterns like timestamp
    clustering, repeated URLs, or known bad-actor naming conventions.
    """

    def __init__(self, name: str, patterns: list[str], weight: float = 1.0):
        super().__init__(name, weight)
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]

    def evaluate(self, text: str, language: str = "en") -> float:
        matches = sum(1 for p in self.patterns if p.search(text))
        if matches == 0:
            return 0.0
        return min(0.4 + (matches / len(self.patterns)) * 0.6, 1.0)


class RuleEngine:
    """Orchestrates evaluation across multiple detection rules.

    Combines scores using weighted averaging with a configurable ceiling.
    The engine also handles deduplication within a sliding window.
    """

    TYPE_MAPPING: dict[str, IncidentType] = {
        "violence": IncidentType.VIOLENT_EXTREMISM,
        "natural_disaster": IncidentType.NATURAL_DISASTER,
        "platform_abuse": IncidentType.PLATFORM_MANIPULATION,
        "child_safety": IncidentType.CHILD_SAFETY,
        "self_harm": IncidentType.SELF_HARM,
    }

    def __init__(
        self,
        rules: list[DetectionRule] | None = None,
        thresholds: SeverityThresholds | None = None,
    ):
        self.rules = rules or self._default_rules()
        self.thresholds = thresholds or SeverityThresholds()
        self._seen_keys: dict[str, float] = {}

    @staticmethod
    def _default_rules() -> list[DetectionRule]:
        return [
            KeywordRule("keyword_violence", categories=["violence"], weight=1.2),
            KeywordRule("keyword_disaster", categories=["natural_disaster"], weight=1.0),
            KeywordRule("keyword_platform", categories=["platform_abuse"], weight=1.1),
            KeywordRule("keyword_csam", categories=["child_safety"], weight=1.5),
            KeywordRule("keyword_self_harm", categories=["self_harm"], weight=1.3),
            PatternRule(
                "url_flood",
                patterns=[r"https?://\S+"] * 1,  # single pattern, scored by density
                weight=0.5,
            ),
        ]

    def evaluate(self, text: str, language: str = "en", source: str = "unknown") -> Signal | None:
        """Run all rules against text and produce a scored signal."""
        if not text.strip():
            return None

        total_weight = 0.0
        weighted_score = 0.0
        matched: list[str] = []
        suggested: list[IncidentType] = []

        for rule in self.rules:
            score = rule.evaluate(text, language)
            if score > 0:
                weighted_score += score * rule.weight
                total_weight += rule.weight
                matched.append(rule.name)

                # Map keyword rules to incident types
                if isinstance(rule, KeywordRule):
                    for cat in rule.categories:
                        t = self.TYPE_MAPPING.get(cat)
                        if t and t not in suggested:
                            suggested.append(t)

        if total_weight == 0:
            return None

        final_score = weighted_score / total_weight
        severity = self.thresholds.assign_severity(final_score)

        signal = Signal(
            signal_id="",
            text=text,
            source=source,
            language=language,
            score=final_score,
            severity=severity,
            matched_rules=matched,
            suggested_types=suggested,
        )

        # Dedup check
        if signal.dedup_key in self._seen_keys:
            prev_score = self._seen_keys[signal.dedup_key]
            if final_score <= prev_score:
                return None
        self._seen_keys[signal.dedup_key] = final_score
        signal.signal_id = f"SIG-{signal.dedup_key[:8].upper()}"

        return signal

    def clear_dedup_cache(self) -> None:
        self._seen_keys.clear()
