"""Tests for the classification subsystem."""

import pytest

from crisis_lens.classification.classifier import (
    ClassificationResult,
    CrisisClassifier,
    LabelScore,
)
from crisis_lens.classification.prompts import (
    build_boosted_prompt,
    build_classification_prompt,
)
from crisis_lens.classification.providers import LLMResponse
from crisis_lens.config import ClassificationConfig, IncidentType, Severity


class TestPrompts:
    def test_classification_prompt_contains_text(self):
        prompt = build_classification_prompt("Test crisis text", source="twitter")
        assert "Test crisis text" in prompt
        assert "twitter" in prompt

    def test_boosted_prompt_includes_failures(self):
        prompt = build_boosted_prompt(
            failure_modes=["Over-triggering on satire content"],
            calibration_notes=["Reduce confidence for humor-adjacent content"],
        )
        assert "Over-triggering on satire content" in prompt
        assert "Reduce confidence" in prompt

    def test_boosted_prompt_empty_failures(self):
        prompt = build_boosted_prompt([], [])
        assert "None identified yet" in prompt


class TestLabelScore:
    def test_valid_label(self):
        label = LabelScore(
            incident_type=IncidentType.VIOLENT_EXTREMISM,
            confidence=0.9,
            reasoning="Active threat indicators present",
        )
        assert label.confidence == 0.9

    def test_confidence_bounds(self):
        with pytest.raises(ValueError):
            LabelScore(incident_type=IncidentType.OTHER, confidence=1.5)


class TestClassificationResult:
    def test_primary_type(self):
        result = ClassificationResult(
            signal_id="SIG-001",
            labels=[
                LabelScore(incident_type=IncidentType.POLITICAL_UNREST, confidence=0.7),
                LabelScore(incident_type=IncidentType.MISINFORMATION, confidence=0.9),
            ],
            severity=Severity.P2,
        )
        assert result.primary_type == IncidentType.MISINFORMATION

    def test_primary_type_empty(self):
        result = ClassificationResult(signal_id="SIG-002")
        assert result.primary_type is None

    def test_max_confidence(self):
        result = ClassificationResult(
            signal_id="SIG-003",
            labels=[
                LabelScore(incident_type=IncidentType.NATURAL_DISASTER, confidence=0.6),
                LabelScore(incident_type=IncidentType.SELF_HARM, confidence=0.85),
            ],
        )
        assert result.max_confidence == 0.85


class TestCrisisClassifier:
    def test_parse_valid_response(self):
        config = ClassificationConfig(confidence_threshold=0.5)
        classifier = CrisisClassifier(config=config)

        response = LLMResponse({
            "labels": [
                {
                    "type": "violent_extremism",
                    "confidence": 0.92,
                    "reasoning": "Active threat language detected",
                },
                {
                    "type": "political_unrest",
                    "confidence": 0.65,
                    "reasoning": "Political context present",
                },
            ],
            "overall_severity": "P1",
            "requires_human_review": True,
            "escalation_note": "Possible active threat, route to senior on-call",
        })

        result = classifier._parse_response("SIG-TEST", response)
        assert result.severity == Severity.P1
        assert len(result.labels) == 2
        assert result.requires_human_review is True

    def test_parse_filters_low_confidence(self):
        config = ClassificationConfig(confidence_threshold=0.7)
        classifier = CrisisClassifier(config=config)

        response = LLMResponse({
            "labels": [
                {"type": "violent_extremism", "confidence": 0.92},
                {"type": "political_unrest", "confidence": 0.4},  # Below threshold
            ],
            "overall_severity": "P2",
        })

        result = classifier._parse_response("SIG-TEST", response)
        assert len(result.labels) == 1
        assert result.labels[0].incident_type == IncidentType.VIOLENT_EXTREMISM

    def test_parse_respects_max_labels(self):
        config = ClassificationConfig(max_labels=1)
        classifier = CrisisClassifier(config=config)

        response = LLMResponse({
            "labels": [
                {"type": "violent_extremism", "confidence": 0.9},
                {"type": "political_unrest", "confidence": 0.85},
                {"type": "misinformation", "confidence": 0.8},
            ],
            "overall_severity": "P1",
        })

        result = classifier._parse_response("SIG-TEST", response)
        assert len(result.labels) == 1

    def test_parse_error_response(self):
        classifier = CrisisClassifier()
        response = LLMResponse({"_raw_text": "invalid", "_parse_error": True})
        result = classifier._parse_response("SIG-ERR", response)
        assert result.requires_human_review is True

    def test_parse_invalid_incident_type(self):
        classifier = CrisisClassifier()
        response = LLMResponse({
            "labels": [
                {"type": "nonexistent_type", "confidence": 0.9},
                {"type": "violent_extremism", "confidence": 0.8},
            ],
            "overall_severity": "P2",
        })
        result = classifier._parse_response("SIG-TEST", response)
        # Should skip the invalid type and keep the valid one
        assert len(result.labels) == 1

    def test_boosted_prompt_integration(self):
        classifier = CrisisClassifier()
        classifier.add_failure_mode("False positives on satire about political events")
        classifier.add_calibration_note("Lower confidence for content with humor markers")

        prompt = classifier.system_prompt
        assert "satire" in prompt
        assert "humor markers" in prompt

    def test_reset_boost(self):
        classifier = CrisisClassifier()
        classifier.add_failure_mode("Some failure")
        classifier.reset_boost()
        # After reset, should use base prompt (no FAILURE MODES section)
        assert "FAILURE MODES" not in classifier.system_prompt

    def test_classify_from_dict(self):
        classifier = CrisisClassifier()
        result = classifier.classify_from_dict("SIG-001", {
            "labels": [
                {"type": "natural_disaster", "confidence": 0.88, "reasoning": "Earthquake signals"},
            ],
            "overall_severity": "P2",
            "requires_human_review": False,
        })
        assert result.signal_id == "SIG-001"
        assert result.primary_type == IncidentType.NATURAL_DISASTER
