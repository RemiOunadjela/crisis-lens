"""Tests for the pipeline orchestration layer."""

import json
import tempfile
from pathlib import Path

import pytest

from crisis_lens.config import PipelineConfig, Severity
from crisis_lens.detection.monitors import TextRecord
from crisis_lens.pipeline import CrisisDetectionPipeline, PipelineResult
from crisis_lens.reports.handover import HandoverGenerator, ShiftAction
from crisis_lens.reports.sitrep import SitRepGenerator


class TestPipelineResult:
    def test_empty_result(self):
        result = PipelineResult()
        assert result.signal_count == 0
        assert result.escalation_count == 0
        summary = result.summary()
        assert summary["signals_detected"] == 0

    def test_summary_structure(self):
        result = PipelineResult()
        summary = result.summary()
        assert "signals_detected" in summary
        assert "classifications" in summary
        assert "sitreps_generated" in summary
        assert "escalations" in summary
        assert "errors" in summary


class TestPipelineConfig:
    def test_default_config(self):
        config = PipelineConfig.default()
        assert config.detection.enabled is True
        assert "en" in config.detection.languages

    def test_yaml_roundtrip(self, tmp_path: Path):
        config = PipelineConfig.default()
        yaml_path = tmp_path / "test_config.yaml"
        config.to_yaml(yaml_path)
        loaded = PipelineConfig.from_yaml(yaml_path)
        assert loaded.detection.batch_size == config.detection.batch_size
        assert loaded.classification.llm.model == config.classification.llm.model

    def test_missing_config_file(self):
        with pytest.raises(FileNotFoundError):
            PipelineConfig.from_yaml("/nonexistent/path.yaml")


@pytest.mark.asyncio
async def test_pipeline_process_records():
    pipeline = CrisisDetectionPipeline()
    records = [
        TextRecord(text="Major earthquake strikes the capital, buildings collapsed", source="test"),
        TextRecord(text="Beautiful sunset over the mountains", source="test"),
        TextRecord(text="Active shooter reported at university campus", source="test"),
    ]
    result = await pipeline.process_records(records)
    assert result.signal_count >= 1


@pytest.mark.asyncio
async def test_pipeline_process_file():
    records = [
        {"text": "Bombing attack near the embassy", "source": "file_test", "language": "en"},
        {"text": "Regular news update about sports", "source": "file_test", "language": "en"},
        {"text": "Tsunami warning for coastal areas after earthquake", "source": "file_test", "language": "en"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        tmp_path = f.name

    try:
        pipeline = CrisisDetectionPipeline()
        result = await pipeline.process_file(tmp_path)
        assert result.signal_count >= 1
    finally:
        Path(tmp_path).unlink()


@pytest.mark.asyncio
async def test_pipeline_stream_signals():
    records = [
        {"text": "Active shooter and bombing attack at the mall, massacre reported", "source": "stream_test"},
        {"text": "Nice weather today", "source": "stream_test"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        tmp_path = f.name

    try:
        pipeline = CrisisDetectionPipeline()
        signals = []
        async for signal in pipeline.stream_signals(tmp_path):
            signals.append(signal)
        assert len(signals) >= 1
    finally:
        Path(tmp_path).unlink()


class TestSitRepGenerator:
    def test_generate_from_signals(self):
        from crisis_lens.config import IncidentType
        from crisis_lens.detection.rules import Signal

        signals = [
            Signal(
                signal_id="SIG-001",
                text="Earthquake detected in region X",
                score=0.8,
                severity=Severity.P2,
                suggested_types=[IncidentType.NATURAL_DISASTER],
                source="monitor",
                language="en",
            ),
            Signal(
                signal_id="SIG-002",
                text="Aftershock reported, tsunami warning",
                score=0.75,
                severity=Severity.P2,
                suggested_types=[IncidentType.NATURAL_DISASTER],
                source="monitor",
                language="en",
            ),
        ]

        gen = SitRepGenerator()
        sitrep = gen.generate(incident_id="INC-001", signals=signals)

        assert sitrep.incident_id == "INC-001"
        assert sitrep.severity == Severity.P2
        assert IncidentType.NATURAL_DISASTER in sitrep.incident_types
        assert sitrep.metrics.signal_count == 2

    def test_generate_empty_signals(self):
        gen = SitRepGenerator()
        sitrep = gen.generate(incident_id="INC-EMPTY", signals=[])
        assert sitrep.severity == Severity.P4

    def test_markdown_rendering(self):
        from crisis_lens.config import IncidentType
        from crisis_lens.detection.rules import Signal

        signals = [
            Signal(
                signal_id="SIG-001",
                text="Crisis event detected",
                score=0.9,
                severity=Severity.P1,
                suggested_types=[IncidentType.VIOLENT_EXTREMISM],
            ),
        ]
        gen = SitRepGenerator()
        sitrep = gen.generate(incident_id="INC-MD", signals=signals)
        md = sitrep.render_markdown(include_signals=True)
        assert "SITREP" in md
        assert "INC-MD" in md
        assert "SIG-001" in md

    def test_markdown_severity_descriptions(self):
        from crisis_lens.config import SEVERITY_DESCRIPTIONS, IncidentType
        from crisis_lens.detection.rules import Signal

        for severity, description in SEVERITY_DESCRIPTIONS.items():
            signal = Signal(
                signal_id="SIG-DESC",
                text="Test signal",
                score=0.9,
                severity=severity,
                suggested_types=[IncidentType.OTHER],
            )
            gen = SitRepGenerator()
            sitrep = gen.generate(incident_id="INC-DESC", signals=[signal])
            md = sitrep.render_markdown()
            assert f"**Severity:** {severity.value} —" in md
            assert description in md


class TestHandoverGenerator:
    def test_generate_handover(self):
        from crisis_lens.config import IncidentType
        from crisis_lens.detection.rules import Signal

        signals = [
            Signal(
                signal_id="SIG-H1",
                text="Ongoing incident",
                score=0.8,
                severity=Severity.P2,
                suggested_types=[IncidentType.POLITICAL_UNREST],
            ),
        ]

        gen_sitrep = SitRepGenerator()
        sitrep = gen_sitrep.generate(incident_id="INC-H1", signals=signals)

        gen = HandoverGenerator()
        handover = gen.generate(
            sitreps=[sitrep],
            resolved_ids=["INC-OLD"],
            actions=[
                ShiftAction(description="Escalated to policy", status="completed", owner="analyst-1"),
                ShiftAction(description="Waiting for legal review", status="pending", owner="legal"),
            ],
            analyst="analyst-1",
            notes="Relatively quiet shift. One developing situation in LATAM.",
        )

        assert len(handover.active_incidents) == 1
        assert "INC-OLD" in handover.resolved_incidents
        assert len(handover.actions_taken) == 1
        assert len(handover.pending_actions) == 1

    def test_handover_markdown(self):
        gen = HandoverGenerator()
        handover = gen.generate(sitreps=[], analyst="test-analyst")
        md = handover.render_markdown()
        assert "Shift Handover" in md
        assert "test-analyst" in md
