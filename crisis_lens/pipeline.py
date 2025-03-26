"""Configurable crisis detection pipeline.

Orchestrates the full flow: ingest -> detect -> classify -> triage -> report.

The pipeline is async-native to handle high-throughput text streams without
blocking. In production deployments at scale, this would sit behind a
message queue (Kafka, SQS), but the async design means it can saturate
network I/O on classification API calls without a queue layer for
moderate-volume use cases.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from crisis_lens.classification.classifier import ClassificationResult, CrisisClassifier
from crisis_lens.config import PipelineConfig, Severity
from crisis_lens.detection.monitors import SignalMonitor, StreamMonitor, TextRecord
from crisis_lens.detection.rules import RuleEngine, Signal
from crisis_lens.reports.sitrep import SitRep, SitRepGenerator

logger = logging.getLogger("crisis_lens.pipeline")


class PipelineResult:
    """Collects pipeline output across all stages."""

    def __init__(self) -> None:
        self.signals: list[Signal] = []
        self.classifications: list[ClassificationResult] = []
        self.sitreps: list[SitRep] = []
        self.errors: list[str] = []

    @property
    def signal_count(self) -> int:
        return len(self.signals)

    @property
    def escalation_count(self) -> int:
        return sum(
            1 for s in self.signals
            if s.severity in (Severity.P0, Severity.P1)
        )

    def summary(self) -> dict[str, Any]:
        return {
            "signals_detected": self.signal_count,
            "classifications": len(self.classifications),
            "sitreps_generated": len(self.sitreps),
            "escalations": self.escalation_count,
            "errors": len(self.errors),
        }


class CrisisDetectionPipeline:
    """End-to-end crisis detection pipeline with async processing."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        classifier: CrisisClassifier | None = None,
    ):
        self.config = config or PipelineConfig.default()
        self.engine = RuleEngine(
            thresholds=self.config.detection.thresholds,
        )
        self.monitor = SignalMonitor(
            engine=self.engine,
            config=self.config.detection,
        )
        self.stream_monitor = StreamMonitor(
            signal_monitor=self.monitor,
            batch_size=self.config.detection.batch_size,
        )
        self.classifier = classifier
        self.sitrep_generator = SitRepGenerator(
            max_signals=self.config.reports.max_signals_per_report,
        )

    async def process_records(self, records: list[TextRecord]) -> PipelineResult:
        """Process a batch of text records through the full pipeline."""
        result = PipelineResult()

        # Detection stage
        signals = self.monitor.process_batch(records)
        result.signals = signals
        logger.info(f"Detection: {len(signals)} signals from {len(records)} records")

        if not signals:
            return result

        # Classification stage (if classifier is configured)
        if self.classifier:
            try:
                classifications = await self.classifier.classify_batch(signals)
                result.classifications = classifications
                logger.info(f"Classification: {len(classifications)} results")
            except Exception as e:
                error_msg = f"Classification failed: {e}"
                result.errors.append(error_msg)
                logger.error(error_msg)

        # Report generation for high-severity signals
        high_sev = [s for s in signals if s.severity in (Severity.P0, Severity.P1, Severity.P2)]
        if high_sev:
            sitrep = self.sitrep_generator.generate(
                incident_id=f"INC-{high_sev[0].signal_id}",
                signals=high_sev,
                classifications=[
                    c for c in result.classifications
                    if c.signal_id in {s.signal_id for s in high_sev}
                ],
            )
            result.sitreps.append(sitrep)

        return result

    async def process_file(self, path: str) -> PipelineResult:
        """Process a JSONL file through the pipeline."""
        result = PipelineResult()
        batch: list[TextRecord] = []
        batch_size = self.config.detection.batch_size

        async for record in self.stream_monitor.read_jsonl(path):
            batch.append(record)

            if len(batch) >= batch_size:
                batch_result = await self.process_records(batch)
                result.signals.extend(batch_result.signals)
                result.classifications.extend(batch_result.classifications)
                result.sitreps.extend(batch_result.sitreps)
                result.errors.extend(batch_result.errors)
                batch = []

        # Process remaining
        if batch:
            batch_result = await self.process_records(batch)
            result.signals.extend(batch_result.signals)
            result.classifications.extend(batch_result.classifications)
            result.sitreps.extend(batch_result.sitreps)
            result.errors.extend(batch_result.errors)

        return result

    async def stream_signals(self, path: str) -> AsyncIterator[Signal]:
        """Stream signals from a file for real-time monitoring UIs."""
        async for signal in self.stream_monitor.monitor_file(path):
            yield signal
