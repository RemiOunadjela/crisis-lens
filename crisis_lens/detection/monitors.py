"""Stream monitors that feed text into the detection pipeline.

Monitors are the ingestion layer -- they read from various sources
(files, APIs, streaming endpoints) and yield text records for processing.
The design supports both batch backfill and real-time monitoring, which
maps to how T&S teams actually operate: reviewing historical data during
investigations while simultaneously watching live feeds.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from crisis_lens.config import DetectionConfig
from crisis_lens.detection.anomaly import VolumeAnomalyDetector
from crisis_lens.detection.rules import RuleEngine, Signal


class TextRecord(BaseModel):
    """A single text record from any source."""

    text: str
    source: str = "unknown"
    language: str = "en"
    timestamp: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class SignalMonitor:
    """Applies detection rules to incoming text records with batching and dedup."""

    def __init__(
        self,
        engine: RuleEngine | None = None,
        config: DetectionConfig | None = None,
    ):
        self.engine = engine or RuleEngine()
        self.config = config or DetectionConfig()
        self._anomaly_detector = VolumeAnomalyDetector()

    def process_record(self, record: TextRecord) -> Signal | None:
        if record.language not in self.config.languages:
            return None

        signal = self.engine.evaluate(
            text=record.text,
            language=record.language,
            source=record.source,
        )

        if signal is None:
            return None

        if signal.score < self.config.min_confidence:
            return None

        signal.metadata.update(record.metadata)

        if record.timestamp > 0:
            anomaly = self._anomaly_detector.observe(record.timestamp)
            if anomaly:
                signal.metadata["volume_anomaly"] = anomaly
                # Boost score for anomalous volume periods
                signal.score = min(signal.score * 1.15, 1.0)

        return signal

    def process_batch(self, records: list[TextRecord]) -> list[Signal]:
        signals: list[Signal] = []
        for record in records:
            sig = self.process_record(record)
            if sig is not None:
                signals.append(sig)
        return signals


class StreamMonitor:
    """Async stream processor for real-time monitoring.

    Reads from file-based sources (JSONL) with support for tailing.
    In production, this would be backed by Kafka or Pub/Sub, but file-based
    input keeps the tool accessible for evaluation and demos.
    """

    def __init__(
        self,
        signal_monitor: SignalMonitor | None = None,
        batch_size: int = 64,
    ):
        self.signal_monitor = signal_monitor or SignalMonitor()
        self.batch_size = batch_size

    async def read_jsonl(self, path: str | Path) -> AsyncIterator[TextRecord]:
        """Read records from a JSONL file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    yield TextRecord(**data)
                except (json.JSONDecodeError, ValueError):
                    continue

    async def monitor_file(self, path: str | Path) -> AsyncIterator[Signal]:
        """Process a JSONL file and yield signals."""
        batch: list[TextRecord] = []

        async for record in self.read_jsonl(path):
            batch.append(record)
            if len(batch) >= self.batch_size:
                for signal in self.signal_monitor.process_batch(batch):
                    yield signal
                batch = []
                await asyncio.sleep(0)  # yield control

        if batch:
            for signal in self.signal_monitor.process_batch(batch):
                yield signal

    async def monitor_records(self, records: list[TextRecord]) -> list[Signal]:
        """Process a list of records and return all signals."""
        return self.signal_monitor.process_batch(records)
